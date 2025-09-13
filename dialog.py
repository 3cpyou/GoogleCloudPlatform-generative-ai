import functions_framework
from google.cloud import firestore
from google.api_core.exceptions import GoogleAPIError
import google.generativeai as genai
import os
from datetime import datetime, timezone
import json

# Initialize Firestore and Google Generative AI
db = firestore.Client(project="ad3-sam", database="ad3gem-gmail-lineage")
chat_db = firestore.Client(project="ad3-sam", database="ad3gem-chat-history")

# Configure Google Generative AI
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    pro_model = genai.GenerativeModel("gemini-1.5-flash")
    flash_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    pro_model = None
    flash_model = None

@functions_framework.http
def conversation_bot_webhook(request):
    """Webhook for conversation bot: Searches Firestore emails and stores history"""

    # Parse request (e.g., from Dialogflow or chat platform)
    req = request.get_json()
    user_id = req.get('user_id', 'default_user')  # User ID for history
    user_input = req.get('text', '')  # Query, e.g., "mail from Julie"

    # Search Firestore with retries
    search_results = query_firestore_with_retries(user_input)

    # Store conversation history
    store_conversation_history(user_id, user_input, search_results)

    # Return response
    bot_response = f"Found {len(search_results)} results: {json.dumps(search_results, default=str)}"
    return {"response": bot_response}

def query_firestore_with_retries(user_input, max_retries=3):
    """Query Firestore with retries, using Pro for interpretation and Flash for assessment"""

    schema_info = """
    Firestore Schema from firestore-structure.json and gmail-firestore.py:
    Database: ad3gem-gmail-lineage
    Collections and Fields:
    1. emails_master
       - Description: Stores full raw email content and attachments; no indexes forcont cost efficiency.
       - Document ID: {messageId} (e.g., "
       6c1a0d9b4e5a2f")
       - Fields:
         - messageId: Unique email ID (string)
         - threadId: Thread ID (string)
         - owner: Mailbox owner email (string, e.g., "craig@lineagecoffee.com")
         - historyId: Gmail history ID (string)
         - payload: Full email content (object, includes mimeType, headers, body, parts)
         - parts: Array of MIME parts (objects with partId, mimeType, body, filename, attachmentId)
         - sizeEstimate: Email size in bytes (integer)
         - snippet: Short email preview (string)
         - labelIds: Gmail labels (array, e.g., ["INBOX", "UNREAD"])
         - attachments: Array of attachment details (objects with filename, mimeType, sizeBytes, gmailAttachmentId, typeNorm, isInline)
         - internalDate: Email sent timestamp (string)
         - ingestAt: Ingestion timestamp (ISO string)
         - updatedAt: Last update timestamp (ISO string)
       - Purpose: Full email storage; queried only for complete details.
       - Indexes: None (fieldOverrides exempt all fields).

    2. email_search_index
       - Description: Lightweight searchable metadata for fast queries.
       - Document ID: {messageId}
       - Fields:
         - owner: Mailbox owner (string)
         - messageId: Unique email ID (string)
         - threadId: Thread ID (string)
         - fromEmailLc: Normalized sender email (string, e.g., "julie@acme.com")
         - fromPerson: Sender name (string, e.g., "Julie")
         - fromFirstName: Extracted first name (string, e.g., "julie")
         - fromDomain: Sender domain (string, e.g., "acme.com")
         - toEmails: Recipient emails (array)
         - ccEmails: CC emails (array)
         - bccEmails: BCC emails (array)
         - allPeople: All participants (array, e.g., ["julie@acme.com", "craig@lineagecoffee.com"])
         - sentAt: Sent timestamp (ISO string)
         - sentDate: Date (string)
         - sentMonth: Month (string)
         - sentYear: Year (integer)
         - sentWeek: Week (string)
         - sentDayOfWeek: Day index (integer, 0-6)
         - sentHour: Hour (integer, 0-23)
         - labelIds: Gmail labels (array)
         - labelNames: Label names (array)
         - isUnread: Unread status (boolean)
         - isImportant: Important status (boolean)
         - isStarred: Starred status (boolean)
         - hasAttachments: Attachment presence (boolean)
         - hasExternal: External participants (boolean)
         - inInbox: In inbox (boolean)
         - isInternal: Internal sender (boolean)
         - category: Gmail category (string, e.g., "PERSONAL")
         - domainsMentioned: Domains in email (array)
         - searchTerms: Keywords for search (array, e.g., ["invoice"])
         - businessTerms: Business keywords (array)
         - invoiceNumbers: Extracted invoice numbers (array)
         - poNumbers: Purchase order numbers (array)
         - amounts: Monetary amounts (array)
         - subject: Email subject (string)
         - subjectLc: Lowercase subject (string)
         - snippet: Short preview (string)
         - bodyPreview: Extracted body text (string)
       - Purpose: Fast searches for keywords, people, or labels.
       - Indexes: owner, sentAt, fromEmailLc, etc.; bodyPreview/snippet/subject exempt.

    3. people_profiles
       - Description: Directory for name resolution.
       - Document ID: {personId} (e.g., "julie@acme.com")
       - Fields: personId, firstName, email
       - Purpose: Maps names to emails for queries like "emails from Julie".

    4. threads
       - Description: Owner-scoped conversation context.
       - Document ID: {owner_threadId} (e.g., "craig@lineagecoffee.com_186c19fe1d2b9c7e")
       - Fields: threadId, owner, participants (array), lastSentAt
       - Purpose: Groups emails for conversation context; handles group sends.

    5. idx_person
       - Description: Pointers for participants.
       - Fields: messageId, owner, fromEmailLc, person (email/name), personFirstName, isUnread, hasAttachments, sentAt, subjectPreview
       - Purpose: Fast lookup by person (sender/recipient).

    6. idx_domain
       - Fields: messageId, owner, fromEmailLc, domain, sentAt, subjectPreview
       - Purpose: Filter by external domain.

    7. idx_day
       - Fields: messageId, owner, fromEmailLc, day, hour, sentAt, subjectPreview
       - Purpose: Time-based queries (daily).

    8. idx_week
       - Fields: messageId, owner, fromEmailLc, week, sentAt, subjectPreview
       - Purpose: Weekly time filters.

    9. idx_label
       - Fields: messageId, owner, fromEmailLc, labelId, sentAt, subjectPreview
       - Purpose: Filter by labels (e.g., "UNREAD").

    10. idx_attachment
        - Fields: messageId, owner, fromEmailLc, type, filename, sentAt, subjectPreview
        - Purpose: Search by attachment type.

    11. idx_token
        - Fields: messageId, owner, fromEmailLc, token, contextPreview, sentAt, subjectPreview
        - Purpose: Keyword-based search.

    12. idx_recent
        - Fields: messageId, owner, fromEmailLc, labelIds, hasAttachments, sentAt, subjectPreview
        - Purpose: Quick access to recent emails.

    13. idx_thread_membership
        - Fields: messageId, owner, fromEmailLc, threadId, position, sentAt, subjectPreview
        - Purpose: Reconstruct thread order.

    Ambiguity Handling:
    - For queries like "mail from X":
      - Check `fromEmailLc`, `fromPerson`, `fromFirstName` (sender in email_search_index).
      - Check `allPeople` (all participants, including recipients).
      - Check `owner` (mailbox owner for group sends).
      - Use `idx_person` for pointers to sender/recipient.
      - Example: "mail from Julie" searches `fromFirstName="julie"`, `allPeople` contains "julie@%", or `owner="julie@lineagecoffee.com"`.
    """

    results = []
    attempts = 0
    query_strategy = None

    while attempts < max_retries:
        attempts += 1

        # Step 1: Gemini 1.5 Pro - Interpret query
        interpret_prompt = f"""
        Interpret query: '{user_input}'.
        Use schema: {schema_info}.
        Suggest Firestore query to cast a wide net (e.g., check sent/received for 'mail from X').
        For attempt {attempts}, broaden if needed (e.g., use allPeople if fromEmailLc fails).
        Output JSON: {{'collection': str, 'field': str, 'value': str, 'operator': str}}
        """
        try:
            if pro_model:
                response1 = pro_model.generate_content(interpret_prompt)
                query_strategy = json.loads(response1.text.strip("```json\n```"))
            else:
                return [{"error": "GEMINI_API_KEY not configured"}]
        except Exception as e:
            if attempts == max_retries:
                return [{"error": f"Failed to interpret query after {attempts} attempts: {str(e)}"}]
            continue

        # Step 2: Execute Firestore query
        try:
            query = db.collection(query_strategy['collection'])\
                      .where(query_strategy['field'], query_strategy['operator'], query_strategy['value'])\
                      .limit(10).stream()
            results = [doc.to_dict() for doc in query]
        except GoogleAPIError as e:
            results = []

        # Step 3: Gemini 1.5 Flash - Assess relevance
        assess_prompt = f"""
        Assess results for '{user_input}': {json.dumps(results, default=str)}.
        Score 1-10 (10=perfect match, e.g., exact sender or keyword).
        If score <7 or empty, suggest wider query (e.g., alt field like allPeople or wildcard).
        Use schema: {schema_info}.
        Output JSON: {{'score': int, 'suggestion': dict or null}}
        """
        try:
            if flash_model:
                response3 = flash_model.generate_content(assess_prompt)
                assessment = json.loads(response3.text.strip("```json\n```"))
            else:
                assessment = {'score': 0, 'suggestion': None}
        except:
            assessment = {'score': 0, 'suggestion': None}

        # Break if good results
        if results and assessment['score'] >= 7:
            break

        # Step 4: Gemini 1.5 Pro - Refine query if needed
        if assessment['suggestion'] and attempts < max_retries:
            refine_prompt = f"""
            Refine query for '{user_input}' based on suggestion: {json.dumps(assessment['suggestion'])}.
            Use schema: {schema_info}.
            Cast wider net (e.g., check toEmails or idx_person).
            Output JSON: {{'collection': str, 'field': str, 'value': str, 'operator': str}}
            """
            try:
                if pro_model:
                    response4 = pro_model.generate_content(refine_prompt)
                    query_strategy = json.loads(response4.text.strip("```json\n```"))
                else:
                    continue
            except:
                continue

    # Format results
    if not results:
        return [{"error": f"No relevant results after {attempts} attempts"}]

    formatted_results = [{
        'messageId': r.get('messageId'),
        'subject': r.get('subject'),
        'fromEmailLc': r.get('fromEmailLc'),
        'owner': r.get('owner'),
        'snippet': r.get('snippet')
    } for r in results]
    return formatted_results

def store_conversation_history(user_id, user_input, search_results):
    """Store conversation in ad3gem-chat-history"""

    # Store user message
    user_doc = chat_db.collection(f'users/{user_id}/recent_messages').document()
    user_doc.set({
        'text': user_input,
        'role': 'user',
        'timestamp': datetime.now(timezone.utc),
        'intent': 'email_search'
    })

    # Store bot response
    bot_doc = chat_db.collection(f'users/{user_id}/recent_messages').document()
    bot_doc.set({
        'text': json.dumps(search_results, default=str),
        'role': 'bot',
        'timestamp': datetime.now(timezone.utc),
        'intent': 'email_search'
    })
