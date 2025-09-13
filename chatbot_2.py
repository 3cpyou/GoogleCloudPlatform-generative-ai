"""
Intelligent AD3Gem Chatbot with Knowledge Base - Updated for ad3gem-emails
Uses the new email database structure and knowledge base for intelligent responses

This chatbot:
1. Connects to the ad3gem-knowledge database for context
2. Queries the new ad3gem-emails database structure for email data
3. Uses Gemini 1.5 Pro for intelligent understanding
4. Provides context-aware responses with learning capabilities

Database Connections:
- ad3gem-emails: New email storage (threads, messages)
- ad3gem-knowledge: Processed knowledge and patterns
- ad3sam-email: Legacy database (optional fallback)
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
from google.cloud import firestore

from firestore_client import (
    FirestoreClient,
    append_to_conversation,
)


class IntelligentAD3GemChatbot:
    """
    Chatbot that uses the centralized knowledge base and new email structure.
    """

    def __init__(self, user_id: str = "default_user"):
        """Initialize the chatbot with all database connections."""
        self.user_id = user_id
        self.conversation_id = f"chat_{user_id}_{datetime.now().strftime('%Y%m%d')}"

        # Configure Google Generative AI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        # Initialize Firestore clients
        self.firestore_client = FirestoreClient()  # Legacy client

        # Knowledge base connection
        self.knowledge_db = firestore.Client(
            project="ad3-sam", database="ad3gem-knowledge"
        )

        # New email database connection
        self.email_db = firestore.Client(project="ad3-sam", database="ad3gem-emails")

        # Initialize Gemini 1.5 Pro
        self.model = genai.GenerativeModel(
            "gemini-1.5-pro",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            },
        )

        # Session context memory
        self.session_context = {
            "last_thread_id": None,
            "last_sender_mentioned": None,
            "last_time_period": None,
            "last_topic": None,
            "clarification_attempts": 0,
        }

        print("🤖 Intelligent AD3Gem Chatbot initialized")
        print("🧠 Knowledge DB: ad3gem-knowledge")
        print("📧 Email DB: ad3gem-emails (new structure)")
        print(f"💬 Conversation ID: {self.conversation_id}")

    def get_knowledge_context(self, user_message: str) -> Dict:
        """Retrieve relevant context from the knowledge base."""
        context = {
            "entities": {},
            "relationships": [],
            "facts": [],
            "patterns": {},
            "topics": [],
            "temporal": {},
            "thread_summaries": [],
            "corrections": [],
        }

        query_lower = user_message.lower()

        try:
            # Search for mentioned entities
            entities = self.knowledge_db.collection("entities").limit(100).stream()
            for entity_doc in entities:
                entity_data = entity_doc.to_dict()

                # Check if entity is mentioned
                mentioned = False

                # Check names
                for name in entity_data.get("names", []):
                    if name.lower() in query_lower:
                        mentioned = True
                        break

                # Check email addresses
                if not mentioned:
                    for email in entity_data.get("email_addresses", []):
                        email_parts = email.lower().split("@")[0]
                        if email_parts in query_lower:
                            mentioned = True
                            break

                if mentioned:
                    context["entities"][entity_doc.id] = entity_data

                    # Get relationships
                    rels = (
                        self.knowledge_db.collection("relationships")
                        .where("from_entity", "==", entity_doc.id)
                        .limit(5)
                        .stream()
                    )
                    for rel in rels:
                        context["relationships"].append(rel.to_dict())

            # Get relevant facts
            facts = self.knowledge_db.collection("facts").limit(30).stream()
            for fact_doc in facts:
                fact_data = fact_doc.to_dict()
                fact_text = fact_data.get("fact", "").lower()
                # Check relevance
                if any(word in query_lower for word in fact_text.split()[:10]):
                    context["facts"].append(fact_data)

            # Get patterns
            patterns_doc = (
                self.knowledge_db.collection("patterns")
                .document("communication_patterns")
                .get()
            )
            if patterns_doc.exists:
                context["patterns"] = patterns_doc.to_dict()

            # Get temporal patterns
            temporal_doc = (
                self.knowledge_db.collection("temporal")
                .document("email_patterns")
                .get()
            )
            if temporal_doc.exists:
                context["temporal"] = temporal_doc.to_dict()

            # Get thread summaries
            summaries_doc = (
                self.knowledge_db.collection("thread_summaries")
                .document("active_threads")
                .get()
            )
            if summaries_doc.exists:
                context["thread_summaries"] = summaries_doc.to_dict().get("threads", [])

            # Get active corrections
            corrections = (
                self.knowledge_db.collection("corrections")
                .where("active", "==", True)
                .limit(10)
                .stream()
            )
            for correction in corrections:
                context["corrections"].append(correction.to_dict())

        except Exception as e:
            print(f"⚠️ Error fetching knowledge context: {e}")

        return context

    def search_emails_new_structure(
        self, query: str, knowledge_context: Dict, time_filter: Optional[Dict] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Search emails using the new database structure.
        Returns (threads, messages)
        """
        threads = []
        messages = []

        try:
            # Build search based on entities in context
            mentioned_emails = []
            for entity_data in knowledge_context.get("entities", {}).values():
                mentioned_emails.extend(entity_data.get("email_addresses", []))

            # Search threads
            thread_query = self.email_db.collection("threads")

            # Apply filters based on query understanding
            if mentioned_emails:
                # Search for threads with these participants
                for email in mentioned_emails[:3]:  # Limit to avoid too many queries
                    thread_results = (
                        thread_query.where(
                            "participants", "array_contains", email.lower()
                        )
                        .order_by("lastMessageAt", direction=firestore.Query.DESCENDING)
                        .limit(10)
                        .stream()
                    )

                    for thread_doc in thread_results:
                        thread_data = thread_doc.to_dict()
                        thread_data["thread_id"] = thread_doc.id
                        threads.append(thread_data)
            else:
                # Get recent threads
                thread_results = (
                    thread_query.order_by(
                        "lastMessageAt", direction=firestore.Query.DESCENDING
                    )
                    .limit(20)
                    .stream()
                )

                for thread_doc in thread_results:
                    thread_data = thread_doc.to_dict()
                    thread_data["thread_id"] = thread_doc.id
                    threads.append(thread_data)

            # Apply time filter if specified
            if time_filter:
                cutoff_date = self._calculate_cutoff_date(time_filter)
                threads = [
                    t
                    for t in threads
                    if t.get("lastMessageAt", datetime.min.replace(tzinfo=timezone.utc))
                    >= cutoff_date
                ]

            # Get messages from relevant threads
            for thread in threads[:5]:  # Limit to top 5 threads
                thread_id = thread.get("thread_id")
                if not thread_id:
                    continue

                message_docs = (
                    self.email_db.collection("threads")
                    .document(thread_id)
                    .collection("messages")
                    .order_by("sentAt", direction=firestore.Query.DESCENDING)
                    .limit(5)
                    .stream()
                )

                for msg_doc in message_docs:
                    msg_data = msg_doc.to_dict()
                    msg_data["message_id"] = msg_doc.id
                    msg_data["thread_id"] = thread_id
                    msg_data["thread_subject"] = thread.get("subject", "")
                    messages.append(msg_data)

            # Sort messages by date
            messages.sort(
                key=lambda x: x.get(
                    "sentAt", datetime.min.replace(tzinfo=timezone.utc)
                ),
                reverse=True,
            )

        except Exception as e:
            print(f"⚠️ Error searching emails: {e}")

        return threads, messages

    def _calculate_cutoff_date(self, time_filter: Dict) -> datetime:
        """Calculate cutoff date from time filter."""
        now = datetime.now(timezone.utc)
        value = time_filter.get("value", "")

        if value == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif value == "yesterday":
            return (now - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif value == "last_week":
            return now - timedelta(days=7)
        elif value == "last_month":
            return now - timedelta(days=30)
        else:
            return now - timedelta(days=7)  # Default to last week

    def understand_query(self, user_message: str, knowledge_context: Dict) -> Dict:
        """Use Gemini to understand the user's query with knowledge context."""

        # Format entity information for prompt
        entity_info = ""
        for entity_id, entity_data in knowledge_context.get("entities", {}).items():
            names = ", ".join(entity_data.get("names", []))
            emails = ", ".join(entity_data.get("email_addresses", [])[:2])
            entity_info += f"- {names} ({emails})\n"

        prompt = f"""
        Analyze this email query with available context.

        User Query: "{user_message}"

        Known Entities Mentioned:
        {entity_info if entity_info else "None identified"}

        Session Context:
        - Last sender mentioned: {self.session_context.get("last_sender_mentioned")}
        - Last time period: {self.session_context.get("last_time_period")}
        - Last topic: {self.session_context.get("last_topic")}

        Provide analysis in JSON:
        {{
            "intent": "search_emails/count_emails/find_specific/check_status/general_question",
            "entities_mentioned": ["list of entity IDs from context"],
            "time_filter": {{
                "value": "today/yesterday/last_week/last_month/all",
                "specific_date": null
            }},
            "topics": ["relevant topics"],
            "needs_clarification": true/false,
            "clarification_question": "question if needed",
            "confidence": 0-100,
            "search_query": "suggested search terms"
        }}

        Return ONLY valid JSON.
        """

        try:
            response = self.model.generate_content(prompt)
            json_str = response.text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:-3]

            understanding = json.loads(json_str)

            # Update session context
            if understanding.get("entities_mentioned"):
                self.session_context["last_sender_mentioned"] = understanding[
                    "entities_mentioned"
                ][0]
            if understanding.get("time_filter", {}).get("value"):
                self.session_context["last_time_period"] = understanding["time_filter"][
                    "value"
                ]
            if understanding.get("topics"):
                self.session_context["last_topic"] = understanding["topics"][0]

            return understanding

        except Exception as e:
            print(f"Error understanding query: {e}")
            return {
                "intent": "search_emails",
                "confidence": 30,
                "needs_clarification": True,
            }

    def generate_intelligent_response(
        self,
        user_message: str,
        understanding: Dict,
        knowledge_context: Dict,
        threads: List[Dict],
        messages: List[Dict],
    ) -> str:
        """Generate response using all available context."""

        # Format data for prompt
        thread_info = self._format_threads(threads[:3])
        message_info = self._format_messages(messages[:5])
        entity_info = self._format_entities(knowledge_context.get("entities", {}))
        fact_info = self._format_facts(knowledge_context.get("facts", [])[:5])
        pattern_info = self._format_patterns(knowledge_context.get("patterns", {}))
        temporal_info = self._format_temporal(knowledge_context.get("temporal", {}))

        prompt = f"""
        You are AD3Gem, an intelligent email assistant with comprehensive knowledge.

        USER QUERY: "{user_message}"

        UNDERSTANDING:
        Intent: {understanding.get("intent")}
        Confidence: {understanding.get("confidence")}%

        KNOWLEDGE BASE CONTEXT:

        Entities:
        {entity_info}

        Known Facts:
        {fact_info}

        Communication Patterns:
        {pattern_info}

        Temporal Patterns:
        {temporal_info}

        EMAIL SEARCH RESULTS:

        Threads Found ({len(threads)} total):
        {thread_info}

        Recent Messages ({len(messages)} total):
        {message_info}

        INSTRUCTIONS:
        1. Answer the user's question directly and accurately
        2. Reference specific emails, facts, or patterns when relevant
        3. If you notice patterns (like someone always emails at 9 AM), mention them
        4. Be conversational but precise
        5. If results seem incomplete, mention that
        6. Suggest follow-up actions if helpful
        7. Keep response under 200 words unless listing multiple items

        Generate a helpful response:
        """

        response = self.model.generate_content(prompt)
        return response.text

    def _format_threads(self, threads: List[Dict]) -> str:
        """Format thread information for prompt."""
        if not threads:
            return "No threads found"

        formatted = []
        for thread in threads:
            participants = ", ".join(thread.get("participants", [])[:3])
            formatted.append(
                f"- {thread.get('subject', 'No subject')}\n"
                f"  Participants: {participants}\n"
                f"  Messages: {thread.get('messageCount', 0)}\n"
                f"  Last activity: {thread.get('lastMessageAt', 'Unknown')}\n"
                f"  Preview: {thread.get('latestSnippet', '')[:100]}..."
            )

        return "\n".join(formatted)

    def _format_messages(self, messages: List[Dict]) -> str:
        """Format message information for prompt."""
        if not messages:
            return "No messages found"

        formatted = []
        for msg in messages:
            from_data = msg.get("from", {})
            from_email = from_data.get("email", "Unknown")
            from_name = from_data.get("name", "")
            sender = f"{from_name} ({from_email})" if from_name else from_email

            # Format date
            sent_at = msg.get("sentAt")
            if sent_at:
                if isinstance(sent_at, str):
                    date_str = sent_at
                else:
                    date_str = sent_at.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = "Unknown"

            formatted.append(
                f"- From: {sender}\n"
                f"  Subject: {msg.get('thread_subject', 'No subject')}\n"
                f"  Date: {date_str}\n"
                f"  Preview: {msg.get('bodyPreview', '')[:150]}..."
            )

        return "\n".join(formatted)

    def _format_entities(self, entities: Dict) -> str:
        """Format entity information."""
        if not entities:
            return "No specific entities identified"

        formatted = []
        for entity_id, data in entities.items():
            names = ", ".join(data.get("names", ["Unknown"]))
            emails = ", ".join(data.get("email_addresses", [])[:2])
            company = ", ".join(data.get("companies", ["Unknown"]))

            formatted.append(
                f"- {names} ({emails})\n"
                f"  Type: {data.get('type', 'unknown')}\n"
                f"  Company: {company}\n"
                f"  Email count: {data.get('email_count', 0)}"
            )

        return "\n".join(formatted) if formatted else "None identified"

    def _format_facts(self, facts: List) -> str:
        """Format facts for prompt."""
        if not facts:
            return "No relevant facts"

        formatted = []
        for fact in facts:
            formatted.append(
                f"- {fact.get('fact', '')}\n"
                f"  Type: {fact.get('type', 'general')}, "
                f"Importance: {fact.get('importance', 'medium')}"
            )

        return "\n".join(formatted)

    def _format_patterns(self, patterns: Dict) -> str:
        """Format communication patterns."""
        if not patterns:
            return "No patterns available"

        top_senders = patterns.get("top_senders", {})
        if top_senders:
            sender_list = [f"{s}: {c} emails" for s, c in list(top_senders.items())[:3]]
            return f"Top senders: {', '.join(sender_list)}"

        return "No patterns available"

    def _format_temporal(self, temporal: Dict) -> str:
        """Format temporal patterns."""
        if not temporal:
            return "No temporal patterns"

        schedules = temporal.get("sender_schedules", {})
        response_times = temporal.get("response_times", {})

        formatted = []

        # Format sender schedules
        for sender, schedule in list(schedules.items())[:3]:
            hour = schedule.get("typical_hour")
            day = schedule.get("typical_day_name")
            if hour is not None:
                formatted.append(f"{sender} usually emails around {hour}:00")
            if day:
                formatted[-1] += f" on {day}s"

        # Format response times
        for responder, times in list(response_times.items())[:2]:
            avg_hours = times.get("avg_response_hours", 0)
            if avg_hours < 1:
                formatted.append(f"{responder} typically responds within an hour")
            elif avg_hours < 24:
                formatted.append(
                    f"{responder} typically responds within {int(avg_hours)} hours"
                )
            else:
                days = int(avg_hours / 24)
                formatted.append(f"{responder} typically responds within {days} days")

        return "; ".join(formatted) if formatted else "No temporal patterns"

    def send_message(self, user_message: str) -> str:
        """Process user message with full knowledge base context."""
        print(f"\n💭 User: {user_message}")

        # Save user message
        append_to_conversation(self.conversation_id, "user", user_message)

        # Get knowledge base context
        print("🧠 Fetching knowledge context...")
        knowledge_context = self.get_knowledge_context(user_message)

        entity_count = len(knowledge_context.get("entities", {}))
        fact_count = len(knowledge_context.get("facts", []))
        print(f"   Found {entity_count} entities, {fact_count} facts")

        # Understand the query
        understanding = self.understand_query(user_message, knowledge_context)
        print(
            f"   Intent: {understanding.get('intent')}, Confidence: {understanding.get('confidence')}%"
        )

        # Check if clarification needed
        if (
            understanding.get("needs_clarification")
            and understanding.get("confidence", 0) < 50
        ):
            response = understanding.get(
                "clarification_question",
                "Could you be more specific? You can ask about:\n"
                "- Emails from specific people (e.g., 'emails from Julie')\n"
                "- Time periods (e.g., 'emails yesterday')\n"
                "- Topics (e.g., 'coffee orders', 'invoices')",
            )
        else:
            # Search emails using new structure
            print("📧 Searching emails...")
            threads, messages = self.search_emails_new_structure(
                user_message, knowledge_context, understanding.get("time_filter")
            )
            print(f"   Found {len(threads)} threads, {len(messages)} messages")

            # Generate response
            response = self.generate_intelligent_response(
                user_message, understanding, knowledge_context, threads, messages
            )

        # Save response
        append_to_conversation(self.conversation_id, "assistant", response)

        print(f"🤖 Assistant: {response}")
        return response

    def add_correction(self, original: str, corrected: str):
        """Add a correction to the knowledge base."""
        doc_ref = self.knowledge_db.collection("corrections").document()
        doc_ref.set(
            {
                "type": "user_correction",
                "original": original,
                "corrected": corrected,
                "user_id": self.user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "active": True,
            }
        )

        return f"✅ I'll remember that '{original}' should be '{corrected}'"

    def get_email_stats(self) -> str:
        """Get statistics about the email database."""
        try:
            # Count threads
            threads = self.email_db.collection("threads").limit(1000).stream()
            thread_count = sum(1 for _ in threads)

            # Count recent threads (last 7 days)
            recent_date = datetime.now(timezone.utc) - timedelta(days=7)
            recent_threads = (
                self.email_db.collection("threads")
                .where("lastMessageAt", ">=", recent_date)
                .stream()
            )
            recent_count = sum(1 for _ in recent_threads)

            # Get patterns
            patterns_doc = (
                self.knowledge_db.collection("patterns")
                .document("communication_patterns")
                .get()
            )
            top_senders = []
            if patterns_doc.exists:
                top_senders_dict = patterns_doc.to_dict().get("top_senders", {})
                top_senders = list(top_senders_dict.items())[:5]

            # Format response
            response = f"""📊 Email Database Statistics:

Total Threads: {thread_count}
Active Threads (last 7 days): {recent_count}

Top Senders:"""

            for sender, count in top_senders:
                response += f"\n  • {sender}: {count} emails"

            return response

        except Exception as e:
            return f"Error getting statistics: {e}"

    def start_interactive_session(self):
        """Start an interactive chat session."""
        print("\n" + "=" * 60)
        print("🚀 Intelligent AD3Gem Chatbot")
        print("=" * 60)
        print("Connected to:")
        print("  📧 Email DB: ad3gem-emails (new structure)")
        print("  🧠 Knowledge DB: ad3gem-knowledge")
        print("\nI understand natural language and learn from context.")
        print("\nCommands:")
        print("  'stats' - Show email statistics")
        print("  'correct: X should be Y' - Add a correction")
        print("  'quit/exit/bye' - End session")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n💭 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "stats":
                    print(self.get_email_stats())
                    continue

                if user_input.lower().startswith("correct:"):
                    parts = user_input[8:].split(" should be ")
                    if len(parts) == 2:
                        result = self.add_correction(parts[0].strip(), parts[1].strip())
                        print(result)
                    else:
                        print("Format: 'correct: X should be Y'")
                    continue

                if not user_input:
                    continue

                # Process normal message
                response = self.send_message(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Let me try to continue...")
                self.session_context["clarification_attempts"] = 0


def main():
    """Main function to run the chatbot."""
    print("🔥 Initializing Intelligent AD3Gem Chatbot...")
    print("   Using new ad3gem-emails database structure")

    user_id = os.getenv("USER_ID", "default_user")

    try:
        chatbot = IntelligentAD3GemChatbot(user_id=user_id)
        chatbot.start_interactive_session()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\nPlease ensure:")
        print("1. GEMINI_API_KEY environment variable is set")
        print("2. Firestore databases exist: ad3gem-emails, ad3gem-knowledge")
        print("3. You have proper IAM permissions")


if __name__ == "__main__":
    main()
