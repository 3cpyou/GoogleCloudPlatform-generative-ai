#!/usr/bin/env python3

from google.cloud import firestore

# Initialize Firestore
db = firestore.Client(project="ad3gem-email")

# Execute the exact query from your JSON
print("🔍 Executing query: Get the most recent message")
print(
    "Query: db.collection('email_search_index').order_by('sentAt', direction=firestore.Query.DESCENDING).limit(1)"
)
print("-" * 60)

# Run the query
results = (
    db.collection("email_search_index")
    .order_by("sentAt", direction=firestore.Query.DESCENDING)
    .limit(1)
    .get()
)

# Display results
for doc in results:
    data = doc.to_dict()

    print("\n📧 MOST RECENT MESSAGE:")
    print(f"  Message ID: {data.get('messageId', 'N/A')}")
    print(f"  Thread ID: {data.get('threadId', 'N/A')}")
    print(f"  From Person: {data.get('fromPerson', 'N/A')}")
    print(f"  From Email: {data.get('fromEmail', 'N/A')}")
    print(f"  Subject: {data.get('subject', 'N/A')}")
    print(f"  Sent At: {data.get('sentAt', 'N/A')}")
    print(f"\n  Snippet: {data.get('snippet', 'N/A')}")
    print(f"\n  Body Preview:\n  {data.get('bodyPreview', 'N/A')}")
    print("-" * 60)

if not results:
    print("❌ No messages found in the database")
