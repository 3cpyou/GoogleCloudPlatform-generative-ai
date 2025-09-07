import os

from google.cloud import firestore

# Initialize Firestore client with your project and database
db = firestore.Client(
    project=os.getenv("PROJECT_ID"), database=os.getenv("FIRESTORE_DATABASE")
)  # type: ignore

# Sample data to add
sample_users = [
    {"id": "user123", "name": "John Doe", "email": "john@example.com", "age": 30},
    {"id": "user456", "name": "Jane Smith", "email": "jane@example.com", "age": 28},
    {"id": "user789", "name": "Bob Johnson", "email": "bob@example.com", "age": 35},
]

# Add each user to the 'users' collection
for user in sample_users:
    doc_ref = db.collection("users").document(user["id"])
    doc_ref.set({"name": user["name"], "email": user["email"], "age": user["age"]})
    print(f"Added user: {user['id']}")

print("Sample data added successfully to 'users' collection.")
