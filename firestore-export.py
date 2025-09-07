#!/usr/bin/env python3
"""
Firestore Data Export Script
Downloads all data from Firestore database to JSON file
Usage: python export_firestore.py
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from google.api_core import exceptions
from google.cloud import firestore
from google.cloud.firestore import CollectionReference, DocumentReference

# Configuration
DATABASE_ID = "ad3sam-database"
PROJECT_ID = "ad3-sam"  # Update if different
OUTPUT_FILE = f"firestore_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def convert_firestore_data(data: Any) -> Any:
    """Convert Firestore-specific data types to JSON-serializable formats"""
    if isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, DocumentReference):
        return f"__doc_ref__{data.path}"
    elif isinstance(data, dict):
        return {key: convert_firestore_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_firestore_data(item) for item in data]
    else:
        return data


def get_all_documents(collection_ref: CollectionReference) -> List[Dict[str, Any]]:
    """Get all documents from a collection with their subcollections"""
    documents = []

    try:
        for doc in collection_ref.stream():
            doc_data = {
                "id": doc.id,
                "data": convert_firestore_data(doc.to_dict()),
                "subcollections": {},
            }

            # Get subcollections
            for subcol in doc.reference.collections():
                subcol_docs = get_all_documents(subcol)
                if subcol_docs:
                    doc_data["subcollections"][subcol.id] = subcol_docs

            documents.append(doc_data)

    except exceptions.NotFound:
        print(f"Collection {collection_ref.id} not found")
    except Exception as e:
        print(f"Error reading collection {collection_ref.id}: {str(e)}")

    return documents


def export_firestore_database():
    """Export entire Firestore database to JSON"""

    # Initialize Firestore client
    try:
        if DATABASE_ID == "(default)":
            db = firestore.Client(project=PROJECT_ID)
        else:
            db = firestore.Client(project=PROJECT_ID, database=DATABASE_ID)

        print(f"Connected to Firestore database: {DATABASE_ID}")
    except Exception as e:
        print(f"Failed to connect to Firestore: {str(e)}")
        return

    # Get all root collections
    try:
        collections = db.collections()
        collection_names = [col.id for col in collections]
        print(f"Found collections: {collection_names}")
    except Exception as e:
        print(f"Failed to list collections: {str(e)}")
        return

    # Export data
    export_data = {
        "database_id": DATABASE_ID,
        "project_id": PROJECT_ID,
        "export_timestamp": datetime.now().isoformat(),
        "collections": {},
    }

    total_docs = 0

    for collection_name in collection_names:
        print(f"\nExporting collection: {collection_name}")

        collection_ref = db.collection(collection_name)
        documents = get_all_documents(collection_ref)

        export_data["collections"][collection_name] = documents
        doc_count = len(documents)
        total_docs += doc_count

        print(f"  └─ Exported {doc_count} documents")

    # Save to JSON file
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)  # MB

        print("\n✅ Export completed successfully!")
        print(f"📁 File: {OUTPUT_FILE}")
        print(f"📊 Total documents: {total_docs}")
        print(f"💾 File size: {file_size:.2f} MB")

    except Exception as e:
        print(f"❌ Failed to save export file: {str(e)}")


def validate_environment():
    """Check if required environment is set up"""

    # Check for authentication
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        print("⚠️  Warning: GOOGLE_APPLICATION_CREDENTIALS not set")
        print(
            "   Make sure you're authenticated with gcloud or have a service account key"
        )

    # Check if credentials file exists
    if cred_path and not os.path.exists(cred_path):
        print(f"❌ Credentials file not found: {cred_path}")
        return False

    return True


if __name__ == "__main__":
    print("🔥 Firestore Database Export Tool")
    print("=" * 40)

    # Validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed")
        exit(1)

    # Show configuration
    print("📋 Configuration:")
    print(f"   Project ID: {PROJECT_ID}")
    print(f"   Database ID: {DATABASE_ID}")
    print(f"   Output file: {OUTPUT_FILE}")

    # Confirm before proceeding
    confirm = input("\n🤔 Proceed with export? (y/N): ").lower().strip()

    if confirm in ["y", "yes"]:
        print("\n🚀 Starting export...")
        export_firestore_database()
    else:
        print("❌ Export cancelled")
