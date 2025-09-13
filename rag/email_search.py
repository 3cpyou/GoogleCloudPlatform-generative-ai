#!/usr/bin/env python3
"""
Email Search Tool - Works without LLM access
Semantic search through your email vectors
"""

import os
import sys
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import vertexai
from google.cloud import firestore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore


def initialize_gcp(database: Optional[str] = None) -> firestore.Client:
    """Initialize GCP and return Firestore client."""
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    database = database or os.getenv("GOOGLE_CLOUD_FIRESTORE_DB")

    if not project_id:
        print("Missing GOOGLE_CLOUD_PROJECT environment variable", file=sys.stderr)
        sys.exit(1)

    vertexai.init(project=project_id, location=region)
    if database:
        return firestore.Client(project=project_id, database=database)
    return firestore.Client(project=project_id)


def search_emails(
    query: str,
    vector_collection: str = "email_vectors",
    k: int = 5,
    database_read: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search emails using vector similarity."""

    # Initialize
    db = initialize_gcp(database=database_read)
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    # Create vector store
    store = FirestoreVectorStore(
        client=db,
        collection=vector_collection,
        embedding_service=embeddings
    )

    # Perform search
    docs = store.similarity_search(query, k=k)

    # Format results
    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        results.append({
            "rank": i,
            "messageId": meta.get("messageId"),
            "threadId": meta.get("threadId"),
            "fromEmail": meta.get("fromEmail"),
            "sentDate": meta.get("sentDate"),
            "labelNames": meta.get("labelNames", []),
            "category": meta.get("category"),
            "hasAttachments": meta.get("hasAttachments", False),
            "attachmentTypes": meta.get("attachmentTypes", []),
            "preview": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            "full_content": doc.page_content
        })

    return results


def interactive_search(database_read: Optional[str] = None):
    """Interactive email search session."""
    print("\n🔍 Email Search Tool")
    print("=" * 50)
    print("Enter search queries to find relevant emails.")
    print("Type 'quit' to exit, 'help' for examples.\n")

    while True:
        try:
            query = input("📧 Search: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            elif query.lower() in ['help', 'h']:
                print("\n💡 Example searches:")
                print("• 'urgent deadlines this week'")
                print("• 'meeting schedules'")
                print("• 'project updates from John'")
                print("• 'invoices and payments'")
                print("• 'customer complaints'")
                print("• 'budget discussions'")
                print()
                continue

            elif not query:
                continue

            # Perform search
            print(f"\n🔍 Searching for: '{query}'...")
            results = search_emails(query, database_read=database_read, k=5)

            if not results:
                print("❌ No results found.")
                continue

            print(f"\n📋 Found {len(results)} relevant emails:")
            print("-" * 60)

            for result in results:
                print(f"\n🔹 Rank {result['rank']}")
                print(f"   From: {result['fromEmail'] or 'Unknown'}")
                print(f"   Date: {result['sentDate'] or 'Unknown'}")
                print(f"   Labels: {', '.join(result['labelNames'][:3]) or 'None'}")
                if result['hasAttachments']:
                    print(f"   Attachments: {', '.join(result['attachmentTypes'])}")
                print(f"   Preview: {result['preview']}")

                # Ask if user wants full content
                if len(result['full_content']) > 300:
                    show_full = input("     📄 Show full content? (y/N): ").strip().lower()
                    if show_full in ['y', 'yes']:
                        print(f"     Full: {result['full_content']}")

            print("\n" + "-" * 60)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Email Search Tool")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--db-read", type=str, default=None, help="Firestore database")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    if args.interactive or not args.query:
        interactive_search(database_read=args.db_read)
    else:
        # Single query mode
        results = search_emails(args.query, database_read=args.db_read, k=args.k)

        print(f"\n🔍 Search: '{args.query}'")
        print(f"📋 Found {len(results)} results:")
        print("=" * 60)

        for result in results:
            print(f"\n🔹 Rank {result['rank']}")
            print(f"   From: {result['fromEmail'] or 'Unknown'}")
            print(f"   Date: {result['sentDate'] or 'Unknown'}")
            print(f"   Preview: {result['preview']}")


if __name__ == "__main__":
    main()
