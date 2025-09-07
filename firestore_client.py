"""
Firestore Client for AD3Gem Chatbot System

This module provides a unified interface to interact with the three Firestore databases:
- ad3gem-database: Main application data (users, business logic)
- ad3gem-conversation: Chat history and conversation logs
- ad3gem-memory: Refined knowledge, heads, and claims

Based on patterns from the ad3gem repository and the ad3gem-CI.md PRD.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter


class FirestoreClient:
    """Unified client for managing AD3Gem Firestore databases."""

    def __init__(self):
        """Initialize connections to all three databases."""
        self.project_id = os.getenv("PROJECT_ID", "ad3-sam")
        self.region = os.getenv("REGION", "us-central1")

        # Initialize database connections
        self.main_db = firestore.Client(
            project=self.project_id,
            database=os.getenv("FIRESTORE_DATABASE", "ad3gem-database"),
        )

        self.conversation_db = firestore.Client(
            project=self.project_id,
            database=os.getenv("FIRESTORE_CONVERSATION_DB", "ad3gem-conversation"),
        )

        self.memory_db = firestore.Client(
            project=self.project_id,
            database=os.getenv("FIRESTORE_MEMORY_DB", "ad3gem-memory"),
        )

        # Email database connection
        self.email_db = firestore.Client(
            project=self.project_id,
            database=os.getenv("FIRESTORE_EMAIL_DB", "ad3sam-email"),
        )

    # ================================
    # MAIN DATABASE OPERATIONS
    # ================================

    def get_document(
        self, collection: str, document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a document from the main database."""
        try:
            doc_ref = self.main_db.collection(collection).document(document_id)
            doc = doc_ref.get()

            if doc.exists:
                data = doc.to_dict()
                data["_id"] = doc.id
                return data
            return None

        except Exception as e:
            print(f"Error getting document {document_id} from {collection}: {e}")
            return None

    def add_document(
        self, collection: str, data: Dict[str, Any], document_id: Optional[str] = None
    ) -> Optional[str]:
        """Add a document to the main database."""
        try:
            # Add timestamp
            data["created_at"] = datetime.now(timezone.utc).isoformat()

            if document_id:
                doc_ref = self.main_db.collection(collection).document(document_id)
                doc_ref.set(data)
                return document_id
            else:
                doc_ref = self.main_db.collection(collection).add(data)
                return doc_ref[1].id  # Returns the document reference

        except Exception as e:
            print(f"Error adding document to {collection}: {e}")
            return None

    def update_document(
        self, collection: str, document_id: str, data: Dict[str, Any]
    ) -> bool:
        """Update a document in the main database."""
        try:
            # Add timestamp
            data["updated_at"] = datetime.now(timezone.utc).isoformat()

            doc_ref = self.main_db.collection(collection).document(document_id)
            doc_ref.update(data)
            return True

        except Exception as e:
            print(f"Error updating document {document_id} in {collection}: {e}")
            return False

    def delete_document(self, collection: str, document_id: str) -> bool:
        """Delete a document from the main database."""
        try:
            doc_ref = self.main_db.collection(collection).document(document_id)
            doc_ref.delete()
            return True

        except Exception as e:
            print(f"Error deleting document {document_id} from {collection}: {e}")
            return False

    def query_collection(
        self,
        collection: str,
        field: Optional[str] = None,
        value: Any = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query documents from the main database."""
        try:
            query = self.main_db.collection(collection)

            if field and value is not None:
                query = query.where(filter=FieldFilter(field, "==", value))

            query = query.limit(limit)
            docs = query.stream()

            results = []
            for doc in docs:
                data = doc.to_dict()
                data["_id"] = doc.id
                results.append(data)

            return results

        except Exception as e:
            print(f"Error querying collection {collection}: {e}")
            return []

    # ================================
    # CONVERSATION DATABASE OPERATIONS
    # ================================

    def append_to_conversation(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """Add a message to the conversation history."""
        try:
            message = {
                "role": role,  # 'user' or 'assistant'
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            # Add to messages subcollection
            doc_ref = self.conversation_db.collection("conversations").document(
                conversation_id
            )
            message_ref = doc_ref.collection("messages").add(message)

            # Update conversation summary
            conversation_data = {
                "last_message_at": message["timestamp"],
                "message_count": firestore.Increment(1),
                "updated_at": message["timestamp"],
            }
            doc_ref.set(conversation_data, merge=True)

            return message_ref[1].id

        except Exception as e:
            print(f"Error appending to conversation {conversation_id}: {e}")
            return None

    def get_recent_conversation(
        self, conversation_id: str, limit: int = 10
    ) -> List[Dict]:
        """Get recent messages from a conversation."""
        try:
            doc_ref = self.conversation_db.collection("conversations").document(
                conversation_id
            )
            messages_ref = doc_ref.collection("messages")

            # Get recent messages ordered by timestamp
            query = messages_ref.order_by(
                "timestamp", direction=firestore.Query.DESCENDING
            ).limit(limit)
            docs = query.stream()

            messages = []
            for doc in docs:
                data = doc.to_dict()
                data["_id"] = doc.id
                messages.append(data)

            # Reverse to get chronological order
            return messages[::-1]

        except Exception as e:
            print(f"Error getting conversation history for {conversation_id}: {e}")
            return []

    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation metadata and summary."""
        try:
            doc_ref = self.conversation_db.collection("conversations").document(
                conversation_id
            )
            doc = doc_ref.get()

            if doc.exists:
                return doc.to_dict()
            return None

        except Exception as e:
            print(f"Error getting conversation summary for {conversation_id}: {e}")
            return None

    # ================================
    # MEMORY DATABASE OPERATIONS
    # ================================

    def store_memory_head(
        self,
        facet: str,
        scope: str,
        claim: str,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """Store a memory head (current belief) in the memory database."""
        try:
            head_data = {
                "facet": facet,
                "scope": scope,
                "claim": claim,
                "confidence": confidence,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            # Use facet_scope as document ID for easy retrieval
            doc_id = f"{facet}_{scope}".replace(" ", "_").lower()
            doc_ref = self.memory_db.collection("heads").document(doc_id)
            doc_ref.set(head_data, merge=True)

            return doc_id

        except Exception as e:
            print(f"Error storing memory head: {e}")
            return None

    def get_memory_heads(
        self, facet: Optional[str] = None, scope: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current memory heads (beliefs) from the memory database."""
        try:
            query = self.memory_db.collection("heads")

            if facet:
                query = query.where(filter=FieldFilter("facet", "==", facet))
            if scope:
                query = query.where(filter=FieldFilter("scope", "==", scope))

            docs = query.stream()

            heads = {}
            for doc in docs:
                data = doc.to_dict()
                heads[doc.id] = data

            return heads

        except Exception as e:
            print(f"Error getting memory heads: {e}")
            return {}

    def store_memory_claim(
        self,
        conversation_id: str,
        claim: str,
        evidence: str,
        confidence: float,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """Store a claim from conversation for later processing by the refiner."""
        try:
            claim_data = {
                "conversation_id": conversation_id,
                "claim": claim,
                "evidence": evidence,
                "confidence": confidence,
                "processed": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            doc_ref = self.memory_db.collection("claims").add(claim_data)
            return doc_ref[1].id

        except Exception as e:
            print(f"Error storing memory claim: {e}")
            return None

    # ================================
    # EMAIL DATABASE OPERATIONS
    # ================================

    def search_emails_by_sender(
        self, sender_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for emails from a specific sender across all reports."""
        try:
            results = []

            # Get all reports
            reports = self.email_db.collection("reports").limit(5).stream()

            for report in reports:
                # Search in sample_emails subcollection
                emails_ref = report.reference.collection("sample_emails")

                # Get all emails and filter by sender name (more flexible search)
                all_emails = emails_ref.limit(
                    100
                ).stream()  # Get more emails for filtering

                for email_doc in all_emails:
                    email_data = email_doc.to_dict()
                    from_addr = email_data.get("from", "").lower()

                    # Check if sender name is in the 'from' field (flexible matching)
                    if sender_name.lower() in from_addr:
                        email_data["_id"] = email_doc.id
                        email_data["_report_id"] = report.id
                        results.append(email_data)

                        if len(results) >= limit:
                            break

                if len(results) >= limit:
                    break

            # Sort by processed_at timestamp (most recent first)
            results.sort(key=lambda x: x.get("processed_at", ""), reverse=True)

            return results[:limit]

        except Exception as e:
            print(f"Error searching emails by sender {sender_name}: {e}")
            return []

    def get_sender_summary(self, sender_name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a specific sender."""
        try:
            # Encode sender name for document ID
            sender_encoded = (
                sender_name.lower()
                .replace("@", "_at_")
                .replace(".", "_dot_")
                .replace("/", "_")
                .replace("#", "_hash_")
            )

            # Search across all reports
            reports = self.email_db.collection("reports").limit(5).stream()

            for report in reports:
                sender_ref = report.reference.collection("top_senders").document(
                    sender_encoded
                )
                sender_doc = sender_ref.get()

                if sender_doc.exists:
                    data = sender_doc.to_dict()
                    data["_report_id"] = report.id

                    # Calculate average confidence if available
                    if data.get("confidence_count", 0) > 0:
                        data["average_confidence"] = data.get(
                            "confidence_sum", 0
                        ) / data.get("confidence_count", 1)

                    return data

            return None

        except Exception as e:
            print(f"Error getting sender summary for {sender_name}: {e}")
            return None

    def search_emails_by_content(
        self, search_term: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for emails containing specific content in subject or body."""
        try:
            results = []
            search_term_lower = search_term.lower()

            # Get all reports
            reports = self.email_db.collection("reports").limit(3).stream()

            for report in reports:
                # Get sample emails
                emails_ref = report.reference.collection("sample_emails")
                emails = emails_ref.limit(50).stream()  # Limit for performance

                for email_doc in emails:
                    email_data = email_doc.to_dict()
                    subject = email_data.get("subject", "").lower()
                    body = email_data.get("body", "").lower()

                    # Check if search term is in subject or body
                    if search_term_lower in subject or search_term_lower in body:
                        email_data["_id"] = email_doc.id
                        email_data["_report_id"] = report.id
                        results.append(email_data)

                        if len(results) >= limit:
                            break

                if len(results) >= limit:
                    break

            # Sort by processed_at timestamp (most recent first)
            results.sort(key=lambda x: x.get("processed_at", ""), reverse=True)

            return results[:limit]

        except Exception as e:
            print(f"Error searching emails by content '{search_term}': {e}")
            return []

    def get_recent_emails(
        self, user_email: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent emails, optionally filtered by user account."""
        try:
            results = []

            # Get all reports
            reports = self.email_db.collection("reports").limit(3).stream()

            for report in reports:
                emails_ref = report.reference.collection("sample_emails")

                # Apply user filter if specified
                if user_email:
                    query = emails_ref.where(
                        filter=FieldFilter("user_account", "==", user_email)
                    ).limit(limit)
                else:
                    query = emails_ref.limit(limit)

                for email_doc in query.stream():
                    email_data = email_doc.to_dict()
                    email_data["_id"] = email_doc.id
                    email_data["_report_id"] = report.id
                    results.append(email_data)

                if len(results) >= limit:
                    break

            # Sort by processed_at timestamp (most recent first)
            results.sort(key=lambda x: x.get("processed_at", ""), reverse=True)

            return results[:limit]

        except Exception as e:
            print(f"Error getting recent emails: {e}")
            return []

    # ================================
    # UTILITY METHODS
    # ================================

    def health_check(self) -> Dict[str, bool]:
        """Check the health of all database connections."""
        health = {}

        try:
            # Test main database
            self.main_db.collection("_health").limit(1).get()
            health["main_db"] = True
        except Exception:
            health["main_db"] = False

        try:
            # Test conversation database
            self.conversation_db.collection("_health").limit(1).get()
            health["conversation_db"] = True
        except Exception:
            health["conversation_db"] = False

        try:
            # Test memory database
            self.memory_db.collection("_health").limit(1).get()
            health["memory_db"] = True
        except Exception:
            health["memory_db"] = False

        try:
            # Test email database
            self.email_db.collection("_health").limit(1).get()
            health["email_db"] = True
        except Exception:
            health["email_db"] = False

        return health

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about all databases."""
        stats = {
            "main_db": {},
            "conversation_db": {},
            "memory_db": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Main database stats
            collections = list(self.main_db.collections())
            stats["main_db"]["collections"] = len(collections)
            stats["main_db"]["collection_names"] = [c.id for c in collections]
        except Exception:
            stats["main_db"]["error"] = "Failed to get stats"

        try:
            # Conversation database stats
            conversations = list(
                self.conversation_db.collection("conversations").limit(1).stream()
            )
            stats["conversation_db"]["has_conversations"] = len(conversations) > 0
        except Exception:
            stats["conversation_db"]["error"] = "Failed to get stats"

        try:
            # Memory database stats
            heads = list(self.memory_db.collection("heads").limit(1).stream())
            claims = list(self.memory_db.collection("claims").limit(1).stream())
            stats["memory_db"]["has_heads"] = len(heads) > 0
            stats["memory_db"]["has_claims"] = len(claims) > 0
        except Exception:
            stats["memory_db"]["error"] = "Failed to get stats"

        return stats


# ================================
# CONVENIENCE FUNCTIONS
# ================================

# Global client instance
_client = None


def get_client() -> FirestoreClient:
    """Get or create the global Firestore client instance."""
    global _client
    if _client is None:
        _client = FirestoreClient()
    return _client


# Direct access functions for easier usage
def get_document(collection: str, document_id: str) -> Optional[Dict[str, Any]]:
    """Get a document from the main database."""
    return get_client().get_document(collection, document_id)


def add_document(
    collection: str, data: Dict[str, Any], document_id: Optional[str] = None
) -> Optional[str]:
    """Add a document to the main database."""
    return get_client().add_document(collection, data, document_id)


def update_document(collection: str, document_id: str, data: Dict[str, Any]) -> bool:
    """Update a document in the main database."""
    return get_client().update_document(collection, document_id, data)


def delete_document(collection: str, document_id: str) -> bool:
    """Delete a document from the main database."""
    return get_client().delete_document(collection, document_id)


def query_collection(
    collection: str, field: Optional[str] = None, value: Any = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """Query documents from the main database."""
    return get_client().query_collection(collection, field, value, limit)


def append_to_conversation(
    conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None
) -> Optional[str]:
    """Add a message to conversation history."""
    return get_client().append_to_conversation(conversation_id, role, content, metadata)


def get_recent_conversation(conversation_id: str, limit: int = 10) -> List[Dict]:
    """Get recent messages from a conversation."""
    return get_client().get_recent_conversation(conversation_id, limit)


def get_memory_heads(
    facet: Optional[str] = None, scope: Optional[str] = None
) -> Dict[str, Any]:
    """Get current memory heads from the memory database."""
    return get_client().get_memory_heads(facet, scope)


def store_memory_head(
    facet: str,
    scope: str,
    claim: str,
    confidence: float = 1.0,
    metadata: Optional[Dict] = None,
) -> Optional[str]:
    """Store a memory head in the memory database."""
    return get_client().store_memory_head(facet, scope, claim, confidence, metadata)


# Email database convenience functions
def search_emails_by_sender(sender_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for emails from a specific sender."""
    return get_client().search_emails_by_sender(sender_name, limit)


def get_sender_summary(sender_name: str) -> Optional[Dict[str, Any]]:
    """Get summary statistics for a specific sender."""
    return get_client().get_sender_summary(sender_name)


def search_emails_by_content(search_term: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for emails containing specific content."""
    return get_client().search_emails_by_content(search_term, limit)


def get_recent_emails(
    user_email: Optional[str] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """Get recent emails, optionally filtered by user account."""
    return get_client().get_recent_emails(user_email, limit)


if __name__ == "__main__":
    # Test script
    print("🔥 Testing Firestore Client...")

    client = FirestoreClient()

    # Health check
    health = client.health_check()
    print(f"📋 Health Check: {health}")

    # Database stats
    stats = client.get_database_stats()
    print(f"📊 Database Stats: {json.dumps(stats, indent=2)}")

    print("✅ Firestore Client test completed!")
