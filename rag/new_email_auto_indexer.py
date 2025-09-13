#!/usr/bin/env python3
"""
Auto-Indexer for New Emails - Bolt-On Service
Guarantees all new emails are automatically vector-indexed for RAG search

This service:
1. Monitors email_search_index for unindexed emails
2. Vector indexes them using the same logic as firestore_rag.py
3. Marks them as indexed to prevent duplicates
4. Can run as Cloud Function, Cloud Run, or scheduled script

Usage:
- Cloud Function: Deploy with gcloud functions deploy
- Manual run: python new_email_auto_indexer.py
- Scheduled: Set up Cloud Scheduler to trigger periodically
"""

import os
import sys
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import json

# Google Cloud imports
import vertexai
from google.cloud import firestore

# RAG imports
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_vertexai import VertexAIEmbeddings
    from langchain_google_firestore import FirestoreVectorStore
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAG dependencies not available: {e}")
    RAG_AVAILABLE = False

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "ad3-sam")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "ad3gem-gmail-lineage")
REGION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
BATCH_SIZE = int(os.environ.get("INDEXING_BATCH_SIZE", "50"))

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewEmailAutoIndexer:
    """Automatically indexes new emails for RAG search"""

    def __init__(self):
        self.project_id = PROJECT_ID
        self.database_name = DATABASE_NAME
        self.region = REGION
        self.db = None
        self.embeddings = None

    def initialize(self):
        """Initialize Firestore and Vertex AI connections"""
        try:
            # Initialize Firestore
            self.db = firestore.Client(project=self.project_id, database=self.database_name)
            logger.info(f"✅ Connected to Firestore database: {self.database_name}")

            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.region)
            self.embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
            logger.info(f"✅ Initialized Vertex AI embeddings in region: {self.region}")

            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            return False

    def get_unindexed_emails(self, limit: int = BATCH_SIZE) -> List[Dict]:
        """Get emails that haven't been vector indexed yet"""
        try:
            # Query for emails without 'indexed' field or where indexed=False
            query = (
                self.db.collection('email_search_index')
                .where('indexed', '==', False)
                .limit(limit)
            )

            docs = list(query.stream())
            logger.info(f"Found {len(docs)} unindexed emails")

            return [(doc.id, doc.to_dict()) for doc in docs]

        except Exception as e:
            logger.error(f"❌ Error getting unindexed emails: {e}")
            return []

    def get_emails_without_indexed_field(self, limit: int = BATCH_SIZE) -> List[Dict]:
        """Get emails that don't have the 'indexed' field at all (legacy emails)"""
        try:
            # Get all emails and filter out ones that have 'indexed' field
            query = self.db.collection('email_search_index').limit(limit * 2)  # Get extra to filter
            docs = list(query.stream())

            unindexed = []
            for doc in docs:
                data = doc.to_dict()
                if 'indexed' not in data:  # No indexed field = needs indexing
                    unindexed.append((doc.id, data))
                    if len(unindexed) >= limit:
                        break

            logger.info(f"Found {len(unindexed)} emails without indexed field")
            return unindexed

        except Exception as e:
            logger.error(f"❌ Error getting emails without indexed field: {e}")
            return []

    def compose_page_content(self, email_data: Dict[str, Any]) -> str:
        """Convert email data to text content (same logic as firestore_rag.py)"""
        lines = []

        # Add subject
        if email_data.get('subject'):
            lines.append(f"Subject: {email_data['subject']}")

        # Add sender info
        if email_data.get('fromEmailLc'):
            lines.append(f"From: {email_data['fromEmailLc']}")

        # Add date
        if email_data.get('sentDate'):
            lines.append(f"Date: {email_data['sentDate']}")

        # Add snippet
        if email_data.get('snippet'):
            lines.append(f"Snippet: {email_data['snippet']}")

        # Add body preview
        if email_data.get('bodyPreview'):
            lines.append(f"Body: {email_data['bodyPreview']}")

        return "\n".join(lines).strip()

    def create_langchain_documents(self, email_batch: List[tuple]) -> List[Document]:
        """Convert email batch to LangChain Documents"""
        documents = []

        for email_id, email_data in email_batch:
            try:
                page_content = self.compose_page_content(email_data)

                if not page_content:
                    logger.warning(f"Empty content for email {email_id}, skipping")
                    continue

                metadata = {
                    "messageId": email_data.get("messageId", email_id),
                    "threadId": email_data.get("threadId", ""),
                    "fromEmail": email_data.get("fromEmailLc", ""),
                    "sentDate": email_data.get("sentDate", ""),
                    "owner": email_data.get("owner", ""),
                    "hasAttachments": email_data.get("hasAttachments", False),
                }

                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)

            except Exception as e:
                logger.warning(f"Failed to process email {email_id}: {e}")
                continue

        logger.info(f"Created {len(documents)} LangChain documents from {len(email_batch)} emails")
        return documents

    def index_documents(self, documents: List[Document]) -> int:
        """Index documents in Firestore vector store"""
        if not documents:
            return 0

        try:
            # Split documents into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

            # Add to vector store
            FirestoreVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection="email_vectors",
                client=self.db,
            )

            logger.info(f"✅ Successfully indexed {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"❌ Failed to index documents: {e}")
            raise

    def mark_as_indexed(self, email_ids: List[str]) -> bool:
        """Mark emails as indexed to prevent reprocessing"""
        if not email_ids:
            return True

        try:
            batch = self.db.batch()
            timestamp = datetime.now(timezone.utc)

            for email_id in email_ids:
                doc_ref = self.db.collection('email_search_index').document(email_id)
                batch.update(doc_ref, {
                    'indexed': True,
                    'indexedAt': timestamp
                })

            batch.commit()
            logger.info(f"✅ Marked {len(email_ids)} emails as indexed")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to mark emails as indexed: {e}")
            return False

    def process_batch(self, batch_size: int = BATCH_SIZE) -> Dict[str, Any]:
        """Process a batch of unindexed emails"""
        if not RAG_AVAILABLE:
            return {
                "status": "error",
                "message": "RAG dependencies not available",
                "processed": 0
            }

        try:
            # Get unindexed emails (try both methods)
            unindexed_emails = self.get_unindexed_emails(batch_size)

            # If no emails with indexed=False, check for emails without indexed field
            if not unindexed_emails:
                unindexed_emails = self.get_emails_without_indexed_field(batch_size)

            if not unindexed_emails:
                return {
                    "status": "success",
                    "message": "No unindexed emails found",
                    "processed": 0
                }

            # Convert to LangChain documents
            documents = self.create_langchain_documents(unindexed_emails)

            if not documents:
                return {
                    "status": "error",
                    "message": "No valid documents created",
                    "processed": 0
                }

            # Index the documents
            chunks_created = self.index_documents(documents)

            # Mark as indexed
            email_ids = [email_id for email_id, _ in unindexed_emails]
            marked_success = self.mark_as_indexed(email_ids)

            if not marked_success:
                logger.warning("Failed to mark some emails as indexed")

            return {
                "status": "success",
                "message": f"Successfully processed {len(email_ids)} emails",
                "processed": len(email_ids),
                "chunks_created": chunks_created,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "processed": 0
            }

def index_new_emails(request=None):
    """
    Main entry point for Cloud Function or HTTP trigger
    Can also be called directly for manual execution
    """
    logger.info("🚀 Starting new email auto-indexer")

    # Create indexer instance
    indexer = NewEmailAutoIndexer()

    # Initialize connections
    if not indexer.initialize():
        return {
            "status": "error",
            "message": "Failed to initialize indexer"
        }

    # Process a batch
    result = indexer.process_batch()

    logger.info(f"🏁 Indexing completed: {result}")
    return result

# For direct execution
if __name__ == "__main__":
    # Manual execution
    result = index_new_emails()
    print(json.dumps(result, indent=2, default=str))
