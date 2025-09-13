#!/usr/bin/env python3
"""
Bulk Email Indexer - Cloud Run Service
=====================================

One-time service to index all remaining emails that are marked as indexed=True
but don't actually have vector embeddings.

Uses the same working setup as the successful Cloud Function deployment.

HTTP Endpoints:
- GET / -> health + status
- POST / -> start bulk indexing
- POST /?dry_run=1 -> simulate without changes
- POST /?batch_size=100 -> override batch size
- POST /?limit=1000 -> limit number of emails to process

Environment Variables:
- GOOGLE_CLOUD_PROJECT_ID (e.g. ad3-sam)
- DATABASE_NAME (e.g. ad3gem-gmail-lineage)
- GOOGLE_CLOUD_REGION (e.g. us-central1)
- BATCH_SIZE (default 100)
"""

import os
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Any, Set, Tuple, Optional

import vertexai
from google.cloud import firestore

# LangChain (0.3 line) - same as working Cloud Function
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore

# Flask for Cloud Run
from flask import Flask, jsonify, request

# ------------ Config ------------

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "ad3-sam")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "ad3gem-gmail-lineage")
REGION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
VECTOR_COLLECTION = os.environ.get("VECTOR_COLLECTION", "email_vectors")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bulk_email_indexer")

app = Flask(__name__)


class BulkEmailIndexer:
    """Bulk indexer for all remaining emails"""

    def __init__(self):
        self.project_id = PROJECT_ID
        self.database_name = DATABASE_NAME
        self.region = REGION
        self.db: Optional[firestore.Client] = None
        self.embeddings: Optional[VertexAIEmbeddings] = None

    def initialize(self) -> bool:
        """Initialize Firestore + Vertex AI."""
        try:
            if not self.project_id or not self.database_name:
                raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT_ID or DATABASE_NAME")

            # Firestore (Native) client
            self.db = firestore.Client(project=self.project_id, database=self.database_name)
            logger.info("✅ Firestore connected | db=%s", self.database_name)

            # Vertex AI
            vertexai.init(project=self.project_id, location=self.region)
            self.embeddings = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
            logger.info("✅ Vertex AI ready | region=%s model=%s", self.region, EMBEDDING_MODEL)
            return True

        except Exception as e:
            logger.exception("❌ init failed: %s", e)
            return False

    def get_vector_indexed_email_ids(self) -> Set[str]:
        """Get all email IDs that actually have vector embeddings"""
        logger.info("🔍 Checking which emails are actually vector indexed...")

        vector_docs = list(self.db.collection(VECTOR_COLLECTION).stream())
        vector_ids = set()

        for doc in vector_docs:
            data = doc.to_dict()
            if data and 'emailDocId' in data:
                vector_ids.add(data['emailDocId'])

        logger.info("✅ Found %d emails with actual vector embeddings", len(vector_ids))
        return vector_ids

    def get_marked_indexed_emails(self, limit: int = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Get emails marked as indexed=True"""
        logger.info("🔍 Getting emails marked as indexed=True...")

        query = self.db.collection('email_search_index').where('indexed', '==', True)
        if limit:
            query = query.limit(limit)

        docs = list(query.stream())
        emails = [(doc.id, doc.to_dict() or {}) for doc in docs]

        logger.info("✅ Found %d emails marked as indexed=True", len(emails))
        return emails

    def find_missing_vector_emails(self, limit: int = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Find emails marked as indexed=True but missing from vector collection"""
        logger.info("🔍 Finding emails that need re-indexing...")

        # Get all vector indexed IDs
        vector_ids = self.get_vector_indexed_email_ids()

        # Get all marked as indexed emails
        marked_emails = self.get_marked_indexed_emails(limit=limit)

        # Find missing ones
        missing_emails = []
        for email_id, email_data in marked_emails:
            if email_id not in vector_ids:
                missing_emails.append((email_id, email_data))

        logger.info("🎯 Found %d emails that need re-indexing", len(missing_emails))
        return missing_emails

    def reset_emails_to_unindexed(self, email_ids: List[str], dry_run: bool = False) -> bool:
        """Reset emails to indexed=False so they can be processed"""
        if not email_ids:
            return True

        logger.info("🔄 Resetting %d emails to indexed=False...", len(email_ids))

        if dry_run:
            logger.info("DRY RUN: Would reset emails to indexed=False")
            return True

        try:
            # Process in batches of 500 (Firestore batch limit)
            batch_size = 500
            for i in range(0, len(email_ids), batch_size):
                batch_ids = email_ids[i:i + batch_size]

                batch = self.db.batch()
                for email_id in batch_ids:
                    doc_ref = self.db.collection('email_search_index').document(email_id)
                    batch.update(doc_ref, {
                        'indexed': False,
                        'resetForReindexing': True,
                        'resetAt': datetime.now(timezone.utc)
                    })

                batch.commit()
                logger.info("✅ Reset batch %d: %d emails", i//batch_size + 1, len(batch_ids))

            return True

        except Exception as e:
            logger.exception("❌ Failed to reset emails: %s", e)
            return False

    @staticmethod
    def compose_page_content(email: Dict[str, Any]) -> str:
        """Convert email data to text content"""
        parts = []
        if email.get("subject"):
            parts.append(f"Subject: {email['subject']}")
        if email.get("fromEmailLc"):
            parts.append(f"From: {email['fromEmailLc']}")
        if email.get("sentDate"):
            parts.append(f"Date: {email['sentDate']}")
        if email.get("snippet"):
            parts.append(f"Snippet: {email['snippet']}")
        if email.get("bodyPreview"):
            parts.append(f"Body: {email['bodyPreview']}")
        return "\n".join(parts).strip()

    def to_documents(self, batch: List[Tuple[str, Dict[str, Any]]]) -> List[Document]:
        """Convert email batch to LangChain Documents"""
        docs: List[Document] = []
        for email_id, email in batch:
            content = self.compose_page_content(email)
            if not content:
                logger.warning("Skipping empty email content id=%s", email_id)
                continue
            metadata = {
                "messageId": email.get("messageId", email_id),
                "threadId": email.get("threadId", ""),
                "fromEmail": email.get("fromEmailLc", ""),
                "sentDate": email.get("sentDate", ""),
                "owner": email.get("owner", ""),
                "hasAttachments": email.get("hasAttachments", False),
                "source": "email_search_index",
                "emailDocId": email_id,
            }
            docs.append(Document(page_content=content, metadata=metadata))
        logger.info("Built %d Document objects from %d emails", len(docs), len(batch))
        return docs

    def index_documents(self, documents: List[Document], dry_run: bool = False) -> int:
        """Index documents in Firestore vector store"""
        if not documents:
            return 0
        try:
            assert self.embeddings is not None and self.db is not None

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_documents(documents)
            logger.info("Split %d docs -> %d chunks", len(documents), len(chunks))

            if dry_run:
                logger.info("DRY RUN: skipping vector upsert")
                return len(chunks)

            # Upsert into Firestore Vector Store
            FirestoreVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection=VECTOR_COLLECTION,
                client=self.db,
            )
            logger.info("✅ Upserted %d chunks into '%s'", len(chunks), VECTOR_COLLECTION)
            return len(chunks)

        except Exception as e:
            logger.exception("index_documents failed: %s", e)
            raise

    def mark_indexed(self, email_ids: List[str], dry_run: bool = False) -> bool:
        """Mark emails as indexed to prevent reprocessing"""
        if not email_ids:
            return True
        try:
            assert self.db is not None
            if dry_run:
                logger.info("DRY RUN: skipping mark_indexed for %d ids", len(email_ids))
                return True

            batch = self.db.batch()
            now = datetime.now(timezone.utc)
            for eid in email_ids:
                ref = self.db.collection("email_search_index").document(eid)
                batch.update(ref, {"indexed": True, "indexedAt": now})
            batch.commit()
            logger.info("✅ Marked %d emails as indexed", len(email_ids))
            return True
        except Exception as e:
            logger.exception("mark_indexed failed: %s", e)
            return False

    def run_bulk_indexing(self, batch_size: int, limit: int = None, dry_run: bool = False) -> Dict[str, Any]:
        """Run bulk indexing of all remaining emails"""
        try:
            # Find emails that need re-indexing
            missing_emails = self.find_missing_vector_emails(limit=limit)

            if not missing_emails:
                return {
                    "status": "success",
                    "message": "All emails are already properly indexed!",
                    "processed": 0,
                    "chunks_created": 0
                }

            logger.info("🎯 Found %d emails that need re-indexing", len(missing_emails))

            # Reset emails to unindexed so they can be processed
            email_ids = [email_id for email_id, _ in missing_emails]
            if not self.reset_emails_to_unindexed(email_ids, dry_run=dry_run):
                return {
                    "status": "error",
                    "message": "Failed to reset emails to unindexed",
                    "processed": 0
                }

            if dry_run:
                return {
                    "status": "success",
                    "message": f"DRY RUN: Would process {len(missing_emails)} emails",
                    "processed": 0,
                    "chunks_created": 0,
                    "dry_run": True
                }

            # Process emails in batches
            total_processed = 0
            total_chunks = 0

            for i in range(0, len(missing_emails), batch_size):
                batch = missing_emails[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(missing_emails) + batch_size - 1) // batch_size

                logger.info("📦 Processing batch %d/%d (%d emails)", batch_num, total_batches, len(batch))

                # Convert to documents
                documents = self.to_documents(batch)
                if not documents:
                    logger.warning("⚠️ No valid documents created for batch %d", batch_num)
                    continue

                # Index the documents
                chunks = self.index_documents(documents, dry_run=False)

                # Mark as indexed
                batch_email_ids = [email_id for email_id, _ in batch]
                self.mark_indexed(batch_email_ids, dry_run=False)

                total_processed += len(batch)
                total_chunks += chunks

                logger.info("✅ Batch %d complete: %d emails, %d chunks", batch_num, len(batch), chunks)
                logger.info("📊 Progress: %d/%d emails processed", total_processed, len(missing_emails))

            return {
                "status": "success",
                "message": f"Bulk indexing complete! Processed {total_processed} emails",
                "processed": total_processed,
                "chunks_created": total_chunks,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error("run_bulk_indexing failed: %s", e)
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e),
                "processed": 0
            }


# ------------ Flask Routes ------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service": "Bulk Email Indexer",
        "description": "One-time service to index all remaining emails",
        "database": DATABASE_NAME,
        "project": PROJECT_ID,
        "region": REGION,
        "batch_size": BATCH_SIZE,
        "model": EMBEDDING_MODEL,
        "vector_collection": VECTOR_COLLECTION,
        "status": "ready"
    }), 200

@app.route('/index', methods=['POST'])
def start_bulk_indexing():
    """Start bulk indexing of all remaining emails"""
    try:
        # Get parameters
        batch_size = int(request.args.get("batch_size", BATCH_SIZE))
        limit = request.args.get("limit")
        limit = int(limit) if limit else None
        dry_run = request.args.get("dry_run", "0") in ("1", "true", "True")

        logger.info("🚀 Starting bulk indexing")
        logger.info("📊 Batch size: %d", batch_size)
        logger.info("🔍 Dry run: %s", dry_run)
        if limit:
            logger.info("🎯 Limit: %d emails", limit)

        # Initialize indexer
        indexer = BulkEmailIndexer()
        if not indexer.initialize():
            return jsonify({
                "status": "error",
                "message": "Failed to initialize indexer"
            }), 500

        # Run bulk indexing
        result = indexer.run_bulk_indexing(
            batch_size=batch_size,
            limit=limit,
            dry_run=dry_run
        )

        code = 200 if result.get("status") == "success" else 500
        return jsonify(result), code

    except Exception as e:
        logger.exception("Unhandled error in bulk indexing")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
