#!/usr/bin/env python3
"""
One-Time Script: Index All Remaining Emails
===========================================

This script processes all emails that are marked as indexed=True but don't actually
have vector embeddings in the email_vectors collection.

The issue: Previous runs marked emails as indexed=True even when vector indexing failed.
This script will:
1. Find emails marked as indexed=True
2. Check if they actually exist in email_vectors collection
3. Re-index the ones that are missing
4. Process in batches to avoid timeouts

Usage:
    python index_all_remaining_emails.py
    python index_all_remaining_emails.py --batch-size 100
    python index_all_remaining_emails.py --dry-run
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Set

# Add the rag directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import NewEmailAutoIndexer

# Configuration
PROJECT_ID = "ad3-sam"
DATABASE_NAME = "ad3gem-gmail-lineage"
REGION = "us-central1"
DEFAULT_BATCH_SIZE = 100

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_vector_indexed_email_ids(db) -> Set[str]:
    """Get all email IDs that actually have vector embeddings"""
    logger.info("🔍 Checking which emails are actually vector indexed...")

    vector_docs = list(db.collection('email_vectors').stream())
    vector_ids = set()

    for doc in vector_docs:
        data = doc.to_dict()
        if data and 'emailDocId' in data:
            vector_ids.add(data['emailDocId'])

    logger.info(f"✅ Found {len(vector_ids)} emails with actual vector embeddings")
    return vector_ids


def get_marked_indexed_emails(db, limit: int = None) -> List[tuple]:
    """Get emails marked as indexed=True"""
    logger.info("🔍 Getting emails marked as indexed=True...")

    query = db.collection('email_search_index').where('indexed', '==', True)
    if limit:
        query = query.limit(limit)

    docs = list(query.stream())
    emails = [(doc.id, doc.to_dict()) for doc in docs]

    logger.info(f"✅ Found {len(emails)} emails marked as indexed=True")
    return emails


def find_missing_vector_emails(db, batch_size: int = 1000) -> List[tuple]:
    """Find emails marked as indexed=True but missing from vector collection"""
    logger.info("🔍 Finding emails that need re-indexing...")

    # Get all vector indexed IDs
    vector_ids = get_vector_indexed_email_ids(db)

    # Get all marked as indexed emails in batches
    missing_emails = []
    offset = 0

    while True:
        logger.info(f"📊 Processing batch starting at offset {offset}...")

        # Get batch of marked emails
        marked_emails = get_marked_indexed_emails(db, limit=batch_size)

        if not marked_emails:
            break

        # Check which ones are missing from vector collection
        for email_id, email_data in marked_emails:
            if email_id not in vector_ids:
                missing_emails.append((email_id, email_data))

        offset += len(marked_emails)

        # If we got fewer than batch_size, we're done
        if len(marked_emails) < batch_size:
            break

    logger.info(f"🎯 Found {len(missing_emails)} emails that need re-indexing")
    return missing_emails


def reset_emails_to_unindexed(db, email_ids: List[str], dry_run: bool = False):
    """Reset emails to indexed=False so they can be processed"""
    if not email_ids:
        return

    logger.info(f"🔄 Resetting {len(email_ids)} emails to indexed=False...")

    if dry_run:
        logger.info("DRY RUN: Would reset emails to indexed=False")
        return

    # Process in batches of 500 (Firestore batch limit)
    batch_size = 500
    for i in range(0, len(email_ids), batch_size):
        batch_ids = email_ids[i:i + batch_size]

        batch = db.batch()
        for email_id in batch_ids:
            doc_ref = db.collection('email_search_index').document(email_id)
            batch.update(doc_ref, {
                'indexed': False,
                'resetForReindexing': True,
                'resetAt': datetime.now(timezone.utc)
            })

        batch.commit()
        logger.info(f"✅ Reset batch {i//batch_size + 1}: {len(batch_ids)} emails")


def main():
    parser = argparse.ArgumentParser(description='Index all remaining emails')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size for processing (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--limit', type=int,
                       help='Limit number of emails to process (for testing)')

    args = parser.parse_args()

    logger.info("🚀 Starting one-time email re-indexing script")
    logger.info(f"📊 Batch size: {args.batch_size}")
    logger.info(f"🔍 Dry run: {args.dry_run}")
    if args.limit:
        logger.info(f"🎯 Limit: {args.limit} emails")

    try:
        # Initialize indexer
        indexer = NewEmailAutoIndexer()
        if not indexer.initialize():
            logger.error("❌ Failed to initialize indexer")
            return 1

        db = indexer.db

        # Find emails that need re-indexing
        missing_emails = find_missing_vector_emails(db, batch_size=1000)

        if not missing_emails:
            logger.info("🎉 All emails are already properly indexed!")
            return 0

        # Apply limit if specified
        if args.limit and len(missing_emails) > args.limit:
            missing_emails = missing_emails[:args.limit]
            logger.info(f"🎯 Limited to {args.limit} emails for processing")

        # Reset emails to unindexed so they can be processed
        email_ids = [email_id for email_id, _ in missing_emails]
        reset_emails_to_unindexed(db, email_ids, dry_run=args.dry_run)

        if args.dry_run:
            logger.info("🔍 DRY RUN: Would process the following emails:")
            for i, (email_id, email_data) in enumerate(missing_emails[:10]):
                subject = email_data.get('subject', 'No subject')[:50]
                logger.info(f"  {i+1}. {email_id}: {subject}...")
            if len(missing_emails) > 10:
                logger.info(f"  ... and {len(missing_emails) - 10} more")
            return 0

        # Process emails in batches
        total_processed = 0
        total_chunks = 0

        for i in range(0, len(missing_emails), args.batch_size):
            batch = missing_emails[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (len(missing_emails) + args.batch_size - 1) // args.batch_size

            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} emails)")

            # Convert to documents
            documents = indexer.to_documents(batch)
            if not documents:
                logger.warning(f"⚠️ No valid documents created for batch {batch_num}")
                continue

            # Index the documents
            chunks = indexer.index_documents(documents, dry_run=False)

            # Mark as indexed
            batch_email_ids = [email_id for email_id, _ in batch]
            indexer.mark_indexed(batch_email_ids, dry_run=False)

            total_processed += len(batch)
            total_chunks += chunks

            logger.info(f"✅ Batch {batch_num} complete: {len(batch)} emails, {chunks} chunks")
            logger.info(f"📊 Progress: {total_processed}/{len(missing_emails)} emails processed")

        logger.info("🎉 Re-indexing complete!")
        logger.info(f"📊 Total processed: {total_processed} emails")
        logger.info(f"📊 Total chunks created: {total_chunks}")

        return 0

    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
