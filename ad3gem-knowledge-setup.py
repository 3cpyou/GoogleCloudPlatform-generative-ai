"""
AD3Gem Knowledge Base Initial Setup
One-time seeding script for canonical knowledge

Features:
- Seeds canonical people, companies, and corrections
- Compatible with ad3gem-emails database structure
- Sets up knowledge base structure
- Processes recent emails for initial entity enrichment

Usage:
- Run once to initialize the knowledge base with canonical data
- Can be re-run safely (uses merge=True for updates)
- Works with the ad3gem-emails database created by Gmail ingestion
"""

import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import google.generativeai as genai
from google.cloud import firestore
from google.cloud.firestore_v1 import Increment

# =========================
# CANONICAL DATA
# =========================

# Monitored email addresses for Craig
MONITORED_EMAILS = {
    "craig@lineagecoffee.com",
    "craig@completecoffeesolutions.com",
    "craigcharity@gmail.com",
    "craig@adddesigns.co.za",
}

# Canonical companies with classifications
STATIC_COMPANIES = {
    # Internal companies
    "Lineage Coffee": {"type": "internal", "aliases": [], "tags": ["cafe", "roastery"]},
    "Complete Coffee Solutions": {
        "type": "internal",
        "aliases": [],
        "tags": ["tech", "training", "services"],
    },
    "ADD Designs": {
        "type": "internal",
        "aliases": ["AD3"],
        "tags": ["software", "automation"],
    },
    "Resileage": {
        "type": "internal",
        "aliases": ["Resileage trading as Lineage Coffee Westville"],
        "tags": ["legal-entity", "westville"],
    },
    # Partners
    "Resilient Flooring": {
        "type": "partner",
        "aliases": [],
        "tags": ["owned-by-pascal"],
    },
    "Wildner and Associates": {
        "type": "partner",
        "aliases": [],
        "tags": ["accountants"],
    },
    "Collective Accounting": {
        "type": "partner",
        "aliases": [],
        "tags": ["accountants"],
    },
    # Suppliers
    "Jersey Cow Company": {
        "type": "supplier",
        "aliases": ["Jersey Cow Company (our dairy)"],
        "tags": ["milk"],
    },
    "La Marzocco South Africa": {
        "type": "supplier",
        "aliases": [],
        "tags": ["equipment"],
    },
    "Equipment Cafe": {"type": "supplier", "aliases": [], "tags": ["equipment"]},
    "Gourmet Coffee": {"type": "supplier", "aliases": [], "tags": ["supplier"]},
    "Parts Hub": {"type": "supplier", "aliases": [], "tags": ["spares"]},
    # Tools/Services
    "Xero": {"type": "tool", "aliases": [], "tags": ["accounting"]},
    "Lightspeed": {"type": "tool", "aliases": [], "tags": ["pos"]},
    "Yoco": {"type": "tool", "aliases": [], "tags": ["card"]},
    "Zapper": {"type": "tool", "aliases": [], "tags": ["payments"]},
    "monday.com": {"type": "tool", "aliases": [], "tags": ["orders", "tracking"]},
    "The Courier Guy": {"type": "tool", "aliases": [], "tags": ["courier"]},
    "Tradify": {"type": "tool", "aliases": [], "tags": ["jobs", "tech"]},
    # Competitors
    "Colombo Coffee": {
        "type": "competitor",
        "aliases": ["Colombo"],
        "tags": ["roaster"],
    },
    "Coastal Coffee": {"type": "competitor", "aliases": [], "tags": []},
    "The Coffee Merchant": {"type": "competitor", "aliases": [], "tags": []},
    "Brewstar": {"type": "competitor", "aliases": [], "tags": []},
    "Starbucks": {"type": "competitor", "aliases": [], "tags": []},
    # Clients
    "Events Coffee": {"type": "client", "aliases": [], "tags": []},
    "Champagne Sports Resort": {"type": "client", "aliases": [], "tags": []},
    "Underberg Superspar": {
        "type": "client",
        "aliases": ["Underberg Spar"],
        "tags": [],
    },
}

# Canonical people with roles and relationships
STATIC_PEOPLE = {
    # Owners & partners
    "Craig Charity": {
        "type": "owner",
        "aliases": [],
        "roles": [
            "Owner Lineage Coffee",
            "50% Complete Coffee Solutions",
            "50% Resileage",
            "Owner ADD Designs",
        ],
        "emails": list(MONITORED_EMAILS),
        "companies": [
            "Lineage Coffee",
            "Complete Coffee Solutions",
            "Resileage",
            "ADD Designs",
        ],
    },
    "Pascal Marot": {
        "type": "partner",
        "aliases": ["Pas"],
        "roles": ["Partner in Resileage", "Owner Resilient Flooring"],
        "companies": ["Resileage", "Resilient Flooring"],
    },
    "Brandon Rochat": {
        "type": "partner",
        "aliases": [],
        "roles": ["Partner in Complete Coffee Solutions"],
        "companies": ["Complete Coffee Solutions"],
    },
    # Staff
    "Julie": {
        "type": "staff",
        "aliases": ["Jules"],
        "roles": ["General Manager, administrator"],
        "companies": ["Lineage Coffee"],
    },
    "Zanele": {
        "type": "staff",
        "aliases": ["Zan"],
        "roles": ["Head of Cafe (Westville)"],
        "companies": ["Lineage Coffee"],
    },
    "Mark Louw": {
        "type": "staff",
        "aliases": ["Mark"],
        "roles": ["Barista training, green coffee roaster"],
        "companies": ["Complete Coffee Solutions", "Lineage Coffee"],
    },
    "Danny": {
        "type": "staff",
        "aliases": [],
        "roles": ["Technical repair division"],
        "companies": ["Complete Coffee Solutions", "Lineage Coffee"],
    },
    # Family
    "Claudine Schafli": {
        "type": "family",
        "aliases": [],
        "roles": ["Craig's girlfriend", "Social media"],
        "companies": ["Lineage Coffee"],
    },
    "Wendy Charity": {
        "type": "family",
        "aliases": [],
        "roles": ["Craig's ex-wife, mother of his children"],
    },
    # External contacts
    "Charles Denison": {
        "type": "external",
        "aliases": ["Charlie"],
        "roles": ["Co-owner of Colombo Coffee"],
        "companies": ["Colombo Coffee"],
    },
    "Ryan Moore": {
        "type": "external",
        "aliases": [],
        "roles": ["Co-owner of Colombo Coffee"],
        "companies": ["Colombo Coffee"],
    },
}

# Common spelling/alias corrections
CORRECTIONS = {
    "Zam": "Zan",
    "Nduli": "Noodles",
    "Nomfundo": "Noodles",
    "My Sindi": "My Sind",
    "Underberg Spar": "Underberg Superspar",
    "Stretta Café": "Stretta Cafe",
}


class AD3GemKnowledgeSetup:
    """Initial setup and seeding for the AD3Gem knowledge base."""

    def __init__(self):
        """Initialize the knowledge base setup."""
        self.PROJECT_ID = "ad3-sam"
        self.KNOWLEDGE_DB_ID = "ad3gem-knowledge"
        self.EMAIL_DB_ID = "ad3gem-emails"

        # Initialize Firestore clients
        self.knowledge_db = firestore.Client(
            project=self.PROJECT_ID, database=self.KNOWLEDGE_DB_ID
        )

        self.email_db = firestore.Client(
            project=self.PROJECT_ID, database=self.EMAIL_DB_ID
        )

        # Initialize Gemini (optional for fact extraction)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
        else:
            print("⚠️ GEMINI_API_KEY not set - fact extraction disabled")
            self.model = None

        print("🧠 AD3Gem Knowledge Base Setup initialized")
        print(f"📚 Knowledge DB: {self.KNOWLEDGE_DB_ID}")
        print(f"📧 Email DB: {self.EMAIL_DB_ID}")

    def initialize_knowledge_structure(self):
        """Create the initial knowledge base collections."""
        print("📋 Initializing knowledge base structure...")

        collections = [
            "entities",  # People, companies, projects
            "relationships",  # Who emails whom, works with whom
            "patterns",  # Email patterns, schedules, habits
            "facts",  # Extracted facts and information
            "topics",  # Common topics and threads
            "temporal",  # Time-based patterns
            "corrections",  # Manual corrections and overrides
            "metadata",  # System metadata
        ]

        for collection in collections:
            doc_ref = self.knowledge_db.collection(collection).document("_init")
            doc_ref.set(
                {
                    "initialized": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "description": f"Knowledge base for {collection}",
                },
                merge=True,
            )

        # Store settings
        self.knowledge_db.collection("metadata").document("settings").set(
            {
                "monitored_emails": list(MONITORED_EMAILS),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            merge=True,
        )

        print("✅ Knowledge base structure initialized")

    def seed_canonical_knowledge(self):
        """Seed canonical people, companies, and corrections."""
        print("🌱 Seeding canonical knowledge...")

        batch = self.knowledge_db.batch()
        batch_count = 0

        # Seed companies
        for name, data in STATIC_COMPANIES.items():
            norm_name = self._normalize_company_name(name)
            doc_ref = self.knowledge_db.collection("entities").document(
                f"company::{norm_name}"
            )

            payload = {
                "entity_id": f"company::{norm_name}",
                "type": "company",
                "names": [name] + data.get("aliases", []),
                "companies": [name],
                "classification": data.get("type", "unknown"),
                "tags": data.get("tags", []),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            batch.set(doc_ref, payload, merge=True)
            batch_count += 1

        # Seed people
        for name, data in STATIC_PEOPLE.items():
            norm_id = self._normalize_person_id(name)
            doc_ref = self.knowledge_db.collection("entities").document(
                f"person::{norm_id}"
            )

            payload = {
                "entity_id": f"person::{norm_id}",
                "type": "person",
                "names": [name] + data.get("aliases", []),
                "email_addresses": data.get("emails", []),
                "roles": data.get("roles", []),
                "companies": data.get("companies", []),
                "category": data.get("type", "unknown"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            batch.set(doc_ref, payload, merge=True)
            batch_count += 1

        # Seed corrections
        for original, corrected in CORRECTIONS.items():
            doc_ref = self.knowledge_db.collection("corrections").document()
            batch.set(
                doc_ref,
                {
                    "type": "spelling_or_alias",
                    "original": original,
                    "corrected": corrected,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "active": True,
                },
                merge=True,
            )
            batch_count += 1

        batch.commit()
        print(
            f"✅ Seeded {len(STATIC_COMPANIES)} companies, {len(STATIC_PEOPLE)} people, {len(CORRECTIONS)} corrections"
        )

    def process_recent_emails_for_enrichment(self, days_back: int = 7):
        """Process recent emails to enrich entities with real data."""
        print("📧 Processing recent emails for entity enrichment...")

        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            emails = self._fetch_recent_emails(cutoff_date)

            if not emails:
                print("📭 No recent emails found")
                return

            print(f"📨 Processing {len(emails)} emails...")

            # Update entity activity
            self._update_entity_activity(emails)

            # Extract basic patterns
            self._extract_basic_patterns(emails)

            # Extract facts if Gemini is available
            if self.model:
                self._extract_facts_sample(emails[:10])  # Limit for initial setup

        except Exception as e:
            print(f"⚠️ Email processing failed: {e}")
            print("Canonical knowledge seeding was successful.")

    def _fetch_recent_emails(self, cutoff_date: datetime) -> List[Dict]:
        """Fetch recent emails from the Gmail ingestion database."""
        emails = []
        monitored_emails = {e.lower() for e in MONITORED_EMAILS}

        # Get recent threads
        thread_docs = (
            self.email_db.collection("threads")
            .where("lastMessageAt", ">=", cutoff_date)
            .order_by("lastMessageAt", direction=firestore.Query.DESCENDING)
            .limit(50)  # Limit for initial setup
            .stream()
        )

        for thread_doc in thread_docs:
            thread_data = thread_doc.to_dict()
            thread_id = thread_doc.id

            # Check if thread involves monitored addresses
            participants = thread_data.get("participantEmails", [])
            if not any(addr.lower() in monitored_emails for addr in participants):
                continue

            # Get messages from this thread
            message_docs = (
                self.email_db.collection("threads")
                .document(thread_id)
                .collection("messages")
                .where("sentAt", ">=", cutoff_date)
                .limit(10)  # Limit messages per thread
                .stream()
            )

            for msg_doc in message_docs:
                msg_data = msg_doc.to_dict()

                # Convert to simple format
                email_data = {
                    "from": self._extract_email_from_field(msg_data.get("from", {})),
                    "to": [
                        self._extract_email_from_field(r)
                        for r in msg_data.get("to", [])
                    ],
                    "subject": msg_data.get("subject", ""),
                    "body": msg_data.get("body", ""),
                    "sentAt": msg_data.get("sentAt"),
                    "thread_subject": thread_data.get("subject", ""),
                }

                emails.append(email_data)

        return emails

    def _extract_email_from_field(self, field):
        """Extract email address from Gmail API field format."""
        if isinstance(field, dict):
            return field.get("email", "")
        return str(field) if field else ""

    def _update_entity_activity(self, emails: List[Dict]):
        """Update entity activity based on email data."""
        entity_activity: Dict[str, int] = defaultdict(int)

        for email in emails:
            # Count sender activity
            from_email = self._normalize_email(email.get("from", ""))
            if from_email:
                entity_activity[from_email] += 1

        # Update entity documents with activity counts
        batch = self.knowledge_db.batch()
        for email, count in entity_activity.items():
            # Try to find existing entity
            entity_docs = (
                self.knowledge_db.collection("entities")
                .where("email_addresses", "array_contains", email)
                .limit(1)
                .stream()
            )

            for entity_doc in entity_docs:
                doc_ref = self.knowledge_db.collection("entities").document(
                    entity_doc.id
                )
                batch.update(
                    doc_ref,
                    {
                        "email_count": Increment(count),
                        "last_seen": datetime.now(timezone.utc).isoformat(),
                    },
                )

        batch.commit()

    def _extract_basic_patterns(self, emails: List[Dict]):
        """Extract basic communication patterns."""
        patterns: Dict[str, Dict] = {
            "sender_frequency": defaultdict(int),
            "hourly_distribution": defaultdict(int),
        }

        for email in emails:
            # Sender frequency
            from_email = self._normalize_email(email.get("from", ""))
            if from_email:
                patterns["sender_frequency"][from_email] += 1

            # Time patterns
            sent_at = email.get("sentAt")
            if sent_at:
                if hasattr(sent_at, "hour"):
                    patterns["hourly_distribution"][sent_at.hour] += 1

        # Store patterns
        doc_ref = self.knowledge_db.collection("patterns").document(
            "communication_patterns"
        )
        doc_ref.set(
            {
                "top_senders": dict(
                    sorted(
                        patterns["sender_frequency"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:20]
                ),
                "hourly_distribution": dict(patterns["hourly_distribution"]),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            merge=True,
        )

    def _extract_facts_sample(self, emails: List[Dict]):
        """Extract facts from a sample of emails using Gemini."""
        if not self.model:
            return

        print("💡 Extracting facts from sample emails...")
        facts = []

        for email in emails[:5]:  # Limit for initial setup
            if len(email.get("body", "")) < 100:
                continue

            prompt = f"""
            Extract important business facts from this email:

            From: {email.get("from", "Unknown")}
            Subject: {email.get("subject", "No subject")}
            Body: {email.get("body", "")[:800]}

            Extract facts in JSON format:
            {{
                "facts": [
                    {{
                        "type": "person/date/price/product/decision/action",
                        "fact": "clear statement of fact",
                        "confidence": 0-100,
                        "entities": ["related entities"],
                        "category": "business/financial/scheduling/operational"
                    }}
                ]
            }}

            Only extract clear, useful facts. Return empty list if none found.
            Return ONLY JSON.
            """

            try:
                response = self.model.generate_content(prompt)
                json_str = response.text.strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:-3]

                result = json.loads(json_str)
                for fact in result.get("facts", []):
                    fact["source_email"] = email.get("from", "")
                    fact["extracted_at"] = datetime.now(timezone.utc).isoformat()
                    facts.append(fact)

            except Exception as e:
                print(f"⚠️ Error extracting facts: {e}")
                continue

        # Store facts
        if facts:
            batch = self.knowledge_db.batch()
            for fact in facts:
                fact_id = hashlib.md5(fact["fact"].encode()).hexdigest()[:12]
                doc_ref = self.knowledge_db.collection("facts").document(fact_id)
                batch.set(doc_ref, fact, merge=True)
            batch.commit()
            print(f"✅ Extracted {len(facts)} facts")

    def _normalize_email(self, email: str) -> str:
        """Normalize email address."""
        if not email:
            return ""

        # Extract from "Name <email>" format
        if "<" in email and ">" in email:
            email = email[email.index("<") + 1 : email.index(">")]

        return email.lower().strip()

    def _normalize_company_name(self, name: str) -> str:
        """Normalize company name for entity ID."""
        # Remove common suffixes and normalize
        cleaned = re.sub(
            r"\b(pty|ltd|company|inc|corp)\b\.?", "", name, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", cleaned.lower()).strip("-")
        return cleaned

    def _normalize_person_id(self, name: str) -> str:
        """Normalize person name for entity ID."""
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
        return cleaned


def main():
    """Main function for initial knowledge base setup."""
    print("🚀 AD3Gem Knowledge Base Initial Setup")
    print("=" * 60)
    print("This script seeds canonical knowledge and processes recent emails")
    print("for initial entity enrichment.")
    print("=" * 60)

    setup = AD3GemKnowledgeSetup()

    try:
        # Initialize structure (safe to run multiple times)
        setup.initialize_knowledge_structure()

        # Seed canonical knowledge (safe to run multiple times)
        setup.seed_canonical_knowledge()

        # Optional: Process recent emails for enrichment
        setup.process_recent_emails_for_enrichment(days_back=7)

        print("\n✅ Knowledge base initial setup complete!")
        print("\n💡 Next steps:")
        print(
            "   1. Run the Gmail ingestion service (main.py) to populate ad3gem-emails"
        )
        print("   2. Use ad3gem-knowledge.py for daily knowledge updates")
        print("   3. Use ad3gem-chatbot.py for intelligent conversations")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check your Firestore permissions and database configuration.")


if __name__ == "__main__":
    main()
