"""
AD3Gem Knowledge Base Initial Setup
One-time seeding and initialization script for the knowledge base

Features:
- Static knowledge seeding (people, aliases, companies, ownership)
- Canonical entity definitions with roles and relationships
- Misspelling & alias corrections (strip 'Pty Ltd', nicknames, etc.)
- Company classification (internal/supplier/competitor/partner/client)
- Entity enrichment using predefined canonical data
- Compatible with ad3gem-emails database structure from Gmail ingestion

Usage:
- Run once to initialize the knowledge base with canonical data
- Can be re-run safely (uses merge=True for updates)
- Works with the ad3gem-emails database created by main.py Gmail ingestion
"""

import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import google.generativeai as genai
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter, Increment

# =========================
# CONFIG & CANONICAL DATA
# =========================

CONFIG = {
    # Only process emails sent to/cc’d to these inboxes
    "MONITORED_EMAILS": {
        "craig@lineagecoffee.com",
        "craig@completecoffeesolutions.com",
        "craigcharity@gmail.com",
        "craig@adddesigns.co.za",
    },
    # Normalization flags/policies
    "REMOVE_PTY_LTD": True,
    "MAX_EMAILS_FACTS": 20,
    "EMAIL_FETCH_LIMIT": 500,
    "SUBJECT_SNIPPET_LEN": 100,
    "BODY_SNIPPET_LEN": 500,
}

# Canonical companies (merged per your latest instructions; no “Pty Ltd”)
STATIC_COMPANIES: Dict[str, Dict[str, Any]] = {
    # Internal / yours
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
    # Partners / related
    "Resilient Flooring": {
        "type": "partner",
        "aliases": [],
        "tags": ["owned-by-pascal"],
    },
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
    "Xero": {"type": "tool", "aliases": [], "tags": ["accounting"]},
    "Lightspeed": {"type": "tool", "aliases": [], "tags": ["pos"]},
    "Yoco": {"type": "tool", "aliases": [], "tags": ["card"]},
    "Zapper": {"type": "tool", "aliases": [], "tags": ["payments"]},
    "monday.com": {"type": "tool", "aliases": [], "tags": ["orders", "tracking"]},
    "The Courier Guy": {"type": "tool", "aliases": [], "tags": ["courier"]},
    "Tradify": {"type": "tool", "aliases": [], "tags": ["jobs", "tech"]},
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
    # Competitors / market
    "Colombo Coffee": {
        "type": "competitor",
        "aliases": ["Colombo"],
        "tags": ["roaster"],
    },
    "Coastal Coffee": {"type": "competitor", "aliases": [], "tags": []},
    "The Coffee Merchant": {"type": "competitor", "aliases": [], "tags": []},
    "Brewstar": {"type": "competitor", "aliases": [], "tags": []},
    "Tribeca": {"type": "competitor", "aliases": [], "tags": []},
    "Ciro": {
        "type": "competitor",
        "aliases": ["Ciro (Lavazza contract)"],
        "tags": ["Lavazza"],
    },
    "Manna Coffee": {"type": "competitor", "aliases": [], "tags": []},
    "Terbadore": {"type": "competitor", "aliases": [], "tags": []},
    "Stretta Cafe": {
        "type": "competitor",
        "aliases": ["Stretta Café"],
        "tags": ["Hillcrest"],
    },
    "Oscar's": {"type": "competitor", "aliases": [], "tags": []},
    "Seattle Coffee Company": {
        "type": "competitor",
        "aliases": ["Seattle"],
        "tags": [],
    },
    "Plato": {"type": "competitor", "aliases": [], "tags": []},
    "Bootlegger Coffee Company": {
        "type": "competitor",
        "aliases": ["Bootlegger"],
        "tags": [],
    },
    "Woolworths Café": {
        "type": "competitor",
        "aliases": ["Woolworths Cafe"],
        "tags": [],
    },
    "Mugg & Bean": {"type": "competitor", "aliases": [], "tags": []},
    "Vida e Caffè": {"type": "competitor", "aliases": [], "tags": []},
    "Starbucks": {"type": "competitor", "aliases": [], "tags": []},
    "The Coffee Minista": {"type": "competitor", "aliases": [], "tags": []},
    # Clients / venues
    "Events Coffee": {"type": "client", "aliases": [], "tags": []},
    "Champagne Sports Resort": {"type": "client", "aliases": [], "tags": []},
    "Underberg Superspar": {
        "type": "client",
        "aliases": ["Underberg Spar"],
        "tags": [],
    },
}

# Canonical people (owners/partners/staff/family/external)
STATIC_PEOPLE: Dict[str, Dict[str, Any]] = {
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
        "emails": list(CONFIG["MONITORED_EMAILS"]),
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
    # Staff (per your last list)
    "Zanele": {
        "type": "staff",
        "aliases": ["Zan"],
        "roles": ["Head of Cafe (Westville)"],
        "companies": ["Lineage Coffee"],
    },
    "Sharon": {
        "type": "staff",
        "aliases": [],
        "roles": ["Cafe Assistant (Westville)"],
        "companies": ["Lineage Coffee"],
    },
    "Jabu": {
        "type": "staff",
        "aliases": [],
        "roles": ["Barista (Westville)"],
        "companies": ["Lineage Coffee"],
    },
    "Cyprian": {
        "type": "staff",
        "aliases": ["CP"],
        "roles": ["Barista (Westville)"],
        "companies": ["Lineage Coffee"],
    },
    "Fani": {
        "type": "staff",
        "aliases": ["Fana Man"],
        "roles": ["Roastery staff, explains coffee"],
        "companies": ["Lineage Coffee"],
    },
    "Julie": {
        "type": "staff",
        "aliases": ["Jules"],
        "roles": ["General Manager, administrator"],
        "companies": ["Lineage Coffee"],
    },
    "Sindi": {
        "type": "staff",
        "aliases": ["My Sind"],
        "roles": ["Cafe staff"],
        "companies": ["Lineage Coffee"],
    },
    "Ntokozo": {
        "type": "staff",
        "aliases": ["TK"],
        "roles": ["Cafe staff"],
        "companies": ["Lineage Coffee"],
    },
    "Andy": {
        "type": "staff",
        "aliases": [],
        "roles": ["Cafe staff"],
        "companies": ["Lineage Coffee"],
    },
    "Noodles": {
        "type": "staff",
        "aliases": ["Nduli", "Nomfundo"],
        "roles": ["Cafe staff"],
        "companies": ["Lineage Coffee"],
    },
    "Roxanne": {
        "type": "staff",
        "aliases": ["Rox"],
        "roles": ["Cafe staff"],
        "companies": ["Lineage Coffee"],
    },
    "Nomvula": {
        "type": "staff",
        "aliases": ["Noms"],
        "roles": ["Cafe staff"],
        "companies": ["Lineage Coffee"],
    },
    "Danny": {
        "type": "staff",
        "aliases": [],
        "roles": ["Technical repair division"],
        "companies": ["Complete Coffee Solutions", "Lineage Coffee"],
    },
    "Mark Louw": {
        "type": "staff",
        "aliases": ["Mark"],
        "roles": ["Barista training, green coffee roaster"],
        "companies": ["Complete Coffee Solutions", "Lineage Coffee"],
    },
    # Family
    "Claudine Schafli": {
        "type": "family",
        "aliases": [],
        "roles": ["Craig’s girlfriend", "Social media"],
        "companies": ["Lineage Coffee"],
    },
    "Patrick Schafli": {"type": "family", "aliases": [], "roles": ["Claudine’s son"]},
    "Grace Schafli": {
        "type": "family",
        "aliases": [],
        "roles": ["Claudine’s daughter"],
    },
    "Levi Charity": {"type": "family", "aliases": [], "roles": ["Craig’s child"]},
    "Emily Charity": {"type": "family", "aliases": [], "roles": ["Craig’s child"]},
    "Rebecca Charity": {"type": "family", "aliases": [], "roles": ["Craig’s child"]},
    "Wendy Charity": {
        "type": "family",
        "aliases": [],
        "roles": ["Craig’s ex-wife, mother of his children"],
    },
    "Elizabeth": {
        "type": "family",
        "aliases": ["Liz"],
        "roles": ["Marketing contributor", "Married to Jared"],
        "children": ["Sebastian"],
    },
    "Jared": {
        "type": "family",
        "aliases": [],
        "roles": ["Brandon’s son", "Married to Elizabeth"],
    },
    "Sebastian": {
        "type": "family",
        "aliases": [],
        "roles": ["Child of Jared and Elizabeth"],
    },
    "Carolyn": {"type": "family", "aliases": [], "roles": ["Brandon’s wife"]},
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
    "Surette": {
        "type": "external",
        "aliases": [],
        "roles": ["Administrator for Pascal’s company"],
        "companies": ["Resilient Flooring"],
    },
    "Clayton": {
        "type": "external",
        "aliases": [],
        "roles": ["Wildner and Associates"],
        "companies": ["Wildner and Associates"],
    },
    "Ross": {
        "type": "external",
        "aliases": [],
        "roles": ["Collective Accounting"],
        "companies": ["Collective Accounting"],
    },
}

# Common spelling/alias corrections (applied to inbound text)
CORRECTIONS = {
    # People nicknames
    "Zam": "Zan",
    "Nduli": "Noodles",
    "Nomfundo": "Noodles",
    "My Sindi": "My Sind",
    # Companies & brands
    "Underberg Spar": "Underberg Superspar",
    "Stretta Café": "Stretta Cafe",
    # Optionally add: "Terbodore": "Terbadore" or vice-versa if you want to force one spelling
}

# ====================================
# ORIGINAL CLASS WITH ENHANCEMENTS
# ====================================


class AD3GemKnowledgeBase:
    """
    Manages the central knowledge base that gets updated daily
    and provides context for the chatbot.
    """

    def __init__(self):
        """Initialize the knowledge base manager."""
        # Setup Firestore for knowledge base
        self.PROJECT_ID = "ad3-sam"
        self.KNOWLEDGE_DB_ID = "ad3gem-knowledge"
        self.EMAIL_DB_ID = "ad3gem-emails"  # Updated to match Gmail ingestion

        # Initialize Firestore clients
        self.knowledge_db = firestore.Client(
            project=self.PROJECT_ID, database=self.KNOWLEDGE_DB_ID
        )

        self.email_db = firestore.Client(
            project=self.PROJECT_ID, database=self.EMAIL_DB_ID
        )

        # Initialize Gemini for analysis
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            "gemini-1.5-pro",
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            },
        )

        print("🧠 AD3Gem Knowledge Base initialized")
        print(f"📚 Knowledge DB: {self.KNOWLEDGE_DB_ID}")
        print(f"📧 Email DB: {self.EMAIL_DB_ID} (Gmail ingestion structure)")

    # ---------------------
    # Initialization / Seed
    # ---------------------

    def initialize_knowledge_structure(self):
        """Create the initial knowledge base structure if it doesn't exist."""
        collections = [
            "entities",  # People, companies, projects
            "relationships",  # Who emails whom, works with whom
            "patterns",  # Email patterns, schedules, habits
            "facts",  # Extracted facts and information
            "topics",  # Common topics and threads
            "preferences",  # User and sender preferences
            "temporal",  # Time-based patterns
            "business_rules",  # Business logic and rules
            "corrections",  # Manual corrections and overrides
            "metadata",
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

        # Store monitored emails & normalization policies
        self.knowledge_db.collection("metadata").document("settings").set(
            {
                "monitored_emails": list(CONFIG["MONITORED_EMAILS"]),
                "remove_pty_ltd": CONFIG["REMOVE_PTY_LTD"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            merge=True,
        )

        print("✅ Knowledge base structure initialized")

    def seed_static_knowledge(self):
        """
        Seed canonical people, companies, and corrections into Firestore.
        Safe to run multiple times (merge=True).
        """
        print("🌱 Seeding static knowledge...")

        batch = self.knowledge_db.batch()

        # Companies
        for name, data in STATIC_COMPANIES.items():
            norm_name = self._normalize_company(name)
            doc_ref = self.knowledge_db.collection("entities").document(
                f"company::{norm_name}"
            )
            payload = {
                "entity_id": f"company::{norm_name}",
                "type": "company",
                "names": [name] + data.get("aliases", []),
                "companies": [name],  # Company entity references itself
                "class": data.get("type", "unknown"),
                "tags": data.get("tags", []),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            batch.set(doc_ref, payload, merge=True)

        # People
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
                "children": data.get("children", []),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            batch.set(doc_ref, payload, merge=True)

        # Corrections
        for original, corrected in CORRECTIONS.items():
            cdoc = self.knowledge_db.collection("corrections").document()
            batch.set(
                cdoc,
                {
                    "type": "spelling_or_alias",
                    "original": original,
                    "corrected": corrected,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "active": True,
                },
                merge=True,
            )

        batch.commit()
        print("✅ Static knowledge seeded")

    # ---------------
    # Daily Updater
    # ---------------

    def update_daily_knowledge(self, days_back: int = 1):
        """Main method to update knowledge base from recent emails."""
        print("\n🔄 Starting daily knowledge update...")

        # Get yesterday's emails (or specified days back)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        emails = self._fetch_recent_emails(cutoff_date)

        if not emails:
            print("📭 No new emails to process")
            return

        print(f"📧 Processing {len(emails)} emails...")

        # Process different aspects of knowledge
        self._update_entities(emails)
        self._update_relationships(emails)
        self._update_patterns(emails)
        self._extract_facts(emails)
        self._update_topics(emails)
        self._update_temporal_patterns(emails)

        # Mark update completion
        self._mark_update_complete()

        print("✅ Knowledge base updated successfully")

    # -----------------
    # Email collection
    # -----------------

    def _fetch_recent_emails(self, cutoff_date: datetime) -> List[Dict]:
        """
        Fetch emails from the Gmail ingestion database since cutoff date.
        Works with the ad3gem-emails database structure:
          threads/{threadId} - Thread documents
          threads/{threadId}/messages/{messageId} - Message documents
        """
        emails = []
        monitored_emails = {e.lower() for e in CONFIG["MONITORED_EMAILS"]}

        try:
            # Get recent threads
            thread_docs = (
                self.email_db.collection("threads")
                .where("lastMessageAt", ">=", cutoff_date)
                .order_by("lastMessageAt", direction=firestore.Query.DESCENDING)
                .limit(100)  # Process up to 100 recent threads
                .stream()
            )

            for thread_doc in thread_docs:
                thread_data = thread_doc.to_dict()
                thread_id = thread_doc.id

                # Check if thread involves monitored addresses
                participants = thread_data.get("participantEmails", [])
                if not any(addr.lower() in monitored_emails for addr in participants):
                    continue

                try:
                    # Get messages from this thread
                    message_docs = (
                        self.email_db.collection("threads")
                        .document(thread_id)
                        .collection("messages")
                        .where("sentAt", ">=", cutoff_date)
                        .order_by("sentAt", direction=firestore.Query.DESCENDING)
                        .limit(20)  # Limit messages per thread
                        .stream()
                    )

                    for msg_doc in message_docs:
                        msg_data = msg_doc.to_dict()

                        # Convert Gmail API format to intake format for compatibility
                        email_data = self._convert_gmail_to_intake_format(
                            msg_data, thread_data
                        )

                        # Filter to only monitored addresses
                        if self._involves_monitored_address(
                            email_data, monitored_emails
                        ):
                            emails.append(email_data)

                except Exception as e:
                    print(f"⚠️ Error fetching messages for thread {thread_id}: {e}")
                    continue

            print(
                f"📥 Fetched {len(emails)} emails since {cutoff_date.date()} (filtered to monitored inboxes)"
            )

        except Exception as e:
            print(f"❌ Error fetching emails: {e}")

        return emails

    def _convert_gmail_to_intake_format(
        self, msg_data: Dict, thread_data: Dict
    ) -> Dict:
        """
        Convert Gmail API message format to the format expected by intake processing.
        """
        # Extract sender email from Gmail API format
        from_data = msg_data.get("from", {})
        from_email = (
            from_data.get("email", "")
            if isinstance(from_data, dict)
            else str(from_data)
        )

        # Extract recipient emails
        to_list = msg_data.get("to", [])
        cc_list = msg_data.get("cc", [])

        # Convert to simple email lists
        to_emails = []
        for recipient in to_list:
            if isinstance(recipient, dict):
                email = recipient.get("email", "")
                if email:
                    to_emails.append(email)
            elif isinstance(recipient, str):
                to_emails.append(recipient)

        cc_emails = []
        for recipient in cc_list:
            if isinstance(recipient, dict):
                email = recipient.get("email", "")
                if email:
                    cc_emails.append(email)
            elif isinstance(recipient, str):
                cc_emails.append(recipient)

        # Convert timestamp
        sent_at = msg_data.get("sentAt")
        if sent_at:
            if hasattr(sent_at, "isoformat"):
                processed_at = sent_at.isoformat()
            else:
                processed_at = str(sent_at)
        else:
            processed_at = datetime.now().isoformat()

        # Return in intake format
        return {
            "from": from_email,
            "to": to_emails,
            "cc": cc_emails,
            "subject": msg_data.get("subject", ""),
            "body": msg_data.get("body", ""),
            "processed_at": processed_at,
            "thread_id": thread_data.get("convKey", ""),
            "thread_subject": thread_data.get("subject", ""),
        }

    def _involves_monitored_address(
        self, email_data: Dict, monitored_emails: set
    ) -> bool:
        """
        Check if the email involves any of the monitored addresses.
        """
        # Check sender
        from_email = self._normalize_email(email_data.get("from", ""))
        if from_email in monitored_emails:
            return True

        # Check recipients
        to_list = self._ensure_list(email_data.get("to", []))
        cc_list = self._ensure_list(email_data.get("cc", []))

        all_recipients = [self._normalize_email(email) for email in to_list + cc_list]
        return any(email in monitored_emails for email in all_recipients)

    # -----------------
    # Entity extraction
    # -----------------

    def _update_entities(self, emails: List[Dict]):
        """
        Extract and update entities (people, companies) from emails.
        Enrich with canonical people/companies when possible.
        """
        print("👥 Updating entities...")

        entities = defaultdict(
            lambda: {
                "type": "unknown",
                "email_addresses": set(),
                "names": set(),
                "roles": set(),
                "companies": set(),
                "first_seen": None,
                "last_seen": None,
                "email_count": 0,
                "attributes": {},
            }
        )

        for email in emails:
            processed_at = email.get("processed_at")
            # Process sender
            sender_raw = email.get("from", "")
            sender_email = self._normalize_email(sender_raw)

            if sender_email:
                entity_id = f"person::{sender_email}"
                entities[entity_id]["email_addresses"].add(sender_email)
                entities[entity_id]["type"] = "person"
                entities[entity_id]["email_count"] += 1
                entities[entity_id]["last_seen"] = processed_at

                # Extract a display name from email localpart if nothing else
                if "@" in sender_email:
                    guess_name = (
                        sender_email.split("@")[0]
                        .replace(".", " ")
                        .replace("_", " ")
                        .title()
                    )
                    guess_name = self._apply_corrections(guess_name)
                    entities[entity_id]["names"].add(guess_name)

                # If in your domain, tag internal
                domain = sender_email.split("@")[-1]
                if domain in {
                    "lineagecoffee.com",
                    "completecoffeesolutions.com",
                    "adddesigns.co.za",
                }:
                    entities[entity_id]["companies"].add("Lineage Coffee")
                    entities[entity_id]["type"] = "internal_user"

                # Canonical enrichment if this sender matches a known person email (by local part/name)
                self._enrich_person_from_static(entities, entity_id)

            # Extract companies mentioned in subject/body (rough heuristic)
            text = f"{email.get('subject', '')} {email.get('body', '')}"
            mentioned_companies = self._find_companies_in_text(text)
            for comp in mentioned_companies:
                comp_id = f"company::{self._normalize_company(comp)}"
                entities[comp_id]["type"] = "company"
                entities[comp_id]["names"].add(comp)
                entities[comp_id]["last_seen"] = processed_at
                self._enrich_company_from_static(entities, comp_id)

        # Store in Firestore
        batch = self.knowledge_db.batch()
        for entity_id, data in entities.items():
            doc_ref = self.knowledge_db.collection("entities").document(entity_id)

            entity_dict = {
                "entity_id": entity_id,
                "type": data["type"],
                "email_addresses": sorted(list(data["email_addresses"])),
                "names": sorted(list(data["names"])),
                "companies": sorted(list(data["companies"])),
                "roles": sorted(list(data["roles"])),
                "last_seen": data["last_seen"],
                "email_count": Increment(data["email_count"]),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            batch.set(doc_ref, entity_dict, merge=True)

        batch.commit()
        print(f"   ✅ Updated {len(entities)} entities")

    def _enrich_person_from_static(self, entities: Dict[str, Any], entity_id: str):
        """If the person matches a known canonical person by name/alias, merge canonical data."""
        names = entities[entity_id]["names"]
        if not names:
            return
        all_names_lower = {n.lower() for n in names}

        for pname, pdata in STATIC_PEOPLE.items():
            candidates = {pname.lower()} | {a.lower() for a in pdata.get("aliases", [])}
            if all_names_lower & candidates:
                entities[entity_id]["names"].add(pname)
                for a in pdata.get("aliases", []):
                    entities[entity_id]["names"].add(a)
                for r in pdata.get("roles", []):
                    entities[entity_id]["roles"].add(r)
                for c in pdata.get("companies", []):
                    entities[entity_id]["companies"].add(c)
                # Keep category in attributes
                entities[entity_id]["attributes"]["category"] = pdata.get(
                    "type", "unknown"
                )
                break

    def _enrich_company_from_static(self, entities: Dict[str, Any], entity_id: str):
        """If the company matches a known canonical company, merge tags/aliases/class."""
        names = entities[entity_id]["names"]
        if not names:
            return
        all_names_lower = {self._normalize_company(n) for n in names}

        for cname, cdata in STATIC_COMPANIES.items():
            candidates = {self._normalize_company(cname)} | {
                self._normalize_company(a) for a in cdata.get("aliases", [])
            }
            if all_names_lower & candidates:
                entities[entity_id]["names"].add(cname)
                for a in cdata.get("aliases", []):
                    entities[entity_id]["names"].add(a)
                entities[entity_id]["attributes"]["class"] = cdata.get(
                    "type", "unknown"
                )
                entities[entity_id]["attributes"]["tags"] = cdata.get("tags", [])
                break

    # -----------------
    # Relationships
    # -----------------

    def _update_relationships(self, emails: List[Dict]):
        """Track relationships between entities."""
        print("🔗 Updating relationships...")

        relationships = defaultdict(
            lambda: {
                "email_count": 0,
                "subjects": [],
                "last_interaction": None,
                "interaction_types": set(),
            }
        )

        for email in emails:
            sender = self._normalize_email(email.get("from", ""))
            # handle multi-recipient
            recipients = self._ensure_list(email.get("to", []))
            recipients = [self._normalize_email(r) for r in recipients]

            for rcpt in recipients:
                if not sender or not rcpt:
                    continue
                rel_id = f"{sender}→{rcpt}"
                relationships[rel_id]["email_count"] += 1
                relationships[rel_id]["subjects"].append(
                    email.get("subject", "")[: CONFIG["SUBJECT_SNIPPET_LEN"]]
                )
                relationships[rel_id]["last_interaction"] = email.get("processed_at")

                # Determine interaction type
                subject_lower = email.get("subject", "").lower()
                if "invoice" in subject_lower:
                    relationships[rel_id]["interaction_types"].add("financial")
                elif "order" in subject_lower:
                    relationships[rel_id]["interaction_types"].add("orders")
                elif (
                    "meeting" in subject_lower
                    or "schedule" in subject_lower
                    or "calendar" in subject_lower
                ):
                    relationships[rel_id]["interaction_types"].add("scheduling")

        # Store in Firestore
        batch = self.knowledge_db.batch()
        for rel_id, rel_data in relationships.items():
            doc_ref = self.knowledge_db.collection("relationships").document(rel_id)

            rel_dict = {
                "relationship_id": rel_id,
                "from_entity": rel_id.split("→")[0],
                "to_entity": rel_id.split("→")[1],
                "email_count": Increment(rel_data["email_count"]),
                "recent_subjects": rel_data["subjects"][-5:],
                "last_interaction": rel_data["last_interaction"],
                "interaction_types": list(rel_data["interaction_types"]),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            batch.set(doc_ref, rel_dict, merge=True)

        batch.commit()
        print(f"   ✅ Updated {len(relationships)} relationships")

    # -----------------
    # Patterns & Facts
    # -----------------

    def _update_patterns(self, emails: List[Dict]):
        """Identify and update communication patterns."""
        print("📊 Updating patterns...")

        patterns = {
            "daily_email_volume": defaultdict(int),
            "hourly_distribution": defaultdict(int),
            "sender_frequency": defaultdict(int),
            "subject_patterns": defaultdict(list),
        }

        for email in emails:
            try:
                processed_at = email.get("processed_at", "")
                if processed_at:
                    dt = datetime.fromisoformat(processed_at.replace("Z", "+00:00"))
                    patterns["daily_email_volume"][dt.date().isoformat()] += 1
                    patterns["hourly_distribution"][dt.hour] += 1
            except Exception:
                pass

            sender = self._normalize_email(email.get("from", ""))
            if sender:
                patterns["sender_frequency"][sender] += 1

            subject = email.get("subject", "")
            if subject:
                key = self._extract_subject_pattern(subject)
                patterns["subject_patterns"][key].append(subject[:50])

        doc_ref = self.knowledge_db.collection("patterns").document(
            "communication_patterns"
        )
        doc_ref.set(
            {
                "daily_volume": dict(patterns["daily_email_volume"]),
                "hourly_distribution": dict(patterns["hourly_distribution"]),
                "top_senders": dict(
                    sorted(
                        patterns["sender_frequency"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:20]
                ),
                "common_subject_patterns": {
                    k: v[:5] for k, v in patterns["subject_patterns"].items()
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            merge=True,
        )

        print("   ✅ Updated communication patterns")

    def _extract_facts(self, emails: List[Dict]):
        """Use Gemini to extract facts from email content."""
        print("💡 Extracting facts...")

        facts = []
        for email in emails[: CONFIG["MAX_EMAILS_FACTS"]]:
            prompt = f"""
            Extract important facts from this email:

            From: {email.get("from", "Unknown")}
            To: {email.get("to", "Unknown")}
            Subject: {email.get("subject", "No subject")}
            Body: {email.get("body", "")[: CONFIG["BODY_SNIPPET_LEN"]]}

            Extract facts in JSON format: {
                {
                    "facts": [
                        {
                            {
                                "type": "person/date/price/product/decision/action",
                                "fact": "clear statement of fact",
                                "confidence": 0 - 100,
                                "entities": ["related entities"],
                                "category": "business/financial/scheduling/general",
                            }
                        }
                    ]
                }
            }

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
                    # Apply corrections to entity mentions
                    fact["entities"] = [
                        self._apply_corrections(e) for e in fact.get("entities", [])
                    ]
                    facts.append(fact)

            except Exception as e:
                print(f"   ⚠️ Error extracting facts: {e}")
                continue

        if facts:
            batch = self.knowledge_db.batch()
            for fact in facts:
                fact_id = hashlib.md5(fact["fact"].encode()).hexdigest()[:12]
                doc_ref = self.knowledge_db.collection("facts").document(fact_id)
                batch.set(doc_ref, fact, merge=True)
            batch.commit()
            print(f"   ✅ Extracted {len(facts)} facts")

    # -----------------
    # Topics & Temporal
    # -----------------

    def _update_topics(self, emails: List[Dict]):
        """Track and categorize email topics."""
        print("🏷️ Updating topics...")

        topics = defaultdict(
            lambda: {
                "occurrences": 0,
                "example_subjects": [],
                "related_senders": set(),
                "keywords": set(),
            }
        )

        topic_keywords = {
            "coffee_orders": ["coffee", "beans", "roast", "arabica", "order"],
            "invoices": ["invoice", "payment", "bill", "due", "amount", "statement"],
            "shipping": [
                "delivery",
                "shipment",
                "tracking",
                "arrived",
                "dispatch",
                "courier",
            ],
            "meetings": ["meeting", "schedule", "calendar", "call", "discussion"],
            "reports": ["report", "summary", "analysis", "metrics", "performance"],
        }

        for email in emails:
            subject = email.get("subject", "").lower()
            body = (email.get("body", "")[: CONFIG["BODY_SNIPPET_LEN"]]).lower()
            content = f"{subject} {body}"

            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    topics[topic]["occurrences"] += 1
                    topics[topic]["example_subjects"].append(
                        email.get("subject", "")[:50]
                    )
                    topics[topic]["related_senders"].add(email.get("from", ""))

                    for keyword in keywords:
                        if keyword in content:
                            topics[topic]["keywords"].add(keyword)

        batch = self.knowledge_db.batch()
        for topic_name, topic_data in topics.items():
            doc_ref = self.knowledge_db.collection("topics").document(topic_name)
            topic_dict = {
                "topic": topic_name,
                "occurrences": Increment(topic_data["occurrences"]),
                "recent_subjects": topic_data["example_subjects"][-10:],
                "related_senders": list(topic_data["related_senders"])[:20],
                "active_keywords": list(topic_data["keywords"]),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            batch.set(doc_ref, topic_dict, merge=True)

        batch.commit()
        print(f"   ✅ Updated {len(topics)} topics")

    def _update_temporal_patterns(self, emails: List[Dict]):
        """Track time-based patterns and schedules."""
        print("⏰ Updating temporal patterns...")

        temporal = {
            "regular_senders": defaultdict(list),
            "peak_hours": defaultdict(int),
        }

        for email in emails:
            try:
                processed_at = email.get("processed_at", "")
                if processed_at:
                    dt = datetime.fromisoformat(processed_at.replace("Z", "+00:00"))
                    sender = self._normalize_email(email.get("from", ""))

                    temporal["regular_senders"][sender].append(
                        {
                            "hour": dt.hour,
                            "day_of_week": dt.weekday(),
                            "day_of_month": dt.day,
                        }
                    )
                    temporal["peak_hours"][dt.hour] += 1

            except Exception:
                continue

        sender_schedules = {}
        for sender, times in temporal["regular_senders"].items():
            if len(times) >= 3:
                hours = [t["hour"] for t in times]
                most_common_hour = max(set(hours), key=hours.count)
                sender_schedules[sender] = {
                    "typical_hour": most_common_hour,
                    "email_times": times[-5:],
                }

        doc_ref = self.knowledge_db.collection("temporal").document("email_schedules")
        doc_ref.set(
            {
                "sender_schedules": sender_schedules,
                "peak_hours": dict(temporal["peak_hours"]),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            merge=True,
        )

        print("   ✅ Updated temporal patterns")

    # ---------------
    # Context & CRUD
    # ---------------

    def get_context_for_query(self, query: str, user_id: str = None) -> Dict:
        """Get relevant context from knowledge base for a query."""
        context = {
            "entities": {},
            "relationships": [],
            "facts": [],
            "patterns": {},
            "topics": [],
            "temporal": {},
            "corrections": [],
        }

        q = self._apply_corrections(query).lower()

        # Entities (quick scan)
        entities = self.knowledge_db.collection("entities").limit(200).get()
        for entity_doc in entities:
            data = entity_doc.to_dict()
            for name in data.get("names", []):
                if name.lower() in q:
                    context["entities"][entity_doc.id] = data
                    break
            for email in data.get("email_addresses", []):
                if email.lower() in q or email.split("@")[0] in q:
                    context["entities"][entity_doc.id] = data
                    break

        # Relationships for mentioned entities
        for entity_id in context["entities"]:
            rels = (
                self.knowledge_db.collection("relationships")
                .where(
                    filter=FieldFilter(
                        "from_entity",
                        "==",
                        entity_id.split("person::")[-1]
                        if entity_id.startswith("person::")
                        else entity_id,
                    )
                )
                .limit(10)
                .get()
            )
            for rel in rels:
                context["relationships"].append(rel.to_dict())

        # Facts
        facts = self.knowledge_db.collection("facts").limit(50).get()
        for fact_doc in facts:
            fact_data = fact_doc.to_dict()
            for ent in fact_data.get("entities", []):
                if ent.lower() in q:
                    context["facts"].append(fact_data)
                    break

        # Patterns
        patterns_doc = (
            self.knowledge_db.collection("patterns")
            .document("communication_patterns")
            .get()
        )
        if patterns_doc.exists:
            context["patterns"] = patterns_doc.to_dict()

        # Temporal
        temporal_doc = (
            self.knowledge_db.collection("temporal").document("email_schedules").get()
        )
        if temporal_doc.exists:
            context["temporal"] = temporal_doc.to_dict()

        # Corrections (active)
        corrections = (
            self.knowledge_db.collection("corrections")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(25)
            .get()
        )
        for correction in corrections:
            c = correction.to_dict()
            if c.get("active"):
                context["corrections"].append(c)

        return context

    def add_correction(
        self, correction_type: str, original: str, corrected: str, user_id: str = None
    ):
        """Add a manual correction to the knowledge base."""
        doc_ref = self.knowledge_db.collection("corrections").document()
        doc_ref.set(
            {
                "type": correction_type,
                "original": original,
                "corrected": corrected,
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "active": True,
            }
        )

        print(f"✅ Added correction: {original} → {corrected}")

    # ---------------
    # Helpers
    # ---------------

    @staticmethod
    def _ensure_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    def _normalize_email(self, email: str) -> str:
        """Normalize email address for consistent storage."""
        if not email:
            return ""

        # Extract email from "Name <email@domain.com>" format
        if "<" in email and ">" in email:
            email = email[email.index("<") + 1 : email.index(">")]

        return email.lower().strip()

    def _apply_corrections(self, s: str) -> str:
        out = s
        # Remove 'Pty Ltd' variants if requested
        if CONFIG["REMOVE_PTY_LTD"]:
            out = re.sub(
                r"\b$begin:math:text$?\\s*pty\\s*ltd\\.?\\s*$end:math:text$?",
                "",
                out,
                flags=re.IGNORECASE,
            ).strip()

        # Apply specific corrections
        for original, corrected in CORRECTIONS.items():
            out = re.sub(
                rf"\b{re.escape(original)}\b", corrected, out, flags=re.IGNORECASE
            )

        # Collapse double spaces after removals
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out

    def _normalize_company(self, name: str) -> str:
        cleaned = self._apply_corrections(name)
        return cleaned.lower()

    def _normalize_person_id(self, name: str) -> str:
        cleaned = self._apply_corrections(name)
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", cleaned.lower()).strip("-")
        return cleaned

    def _find_companies_in_text(self, text: str) -> List[str]:
        """Very light-weight company detection using canonical list."""
        found = set()
        t = self._apply_corrections(text)
        for cname, cdata in STATIC_COMPANIES.items():
            names = [cname] + cdata.get("aliases", [])
            for n in names:
                if n and n.lower() in t.lower():
                    found.add(cname)
        return sorted(found)

    def _extract_subject_pattern(self, subject: str) -> str:
        subject_lower = subject.lower()
        if subject_lower.startswith("re:"):
            return "reply"
        elif subject_lower.startswith("fwd:"):
            return "forward"
        elif "invoice" in subject_lower:
            return "invoice"
        elif "order" in subject_lower:
            return "order"
        elif "report" in subject_lower:
            return "report"
        elif (
            "meeting" in subject_lower
            or "schedule" in subject_lower
            or "calendar" in subject_lower
        ):
            return "meeting"
        else:
            return "general"

    def _mark_update_complete(self):
        doc_ref = self.knowledge_db.collection("metadata").document("last_update")
        doc_ref.set(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "next_update": (
                    datetime.now(timezone.utc) + timedelta(days=1)
                ).isoformat(),
            },
            merge=True,
        )


def main():
    """
    Main function for initial knowledge base setup.
    This is meant to be run once to seed the knowledge base with canonical data.
    """
    print("🚀 AD3Gem Knowledge Base Initial Setup")
    print("=" * 60)
    print("This script seeds canonical knowledge and can process recent emails")
    print("for initial entity enrichment.")
    print("=" * 60)

    kb = AD3GemKnowledgeBase()

    # Initialize structure (safe to run multiple times)
    print("\n📋 Initializing knowledge base structure...")
    kb.initialize_knowledge_structure()

    # Seed canonical knowledge (safe to run multiple times)
    print("\n🌱 Seeding canonical knowledge...")
    kb.seed_static_knowledge()

    # Optional: Process recent emails for initial enrichment
    print("\n📧 Processing recent emails for entity enrichment...")
    try:
        kb.update_daily_knowledge(days_back=7)  # Process last week
    except Exception as e:
        print(f"⚠️ Email processing failed (this is optional): {e}")
        print("The canonical knowledge has been seeded successfully.")

    print("\n✅ Knowledge base initial setup complete!")
    print("\n💡 Next steps:")
    print("   1. Run the Gmail ingestion service (main.py) to populate ad3gem-emails")
    print("   2. Use ad3gem-knowledge.py for daily knowledge updates")
    print("   3. Use ad3gem-chatbot.py for intelligent conversations")


if __name__ == "__main__":
    main()
