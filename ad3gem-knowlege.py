"""
AD3Gem Knowledge Base System - Updated for ad3gem-emails Database
Builds and maintains intelligent context from the new email structure

This script:
1. Reads emails from the new ad3gem-emails database structure
2. Extracts entities, relationships, patterns, and facts
3. Stores processed knowledge in ad3gem-knowledge database
4. Runs daily to keep knowledge current

New Database Structure it reads from:
- ad3gem-emails/incoming_raw: Lean email metadata
- ad3gem-emails/threads: Conversation threads
- ad3gem-emails/threads/{id}/messages: Full email content

Knowledge Database Structure it creates:
- ad3gem-knowledge/entities: People and companies
- ad3gem-knowledge/relationships: Communication patterns
- ad3gem-knowledge/facts: Extracted information
- ad3gem-knowledge/patterns: Communication habits
- ad3gem-knowledge/topics: Common subjects
- ad3gem-knowledge/temporal: Time-based patterns
"""

import json
import os
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from collections import defaultdict

import google.generativeai as genai
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter, Increment, ArrayUnion


class AD3GemKnowledgeBase:
    """
    Manages the central knowledge base using the new email database structure.
    """
    
    def __init__(self):
        """Initialize the knowledge base manager."""
        # Database configuration
        self.PROJECT_ID = "ad3-sam"
        self.KNOWLEDGE_DB_ID = "ad3gem-knowledge"
        self.EMAIL_DB_ID = "ad3gem-emails"  # New database name
        
        # Initialize Firestore clients
        self.knowledge_db = firestore.Client(
            project=self.PROJECT_ID,
            database=self.KNOWLEDGE_DB_ID
        )
        
        self.email_db = firestore.Client(
            project=self.PROJECT_ID,
            database=self.EMAIL_DB_ID
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
            }
        )
        
        print(f"🧠 AD3Gem Knowledge Base initialized")
        print(f"📚 Knowledge DB: {self.KNOWLEDGE_DB_ID}")
        print(f"📧 Email DB: {self.EMAIL_DB_ID} (new structure)")

    def initialize_knowledge_structure(self):
        """Create the initial knowledge base structure if it doesn't exist."""
        collections = [
            "entities",  # People, companies, projects
            "relationships",  # Who emails whom
            "patterns",  # Email patterns, schedules
            "facts",  # Extracted facts
            "topics",  # Common topics
            "temporal",  # Time-based patterns
            "business_rules",  # Business logic
            "corrections",  # Manual corrections
            "thread_summaries",  # Conversation summaries
        ]
        
        for collection in collections:
            doc_ref = self.knowledge_db.collection(collection).document("_init")
            doc_ref.set({
                "initialized": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "description": f"Knowledge base for {collection}"
            }, merge=True)
        
        print(f"✅ Knowledge base structure initialized")

    def update_daily_knowledge(self, days_back: int = 1):
        """Main method to update knowledge base from recent emails."""
        print(f"\n🔄 Starting daily knowledge update...")
        
        # Get cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        # Fetch from new structure
        threads = self._fetch_recent_threads(cutoff_date)
        emails = self._fetch_recent_emails_from_threads(threads, cutoff_date)
        
        if not emails:
            print("📭 No new emails to process")
            return
        
        print(f"📧 Processing {len(emails)} emails from {len(threads)} threads...")
        
        # Process different aspects of knowledge
        self._update_entities(emails)
        self._update_relationships(emails)
        self._update_patterns(emails)
        self._extract_facts(emails)
        self._update_topics(emails)
        self._update_temporal_patterns(emails)
        self._update_thread_summaries(threads)
        
        # Mark update completion
        self._mark_update_complete()
        
        print(f"✅ Knowledge base updated successfully")

    def _fetch_recent_threads(self, cutoff_date: datetime) -> List[Dict]:
        """Fetch recent threads from the new structure."""
        threads = []
        try:
            # Query threads collection for recent activity
            thread_docs = (
                self.email_db.collection("threads")
                .where("lastMessageAt", ">=", cutoff_date)
                .limit(100)  # Process up to 100 threads
                .stream()
            )
            
            for doc in thread_docs:
                thread_data = doc.to_dict()
                thread_data["thread_id"] = doc.id
                threads.append(thread_data)
            
            print(f"📥 Fetched {len(threads)} threads since {cutoff_date.date()}")
            
        except Exception as e:
            print(f"❌ Error fetching threads: {e}")
        
        return threads

    def _fetch_recent_emails_from_threads(self, threads: List[Dict], cutoff_date: datetime) -> List[Dict]:
        """Fetch actual email messages from threads."""
        emails = []
        
        for thread in threads:
            thread_id = thread.get("thread_id")
            if not thread_id:
                continue
            
            try:
                # Get messages subcollection
                message_docs = (
                    self.email_db.collection("threads")
                    .document(thread_id)
                    .collection("messages")
                    .where("sentAt", ">=", cutoff_date)
                    .limit(50)  # Limit per thread
                    .stream()
                )
                
                for msg_doc in message_docs:
                    email_data = msg_doc.to_dict()
                    email_data["message_id"] = msg_doc.id
                    email_data["thread_id"] = thread_id
                    email_data["thread_subject"] = thread.get("subject", "")
                    email_data["thread_participants"] = thread.get("participants", [])
                    emails.append(email_data)
                    
            except Exception as e:
                print(f"⚠️ Error fetching messages for thread {thread_id}: {e}")
                continue
        
        print(f"📨 Fetched {len(emails)} messages from threads")
        return emails

    def _update_entities(self, emails: List[Dict]):
        """Extract and update entities from emails."""
        print("👥 Updating entities...")
        
        entities = defaultdict(lambda: {
            "type": "unknown",
            "email_addresses": set(),
            "names": set(),
            "roles": set(),
            "companies": set(),
            "departments": set(),
            "first_seen": None,
            "last_seen": None,
            "email_count": 0,
            "thread_count": set(),
            "common_subjects": [],
            "attributes": {}
        })
        
        for email in emails:
            # Process sender (from field)
            from_data = email.get("from", {})
            if from_data:
                sender_email = from_data.get("email", "")
                sender_name = from_data.get("name", "")
                
                if sender_email:
                    entity_id = self._normalize_email(sender_email)
                    entities[entity_id]["email_addresses"].add(sender_email)
                    entities[entity_id]["type"] = "person"
                    entities[entity_id]["email_count"] += 1
                    entities[entity_id]["last_seen"] = email.get("sentAt")
                    entities[entity_id]["thread_count"].add(email.get("thread_id"))
                    
                    if sender_name:
                        entities[entity_id]["names"].add(sender_name)
                    
                    # Determine company from domain
                    if "@" in sender_email:
                        domain = sender_email.split("@")[1]
                        if domain == "lineagecoffee.com":
                            entities[entity_id]["companies"].add("Lineage Coffee")
                            entities[entity_id]["type"] = "internal_user"
                        else:
                            entities[entity_id]["companies"].add(domain)
                            entities[entity_id]["type"] = "external_contact"
            
            # Process recipients
            for field in ["to", "cc"]:
                recipients = email.get(field, [])
                for recipient in recipients:
                    if isinstance(recipient, dict):
                        rec_email = recipient.get("email", "")
                        rec_name = recipient.get("name", "")
                        
                        if rec_email:
                            entity_id = self._normalize_email(rec_email)
                            entities[entity_id]["email_addresses"].add(rec_email)
                            entities[entity_id]["thread_count"].add(email.get("thread_id"))
                            
                            if rec_name:
                                entities[entity_id]["names"].add(rec_name)
        
        # Store in Firestore
        batch = self.knowledge_db.batch()
        batch_count = 0
        
        for entity_id, entity_data in entities.items():
            doc_ref = self.knowledge_db.collection("entities").document(entity_id)
            
            entity_dict = {
                "entity_id": entity_id,
                "type": entity_data["type"],
                "email_addresses": list(entity_data["email_addresses"]),
                "names": list(entity_data["names"]),
                "companies": list(entity_data["companies"]),
                "last_seen": entity_data["last_seen"],
                "email_count": Increment(entity_data["email_count"]),
                "thread_count": len(entity_data["thread_count"]),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            batch.set(doc_ref, entity_dict, merge=True)
            batch_count += 1
            
            # Commit batch every 400 operations (Firestore limit is 500)
            if batch_count >= 400:
                batch.commit()
                batch = self.knowledge_db.batch()
                batch_count = 0
        
        if batch_count > 0:
            batch.commit()
        
        print(f"   ✅ Updated {len(entities)} entities")

    def _update_relationships(self, emails: List[Dict]):
        """Track relationships between entities."""
        print("🔗 Updating relationships...")
        
        relationships = defaultdict(lambda: {
            "email_count": 0,
            "subjects": [],
            "threads": set(),
            "last_interaction": None,
            "interaction_types": set(),
            "snippets": []
        })
        
        for email in emails:
            from_data = email.get("from", {})
            sender = self._normalize_email(from_data.get("email", ""))
            
            # Process all recipients
            all_recipients = []
            for field in ["to", "cc"]:
                recipients = email.get(field, [])
                for rec in recipients:
                    if isinstance(rec, dict):
                        all_recipients.append(self._normalize_email(rec.get("email", "")))
            
            # Create relationships
            for recipient in all_recipients:
                if sender and recipient and sender != recipient:
                    rel_id = f"{sender}→{recipient}"
                    relationships[rel_id]["email_count"] += 1
                    relationships[rel_id]["subjects"].append(email.get("thread_subject", "")[:100])
                    relationships[rel_id]["threads"].add(email.get("thread_id"))
                    relationships[rel_id]["last_interaction"] = email.get("sentAt")
                    
                    # Extract snippet
                    body_preview = email.get("bodyPreview", "")
                    if body_preview:
                        relationships[rel_id]["snippets"].append(body_preview[:200])
                    
                    # Determine interaction type from subject/body
                    content = f"{email.get('thread_subject', '')} {body_preview}".lower()
                    if "invoice" in content or "payment" in content:
                        relationships[rel_id]["interaction_types"].add("financial")
                    elif "order" in content:
                        relationships[rel_id]["interaction_types"].add("orders")
                    elif "meeting" in content or "call" in content:
                        relationships[rel_id]["interaction_types"].add("scheduling")
                    elif "report" in content:
                        relationships[rel_id]["interaction_types"].add("reporting")
        
        # Store in Firestore
        batch = self.knowledge_db.batch()
        batch_count = 0
        
        for rel_id, rel_data in relationships.items():
            doc_ref = self.knowledge_db.collection("relationships").document(rel_id)
            
            rel_dict = {
                "relationship_id": rel_id,
                "from_entity": rel_id.split("→")[0],
                "to_entity": rel_id.split("→")[1],
                "email_count": Increment(rel_data["email_count"]),
                "thread_count": len(rel_data["threads"]),
                "recent_subjects": list(set(rel_data["subjects"]))[-10:],
                "recent_snippets": rel_data["snippets"][-5:],
                "last_interaction": rel_data["last_interaction"],
                "interaction_types": list(rel_data["interaction_types"]),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            batch.set(doc_ref, rel_dict, merge=True)
            batch_count += 1
            
            if batch_count >= 400:
                batch.commit()
                batch = self.knowledge_db.batch()
                batch_count = 0
        
        if batch_count > 0:
            batch.commit()
        
        print(f"   ✅ Updated {len(relationships)} relationships")

    def _update_patterns(self, emails: List[Dict]):
        """Identify and update communication patterns."""
        print("📊 Updating patterns...")
        
        patterns = {
            "daily_email_volume": defaultdict(int),
            "hourly_distribution": defaultdict(int),
            "sender_frequency": defaultdict(int),
            "thread_activity": defaultdict(int),
            "label_distribution": defaultdict(int),
            "attachment_patterns": defaultdict(int)
        }
        
        for email in emails:
            # Time patterns
            sent_at = email.get("sentAt")
            if sent_at:
                if isinstance(sent_at, str):
                    try:
                        dt = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
                    except:
                        continue
                else:
                    dt = sent_at
                
                patterns["daily_email_volume"][dt.date().isoformat()] += 1
                patterns["hourly_distribution"][dt.hour] += 1
            
            # Sender patterns
            from_data = email.get("from", {})
            sender = self._normalize_email(from_data.get("email", ""))
            if sender:
                patterns["sender_frequency"][sender] += 1
            
            # Thread patterns
            thread_id = email.get("thread_id")
            if thread_id:
                patterns["thread_activity"][thread_id] += 1
            
            # Label patterns
            for label in email.get("labelNames", []):
                patterns["label_distribution"][label] += 1
            
            # Attachment patterns
            if email.get("hasAttachments"):
                patterns["attachment_patterns"]["with_attachments"] += 1
            else:
                patterns["attachment_patterns"]["without_attachments"] += 1
        
        # Store patterns
        doc_ref = self.knowledge_db.collection("patterns").document("communication_patterns")
        doc_ref.set({
            "daily_volume": dict(patterns["daily_email_volume"]),
            "hourly_distribution": dict(patterns["hourly_distribution"]),
            "top_senders": dict(sorted(patterns["sender_frequency"].items(), 
                                      key=lambda x: x[1], reverse=True)[:30]),
            "active_threads": dict(sorted(patterns["thread_activity"].items(),
                                        key=lambda x: x[1], reverse=True)[:20]),
            "label_usage": dict(patterns["label_distribution"]),
            "attachment_stats": dict(patterns["attachment_patterns"]),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }, merge=True)
        
        print(f"   ✅ Updated communication patterns")

    def _extract_facts(self, emails: List[Dict]):
        """Use Gemini to extract facts from email content."""
        print("💡 Extracting facts...")
        
        facts = []
        processed = 0
        
        # Sample emails for fact extraction (to manage costs)
        sample_size = min(30, len(emails))
        import random
        sampled_emails = random.sample(emails, sample_size) if len(emails) > sample_size else emails
        
        for email in sampled_emails:
            # Only process emails with substantial content
            body = email.get("body", "")
            if len(body) < 100:
                continue
            
            prompt = f"""
            Extract important business facts from this email:
            
            From: {email.get('from', {}).get('email', 'Unknown')}
            Subject: {email.get('thread_subject', 'No subject')}
            Date: {email.get('sentAt', '')}
            Body: {body[:1500]}
            
            Extract facts in JSON format:
            {{
                "facts": [
                    {{
                        "type": "person/date/price/product/decision/action/contact",
                        "fact": "clear, concise statement of fact",
                        "confidence": 0-100,
                        "entities": ["related people or companies"],
                        "category": "business/financial/scheduling/operational/relationship",
                        "importance": "high/medium/low"
                    }}
                ]
            }}
            
            Focus on:
            - Specific dates, deadlines, or schedules
            - Prices, amounts, or financial information
            - Decisions made or actions to be taken
            - Contact information or new relationships
            - Product or service details
            
            Only extract clear, actionable facts. Return empty list if none found.
            Return ONLY valid JSON.
            """
            
            try:
                response = self.model.generate_content(prompt)
                json_str = response.text.strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:-3]
                
                result = json.loads(json_str)
                for fact in result.get("facts", []):
                    fact["source_email_id"] = email.get("message_id", "")
                    fact["source_thread_id"] = email.get("thread_id", "")
                    fact["source_from"] = email.get("from", {}).get("email", "")
                    fact["extracted_at"] = datetime.now(timezone.utc).isoformat()
                    facts.append(fact)
                
                processed += 1
                
            except Exception as e:
                print(f"   ⚠️ Error extracting facts from email: {e}")
                continue
        
        # Store facts
        if facts:
            batch = self.knowledge_db.batch()
            batch_count = 0
            
            for fact in facts:
                # Create unique ID for fact
                fact_text = fact.get("fact", "")
                fact_id = hashlib.md5(f"{fact_text}{fact.get('source_email_id', '')}".encode()).hexdigest()[:12]
                doc_ref = self.knowledge_db.collection("facts").document(fact_id)
                batch.set(doc_ref, fact, merge=True)
                batch_count += 1
                
                if batch_count >= 400:
                    batch.commit()
                    batch = self.knowledge_db.batch()
                    batch_count = 0
            
            if batch_count > 0:
                batch.commit()
            
            print(f"   ✅ Extracted {len(facts)} facts from {processed} emails")

    def _update_topics(self, emails: List[Dict]):
        """Track and categorize email topics."""
        print("🏷️ Updating topics...")
        
        topics = defaultdict(lambda: {
            "occurrences": 0,
            "example_subjects": [],
            "related_senders": set(),
            "related_threads": set(),
            "keywords": set(),
            "recent_activity": []
        })
        
        # Enhanced topic keywords
        topic_keywords = {
            "coffee_business": ["coffee", "beans", "roast", "arabica", "robusta", "cupping", "grind"],
            "orders": ["order", "purchase", "quantity", "delivery", "shipment", "tracking"],
            "financial": ["invoice", "payment", "bill", "due", "amount", "price", "cost", "budget"],
            "logistics": ["shipping", "delivery", "customs", "freight", "container", "port"],
            "quality": ["quality", "grade", "sample", "test", "certification", "standard"],
            "meetings": ["meeting", "call", "schedule", "agenda", "discussion", "conference"],
            "reports": ["report", "summary", "analysis", "metrics", "performance", "statistics"],
            "suppliers": ["supplier", "vendor", "source", "farm", "origin", "producer"],
            "customers": ["customer", "client", "account", "contract", "agreement"],
            "inventory": ["stock", "inventory", "warehouse", "storage", "availability"]
        }
        
        for email in emails:
            subject = email.get("thread_subject", "").lower()
            body_preview = email.get("bodyPreview", "").lower()
            content = f"{subject} {body_preview}"
            
            from_email = email.get("from", {}).get("email", "")
            thread_id = email.get("thread_id", "")
            sent_at = email.get("sentAt", "")
            
            for topic, keywords in topic_keywords.items():
                matched_keywords = [kw for kw in keywords if kw in content]
                if matched_keywords:
                    topics[topic]["occurrences"] += 1
                    topics[topic]["example_subjects"].append(email.get("thread_subject", "")[:100])
                    topics[topic]["related_senders"].add(from_email)
                    topics[topic]["related_threads"].add(thread_id)
                    topics[topic]["keywords"].update(matched_keywords)
                    topics[topic]["recent_activity"].append({
                        "date": sent_at,
                        "from": from_email,
                        "subject": email.get("thread_subject", "")[:50]
                    })
        
        # Store topics
        batch = self.knowledge_db.batch()
        batch_count = 0
        
        for topic_name, topic_data in topics.items():
            doc_ref = self.knowledge_db.collection("topics").document(topic_name)
            
            # Keep only recent activity
            recent_activity = sorted(
                topic_data["recent_activity"], 
                key=lambda x: x.get("date", ""), 
                reverse=True
            )[:20]
            
            topic_dict = {
                "topic": topic_name,
                "occurrences": Increment(topic_data["occurrences"]),
                "recent_subjects": list(set(topic_data["example_subjects"]))[-20:],
                "related_senders": list(topic_data["related_senders"])[:30],
                "related_threads": list(topic_data["related_threads"])[:20],
                "active_keywords": list(topic_data["keywords"]),
                "recent_activity": recent_activity,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            batch.set(doc_ref, topic_dict, merge=True)
            batch_count += 1
            
            if batch_count >= 400:
                batch.commit()
                batch = self.knowledge_db.batch()
                batch_count = 0
        
        if batch_count > 0:
            batch.commit()
        
        print(f"   ✅ Updated {len(topics)} topics")

    def _update_temporal_patterns(self, emails: List[Dict]):
        """Track time-based patterns and schedules."""
        print("⏰ Updating temporal patterns...")
        
        temporal = {
            "sender_schedules": defaultdict(list),
            "thread_patterns": defaultdict(list),
            "peak_hours": defaultdict(int),
            "day_patterns": defaultdict(int),
            "response_times": []
        }
        
        # Group emails by thread for response time calculation
        thread_messages = defaultdict(list)
        for email in emails:
            thread_id = email.get("thread_id")
            if thread_id:
                thread_messages[thread_id].append(email)
        
        for email in emails:
            sent_at = email.get("sentAt")
            if not sent_at:
                continue
            
            if isinstance(sent_at, str):
                try:
                    dt = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
                except:
                    continue
            else:
                dt = sent_at
            
            from_email = email.get("from", {}).get("email", "")
            if from_email:
                sender = self._normalize_email(from_email)
                temporal["sender_schedules"][sender].append({
                    "hour": dt.hour,
                    "day_of_week": dt.weekday(),
                    "day_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()],
                    "date": dt.date().isoformat()
                })
            
            temporal["peak_hours"][dt.hour] += 1
            temporal["day_patterns"][dt.weekday()] += 1
        
        # Calculate response times within threads
        for thread_id, messages in thread_messages.items():
            sorted_msgs = sorted(messages, key=lambda x: x.get("sentAt", ""))
            for i in range(1, len(sorted_msgs)):
                prev_msg = sorted_msgs[i-1]
                curr_msg = sorted_msgs[i]
                
                prev_sender = prev_msg.get("from", {}).get("email", "")
                curr_sender = curr_msg.get("from", {}).get("email", "")
                
                # Only calculate if different senders (actual response)
                if prev_sender != curr_sender:
                    try:
                        prev_time = datetime.fromisoformat(prev_msg.get("sentAt", "").replace("Z", "+00:00"))
                        curr_time = datetime.fromisoformat(curr_msg.get("sentAt", "").replace("Z", "+00:00"))
                        response_hours = (curr_time - prev_time).total_seconds() / 3600
                        
                        if 0 < response_hours < 168:  # Within a week
                            temporal["response_times"].append({
                                "responder": curr_sender,
                                "response_hours": round(response_hours, 2),
                                "thread_id": thread_id
                            })
                    except:
                        continue
        
        # Analyze sender schedules
        sender_analysis = {}
        for sender, times in temporal["sender_schedules"].items():
            if len(times) >= 3:
                hours = [t["hour"] for t in times]
                days = [t["day_of_week"] for t in times]
                
                # Most common hour and day
                most_common_hour = max(set(hours), key=hours.count) if hours else None
                most_common_day = max(set(days), key=days.count) if days else None
                
                sender_analysis[sender] = {
                    "typical_hour": most_common_hour,
                    "typical_day": most_common_day,
                    "typical_day_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][most_common_day] if most_common_day is not None else None,
                    "total_emails": len(times),
                    "recent_times": times[-10:]
                }
        
        # Calculate average response times
        responder_avg = defaultdict(list)
        for rt in temporal["response_times"]:
            responder_avg[rt["responder"]].append(rt["response_hours"])
        
        response_analysis = {}
        for responder, times in responder_avg.items():
            if times:
                response_analysis[responder] = {
                    "avg_response_hours": round(sum(times) / len(times), 2),
                    "min_response_hours": round(min(times), 2),
                    "max_response_hours": round(max(times), 2),
                    "response_count": len(times)
                }
        
        # Store temporal patterns
        doc_ref = self.knowledge_db.collection("temporal").document("email_patterns")
        doc_ref.set({
            "sender_schedules": sender_analysis,
            "peak_hours": dict(temporal["peak_hours"]),
            "day_distribution": dict(temporal["day_patterns"]),
            "response_times": response_analysis,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }, merge=True)
        
        print(f"   ✅ Updated temporal patterns")

    def _update_thread_summaries(self, threads: List[Dict]):
        """Create summaries of important threads."""
        print("📝 Updating thread summaries...")
        
        # Find most active threads
        active_threads = sorted(
            threads, 
            key=lambda x: x.get("messageCount", 0), 
            reverse=True
        )[:10]
        
        summaries = []
        for thread in active_threads:
            summary = {
                "thread_id": thread.get("thread_id"),
                "subject": thread.get("subject", ""),
                "participants": thread.get("participants", []),
                "message_count": thread.get("messageCount", 0),
                "last_activity": thread.get("lastMessageAt"),
                "is_internal_only": thread.get("isInternalOnly", False),
                "has_external": thread.get("hasExternal", False),
                "owner_users": thread.get("ownerUsers", []),
                "latest_snippet": thread.get("latestSnippet", ""),
                "tags": thread.get("tags", [])
            }
            summaries.append(summary)
        
        # Store summaries
        if summaries:
            doc_ref = self.knowledge_db.collection("thread_summaries").document("active_threads")
            doc_ref.set({
                "threads": summaries,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }, merge=True)
            
            print(f"   ✅ Updated {len(summaries)} thread summaries")

    def _mark_update_complete(self):
        """Mark the knowledge base update as complete."""
        doc_ref = self.knowledge_db.collection("metadata").document("last_update")
        doc_ref.set({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "email_database": self.EMAIL_DB_ID,
            "next_update": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        })

    def _normalize_email(self, email: str) -> str:
        """Normalize email address for consistent storage."""
        if not email:
            return ""
        
        # Handle format "Name <email@domain.com>"
        if "<" in email and ">" in email:
            email = email[email.index("<")+1:email.index(">")]
        
        # Remove plus aliases: foo+tag@domain.com -> foo@domain.com
        if "@" in email:
            local, domain = email.split("@", 1)
            if "+" in local:
                local = local.split("+", 1)[0]
            email = f"{local}@{domain}"
        
        return email.lower().strip()

    def get_context_for_query(self, query: str, user_id: str = None) -> Dict:
        """
        Get relevant context from knowledge base for a query.
        This is what the chatbot will call.
        """
        context = {
            "entities": {},
            "relationships": [],
            "facts": [],
            "patterns": {},
            "topics": [],
            "temporal": {},
            "thread_summaries": [],
            "corrections": []
        }
        
        query_lower = query.lower()
        
        # Search for relevant entities
        entities = self.knowledge_db.collection("entities").limit(100).stream()
        for entity_doc in entities:
            entity_data = entity_doc.to_dict()
            
            # Check if entity is mentioned
            mentioned = False
            for name in entity_data.get("names", []):
                if name.lower() in query_lower:
                    mentioned = True
                    break
            
            if not mentioned:
                for email in entity_data.get("email_addresses", []):
                    if email.split("@")[0].lower() in query_lower:
                        mentioned = True
                        break
            
            if mentioned:
                context["entities"][entity_doc.id] = entity_data
                
                # Get relationships for this entity
                rels = (
                    self.knowledge_db.collection("relationships")
                    .where("from_entity", "==", entity_doc.id)
                    .limit(10)
                    .stream()
                )
                for rel in rels:
                    context["relationships"].append(rel.to_dict())
        
        # Get relevant facts
        facts = self.knowledge_db.collection("facts").limit(50).stream()
        for fact_doc in facts:
            fact_data = fact_doc.to_dict()
            fact_text = fact_data.get("fact", "").lower()
            if any(word in query_lower for word in fact_text.split()[:5]):
                context["facts"].append(fact_data)
        
        # Get patterns and temporal data
        patterns_doc = self.knowledge_db.collection("patterns").document("communication_patterns").get()
        if patterns_doc.exists:
            context["patterns"] = patterns_doc.to_dict()
        
        temporal_doc = self.knowledge_db.collection("temporal").document("email_patterns").get()
        if temporal_doc.exists:
            context["temporal"] = temporal_doc.to_dict()
        
        # Get thread summaries
        summaries_doc = self.knowledge_db.collection("thread_summaries").document("active_threads").get()
        if summaries_doc.exists:
            context["thread_summaries"] = summaries_doc.to_dict().get("threads", [])
        
        # Get corrections
        corrections = (
            self.knowledge_db.collection("corrections")
            .where("active", "==", True)
            .limit(10)
            .stream()
        )
        for correction in corrections:
            context["corrections"].append(correction.to_dict())
        
        return context


def main():
    """Main function to run daily knowledge update."""
    print("🚀 AD3Gem Knowledge Base Daily Update")
    print("=" * 50)
    
    kb = AD3GemKnowledgeBase()
    
    # Initialize structure if needed
    kb.initialize_knowledge_structure()
    
    # Run daily update
    kb.update_daily_knowledge(days_back=1)
    
    print("\n✅ Knowledge base update complete!")


if __name__ == "__main__":
    main()
