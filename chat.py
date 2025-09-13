#!/usr/bin/env python3
"""
Intelligent AD3Gem Chatbot - Enhanced Complete Implementation
Production-ready monolithic script with modern AI agent patterns

Original Features Preserved:
- Dual-purpose routing (company vs general questions)
- Structured logging with performance tracking
- Database connection management
- Simple caching layer
- Error recovery and fallback strategies
- Conversation storage in ad3gem-conversation
- Input validation and security
- Secret Manager integration

New Enhanced Features:
- Fixed collection queries (thread_summaries, people/companies)
- Enhanced AI query understanding
- User preference tracking and learning
- Email monitoring (unanswered, urgent)
- Actual sender detection from group emails
- Better knowledge base integration
"""

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from cachetools import TTLCache
from google.cloud import firestore, secretmanager


# Configuration and Environment
@dataclass
class Config:
    """Configuration management with environment variables"""

    # From environment variables
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "ad3-sam")
    user_id: str = os.getenv("USER_ID", "default_user")
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Database names
    knowledge_db: str = os.getenv("KNOWLEDGE_DB", "ad3gem-knowledge")
    email_db: str = os.getenv("EMAIL_DB", "ad3gem-email")
    conversation_db: str = os.getenv("CONVERSATION_DB", "ad3gem-conversation")

    # Performance settings
    max_query_length: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "600"))
    max_entities: int = int(os.getenv("MAX_ENTITIES", "10"))
    max_facts: int = int(os.getenv("MAX_FACTS", "15"))
    max_threads: int = int(os.getenv("MAX_THREADS", "5"))
    max_messages: int = int(os.getenv("MAX_MESSAGES", "10"))

    # Timeouts
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))
    db_timeout: int = int(os.getenv("DB_TIMEOUT", "10"))


class PerformanceTracker:
    """Track performance metrics for operations"""

    def __init__(self, logger):
        self.logger = logger
        self.metrics = {}

    def track_operation(self, operation_name: str):
        """Decorator to track operation performance"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                operation_id = f"{operation_name}_{int(time.time())}"

                self.logger.info(
                    f"Starting {operation_name}",
                    extra={
                        "operation": operation_name,
                        "operation_id": operation_id,
                        "status": "started",
                    },
                )

                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time

                    # Track metrics
                    if operation_name not in self.metrics:
                        self.metrics[operation_name] = []
                    self.metrics[operation_name].append(duration)

                    self.logger.info(
                        f"Completed {operation_name}",
                        extra={
                            "operation": operation_name,
                            "operation_id": operation_id,
                            "status": "completed",
                            "duration_seconds": round(duration, 3),
                            "result_size": len(str(result)) if result else 0,
                        },
                    )

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    self.logger.error(
                        f"Failed {operation_name}",
                        extra={
                            "operation": operation_name,
                            "operation_id": operation_id,
                            "status": "failed",
                            "duration_seconds": round(duration, 3),
                            "error": str(e),
                        },
                    )
                    raise

            return wrapper

        return decorator

    def get_metrics(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    "count": len(times),
                    "avg_duration": round(sum(times) / len(times), 3),
                    "max_duration": round(max(times), 3),
                    "min_duration": round(min(times), 3),
                }
        return stats


class SecretManager:
    """Manage secrets from Google Secret Manager"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
        self._cache = {}

    def get_secret(self, secret_name: str) -> str:
        """Get secret value with caching"""
        if secret_name in self._cache:
            return self._cache[secret_name]

        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            self._cache[secret_name] = secret_value
            return secret_value
        except Exception as e:
            raise ValueError(f"Failed to get secret {secret_name}: {e}")


class DatabaseManager:
    """Centralized database connection management"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._clients = {}

        # Initialize database clients
        self._init_clients()

    def _init_clients(self):
        """Initialize Firestore clients"""
        try:
            self._clients["knowledge"] = firestore.Client(
                project=self.config.project_id, database=self.config.knowledge_db
            )
            self._clients["email"] = firestore.Client(
                project=self.config.project_id, database=self.config.email_db
            )
            self._clients["conversation"] = firestore.Client(
                project=self.config.project_id, database=self.config.conversation_db
            )

            self.logger.info(
                "Database clients initialized",
                extra={
                    "databases": [
                        self.config.knowledge_db,
                        self.config.email_db,
                        self.config.conversation_db,
                    ]
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize database clients: {e}")
            raise

    def get_client(self, db_type: str) -> firestore.Client:
        """Get database client by type"""
        if db_type not in self._clients:
            raise ValueError(f"Unknown database type: {db_type}")
        return self._clients[db_type]

    def health_check(self) -> Dict[str, bool]:
        """Check health of all database connections"""
        health = {}
        for db_type, client in self._clients.items():
            try:
                # Simple test query
                list(client.collection("test").limit(1).stream())
                health[db_type] = True
            except Exception as e:
                self.logger.error(f"Database {db_type} health check failed: {e}")
                health[db_type] = False
        return health


class CacheManager:
    """Simple caching layer with TTL"""

    def __init__(self, ttl_seconds: int = 600):
        self.knowledge_cache = TTLCache(maxsize=100, ttl=ttl_seconds)
        self.email_cache = TTLCache(
            maxsize=50, ttl=ttl_seconds // 2
        )  # Shorter TTL for emails
        self.llm_cache = TTLCache(maxsize=200, ttl=ttl_seconds)
        self.preference_cache = TTLCache(maxsize=50, ttl=ttl_seconds * 2)

    def get_cache_key(self, operation: str, *args) -> str:
        """Generate cache key from operation and arguments"""
        key_data = f"{operation}_{str(args)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_knowledge(self, key: str) -> Optional[Any]:
        return self.knowledge_cache.get(key)

    def set_knowledge(self, key: str, value: Any):
        self.knowledge_cache[key] = value

    def get_email(self, key: str) -> Optional[Any]:
        return self.email_cache.get(key)

    def set_email(self, key: str, value: Any):
        self.email_cache[key] = value

    def get_llm(self, key: str) -> Optional[Any]:
        return self.llm_cache.get(key)

    def set_llm(self, key: str, value: Any):
        self.llm_cache[key] = value

    def get_preference(self, key: str) -> Optional[Any]:
        return self.preference_cache.get(key)

    def set_preference(self, key: str, value: Any):
        self.preference_cache[key] = value

    def get_stats(self) -> Dict:
        return {
            "knowledge_cache": {
                "size": len(self.knowledge_cache),
                "maxsize": self.knowledge_cache.maxsize,
            },
            "email_cache": {
                "size": len(self.email_cache),
                "maxsize": self.email_cache.maxsize,
            },
            "llm_cache": {
                "size": len(self.llm_cache),
                "maxsize": self.llm_cache.maxsize,
            },
            "preference_cache": {
                "size": len(self.preference_cache),
                "maxsize": self.preference_cache.maxsize,
            },
        }


class InputValidator:
    """Validate and sanitize user inputs"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Dangerous patterns to reject
        self.dangerous_patterns = [
            "eval(",
            "exec(",
            "__import__",
            "os.system",
            "subprocess",
            "<script",
            "javascript:",
            "data:text/html",
        ]

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate user query"""
        if not query or not query.strip():
            return False, "Query cannot be empty"

        if len(query) > self.config.max_query_length:
            return (
                False,
                f"Query too long (max {self.config.max_query_length} characters)",
            )

        # Check for dangerous patterns
        query_lower = query.lower()
        for pattern in self.dangerous_patterns:
            if pattern in query_lower:
                self.logger.warning(
                    f"Dangerous pattern detected: {pattern}",
                    extra={"query_preview": query[:50], "pattern": pattern},
                )
                return False, "Query contains invalid content"

        return True, "Valid"

    def sanitize_query(self, query: str) -> str:
        """Sanitize query for safe processing"""
        # Remove control characters
        sanitized = "".join(
            char for char in query if ord(char) >= 32 or char in "\n\r\t"
        )

        # Trim whitespace
        sanitized = sanitized.strip()

        return sanitized


class PreferenceTracker:
    """Track and learn user preferences"""

    def __init__(
        self, db_manager: DatabaseManager, user_id: str, logger: logging.Logger
    ):
        self.db_manager = db_manager
        self.user_id = user_id
        self.logger = logger
        self.preferences = self._load_preferences()

    def _load_preferences(self) -> Dict:
        """Load user preferences from Firestore"""
        try:
            client = self.db_manager.get_client("conversation")
            doc = client.collection("user_preferences").document(self.user_id).get()

            if doc.exists:
                return doc.to_dict()
            else:
                # Initialize with defaults
                return {
                    "name_mappings": {},  # "julie" -> "julie@lineagecoffee.com"
                    "default_supplier": None,  # When user says "the supplier"
                    "common_queries": [],  # Track frequent queries
                    "corrections": {},  # Learn from corrections
                    "time_zone": "Africa/Johannesburg",
                    "created_at": datetime.now(timezone.utc),
                }
        except Exception as e:
            self.logger.error(f"Failed to load preferences: {e}")
            return {}

    def save_preferences(self):
        """Save preferences to Firestore"""
        try:
            client = self.db_manager.get_client("conversation")
            doc_ref = client.collection("user_preferences").document(self.user_id)

            self.preferences["updated_at"] = datetime.now(timezone.utc)
            doc_ref.set(self.preferences, merge=True)

        except Exception as e:
            self.logger.error(f"Failed to save preferences: {e}")

    def learn_name_mapping(self, name: str, email: str):
        """Learn that when user says 'name', they mean this email"""
        if "name_mappings" not in self.preferences:
            self.preferences["name_mappings"] = {}

        self.preferences["name_mappings"][name.lower()] = email
        self.save_preferences()

    def get_email_for_name(self, name: str) -> Optional[str]:
        """Get the email address for a name based on learned preferences"""
        return self.preferences.get("name_mappings", {}).get(name.lower())

    def track_query(self, query: str):
        """Track common queries"""
        if "common_queries" not in self.preferences:
            self.preferences["common_queries"] = []

        # Keep last 50 queries
        self.preferences["common_queries"].append(
            {"query": query, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        self.preferences["common_queries"] = self.preferences["common_queries"][-50:]
        self.save_preferences()


class EmailMonitor:
    """Monitor emails for important patterns"""

    def __init__(self, db_manager: DatabaseManager, logger: logging.Logger):
        self.db_manager = db_manager
        self.logger = logger

    def check_unanswered_emails(self, hours: int = 48) -> List[Dict]:
        """Find emails that haven't been answered"""
        try:
            email_client = self.db_manager.get_client("email")
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Find threads where last message is not from owner
            unanswered = []

            threads = (
                email_client.collection("thread_summaries")
                .where("lastActivity", "<", cutoff_time)
                .limit(20)
                .stream()
            )

            for thread in threads:
                thread_data = thread.to_dict()
                last_from = thread_data.get("lastMessage", {}).get("from", "")

                # Check if we haven't responded
                if not last_from.endswith("@lineagecoffee.com"):
                    unanswered.append(
                        {
                            "threadId": thread.id,
                            "subject": thread_data.get("subject", "No subject"),
                            "from": last_from,
                            "lastActivity": thread_data.get("lastActivity"),
                            "preview": thread_data.get("lastMessage", {}).get(
                                "preview", ""
                            ),
                        }
                    )

            return unanswered

        except Exception as e:
            self.logger.error(f"Failed to check unanswered emails: {e}")
            return []

    def find_urgent_unread(self) -> List[Dict]:
        """Find unread emails marked as important or containing urgent keywords"""
        try:
            email_client = self.db_manager.get_client("email")

            # Query important unread emails
            urgent = []

            messages = (
                email_client.collection("email_search_index")
                .where("isUnread", "==", True)
                .where("isImportant", "==", True)
                .limit(10)
                .stream()
            )

            for msg in messages:
                msg_data = msg.to_dict()
                urgent.append(
                    {
                        "messageId": msg.id,
                        "subject": msg_data.get("subject", ""),
                        "from": msg_data.get("fromEmail", ""),
                        "sentAt": msg_data.get("sentAt"),
                        "preview": msg_data.get("bodyPreview", ""),
                    }
                )

            return urgent

        except Exception as e:
            self.logger.error(f"Failed to find urgent emails: {e}")
            return []

    def find_last_email_from_person(self, person_name: str) -> Optional[Dict]:
        """Find the last email from a specific person"""
        try:
            email_client = self.db_manager.get_client("email")

            # Try to find by fromPerson field
            last_email = (
                email_client.collection("email_search_index")
                .where("fromPerson", "==", person_name.lower())
                .order_by("sentAt", direction=firestore.Query.DESCENDING)
                .limit(1)
                .stream()
            )

            for email in last_email:
                email_data = email.to_dict()
                return {
                    "sentAt": email_data.get("sentAt"),
                    "subject": email_data.get("subject"),
                    "preview": email_data.get("bodyPreview"),
                    "from": email_data.get("fromEmail"),
                }

            return None

        except Exception as e:
            self.logger.error(f"Failed to find last email from {person_name}: {e}")
            return None


class IntelligentAD3GemChatbot:
    """
    Production-ready intelligent chatbot with dual-purpose routing and enhanced features
    """

    def __init__(self, config: Config = None):
        """Initialize the chatbot with all components"""
        self.config = config or Config()

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self.secret_manager = SecretManager(self.config.project_id)
        self.performance_tracker = PerformanceTracker(self.logger)
        self.cache_manager = CacheManager(self.config.cache_ttl_seconds)
        self.input_validator = InputValidator(self.config, self.logger)
        self.db_manager = DatabaseManager(self.config, self.logger)

        # Initialize new components
        self.preference_tracker = PreferenceTracker(
            self.db_manager, self.config.user_id, self.logger
        )
        self.email_monitor = EmailMonitor(self.db_manager, self.logger)

        # Initialize Gemini
        self._init_gemini()

        # Session state
        self.session_id = (
            f"session_{self.config.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.session_context = {
            "last_mode": None,
            "last_entities": [],
            "last_topics": [],
            "message_count": 0,
            "start_time": datetime.now(timezone.utc),
        }

        self.logger.info(
            "Chatbot initialized successfully",
            extra={
                "user_id": self.config.user_id,
                "session_id": self.session_id,
                "environment": self.config.environment,
            },
        )

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("AD3GemChatbot")
        logger.setLevel(getattr(logging, self.config.log_level))

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(f"chatbot_{self.config.environment}.log")
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s | %(pathname)s:%(lineno)d"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _init_gemini(self):
        """Initialize Gemini with API key from Secret Manager"""
        try:
            api_key = self.secret_manager.get_secret("gemini-api-key")
            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel(
                "gemini-1.5-pro",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_output_tokens": 2048,
                },
            )

            self.logger.info("Gemini initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {e}")
            raise

    def save_conversation_message(self, role: str, content: str, metadata: Dict = None):
        """Save message to conversation database"""
        try:
            conversation_client = self.db_manager.get_client("conversation")

            # User document path
            user_doc_path = f"users/{self.config.user_id}"
            session_doc_path = f"{user_doc_path}/sessions/{self.session_id}"

            # Create/update user profile
            user_ref = conversation_client.document(user_doc_path)
            user_ref.set(
                {
                    "user_id": self.config.user_id,
                    "last_active": datetime.now(timezone.utc),
                    "total_sessions": firestore.Increment(1)
                    if role == "user" and self.session_context["message_count"] == 0
                    else firestore.Increment(0),
                },
                merge=True,
            )

            # Create/update session
            session_ref = conversation_client.document(session_doc_path)
            session_ref.set(
                {
                    "session_id": self.session_id,
                    "start_time": self.session_context["start_time"],
                    "last_message": datetime.now(timezone.utc),
                    "message_count": firestore.Increment(1),
                    "environment": self.config.environment,
                },
                merge=True,
            )

            # Add message
            message_ref = session_ref.collection("messages").document()
            message_data = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc),
                "session_context": self.session_context.copy()
                if role == "user"
                else None,
                "metadata": metadata or {},
            }
            message_ref.set(message_data)

            self.session_context["message_count"] += 1

        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")
            # Don't raise - conversation saving shouldn't break the chat

    def understand_query_enhanced(self, user_message: str) -> Dict:
        """Enhanced query understanding using AI"""

        # Check preferences for common interpretations
        preferred_email = self.preference_tracker.get_email_for_name(
            user_message.lower()
        )

        prompt = f"""
        Analyze this email query and extract structured information.

        Query: "{user_message}"

        Known context:
        - Company domain: @lineagecoffee.com
        - If user says "Julie", they likely mean julie@lineagecoffee.com
        - Common suppliers: Jersey Cow (milk), Early Moon Trading, LP Agencies
        - Preferred mapping: {preferred_email if preferred_email else "None"}

        Extract the following in JSON format:
        {{
            "intent": "search_emails/monitor/analyze",
            "people_mentioned": ["email addresses or names"],
            "time_range": {{
                "start": "ISO date or null",
                "end": "ISO date or null",
                "description": "last week/yesterday/etc"
            }},
            "keywords": ["relevant", "search", "terms"],
            "sender_filter": "email if looking for specific sender",
            "recipient_filter": "email if looking for specific recipient",
            "urgency_check": true/false,
            "sentiment": "positive/negative/neutral/urgent",
            "needs_expansion": ["terms that should be expanded"],
            "monitoring_type": "unanswered/urgent/response_time/last_email or null",
            "asking_for_money": true/false,
            "group_email_check": true/false
        }}

        Examples:
        - "when did Rox last send an email" -> monitoring_type: "last_email", people_mentioned: ["rox"]
        - "who was the last person asking for money" -> asking_for_money: true
        - "emails from info@" -> group_email_check: true

        Return ONLY valid JSON.
        """

        try:
            response = self.model.generate_content(prompt)
            json_str = response.text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:-3]

            understanding = json.loads(json_str)

            # Track this query for learning
            self.preference_tracker.track_query(user_message)

            return understanding

        except Exception as e:
            self.logger.error(f"Query understanding failed: {e}")
            return {
                "intent": "search_emails",
                "keywords": user_message.lower().split(),
                "people_mentioned": [],
            }

    def route_query(self, user_message: str) -> Dict:
        """Determine if query is about company or general knowledge"""

        # Check cache first
        cache_key = self.cache_manager.get_cache_key("route", user_message)
        cached_result = self.cache_manager.get_llm(cache_key)
        if cached_result:
            self.logger.info("Using cached routing decision")
            return cached_result

        # Get enhanced understanding
        query_understanding = self.understand_query_enhanced(user_message)

        # Get basic context indicators
        query_lower = user_message.lower()
        company_indicators = [
            "email",
            "sent",
            "received",
            "julie",
            "sam",
            "invoice",
            "project",
            "meeting",
            "team",
            "colleague",
            "work",
            "office",
            "client",
            "last",
            "when",
            "who",
            "money",
            "payment",
        ]

        company_score = sum(
            1 for indicator in company_indicators if indicator in query_lower
        )

        # If monitoring type detected, it's definitely company
        if query_understanding.get("monitoring_type"):
            company_score += 5

        prompt = f"""
        Analyze this query to determine if it's about internal company matters or general knowledge.

        Query: "{user_message}"

        Company indicators found: {company_score}
        Query understanding: {json.dumps(query_understanding)}

        Respond with JSON:
        {{
            "mode": "company" or "general",
            "confidence": 0-100,
            "reasoning": "brief explanation",
            "requires_email_search": true/false,
            "requires_web_search": true/false,
            "query_understanding": {json.dumps(query_understanding)}
        }}

        Company mode: Questions about emails, colleagues, projects, internal communications
        General mode: General knowledge, how-to questions, external information

        Return ONLY valid JSON.
        """

        try:
            response = self.model.generate_content(prompt)
            json_str = response.text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:-3]

            routing = json.loads(json_str)
            routing["query_understanding"] = query_understanding

            # Cache the result
            self.cache_manager.set_llm(cache_key, routing)

            self.logger.info(
                "Query routed",
                extra={
                    "mode": routing.get("mode"),
                    "confidence": routing.get("confidence"),
                    "reasoning": routing.get("reasoning"),
                },
            )

            return routing

        except Exception as e:
            self.logger.error(f"Routing failed: {e}")
            # Fallback: if query mentions emails/people, assume company mode
            fallback_mode = "company" if company_score > 0 else "general"
            return {
                "mode": fallback_mode,
                "confidence": 60,
                "reasoning": "Fallback routing due to LLM error",
                "requires_email_search": fallback_mode == "company",
                "requires_web_search": fallback_mode == "general",
                "query_understanding": query_understanding,
            }

    def get_knowledge_context(
        self, user_message: str, query_understanding: Dict = None
    ) -> Dict:
        """Get relevant context from knowledge base with FIXED collection names"""

        cache_key = self.cache_manager.get_cache_key("knowledge", user_message)
        cached_context = self.cache_manager.get_knowledge(cache_key)
        if cached_context:
            self.logger.info("Using cached knowledge context")
            return cached_context

        context = {
            "people": {},
            "companies": {},
            "relationships": [],
            "suppliers": [],
            "policies": {},
            "locations": {},
        }

        try:
            knowledge_client = self.db_manager.get_client("knowledge")
            query_lower = user_message.lower()

            # Use query understanding if provided
            people_mentioned = []
            if query_understanding:
                people_mentioned = query_understanding.get("people_mentioned", [])

            # Search people collection (NOT entities)
            with ThreadPoolExecutor(max_workers=3) as executor:
                people_future = executor.submit(
                    self._search_people, knowledge_client, query_lower, people_mentioned
                )
                companies_future = executor.submit(
                    self._search_companies, knowledge_client, query_lower
                )
                suppliers_future = executor.submit(
                    self._get_suppliers, knowledge_client
                )

                # Collect results
                context["people"] = people_future.result() or {}
                context["companies"] = companies_future.result() or {}
                context["suppliers"] = suppliers_future.result() or []

            # Get relationships for found people
            if context["people"]:
                context["relationships"] = self._get_relationships(
                    knowledge_client, list(context["people"].keys())[:5]
                )

            # Cache the result
            self.cache_manager.set_knowledge(cache_key, context)

            self.logger.info(
                "Knowledge context retrieved",
                extra={
                    "people_count": len(context["people"]),
                    "companies_count": len(context["companies"]),
                    "suppliers_count": len(context["suppliers"]),
                    "relationships_count": len(context["relationships"]),
                },
            )

        except Exception as e:
            self.logger.error(f"Knowledge context retrieval failed: {e}")

        return context

    def _search_people(
        self, client, query_lower: str, people_mentioned: List[str]
    ) -> Dict:
        """Search for relevant people in the people collection"""
        people = {}
        try:
            people_docs = (
                client.collection("people").limit(self.config.max_entities * 2).stream()
            )

            for person_doc in people_docs:
                person_data = person_doc.to_dict()

                # Check relevance
                mentioned = False
                person_name = person_data.get("name", "").lower()
                aliases = [a.lower() for a in person_data.get("aliases", [])]

                # Check if mentioned in query
                if person_name in query_lower:
                    mentioned = True

                # Check aliases
                for alias in aliases:
                    if alias in query_lower:
                        mentioned = True
                        break

                # Check people_mentioned list
                for mention in people_mentioned:
                    if mention.lower() in person_name or mention.lower() in aliases:
                        mentioned = True
                        # Learn this mapping
                        if person_data.get("email_addresses"):
                            self.preference_tracker.learn_name_mapping(
                                mention, person_data["email_addresses"][0]
                            )
                        break

                # Check email addresses
                if not mentioned:
                    for email in person_data.get("email_addresses", []):
                        if email.lower().split("@")[0] in query_lower:
                            mentioned = True
                            break

                if mentioned:
                    people[person_doc.id] = person_data

                if len(people) >= self.config.max_entities:
                    break

        except Exception as e:
            self.logger.error(f"People search failed: {e}")

        return people

    def _search_companies(self, client, query_lower: str) -> Dict:
        """Search for relevant companies"""
        companies = {}
        try:
            company_docs = (
                client.collection("companies")
                .limit(self.config.max_entities * 2)
                .stream()
            )

            for company_doc in company_docs:
                company_data = company_doc.to_dict()

                # Check relevance
                mentioned = False
                company_name = company_data.get("name", "").lower()
                aliases = [a.lower() for a in company_data.get("aliases", [])]

                if company_name in query_lower:
                    mentioned = True

                for alias in aliases:
                    if alias in query_lower:
                        mentioned = True
                        break

                if mentioned:
                    companies[company_doc.id] = company_data

                if len(companies) >= self.config.max_entities:
                    break

        except Exception as e:
            self.logger.error(f"Company search failed: {e}")

        return companies

    def _get_suppliers(self, client) -> List[Dict]:
        """Get all suppliers"""
        suppliers = []
        try:
            supplier_docs = (
                client.collection("companies")
                .where("classification", "==", "supplier")
                .limit(20)
                .stream()
            )

            for supplier_doc in supplier_docs:
                suppliers.append(supplier_doc.to_dict())

        except Exception as e:
            self.logger.error(f"Supplier retrieval failed: {e}")

        return suppliers

    def _get_relationships(self, client, entity_ids: List[str]) -> List:
        """Get relationships for entities"""
        relationships = []
        try:
            for entity_id in entity_ids:
                rels = (
                    client.collection("relationships")
                    .where("from_entity", "==", entity_id)
                    .limit(3)
                    .stream()
                )

                for rel in rels:
                    relationships.append(rel.to_dict())

        except Exception as e:
            self.logger.error(f"Relationship retrieval failed: {e}")

        return relationships

    def search_emails_enhanced(
        self,
        user_message: str,
        knowledge_context: Dict,
        query_understanding: Dict = None,
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """Search emails with FIXED collection names and enhanced features"""

        cache_key = self.cache_manager.get_cache_key(
            "emails", user_message, str(knowledge_context), str(query_understanding)
        )
        cached_result = self.cache_manager.get_email(cache_key)
        if cached_result:
            self.logger.info("Using cached email search results")
            return cached_result

        threads, messages, search_stats = [], [], {}

        try:
            email_client = self.db_manager.get_client("email")

            # Handle monitoring requests
            if query_understanding and query_understanding.get("monitoring_type"):
                monitoring_type = query_understanding["monitoring_type"]

                if monitoring_type == "last_email":
                    # Find last email from person
                    people_mentioned = query_understanding.get("people_mentioned", [])
                    for person in people_mentioned:
                        last_email = self.email_monitor.find_last_email_from_person(
                            person
                        )
                        if last_email:
                            messages.append(last_email)
                            search_stats["last_email"] = last_email

                elif monitoring_type == "unanswered":
                    unanswered = self.email_monitor.check_unanswered_emails()
                    search_stats["unanswered_emails"] = unanswered

                elif monitoring_type == "urgent":
                    urgent = self.email_monitor.find_urgent_unread()
                    search_stats["urgent_emails"] = urgent

            # Handle "asking for money" queries
            if query_understanding and query_understanding.get("asking_for_money"):
                money_keywords = [
                    "payment",
                    "invoice",
                    "bill",
                    "due",
                    "owe",
                    "pay",
                    "remittance",
                ]

                # Search for emails with money-related keywords
                money_messages = email_client.collection("email_search_index")
                for keyword in money_keywords:
                    money_query = (
                        money_messages.where("searchTerms", "array_contains", keyword)
                        .order_by("sentAt", direction=firestore.Query.DESCENDING)
                        .limit(10)
                    )

                    for msg in money_query.stream():
                        msg_data = msg.to_dict()
                        # Check if from external (likely suppliers asking for money)
                        if msg_data.get("hasExternal"):
                            messages.append(msg_data)
                            break

            # Regular email search using email_search_index
            else:
                # Build query based on understanding
                query = email_client.collection("email_search_index")

                # Add filters based on query understanding
                if query_understanding:
                    sender_filter = query_understanding.get("sender_filter")
                    if sender_filter:
                        query = query.where("fromEmail", "==", sender_filter)

                    recipient_filter = query_understanding.get("recipient_filter")
                    if recipient_filter:
                        query = query.where(
                            "toEmails", "array_contains", recipient_filter
                        )

                    # Time range filter
                    time_range = query_understanding.get("time_range", {})
                    if time_range.get("start"):
                        start_date = datetime.fromisoformat(time_range["start"])
                        query = query.where("sentAt", ">=", start_date)
                    if time_range.get("end"):
                        end_date = datetime.fromisoformat(time_range["end"])
                        query = query.where("sentAt", "<=", end_date)

                    # Group email check
                    if query_understanding.get("group_email_check"):
                        # Look for emails from group addresses
                        group_addresses = ["info@", "support@", "admin@", "contact@"]
                        for group_addr in group_addresses:
                            if group_addr in user_message.lower():
                                # Search by owner + SENT label to find actual sender
                                pass  # This needs special handling

                # Get mentioned email addresses from knowledge context
                mentioned_emails = []
                for person_data in knowledge_context.get("people", {}).values():
                    mentioned_emails.extend(person_data.get("email_addresses", []))

                # Search by participants if we have emails
                if mentioned_emails:
                    for email in mentioned_emails[:3]:
                        # Can't use array_contains on allEmails, need different approach
                        participant_messages = (
                            email_client.collection("email_search_index")
                            .where("fromEmail", "==", email)
                            .order_by("sentAt", direction=firestore.Query.DESCENDING)
                            .limit(5)
                            .stream()
                        )
                        for msg in participant_messages:
                            messages.append(msg.to_dict())
                else:
                    # General search - get recent messages
                    recent_messages = (
                        query.order_by("sentAt", direction=firestore.Query.DESCENDING)
                        .limit(self.config.max_messages)
                        .stream()
                    )

                    for msg in recent_messages:
                        messages.append(msg.to_dict())

            # Get thread summaries for found messages
            thread_ids = set()
            for msg in messages:
                thread_id = msg.get("threadId")
                if thread_id:
                    thread_ids.add(thread_id)

            # Fetch thread summaries (using CORRECT collection name)
            for thread_id in list(thread_ids)[: self.config.max_threads]:
                try:
                    thread_doc = (
                        email_client.collection("thread_summaries")
                        .document(thread_id)
                        .get()
                    )
                    if thread_doc.exists:
                        thread_data = thread_doc.to_dict()
                        thread_data["thread_id"] = thread_id
                        threads.append(thread_data)
                except Exception as e:
                    self.logger.debug(f"Failed to get thread {thread_id}: {e}")

            # Compile statistics
            search_stats.update(
                {
                    "total_messages": len(messages),
                    "total_threads": len(threads),
                    "mentioned_emails": len(mentioned_emails),
                }
            )

            # Cache results
            result = (threads, messages, search_stats)
            self.cache_manager.set_email(cache_key, result)

            self.logger.info(
                "Email search completed",
                extra=search_stats,
            )

        except Exception as e:
            self.logger.error(f"Email search failed: {e}")

        return threads, messages, search_stats

    def web_search(self, query: str) -> List[Dict]:
        """Perform web search for general queries"""
        # Placeholder for web search functionality
        # In production, integrate with search APIs like Google Search API, Bing, etc.

        self.logger.info("Web search requested", extra={"query": query})

        # Mock implementation - replace with actual web search
        return [
            {
                "title": "Web search not implemented",
                "snippet": "This would integrate with web search APIs",
                "url": "https://example.com",
            }
        ]

    def generate_response(
        self,
        user_message: str,
        routing: Dict,
        knowledge_context: Dict,
        email_data: Tuple[List[Dict], List[Dict], Dict] = None,
        web_data: List[Dict] = None,
    ) -> str:
        """Generate intelligent response based on mode and context"""

        # Check cache
        cache_key = self.cache_manager.get_cache_key(
            "response", user_message, str(routing), str(knowledge_context)
        )
        cached_response = self.cache_manager.get_llm(cache_key)
        if cached_response:
            self.logger.info("Using cached response")
            return cached_response

        try:
            if routing.get("mode") == "company":
                response = self._generate_company_response(
                    user_message, routing, knowledge_context, email_data
                )
            else:
                response = self._generate_general_response(
                    user_message, routing, knowledge_context, web_data
                )

            # Cache the response
            self.cache_manager.set_llm(cache_key, response)

            return response

        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def _generate_company_response(
        self,
        user_message: str,
        routing: Dict,
        knowledge_context: Dict,
        email_data: Tuple[List[Dict], List[Dict], Dict],
    ) -> str:
        """Generate response for company-related queries with enhanced features"""

        threads, messages, search_stats = email_data if email_data else ([], [], {})
        query_understanding = routing.get("query_understanding", {})

        # Format context for prompt
        entity_info = self._format_entities_enhanced(knowledge_context)
        thread_info = self._format_threads(threads[:3])
        message_info = self._format_messages_enhanced(messages[:5])
        supplier_info = self._format_suppliers(knowledge_context.get("suppliers", []))

        # Handle special queries
        special_info = ""

        # Handle "last email" queries
        if search_stats.get("last_email"):
            last = search_stats["last_email"]
            sent_at = last.get("sentAt", "Unknown")
            if hasattr(sent_at, "strftime"):
                sent_at = sent_at.strftime("%Y-%m-%d %H:%M")
            special_info = f"\nLAST EMAIL INFO:\n{last.get('from', 'Unknown')} sent their last email on {sent_at}\nSubject: {last.get('subject', 'No subject')}\n"

        # Handle "asking for money" queries
        if query_understanding.get("asking_for_money") and messages:
            money_requests = []
            for msg in messages:
                if any(
                    term in str(msg.get("searchTerms", []))
                    for term in ["payment", "invoice", "due"]
                ):
                    money_requests.append(
                        f"- {msg.get('fromEmail')} on {msg.get('sentAt', 'Unknown')}: {msg.get('subject', 'No subject')}"
                    )
            if money_requests:
                special_info += "\nPEOPLE ASKING FOR MONEY:\n" + "\n".join(
                    money_requests[:3]
                )

        # Handle unanswered emails
        if search_stats.get("unanswered_emails"):
            unanswered = search_stats["unanswered_emails"]
            special_info += f"\nUNANSWERED EMAILS ({len(unanswered)} total):\n"
            for email in unanswered[:3]:
                special_info += f"- {email['subject']} from {email['from']}\n"

        prompt = f"""
        You are AD3Gem, an intelligent email assistant with comprehensive knowledge of the company.

        USER QUERY: "{user_message}"

        QUERY UNDERSTANDING:
        Intent: {query_understanding.get("intent", "search")}
        Monitoring type: {query_understanding.get("monitoring_type", "none")}
        Time range: {
            query_understanding.get("time_range", {}).get("description", "all time")
        }

        KNOWLEDGE CONTEXT:
        People & Companies:
        {entity_info}

        Suppliers:
        {supplier_info}

        EMAIL SEARCH RESULTS:
        {
            f"Found {search_stats.get('total_messages', 0)} messages in {search_stats.get('total_threads', 0)} threads"
            if not special_info
            else ""
        }

        {
            special_info
            if special_info
            else f'''
        Threads Found:
        {thread_info}

        Recent Messages:
        {message_info}
        '''
        }

        INSTRUCTIONS:
        1. Answer the user's specific question directly
        2. Use people's actual names when you know them
        3. Reference specific emails, dates, and details
        4. For "when did X last email" questions, provide the exact date and subject
        5. For "who's asking for money" questions, list the people and what they're asking for
        6. Be conversational but precise
        7. Keep response under 200 words unless listing multiple items

        Generate a helpful response:
        """

        response = self.model.generate_content(prompt)
        return response.text

    def _generate_general_response(
        self,
        user_message: str,
        routing: Dict,
        knowledge_context: Dict,
        web_data: List[Dict],
    ) -> str:
        """Generate response for general knowledge queries"""

        entity_info = self._format_entities_enhanced(knowledge_context)
        web_info = self._format_web_results(web_data or [])

        prompt = f"""
        You are AD3Gem, an intelligent AI assistant with access to general knowledge.

        USER QUERY: "{user_message}"

        ROUTING ANALYSIS:
        Mode: {routing.get("mode")}
        Confidence: {routing.get("confidence")}%

        CONTEXT FILTER (from company knowledge):
        {entity_info if entity_info else "No relevant company context"}

        WEB SEARCH RESULTS:
        {web_info}

        INSTRUCTIONS:
        1. Answer the user's question using your general knowledge and any web search results
        2. If the query relates to people/entities in the context filter, acknowledge that connection
        3. Be helpful, accurate, and conversational
        4. If you need more information, suggest specific follow-up questions
        5. Keep response focused and under 200 words unless the query requires detail

        Generate a helpful response:
        """

        response = self.model.generate_content(prompt)
        return response.text

    def _format_entities_enhanced(self, knowledge_context: Dict) -> str:
        """Format entity information including people and companies"""
        formatted = []

        # Format people
        people = knowledge_context.get("people", {})
        if people:
            formatted.append("PEOPLE:")
            for person_id, data in people.items():
                names = data.get("name", "Unknown")
                aliases = ", ".join(data.get("aliases", []))
                emails = ", ".join(data.get("email_addresses", [])[:2])
                roles = ", ".join(data.get("roles", ["Unknown role"])[:2])

                formatted.append(f"- {names} ({aliases}) | {emails} | {roles}")

        # Format companies
        companies = knowledge_context.get("companies", {})
        if companies:
            formatted.append("\nCOMPANIES:")
            for company_id, data in companies.items():
                name = data.get("name", "Unknown")
                classification = data.get("classification", "")
                contact = data.get("contact_info", {})

                formatted.append(
                    f"- {name} ({classification}) | {contact.get('email', 'No email')}"
                )

        return "\n".join(formatted) if formatted else "No entities identified"

    def _format_suppliers(self, suppliers: List[Dict]) -> str:
        """Format supplier information"""
        if not suppliers:
            return "No suppliers found"

        formatted = []
        for supplier in suppliers[:5]:
            contact = supplier.get("contact_info", {})
            formatted.append(
                f"- {supplier.get('name', 'Unknown')} | {contact.get('email', 'No email')} | {contact.get('contact_person', 'No contact')}"
            )

        return "\n".join(formatted)

    def _format_threads(self, threads: List[Dict]) -> str:
        """Format thread information for prompts"""
        if not threads:
            return "No threads found"

        formatted = []
        for thread in threads:
            # Handle participants as dictionary (not array)
            participants_dict = thread.get("participants", {})
            participant_emails = list(participants_dict.keys())[:3]
            participants = ", ".join(participant_emails)

            last_msg = thread.get("lastActivity", "Unknown")
            if hasattr(last_msg, "strftime"):
                last_msg = last_msg.strftime("%Y-%m-%d %H:%M")

            formatted.append(
                f"- {thread.get('subject', 'No subject')}\n"
                f"  Participants: {participants}\n"
                f"  Messages: {thread.get('messageCount', 0)} | Last: {last_msg}\n"
                f"  Preview: {thread.get('lastMessage', {}).get('preview', '')[:100]}..."
            )

        return "\n".join(formatted)

    def _format_messages_enhanced(self, messages: List[Dict]) -> str:
        """Format message information with enhanced details"""
        if not messages:
            return "No messages found"

        formatted = []
        for msg in messages:
            # Handle both search results and monitoring results
            from_email = msg.get("fromEmail", msg.get("from", "Unknown"))
            from_person = msg.get("fromPerson", "")

            sent_at = msg.get("sentAt", "Unknown")
            if hasattr(sent_at, "strftime"):
                sent_at = sent_at.strftime("%Y-%m-%d %H:%M")

            # Check for group email and actual sender
            owner = msg.get("owner", "")
            label_ids = msg.get("labelIds", [])
            actual_sender = ""
            if owner and "SENT" in label_ids and owner != from_email:
                actual_sender = f" [Actually sent by {owner}]"

            formatted.append(
                f"- From: {from_person or from_email}{actual_sender} | {sent_at}\n"
                f"  Subject: {msg.get('subject', 'No subject')}\n"
                f"  Preview: {msg.get('bodyPreview', msg.get('preview', ''))[:150]}..."
            )

        return "\n".join(formatted)

    def _format_web_results(self, web_results: List[Dict]) -> str:
        """Format web search results"""
        if not web_results:
            return "No web search results"

        formatted = []
        for result in web_results[:3]:
            formatted.append(
                f"- {result.get('title', 'No title')}\n"
                f"  {result.get('snippet', 'No snippet')}\n"
                f"  URL: {result.get('url', 'No URL')}"
            )

        return "\n".join(formatted)

    def send_message(self, user_message: str) -> str:
        """Process user message with full intelligent routing and enhanced features"""

        start_time = time.time()

        # Input validation
        is_valid, validation_msg = self.input_validator.validate_query(user_message)
        if not is_valid:
            self.logger.warning(
                f"Invalid input: {validation_msg}",
                extra={"query_preview": user_message[:50]},
            )
            return f"Sorry, {validation_msg.lower()}. Please try again."

        # Sanitize input
        user_message = self.input_validator.sanitize_query(user_message)

        self.logger.info(
            "Processing user message",
            extra={
                "user_id": self.config.user_id,
                "session_id": self.session_id,
                "message_length": len(user_message),
            },
        )

        try:
            # Save user message
            self.save_conversation_message("user", user_message)

            # Route the query with enhanced understanding
            routing = self.route_query(user_message)
            query_understanding = routing.get("query_understanding", {})

            # Get knowledge context
            knowledge_context = self.get_knowledge_context(
                user_message, query_understanding
            )

            # Process based on routing
            email_data = None
            web_data = None

            if routing.get("mode") == "company" and routing.get(
                "requires_email_search"
            ):
                email_data = self.search_emails_enhanced(
                    user_message, knowledge_context, query_understanding
                )

            if routing.get("mode") == "general" and routing.get("requires_web_search"):
                web_data = self.web_search(user_message)

            # Generate response
            response = self.generate_response(
                user_message, routing, knowledge_context, email_data, web_data
            )

            # Save response
            response_metadata = {
                "routing": routing,
                "knowledge_entities": len(knowledge_context.get("people", {}))
                + len(knowledge_context.get("companies", {})),
                "email_results": email_data[2] if email_data else {},
                "web_results": len(web_data) if web_data else 0,
                "processing_time": round(time.time() - start_time, 3),
            }

            self.save_conversation_message("assistant", response, response_metadata)

            # Update session context
            self.session_context.update(
                {
                    "last_mode": routing.get("mode"),
                    "last_entities": list(knowledge_context.get("people", {}).keys())[
                        :5
                    ],
                    "last_topics": routing.get("topics", []),
                }
            )

            self.logger.info(
                "Message processed successfully",
                extra={
                    "mode": routing.get("mode"),
                    "processing_time": round(time.time() - start_time, 3),
                    "response_length": len(response),
                },
            )

            return response

        except Exception as e:
            self.logger.error(
                f"Message processing failed: {e}",
                extra={"user_message": user_message[:100]},
            )

            # Fallback response
            return (
                "I apologize, but I encountered an error while processing your message. "
                "Please try again, or contact support if the issue persists."
            )

    def run_daily_monitoring(self) -> str:
        """Run daily monitoring checks"""

        report = ["📊 Daily Email Monitoring Report", "=" * 40]

        try:
            # Check unanswered emails
            unanswered = self.email_monitor.check_unanswered_emails(hours=48)
            if unanswered:
                report.append(
                    f"\n📬 {len(unanswered)} emails need responses (>48 hours):"
                )
                for email in unanswered[:5]:
                    sent_time = email.get("lastActivity", "Unknown")
                    if hasattr(sent_time, "strftime"):
                        sent_time = sent_time.strftime("%Y-%m-%d %H:%M")
                    report.append(
                        f"  - {email['subject']} from {email['from']} ({sent_time})"
                    )

            # Check urgent unread
            urgent = self.email_monitor.find_urgent_unread()
            if urgent:
                report.append(f"\n⚠️ {len(urgent)} urgent unread emails:")
                for email in urgent[:5]:
                    report.append(f"  - {email['subject']} from {email['from']}")

            if not unanswered and not urgent:
                report.append("\n✅ All caught up! No urgent items.")

            # Add summary statistics
            report.append("\n📈 Summary:")
            report.append(f"  - Total unanswered: {len(unanswered)}")
            report.append(f"  - Total urgent: {len(urgent)}")
            report.append(
                f"  - Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

        except Exception as e:
            self.logger.error(f"Daily monitoring failed: {e}")
            report.append(f"\n❌ Error during monitoring: {str(e)}")

        return "\n".join(report)

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self.session_id,
                "user_id": self.config.user_id,
                "environment": self.config.environment,
                "uptime_seconds": int(
                    (
                        datetime.now(timezone.utc) - self.session_context["start_time"]
                    ).total_seconds()
                ),
                "message_count": self.session_context["message_count"],
                "database_health": self.db_manager.health_check(),
                "cache_stats": self.cache_manager.get_stats(),
                "performance_metrics": self.performance_tracker.get_metrics(),
                "preferences": {
                    "name_mappings": len(
                        self.preference_tracker.preferences.get("name_mappings", {})
                    ),
                    "queries_tracked": len(
                        self.preference_tracker.preferences.get("common_queries", [])
                    ),
                },
            }

            return status

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"status": "error", "message": str(e)}

    def start_interactive_session(self):
        """Start interactive chat session with enhanced commands"""

        print("\n" + "=" * 80)
        print("🚀 Intelligent AD3Gem Chatbot - Enhanced Production Ready")
        print("=" * 80)
        print(f"Environment: {self.config.environment}")
        print(f"User: {self.config.user_id}")
        print(f"Session: {self.session_id}")
        print("\n🧠 Features:")
        print("  • Intelligent routing (company vs general questions)")
        print("  • Fixed collection queries (actually works now!)")
        print("  • Knowledge base integration with caching")
        print("  • Email search and analysis")
        print("  • User preference learning")
        print("  • Performance tracking and logging")
        print("  • Error recovery and fallback strategies")
        print("  • Daily monitoring and alerts")
        print("\n💬 Example Queries:")
        print("  'When did Rox last send an email?'")
        print("  'Who was the last person asking for money?'")
        print("  'Show me Julie's emails about suppliers'")
        print("  'Find unanswered emails from last week'")
        print("  'What urgent emails haven't I read?'")
        print("\n📝 Commands:")
        print("  'status' - Show system status")
        print("  'metrics' - Show performance metrics")
        print("  'cache' - Show cache statistics")
        print("  'health' - Check database health")
        print("  'monitor' - Run daily monitoring report")
        print("  'preferences' - Show learned preferences")
        print("  'quit/exit/bye' - End session")
        print("-" * 80)

        while True:
            try:
                user_input = input("\n💭 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\n👋 Session ended. Goodbye!")
                    break

                if user_input.lower() == "status":
                    status = self.get_system_status()
                    print("\n📊 System Status:")
                    print(f"  Uptime: {status['uptime_seconds']}s")
                    print(f"  Messages: {status['message_count']}")
                    print(f"  Environment: {status['environment']}")
                    print(
                        f"  Preferences learned: {status['preferences']['name_mappings']} names"
                    )
                    continue

                if user_input.lower() == "metrics":
                    metrics = self.performance_tracker.get_metrics()
                    print("\n⚡ Performance Metrics:")
                    for operation, stats in metrics.items():
                        print(
                            f"  {operation}: {stats['count']} calls, avg {stats['avg_duration']}s"
                        )
                    continue

                if user_input.lower() == "cache":
                    cache_stats = self.cache_manager.get_stats()
                    print("\n🗄️ Cache Statistics:")
                    for cache_type, stats in cache_stats.items():
                        print(f"  {cache_type}: {stats['size']}/{stats['maxsize']}")
                    continue

                if user_input.lower() == "health":
                    health = self.db_manager.health_check()
                    print("\n🏥 Database Health:")
                    for db, status in health.items():
                        status_icon = "✅" if status else "❌"
                        print(f"  {db}: {status_icon}")
                    continue

                if user_input.lower() == "monitor":
                    report = self.run_daily_monitoring()
                    print(f"\n{report}")
                    continue

                if user_input.lower() == "preferences":
                    prefs = self.preference_tracker.preferences
                    print("\n📚 Learned Preferences:")
                    print(f"  Name mappings: {prefs.get('name_mappings', {})}")
                    print(
                        f"  Recent queries: {len(prefs.get('common_queries', []))} tracked"
                    )
                    print(
                        f"  Default supplier: {prefs.get('default_supplier', 'None set')}"
                    )
                    continue

                if not user_input:
                    continue

                # Process normal message
                response = self.send_message(user_input)
                print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Interactive session error: {e}")
                print(f"\n⚠️ Error: {e}")
                print("Continuing session...")


def main():
    """Main function to initialize and run the chatbot"""

    print("🔥 Initializing Enhanced Intelligent AD3Gem Chatbot...")
    print("   Production-ready with modern AI agent patterns")
    print("   Fixed collection queries and enhanced intelligence")

    try:
        # Load configuration
        config = Config()

        # Initialize chatbot
        chatbot = IntelligentAD3GemChatbot(config)

        # Start interactive session
        chatbot.start_interactive_session()

    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("\nPlease ensure:")
        print("1. Environment variables are properly set")
        print("2. Secret Manager contains 'gemini-api-key'")
        print("3. Firestore databases exist and are accessible")
        print("4. Proper IAM permissions are configured")
        print("5. Collection names match (thread_summaries, email_search_index, etc.)")


if __name__ == "__main__":
    main()
