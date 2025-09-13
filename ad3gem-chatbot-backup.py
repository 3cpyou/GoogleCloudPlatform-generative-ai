"""
Intelligent AD3Gem Chatbot - Complete Stage 1+2 Implementation
Production-ready monolithic script with modern AI agent patterns

Features:
- Dual-purpose routing (company vs general questions)
- Structured logging with performance tracking
- Database connection management
- Simple caching layer
- Error recovery and fallback strategies
- Conversation storage in ad3gem-conversation
- Input validation and security
- Secret Manager integration
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
        }


class PreferenceTracker:
    """Track and learn user preferences"""

    def __init__(
        self, db_manager: "DatabaseManager", user_id: str, logger: logging.Logger
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

    def __init__(self, db_manager: "DatabaseManager", logger: logging.Logger):
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


class IntelligentAD3GemChatbot:
    """
    Production-ready intelligent chatbot with dual-purpose routing
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

        # Initialize enhanced email intelligence components
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

    # @performance_tracker.track_operation("save_conversation")
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

    # @performance_tracker.track_operation("route_query")
    def route_query(self, user_message: str) -> Dict:
        """Determine if query is about company or general knowledge"""

        # Check cache first
        cache_key = self.cache_manager.get_cache_key("route", user_message)
        cached_result = self.cache_manager.get_llm(cache_key)
        if cached_result:
            self.logger.info("Using cached routing decision")
            return cached_result

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
        ]

        company_score = sum(
            1 for indicator in company_indicators if indicator in query_lower
        )

        prompt = f"""
        Analyze this query to determine if it's about internal company matters or general knowledge.

        Query: "{user_message}"

        Company indicators found: {company_score}

        Respond with JSON:
        {{
            "mode": "company" or "general",
            "confidence": 0-100,
            "reasoning": "brief explanation",
            "requires_email_search": true/false,
            "requires_web_search": true/false
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
            }

    # @performance_tracker.track_operation("get_knowledge_context")
    def get_knowledge_context(self, user_message: str) -> Dict:
        """Get relevant context from knowledge base with caching"""

        cache_key = self.cache_manager.get_cache_key("knowledge", user_message)
        cached_context = self.cache_manager.get_knowledge(cache_key)
        if cached_context:
            self.logger.info("Using cached knowledge context")
            return cached_context

        context = {
            "entities": {},
            "relationships": [],
            "facts": [],
            "patterns": {},
            "temporal": {},
        }

        try:
            knowledge_client = self.db_manager.get_client("knowledge")
            query_lower = user_message.lower()

            # Search entities with concurrency
            with ThreadPoolExecutor(max_workers=3) as executor:
                entity_future = executor.submit(
                    self._search_entities, knowledge_client, query_lower
                )
                fact_future = executor.submit(
                    self._search_facts, knowledge_client, query_lower
                )
                pattern_future = executor.submit(self._get_patterns, knowledge_client)

                # Collect results
                context["entities"] = entity_future.result() or {}
                context["facts"] = fact_future.result() or []
                patterns = pattern_future.result() or {}
                context["patterns"] = patterns.get("communication_patterns", {})
                context["temporal"] = patterns.get("temporal_patterns", {})

            # Get relationships for found entities
            if context["entities"]:
                context["relationships"] = self._get_relationships(
                    knowledge_client, list(context["entities"].keys())[:5]
                )

            # Cache the result
            self.cache_manager.set_knowledge(cache_key, context)

            self.logger.info(
                "Knowledge context retrieved",
                extra={
                    "entities_count": len(context["entities"]),
                    "facts_count": len(context["facts"]),
                    "relationships_count": len(context["relationships"]),
                },
            )

        except Exception as e:
            self.logger.error(f"Knowledge context retrieval failed: {e}")
            # Return empty context on failure

        return context

    def _search_entities(self, client, query_lower: str) -> Dict:
        """Search for relevant entities"""
        entities = {}
        try:
            entity_docs = (
                client.collection("entities")
                .limit(self.config.max_entities * 2)
                .stream()
            )

            for entity_doc in entity_docs:
                entity_data = entity_doc.to_dict()

                # Check relevance
                mentioned = False
                for name in entity_data.get("names", []):
                    if name.lower() in query_lower:
                        mentioned = True
                        break

                if not mentioned:
                    for email in entity_data.get("email_addresses", []):
                        if email.lower().split("@")[0] in query_lower:
                            mentioned = True
                            break

                if mentioned:
                    entities[entity_doc.id] = entity_data

                if len(entities) >= self.config.max_entities:
                    break

        except Exception as e:
            self.logger.error(f"Entity search failed: {e}")

        return entities

    def _search_facts(self, client, query_lower: str) -> List:
        """Search for relevant facts"""
        facts = []
        try:
            fact_docs = (
                client.collection("facts").limit(self.config.max_facts * 2).stream()
            )

            for fact_doc in fact_docs:
                fact_data = fact_doc.to_dict()
                fact_text = fact_data.get("fact", "").lower()

                # Simple relevance check
                if any(word in query_lower for word in fact_text.split()[:10]):
                    facts.append(fact_data)

                if len(facts) >= self.config.max_facts:
                    break

        except Exception as e:
            self.logger.error(f"Fact search failed: {e}")

        return facts

    def _get_patterns(self, client) -> Dict:
        """Get communication and temporal patterns"""
        patterns = {}
        try:
            # Get communication patterns
            comm_doc = (
                client.collection("patterns").document("communication_patterns").get()
            )
            if comm_doc.exists:
                patterns["communication_patterns"] = comm_doc.to_dict()

            # Get temporal patterns
            temp_doc = client.collection("temporal").document("email_patterns").get()
            if temp_doc.exists:
                patterns["temporal_patterns"] = temp_doc.to_dict()

        except Exception as e:
            self.logger.error(f"Pattern retrieval failed: {e}")

        return patterns

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

    # @performance_tracker.track_operation("search_emails")
    def search_emails(
        self, user_message: str, knowledge_context: Dict
    ) -> Tuple[List[Dict], List[Dict]]:
        """Search emails with caching and error recovery"""

        cache_key = self.cache_manager.get_cache_key(
            "emails", user_message, str(knowledge_context)
        )
        cached_result = self.cache_manager.get_email(cache_key)
        if cached_result:
            self.logger.info("Using cached email search results")
            return cached_result

        threads, messages = [], []

        try:
            email_client = self.db_manager.get_client("email")

            # Get mentioned email addresses from knowledge context
            mentioned_emails = []
            for entity_data in knowledge_context.get("entities", {}).values():
                mentioned_emails.extend(entity_data.get("email_addresses", []))

            # Also search people directory for name-to-email resolution
            mentioned_emails.extend(
                self._resolve_names_to_emails(email_client, user_message)
            )

            # Search threads
            if mentioned_emails:
                threads = self._search_threads_by_participants(
                    email_client, mentioned_emails[:5]
                )
            else:
                threads = self._search_recent_threads(email_client)

            # Get messages from top threads
            if threads:
                messages = self._get_messages_from_threads(
                    email_client, threads[: self.config.max_threads]
                )

            # Cache results
            result = (threads, messages)
            self.cache_manager.set_email(cache_key, result)

            self.logger.info(
                "Email search completed",
                extra={
                    "threads_found": len(threads),
                    "messages_found": len(messages),
                    "mentioned_emails": len(mentioned_emails),
                },
            )

        except Exception as e:
            self.logger.error(f"Email search failed: {e}")
            # Return empty results on failure - don't crash the chat

        return threads, messages

    def _resolve_names_to_emails(self, client, user_message: str) -> List[str]:
        """Resolve first names to email addresses using people_directory"""
        resolved_emails = []
        try:
            # Extract potential first names from user message
            query_lower = user_message.lower()
            words = query_lower.split()

            # Look for common first names in people directory
            people_docs = client.collection("people_directory").limit(50).stream()

            for person_doc in people_docs:
                person_data = person_doc.to_dict()
                search_key = person_data.get("searchKey", "").lower()
                first_name = person_data.get("firstName", "").lower()

                # Check if person's name is mentioned in query
                if search_key in words or first_name in words:
                    email = person_data.get("email", "")
                    if email and email not in resolved_emails:
                        resolved_emails.append(email)

                if len(resolved_emails) >= 5:  # Limit to avoid too many matches
                    break

        except Exception as e:
            self.logger.error(f"Name-to-email resolution failed: {e}")

        return resolved_emails

    def _search_threads_by_participants(self, client, emails: List[str]) -> List[Dict]:
        """Search threads by participant emails"""
        threads = []
        try:
            # Search through thread_summaries collection
            # Since participants is a dict (email -> participant_data),
            # we need to find threads where emails are keys in participants dict

            for email in emails[:3]:  # Limit to avoid too many queries
                email_lower = email.lower()

                # Get all thread summaries and filter by participants
                thread_docs = (
                    client.collection("thread_summaries")
                    .order_by("lastActivity", direction=firestore.Query.DESCENDING)
                    .limit(self.config.max_threads * 5)  # Get more to filter
                    .stream()
                )

                for thread_doc in thread_docs:
                    thread_data = thread_doc.to_dict()

                    # Check if email is in participants dict
                    participants = thread_data.get("participants", {})
                    if email_lower in participants:
                        thread_data["thread_id"] = thread_doc.id
                        # Avoid duplicates
                        existing_ids = [t.get("thread_id") for t in threads]
                        if thread_doc.id not in existing_ids:
                            threads.append(thread_data)

                        if len(threads) >= self.config.max_threads:
                            break

                if len(threads) >= self.config.max_threads:
                    break

        except Exception as e:
            self.logger.error(f"Thread search by participants failed: {e}")

        return threads[: self.config.max_threads]

    def _search_recent_threads(self, client) -> List[Dict]:
        """Search recent threads"""
        threads = []
        try:
            thread_docs = (
                client.collection("thread_summaries")
                .order_by("lastActivity", direction=firestore.Query.DESCENDING)
                .limit(self.config.max_threads * 2)
                .stream()
            )

            for thread_doc in thread_docs:
                thread_data = thread_doc.to_dict()
                thread_data["thread_id"] = thread_doc.id
                threads.append(thread_data)

        except Exception as e:
            self.logger.error(f"Recent thread search failed: {e}")

        return threads

    def _get_messages_from_threads(self, client, threads: List[Dict]) -> List[Dict]:
        """Get messages from thread list using email_search_index collection"""
        messages = []
        try:
            thread_ids = [
                thread.get("thread_id") for thread in threads if thread.get("thread_id")
            ]

            if not thread_ids:
                return messages

            # Search email_search_index for messages in these threads
            # Use search index for better performance and relevant fields
            message_docs = (
                client.collection("email_search_index")
                .where("threadId", "in", thread_ids[:10])  # Firestore limit
                .order_by("sentAt", direction=firestore.Query.DESCENDING)
                .limit(self.config.max_messages)
                .stream()
            )

            # Create thread lookup for subject mapping
            thread_lookup = {t.get("thread_id"): t for t in threads}

            for msg_doc in message_docs:
                msg_data = msg_doc.to_dict()
                msg_data["message_id"] = msg_doc.id
                thread_id = msg_data.get("threadId")

                if thread_id in thread_lookup:
                    msg_data["thread_id"] = thread_id
                    thread_subject = thread_lookup[thread_id].get("subject", "")
                    msg_data["thread_subject"] = thread_subject

                    # Map email_search_index fields to expected message fields
                    msg_data["from"] = {
                        "email": msg_data.get("fromEmail", ""),
                        "name": msg_data.get("fromPerson", ""),
                    }

                    # Use bodyPreview if available, otherwise snippet
                    body_preview = msg_data.get("bodyPreview")
                    snippet = msg_data.get("snippet", "")
                    msg_data["bodyPreview"] = body_preview or snippet

                    messages.append(msg_data)

        except Exception as e:
            self.logger.error(f"Message retrieval failed: {e}")

            # Fallback: try getting from messages_full if search index fails
            try:
                self.logger.info("Falling back to messages_full collection")
                fallback_thread_ids = [
                    thread.get("thread_id")
                    for thread in threads
                    if thread.get("thread_id")
                ][:10]

                message_docs = (
                    client.collection("messages_full")
                    .where("threadId", "in", fallback_thread_ids)
                    .order_by("createdAt", direction=firestore.Query.DESCENDING)
                    .limit(self.config.max_messages)
                    .stream()
                )

                thread_lookup = {t.get("thread_id"): t for t in threads}

                for msg_doc in message_docs:
                    msg_data = msg_doc.to_dict()
                    msg_data["message_id"] = msg_doc.id
                    thread_id = msg_data.get("threadId")

                    if thread_id in thread_lookup:
                        msg_data["thread_id"] = thread_id
                        thread_subject = thread_lookup[thread_id].get("subject", "")
                        msg_data["thread_subject"] = thread_subject

                        # Extract from headers for messages_full
                        headers = msg_data.get("headers", {})
                        from_list = headers.get("from", [])
                        default_from = {"email": "", "name": ""}
                        msg_data["from"] = from_list[0] if from_list else default_from

                        # Use snippet or truncated body
                        snippet = msg_data.get("snippet")
                        body_text = msg_data.get("bodyText", "")[:200]
                        msg_data["bodyPreview"] = snippet or body_text

                        messages.append(msg_data)

            except Exception as fallback_error:
                error_msg = f"Fallback message retrieval also failed: {fallback_error}"
                self.logger.error(error_msg)

        return messages

    # @performance_tracker.track_operation("web_search")
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

    # @performance_tracker.track_operation("generate_response")
    def generate_response(
        self,
        user_message: str,
        routing: Dict,
        knowledge_context: Dict,
        email_data: Tuple[List[Dict], List[Dict]] = None,
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
        email_data: Tuple[List[Dict], List[Dict]],
    ) -> str:
        """Generate response for company-related queries"""

        threads, messages = email_data if email_data else ([], [])

        # Format context for prompt
        entity_info = self._format_entities(knowledge_context.get("entities", {}))
        thread_info = self._format_threads(threads[:3])
        message_info = self._format_messages(messages[:5])
        pattern_info = self._format_patterns(knowledge_context.get("patterns", {}))

        prompt = f"""
        You are AD3Gem, an intelligent email assistant with comprehensive knowledge of the company.

        USER QUERY: "{user_message}"

        ROUTING ANALYSIS:
        Mode: {routing.get("mode")}
        Confidence: {routing.get("confidence")}%
        Reasoning: {routing.get("reasoning")}

        KNOWLEDGE CONTEXT:

        Entities:
        {entity_info}

        Communication Patterns:
        {pattern_info}

        EMAIL SEARCH RESULTS:

        Threads Found ({len(threads)} total):
        {thread_info}

        Recent Messages ({len(messages)} total):
        {message_info}

        INSTRUCTIONS:
        1. Answer the user's question directly using the email data and knowledge context
        2. Reference specific emails, people, or patterns when relevant
        3. Be conversational but precise
        4. If data seems incomplete, mention limitations
        5. Suggest follow-up actions if helpful
        6. Keep response under 200 words unless listing multiple items

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

        entity_info = self._format_entities(knowledge_context.get("entities", {}))
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

    def _format_entities(self, entities: Dict) -> str:
        """Format entity information for prompts"""
        if not entities:
            return "No specific entities identified"

        formatted = []
        for entity_id, data in entities.items():
            names = ", ".join(data.get("names", ["Unknown"]))
            emails = ", ".join(data.get("email_addresses", [])[:2])
            company = ", ".join(data.get("companies", ["Unknown"]))

            formatted.append(
                f"- {names} ({emails}) | {company} | {data.get('email_count', 0)} emails"
            )

        return "\n".join(formatted)

    def _format_threads(self, threads: List[Dict]) -> str:
        """Format thread information for prompts"""
        if not threads:
            return "No threads found"

        formatted = []
        for thread in threads:
            # Handle participants dict structure
            participants_dict = thread.get("participants", {})
            if isinstance(participants_dict, dict):
                # Extract participant emails from dict keys
                participant_emails = list(participants_dict.keys())[:3]
                participants = ", ".join(participant_emails)
            else:
                fallback_participants = (
                    participants_dict[:3] if participants_dict else []
                )
                participants = ", ".join(fallback_participants)

            # Handle lastActivity field
            last_activity = thread.get("lastActivity")
            last_message_at = thread.get("lastMessageAt", "Unknown")
            last_msg = last_activity or last_message_at

            if hasattr(last_msg, "strftime"):
                last_msg = last_msg.strftime("%Y-%m-%d %H:%M")
            elif hasattr(last_msg, "isoformat"):
                last_msg = last_msg.isoformat()[:16].replace("T", " ")

            # Get preview from lastMessage if available
            last_message = thread.get("lastMessage", {})
            latest_snippet = thread.get("latestSnippet", "")
            preview = last_message.get("preview", latest_snippet)

            subject = thread.get("subject", "No subject")
            message_count = thread.get("messageCount", 0)

            formatted.append(
                f"- {subject}\n"
                f"  Participants: {participants}\n"
                f"  Messages: {message_count} | Last: {last_msg}\n"
                f"  Preview: {preview[:100]}..."
            )

        return "\n".join(formatted)

    def _format_messages(self, messages: List[Dict]) -> str:
        """Format message information for prompts"""
        if not messages:
            return "No messages found"

        formatted = []
        for msg in messages:
            from_data = msg.get("from", {})
            sender = (
                f"{from_data.get('name', '')} ({from_data.get('email', 'Unknown')})"
            )

            sent_at = msg.get("sentAt", "Unknown")
            if hasattr(sent_at, "strftime"):
                sent_at = sent_at.strftime("%Y-%m-%d %H:%M")

            formatted.append(
                f"- From: {sender} | {sent_at}\n"
                f"  Subject: {msg.get('thread_subject', 'No subject')}\n"
                f"  Preview: {msg.get('bodyPreview', '')[:150]}..."
            )

        return "\n".join(formatted)

    def _format_patterns(self, patterns: Dict) -> str:
        """Format communication patterns"""
        if not patterns:
            return "No patterns available"

        formatted = []

        # Top senders
        top_senders = patterns.get("top_senders", {})
        if top_senders:
            sender_list = [f"{s}: {c}" for s, c in list(top_senders.items())[:3]]
            formatted.append(f"Top senders: {', '.join(sender_list)}")

        return "; ".join(formatted) if formatted else "No patterns available"

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
        """Process user message with full intelligent routing"""

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

            # Route the query
            routing = self.route_query(user_message)

            # Get knowledge context
            knowledge_context = self.get_knowledge_context(user_message)

            # Process based on routing
            email_data = None
            web_data = None

            if routing.get("mode") == "company" and routing.get(
                "requires_email_search"
            ):
                email_data = self.search_emails(user_message, knowledge_context)

            if routing.get("mode") == "general" and routing.get("requires_web_search"):
                web_data = self.web_search(user_message)

            # Generate response
            response = self.generate_response(
                user_message, routing, knowledge_context, email_data, web_data
            )

            # Save response
            response_metadata = {
                "routing": routing,
                "knowledge_entities": len(knowledge_context.get("entities", {})),
                "email_results": len(email_data[0]) if email_data else 0,
                "web_results": len(web_data) if web_data else 0,
                "processing_time": round(time.time() - start_time, 3),
            }

            self.save_conversation_message("assistant", response, response_metadata)

            # Update session context
            self.session_context.update(
                {
                    "last_mode": routing.get("mode"),
                    "last_entities": list(knowledge_context.get("entities", {}).keys()),
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
            }

            return status

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"status": "error", "message": str(e)}

    def start_interactive_session(self):
        """Start interactive chat session with enhanced commands"""

        print("\n" + "=" * 80)
        print("🚀 Intelligent AD3Gem Chatbot - Production Ready")
        print("=" * 80)
        print(f"Environment: {self.config.environment}")
        print(f"User: {self.config.user_id}")
        print(f"Session: {self.session_id}")
        print("\n🧠 Features:")
        print("  • Intelligent routing (company vs general questions)")
        print("  • Knowledge base integration with caching")
        print("  • Email search and analysis")
        print("  • Performance tracking and logging")
        print("  • Error recovery and fallback strategies")
        print("\n💬 Commands:")
        print("  'status' - Show system status")
        print("  'metrics' - Show performance metrics")
        print("  'cache' - Show cache statistics")
        print("  'health' - Check database health")
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
                    print("\n🗄️  Cache Statistics:")
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
                print(f"\n⚠️  Error: {e}")
                print("Continuing session...")


def main():
    """Main function to initialize and run the chatbot"""

    print("🔥 Initializing Intelligent AD3Gem Chatbot...")
    print("   Production-ready with modern AI agent patterns")

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


if __name__ == "__main__":
    main()
