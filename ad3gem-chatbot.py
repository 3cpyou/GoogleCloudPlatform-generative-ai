"""
Enhanced AD3Gem Chatbot - Complete Implementation with Cast Net Search
Production-ready monolithic script with modern AI agent patterns and cast net email search

Features:
- Cast net email search with parallel field queries and relevance scoring
- Advanced AI query understanding with structured field selection
- User preference learning and intelligent name mapping
- Enhanced email search with precise filtering and monitoring
- Email monitoring (unanswered emails, urgent detection, response times)
- Dual-purpose routing (company vs general questions)
- Structured logging with performance tracking
- Database connection management with health checks
- Multi-layer caching system
- Comprehensive error recovery and fallback strategies
- Conversation storage in ad3gem-conversation
- Input validation and security
- Secret Manager integration
- Interactive commands for monitoring and preferences
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

# Complete field mapping from Gmail ingestion script for cast net searching
EMAIL_FIELD_MAPPING = {
    "person_email_fields": {
        "fromPerson": {
            "collection": "email_search_index",
            "type": "string",
            "weight": 10,
            "search_type": "exact_match",
        },
        "fromEmail": {
            "collection": "email_search_index",
            "type": "string",
            "weight": 10,
            "search_type": "contains",
        },
        "fromFull": {
            "collection": "email_search_index",
            "type": "string",
            "weight": 8,
            "search_type": "contains",
        },
        "toPeople": {
            "collection": "email_search_index",
            "type": "array",
            "weight": 8,
            "search_type": "array_contains",
        },
        "toEmails": {
            "collection": "email_search_index",
            "type": "array",
            "weight": 9,
            "search_type": "array_contains",
        },
        "allPeople": {
            "collection": "email_search_index",
            "type": "array",
            "weight": 7,
            "search_type": "array_contains",
        },
        "allEmails": {
            "collection": "email_search_index",
            "type": "array",
            "weight": 8,
            "search_type": "array_contains",
        },
        "searchKey": {
            "collection": "people_directory",
            "type": "string",
            "weight": 9,
            "search_type": "exact_match",
        },
        "displayName": {
            "collection": "people_directory",
            "type": "string",
            "weight": 7,
            "search_type": "contains",
        },
        "participants": {
            "collection": "thread_summaries",
            "type": "object",
            "weight": 6,
            "search_type": "key_exists",
        },
    },
    "subject_fields": {
        "subject": {
            "collection": "email_search_index",
            "type": "string",
            "weight": 9,
            "search_type": "contains",
        },
        "threadSubject": {
            "collection": "thread_summaries",
            "type": "string",
            "weight": 8,
            "search_type": "contains",
        },
    },
    "document_file_fields": {
        "hasAttachments": {
            "collection": "email_search_index",
            "type": "boolean",
            "weight": 5,
            "search_type": "equals",
        },
        "attachmentTypes": {
            "collection": "email_search_index",
            "type": "array",
            "weight": 10,
            "search_type": "array_contains",
        },
        "attachments.filename": {
            "collection": "messages_full",
            "type": "string",
            "weight": 8,
            "search_type": "contains",
        },
        "attachments.mimeType": {
            "collection": "messages_full",
            "type": "string",
            "weight": 10,
            "search_type": "contains",
        },
        "attachments.sizeBytes": {
            "collection": "messages_full",
            "type": "number",
            "weight": 6,
            "search_type": "range",
        },
    },
    "content_information_fields": {
        "bodySearchable": {
            "collection": "email_search_index",
            "type": "string",
            "weight": 6,
            "search_type": "contains",
        },
        "searchTerms": {
            "collection": "email_search_index",
            "type": "array",
            "weight": 10,
            "search_type": "array_contains_any",
        },
        "bodyText": {
            "collection": "messages_full",
            "type": "string",
            "weight": 7,
            "search_type": "full_text_search",
        },
        "snippet": {
            "collection": "email_search_index",
            "type": "string",
            "weight": 4,
            "search_type": "contains",
        },
        "bodyPreview": {
            "collection": "email_search_index",
            "type": "string",
            "weight": 5,
            "search_type": "contains",
        },
        "keyTopics": {
            "collection": "thread_summaries",
            "type": "array",
            "weight": 8,
            "search_type": "array_contains_any",
        },
    },
    "time_date_fields": {
        "sentAt": {
            "collection": "email_search_index",
            "type": "datetime",
            "weight": 10,
            "search_type": "date_range",
        },
        "startDate": {
            "collection": "thread_summaries",
            "type": "datetime",
            "weight": 8,
            "search_type": "date_range",
        },
        "lastActivity": {
            "collection": "thread_summaries",
            "type": "datetime",
            "weight": 9,
            "search_type": "date_range",
        },
        "lastSeen": {
            "collection": "people_directory",
            "type": "datetime",
            "weight": 6,
            "search_type": "date_range",
        },
        "createdAt": {
            "collection": "email_search_index",
            "type": "datetime",
            "weight": 3,
            "search_type": "date_range",
        },
    },
    "status_metadata_fields": {
        "isInternal": {
            "collection": "email_search_index",
            "type": "boolean",
            "weight": 6,
            "search_type": "equals",
        },
        "hasExternal": {
            "collection": "email_search_index",
            "type": "boolean",
            "weight": 7,
            "search_type": "equals",
        },
        "isUnread": {
            "collection": "email_search_index",
            "type": "boolean",
            "weight": 8,
            "search_type": "equals",
        },
        "isImportant": {
            "collection": "email_search_index",
            "type": "boolean",
            "weight": 9,
            "search_type": "equals",
        },
        "isStarred": {
            "collection": "email_search_index",
            "type": "boolean",
            "weight": 7,
            "search_type": "equals",
        },
        "labelNames": {
            "collection": "email_search_index",
            "type": "array",
            "weight": 5,
            "search_type": "array_contains",
        },
        "messageCount": {
            "collection": "thread_summaries",
            "type": "number",
            "weight": 4,
            "search_type": "range",
        },
    },
}

# Search expansion patterns for cast net approach
SEARCH_EXPANSION_PATTERNS = {
    "person_search": {
        "primary_fields": ["fromPerson", "searchKey", "fromEmail"],
        "secondary_fields": ["allPeople", "toPeople", "displayName", "participants"],
        "tertiary_fields": ["bodySearchable", "searchTerms"],
    },
    "document_search": {
        "primary_fields": ["attachmentTypes", "attachments.mimeType"],
        "secondary_fields": ["hasAttachments", "attachments.filename"],
        "tertiary_fields": ["bodySearchable", "searchTerms"],
    },
    "topic_search": {
        "primary_fields": ["searchTerms", "keyTopics"],
        "secondary_fields": ["subject", "bodySearchable"],
        "tertiary_fields": ["bodyText", "snippet", "bodyPreview"],
    },
    "time_search": {
        "primary_fields": ["sentAt"],
        "secondary_fields": ["lastActivity", "startDate"],
        "tertiary_fields": ["createdAt", "lastSeen"],
    },
    "status_search": {
        "primary_fields": ["isImportant", "isUnread", "isStarred"],
        "secondary_fields": ["labelNames", "isInternal", "hasExternal"],
        "tertiary_fields": ["messageCount"],
    },
}


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


class PreferenceTracker:
    """Track user preferences and learn name mappings"""

    def __init__(self):
        self.preferences = {
            "name_mappings": {},
            "common_queries": [],
        }

    def get_email_for_name(self, name: str) -> Optional[str]:
        """Get preferred email mapping for a name"""
        return self.preferences["name_mappings"].get(name.lower())

    def learn_name_mapping(self, name: str, email: str):
        """Learn name to email mapping"""
        self.preferences["name_mappings"][name.lower()] = email.lower()

    def track_query(self, query: str):
        """Track a query for learning patterns"""
        self.preferences["common_queries"].append(
            {"query": query, "timestamp": datetime.now(timezone.utc)}
        )
        # Keep only last 50 queries
        if len(self.preferences["common_queries"]) > 50:
            self.preferences["common_queries"] = self.preferences["common_queries"][
                -50:
            ]


class EmailMonitor:
    """Monitor email status and find urgent items"""

    def __init__(self, db_manager, logger):
        self.db_manager = db_manager
        self.logger = logger

    def check_unanswered_emails(self, hours: int = 48) -> List[Dict]:
        """Find emails that haven't been answered within specified hours"""
        unanswered = []
        try:
            email_client = self.db_manager.get_client("email")
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Find external emails older than cutoff that don't have replies
            messages = (
                email_client.collection("email_search_index")
                .where("hasExternal", "==", True)
                .where("sentAt", "<=", cutoff_time)
                .order_by("sentAt", direction=firestore.Query.DESCENDING)
                .limit(20)
                .stream()
            )

            for msg in messages:
                msg_data = msg.to_dict()
                # Simple check - if it's from external sender and old, consider unanswered
                if not msg_data.get("isInternal", True):
                    unanswered.append(
                        {
                            "subject": msg_data.get("subject", "No subject"),
                            "from": msg_data.get("fromEmail", "Unknown"),
                            "sentAt": msg_data.get("sentAt"),
                            "threadId": msg_data.get("threadId"),
                        }
                    )

        except Exception as e:
            self.logger.error(f"Failed to check unanswered emails: {e}")

        return unanswered[:10]  # Return top 10

    def find_urgent_unread(self) -> List[Dict]:
        """Find urgent unread emails"""
        urgent = []
        try:
            email_client = self.db_manager.get_client("email")

            messages = (
                email_client.collection("email_search_index")
                .where("isImportant", "==", True)
                .where("isUnread", "==", True)
                .order_by("sentAt", direction=firestore.Query.DESCENDING)
                .limit(10)
                .stream()
            )

            for msg in messages:
                msg_data = msg.to_dict()
                urgent.append(
                    {
                        "subject": msg_data.get("subject", "No subject"),
                        "from": msg_data.get("fromEmail", "Unknown"),
                        "sentAt": msg_data.get("sentAt"),
                        "snippet": msg_data.get("snippet", ""),
                    }
                )

        except Exception as e:
            self.logger.error(f"Failed to find urgent emails: {e}")

        return urgent


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
    Production-ready intelligent chatbot with cast net email search
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
        self.preference_tracker = PreferenceTracker()
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

    def understand_query(self, user_message: str) -> Dict:
        """Enhanced query understanding using AI with cast net field selection"""

        # Check cache first
        cache_key = self.cache_manager.get_cache_key("understand", user_message)
        cached_result = self.cache_manager.get_llm(cache_key)
        if cached_result:
            self.logger.info("Using cached query understanding")
            return cached_result

        # Check preferences for common interpretations
        preferred_email = self.preference_tracker.get_email_for_name(
            user_message.lower()
        )

        # Create field reference for LLM
        field_reference = self._create_field_reference_prompt()

        prompt = f"""
        You are an intelligent email search assistant. Analyze this query and select specific Firestore fields to search using a "cast net" approach.

        Query: "{user_message}"

        Company context:
        - Domain: @lineagecoffee.com
        - Common names: Julie, Craig, Tegan, Sam, Dom, Mark
        - Common suppliers: Jersey Cow, Early Moon Trading, LP Agencies, Durban Packaging
        - Preferred mapping: {preferred_email if preferred_email else "None"}

        AVAILABLE FIRESTORE FIELDS:
        {field_reference}

        CAST NET SEARCH STRATEGY:
        Instead of searching single fields precisely, cast wide nets across related fields, then rank by relevance.

        For "Find emails from Julie":
        - PRIMARY: fromPerson="julie", fromEmail contains "julie", searchKey="julie"
        - SECONDARY: allPeople contains "julie", participants has "julie"
        - TERTIARY: bodySearchable contains "julie", searchTerms contains "julie"

        Extract in JSON format:
        {{
            "intent": "search_emails/monitor/analyze",
            "mode": "company/general",
            "search_strategy": {{
                "primary_fields": [
                    {{"field": "fromPerson", "value": "julie", "search_type": "exact_match", "weight": 10}},
                    {{"field": "fromEmail", "value": "julie", "search_type": "contains", "weight": 10}}
                ],
                "secondary_fields": [
                    {{"field": "allPeople", "value": "julie", "search_type": "array_contains", "weight": 8}}
                ],
                "tertiary_fields": [
                    {{"field": "bodySearchable", "value": "julie", "search_type": "contains", "weight": 3}}
                ]
            }},
            "people_mentioned": ["julie"],
            "time_range": {{
                "start": "ISO date or null",
                "end": "ISO date or null",
                "description": "last week/yesterday/etc"
            }},
            "keywords": ["search", "terms"],
            "document_filters": {{
                "attachment_types": ["pdf", "image", "document"],
                "has_attachments": true/false/null
            }},
            "status_filters": {{
                "is_unread": true/false/null,
                "is_important": true/false/null,
                "is_internal": true/false/null
            }},
            "monitoring_type": "unanswered/urgent/response_time or null",
            "confidence": 0-100,
            "reasoning": "Why these fields were selected",
            "requires_email_search": true/false,
            "requires_web_search": true/false
        }}

        EXAMPLES:

        Query: "Find PDFs from Craig last week"
        - PRIMARY: attachmentTypes=["pdf"], fromPerson="craig", sentAt=last_week_range
        - SECONDARY: attachments.mimeType contains "pdf", fromEmail contains "craig"
        - TERTIARY: bodySearchable contains "pdf craig"

        Query: "Show important unread emails"
        - PRIMARY: isImportant=true, isUnread=true
        - SECONDARY: labelNames contains "important"
        - TERTIARY: searchTerms contains urgent keywords

        Query: "Conversations about payments with suppliers"
        - PRIMARY: searchTerms=["payment", "invoice"], hasExternal=true
        - SECONDARY: keyTopics=["payment"], subject contains "payment"
        - TERTIARY: bodySearchable contains payment terms

        Return ONLY valid JSON with specific field selections for cast net searching.
        """

        try:
            response = self.model.generate_content(prompt)
            json_str = response.text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:-3]

            understanding = json.loads(json_str)

            # Track this query for learning
            self.preference_tracker.track_query(user_message)

            # Validate and enhance search strategy
            understanding = self._validate_search_strategy(understanding, user_message)

            # Cache the result
            self.cache_manager.set_llm(cache_key, understanding)

            self.logger.info(
                "Query understood with cast net strategy",
                extra={
                    "intent": understanding.get("intent"),
                    "primary_fields": len(
                        understanding.get("search_strategy", {}).get(
                            "primary_fields", []
                        )
                    ),
                    "secondary_fields": len(
                        understanding.get("search_strategy", {}).get(
                            "secondary_fields", []
                        )
                    ),
                    "confidence": understanding.get("confidence"),
                },
            )

            return understanding

        except Exception as e:
            self.logger.error(f"Query understanding failed: {e}")
            # Enhanced fallback with basic cast net strategy
            return self._create_fallback_understanding(user_message)

    def _create_field_reference_prompt(self) -> str:
        """Create a reference of available fields for the LLM prompt"""
        reference = []

        for category, fields in EMAIL_FIELD_MAPPING.items():
            reference.append(f"\n{category.upper().replace('_', ' ')}:")
            for field_name, field_info in fields.items():
                reference.append(
                    f"  - {field_name} ({field_info['collection']}): {field_info['type']}, weight={field_info['weight']}"
                )

        return "\n".join(reference)

    def _validate_search_strategy(self, understanding: Dict, user_message: str) -> Dict:
        """Validate and enhance the search strategy from LLM"""

        # Ensure search_strategy exists
        if "search_strategy" not in understanding:
            understanding["search_strategy"] = {
                "primary_fields": [],
                "secondary_fields": [],
                "tertiary_fields": [],
            }

        # Add fallback fields if none selected
        strategy = understanding["search_strategy"]
        if not any(
            [
                strategy.get("primary_fields"),
                strategy.get("secondary_fields"),
                strategy.get("tertiary_fields"),
            ]
        ):
            # Basic fallback strategy
            query_lower = user_message.lower()

            # Add basic search terms
            strategy["primary_fields"] = [
                {
                    "field": "searchTerms",
                    "value": query_lower.split(),
                    "search_type": "array_contains_any",
                    "weight": 8,
                },
                {
                    "field": "bodySearchable",
                    "value": user_message,
                    "search_type": "contains",
                    "weight": 6,
                },
            ]

        # Validate field names against our mapping
        all_fields = {}
        for category_fields in EMAIL_FIELD_MAPPING.values():
            all_fields.update(category_fields)

        for level in ["primary_fields", "secondary_fields", "tertiary_fields"]:
            if level in strategy:
                validated_fields = []
                for field_spec in strategy[level]:
                    field_name = field_spec.get("field")
                    if field_name in all_fields:
                        # Add collection info from our mapping
                        field_spec["collection"] = all_fields[field_name]["collection"]
                        # Ensure weight exists
                        if "weight" not in field_spec:
                            field_spec["weight"] = all_fields[field_name]["weight"]
                        validated_fields.append(field_spec)
                    else:
                        self.logger.warning(f"Unknown field in strategy: {field_name}")

                strategy[level] = validated_fields

        return understanding

    def _create_fallback_understanding(self, user_message: str) -> Dict:
        """Create fallback understanding when LLM fails"""
        query_lower = user_message.lower()

        # Determine mode based on content
        company_indicators = [
            "email",
            "sent",
            "received",
            "julie",
            "craig",
            "tegan",
            "invoice",
            "project",
            "meeting",
        ]
        is_company = any(indicator in query_lower for indicator in company_indicators)

        # Basic search strategy
        search_strategy = {
            "primary_fields": [
                {
                    "field": "searchTerms",
                    "value": query_lower.split(),
                    "search_type": "array_contains_any",
                    "weight": 8,
                    "collection": "email_search_index",
                },
                {
                    "field": "bodySearchable",
                    "value": user_message,
                    "search_type": "contains",
                    "weight": 6,
                    "collection": "email_search_index",
                },
            ],
            "secondary_fields": [
                {
                    "field": "subject",
                    "value": user_message,
                    "search_type": "contains",
                    "weight": 7,
                    "collection": "email_search_index",
                }
            ],
            "tertiary_fields": [],
        }

        return {
            "intent": "search_emails" if is_company else "general_query",
            "mode": "company" if is_company else "general",
            "search_strategy": search_strategy,
            "people_mentioned": [],
            "keywords": query_lower.split(),
            "confidence": 50,
            "reasoning": "Fallback understanding due to LLM error",
            "requires_email_search": is_company,
            "requires_web_search": not is_company,
        }

    def route_query(self, user_message: str) -> Dict:
        """Legacy routing method - now uses enhanced understand_query"""
        understanding = self.understand_query(user_message)

        # Convert to legacy format for backward compatibility
        return {
            "mode": understanding.get("mode", "company"),
            "confidence": understanding.get("confidence", 70),
            "reasoning": understanding.get("reasoning", "Enhanced routing"),
            "requires_email_search": understanding.get("requires_email_search", True),
            "requires_web_search": understanding.get("requires_web_search", False),
        }

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

    def cast_net_search(self, search_strategy: Dict) -> Dict:
        """Execute cast net search across multiple fields and collections"""

        results = {
            "messages": [],
            "threads": [],
            "people": [],
            "scores": {},  # messageId -> relevance_score
            "stats": {
                "primary_results": 0,
                "secondary_results": 0,
                "tertiary_results": 0,
                "total_queries": 0,
            },
        }

        try:
            email_client = self.db_manager.get_client("email")

            # Execute searches by priority level
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = []

                # Primary field searches (highest weight)
                primary_fields = search_strategy.get("primary_fields", [])
                for field_spec in primary_fields:
                    future = executor.submit(
                        self._execute_field_search, email_client, field_spec, "primary"
                    )
                    futures.append(future)

                # Secondary field searches (medium weight)
                secondary_fields = search_strategy.get("secondary_fields", [])
                for field_spec in secondary_fields:
                    future = executor.submit(
                        self._execute_field_search,
                        email_client,
                        field_spec,
                        "secondary",
                    )
                    futures.append(future)

                # Tertiary field searches (lower weight)
                tertiary_fields = search_strategy.get("tertiary_fields", [])
                for field_spec in tertiary_fields:
                    future = executor.submit(
                        self._execute_field_search, email_client, field_spec, "tertiary"
                    )
                    futures.append(future)

                # Collect all results
                all_results = []
                for future in futures:
                    try:
                        field_results = future.result(timeout=10)
                        all_results.extend(field_results)
                        results["stats"]["total_queries"] += 1
                    except Exception as e:
                        self.logger.warning(f"Field search failed: {e}")

            # Process and deduplicate results
            seen_messages = set()
            seen_threads = set()
            seen_people = set()

            for result_item in all_results:
                collection = result_item["collection"]
                doc_data = result_item["data"]
                doc_id = result_item["doc_id"]
                field_weight = result_item["field_weight"]
                priority_level = result_item["priority_level"]

                # Calculate relevance score
                base_score = field_weight
                if priority_level == "primary":
                    score_multiplier = 1.0
                    results["stats"]["primary_results"] += 1
                elif priority_level == "secondary":
                    score_multiplier = 0.8
                    results["stats"]["secondary_results"] += 1
                else:  # tertiary
                    score_multiplier = 0.6
                    results["stats"]["tertiary_results"] += 1

                relevance_score = base_score * score_multiplier

                # Add to appropriate collection
                if collection == "email_search_index" and doc_id not in seen_messages:
                    doc_data["message_id"] = doc_id
                    doc_data["relevance_score"] = relevance_score
                    results["messages"].append(doc_data)
                    results["scores"][doc_id] = relevance_score
                    seen_messages.add(doc_id)

                elif collection == "thread_summaries" and doc_id not in seen_threads:
                    doc_data["thread_id"] = doc_id
                    doc_data["relevance_score"] = relevance_score
                    results["threads"].append(doc_data)
                    seen_threads.add(doc_id)

                elif collection == "people_directory" and doc_id not in seen_people:
                    doc_data["person_id"] = doc_id
                    doc_data["relevance_score"] = relevance_score
                    results["people"].append(doc_data)
                    seen_people.add(doc_id)

            # Sort results by relevance score
            results["messages"].sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )
            results["threads"].sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )
            results["people"].sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )

            # Limit results to manageable amounts
            results["messages"] = results["messages"][:20]
            results["threads"] = results["threads"][:10]
            results["people"] = results["people"][:10]

            # Final stats
            results["stats"]["total_messages"] = len(results["messages"])
            results["stats"]["total_threads"] = len(results["threads"])
            results["stats"]["total_people"] = len(results["people"])

            self.logger.info(
                "Cast net search completed",
                extra={
                    "total_queries": results["stats"]["total_queries"],
                    "messages_found": results["stats"]["total_messages"],
                    "threads_found": results["stats"]["total_threads"],
                    "people_found": results["stats"]["total_people"],
                },
            )

        except Exception as e:
            self.logger.error(f"Cast net search failed: {e}")

        return results

    def _execute_field_search(
        self, client: firestore.Client, field_spec: Dict, priority_level: str
    ) -> List[Dict]:
        """Execute a single field search query"""

        results = []

        try:
            field_name = field_spec.get("field")
            field_value = field_spec.get("value")
            search_type = field_spec.get("search_type", "equals")
            collection_name = field_spec.get("collection")
            field_weight = field_spec.get("weight", 5)

            if not all([field_name, collection_name]) or field_value is None:
                return results

            # Build query based on search type
            query = client.collection(collection_name)

            if search_type == "exact_match":
                query = query.where(field_name, "==", field_value)

            elif search_type == "contains":
                # Note: Firestore doesn't support string contains directly
                # We'll use equality as fallback - implement full-text search externally if needed
                query = query.where(field_name, "==", field_value)

            elif search_type == "array_contains":
                query = query.where(field_name, "array_contains", field_value)

            elif search_type == "array_contains_any":
                if (
                    isinstance(field_value, list) and len(field_value) <= 10
                ):  # Firestore limit
                    query = query.where(field_name, "array_contains_any", field_value)
                else:
                    # Fallback to first value if too many
                    first_value = (
                        field_value[0] if isinstance(field_value, list) else field_value
                    )
                    query = query.where(field_name, "array_contains", first_value)

            elif search_type == "equals":
                query = query.where(field_name, "==", field_value)

            elif search_type == "date_range":
                # Handle date range searches
                if isinstance(field_value, dict):
                    start_date = field_value.get("start")
                    end_date = field_value.get("end")
                    if start_date:
                        query = query.where(field_name, ">=", start_date)
                    if end_date:
                        query = query.where(field_name, "<=", end_date)

            elif search_type == "range":
                # Handle numeric range searches
                if isinstance(field_value, dict):
                    min_val = field_value.get("min")
                    max_val = field_value.get("max")
                    if min_val is not None:
                        query = query.where(field_name, ">=", min_val)
                    if max_val is not None:
                        query = query.where(field_name, "<=", max_val)

            # Add ordering and limits
            try:
                if collection_name == "email_search_index":
                    query = query.order_by(
                        "sentAt", direction=firestore.Query.DESCENDING
                    )
                elif collection_name == "thread_summaries":
                    query = query.order_by(
                        "lastActivity", direction=firestore.Query.DESCENDING
                    )
                elif collection_name == "people_directory":
                    query = query.order_by(
                        "lastSeen", direction=firestore.Query.DESCENDING
                    )
            except Exception:
                # Skip ordering if it fails (e.g., composite index not available)
                pass

            # Execute query with limit
            docs = query.limit(15).stream()

            # Collect results
            for doc in docs:
                doc_data = doc.to_dict()
                results.append(
                    {
                        "collection": collection_name,
                        "doc_id": doc.id,
                        "data": doc_data,
                        "field_weight": field_weight,
                        "priority_level": priority_level,
                        "field_name": field_name,
                        "search_value": field_value,
                    }
                )

        except Exception as e:
            self.logger.warning(f"Field search failed for {field_name}: {e}")

        return results

    def search_emails_enhanced(
        self, query_understanding: Dict, knowledge_context: Dict
    ) -> Dict:
        """Enhanced email search using cast net strategy with multiple field searches"""

        # Use cast net search strategy from query understanding
        search_strategy = query_understanding.get("search_strategy", {})

        if not search_strategy or not any(
            [
                search_strategy.get("primary_fields"),
                search_strategy.get("secondary_fields"),
                search_strategy.get("tertiary_fields"),
            ]
        ):
            # Fallback to basic search if no strategy provided
            self.logger.warning("No search strategy provided, using fallback")
            return self._fallback_email_search(query_understanding)

        # Execute cast net search
        cast_net_results = self.cast_net_search(search_strategy)

        # Enhance results with additional context
        enhanced_results = {
            "messages": cast_net_results["messages"],
            "threads": cast_net_results["threads"],
            "people_directory": cast_net_results["people"],
            "stats": cast_net_results["stats"],
            "scores": cast_net_results["scores"],
            "search_strategy_used": search_strategy,
        }

        # Get thread summaries for messages that don't have threads yet
        self._enhance_with_missing_threads(enhanced_results)

        # Get full message content for top results if needed
        self._enhance_with_message_content(enhanced_results)

        # Apply additional filters from query understanding
        enhanced_results = self._apply_additional_filters(
            enhanced_results, query_understanding
        )

        self.logger.info(
            "Enhanced cast net search completed",
            extra={
                "messages": len(enhanced_results["messages"]),
                "threads": len(enhanced_results["threads"]),
                "people": len(enhanced_results["people_directory"]),
                "search_queries_executed": enhanced_results["stats"]["total_queries"],
            },
        )

        return enhanced_results

    def _fallback_email_search(self, query_understanding: Dict) -> Dict:
        """Fallback search when cast net strategy is not available"""
        results = {"messages": [], "threads": [], "people_directory": [], "stats": {}}

        try:
            email_client = self.db_manager.get_client("email")

            # Build basic query
            query = email_client.collection("email_search_index")

            # Apply time filters
            time_range = query_understanding.get("time_range", {})
            if time_range.get("start"):
                try:
                    start_date = datetime.fromisoformat(time_range["start"])
                    query = query.where("sentAt", ">=", start_date)
                except ValueError:
                    self.logger.warning(f"Invalid start date: {time_range['start']}")

            # Apply status filters
            status_filters = query_understanding.get("status_filters", {})
            if status_filters.get("is_important"):
                query = query.where("isImportant", "==", True)
            if status_filters.get("is_unread"):
                query = query.where("isUnread", "==", True)

            # Execute query
            docs = (
                query.order_by("sentAt", direction=firestore.Query.DESCENDING)
                .limit(15)
                .stream()
            )

            for doc in docs:
                msg_data = doc.to_dict()
                msg_data["message_id"] = doc.id
                results["messages"].append(msg_data)

            results["stats"] = {
                "total_messages": len(results["messages"]),
                "total_threads": 0,
                "total_people": 0,
                "search_type": "fallback",
            }

        except Exception as e:
            self.logger.error(f"Fallback email search failed: {e}")

        return results

    def _enhance_with_missing_threads(self, results: Dict):
        """Get thread summaries for messages that don't have corresponding threads"""
        try:
            email_client = self.db_manager.get_client("email")
            existing_thread_ids = {t.get("thread_id") for t in results["threads"]}

            # Get thread IDs from messages
            message_thread_ids = {
                msg.get("threadId")
                for msg in results["messages"]
                if msg.get("threadId")
            }
            missing_thread_ids = message_thread_ids - existing_thread_ids

            # Fetch missing threads
            for thread_id in list(missing_thread_ids)[
                :5
            ]:  # Limit to avoid too many queries
                thread_doc = (
                    email_client.collection("thread_summaries")
                    .document(thread_id)
                    .get()
                )
                if thread_doc.exists:
                    thread_data = thread_doc.to_dict()
                    thread_data["thread_id"] = thread_id
                    thread_data["relevance_score"] = (
                        5.0  # Default score for missing threads
                    )
                    results["threads"].append(thread_data)

        except Exception as e:
            self.logger.warning(f"Failed to enhance with missing threads: {e}")

    def _enhance_with_message_content(self, results: Dict):
        """Get full message content for top scored messages if needed"""
        try:
            email_client = self.db_manager.get_client("email")

            # Get full content for top 5 messages
            top_messages = sorted(
                results["messages"],
                key=lambda x: x.get("relevance_score", 0),
                reverse=True,
            )[:5]

            for msg in top_messages:
                message_id = msg.get("message_id")
                if message_id and not msg.get(
                    "bodyText"
                ):  # Only if we don't have full content
                    try:
                        full_doc = (
                            email_client.collection("messages_full")
                            .document(message_id)
                            .get()
                        )
                        if full_doc.exists:
                            full_data = full_doc.to_dict()
                            # Add selected full content fields to search result
                            msg["bodyText"] = full_data.get("bodyText", "")
                            msg["attachments"] = full_data.get("attachments", [])
                            msg["full_headers"] = full_data.get("headers", {})
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to get full content for message {message_id}: {e}"
                        )

        except Exception as e:
            self.logger.warning(f"Failed to enhance with message content: {e}")

    def _apply_additional_filters(
        self, results: Dict, query_understanding: Dict
    ) -> Dict:
        """Apply additional filters from query understanding"""
        try:
            # Filter by document types if specified
            document_filters = query_understanding.get("document_filters", {})
            if document_filters.get("has_attachments") is True:
                results["messages"] = [
                    msg for msg in results["messages"] if msg.get("hasAttachments")
                ]
            elif document_filters.get("has_attachments") is False:
                results["messages"] = [
                    msg for msg in results["messages"] if not msg.get("hasAttachments")
                ]

            # Filter by attachment types
            attachment_types = document_filters.get("attachment_types", [])
            if attachment_types:
                filtered_messages = []
                for msg in results["messages"]:
                    msg_attachment_types = msg.get("attachmentTypes", [])
                    if any(
                        att_type in msg_attachment_types
                        for att_type in attachment_types
                    ):
                        filtered_messages.append(msg)
                results["messages"] = filtered_messages

            # Update stats after filtering
            results["stats"]["total_messages"] = len(results["messages"])
            results["stats"]["filters_applied"] = True

        except Exception as e:
            self.logger.warning(f"Failed to apply additional filters: {e}")

        return results

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

    def perform_monitoring(self, query_understanding: Dict) -> Dict:
        """Perform monitoring tasks based on query intent"""

        monitoring_type = query_understanding.get("monitoring_type")
        monitoring_results = {}

        if monitoring_type == "unanswered":
            monitoring_results["unanswered"] = (
                self.email_monitor.check_unanswered_emails()
            )

        elif monitoring_type == "urgent":
            monitoring_results["urgent"] = self.email_monitor.find_urgent_unread()

        elif monitoring_type == "response_time":
            # Find last response time for specific person
            people = query_understanding.get("people_mentioned", [])
            if people:
                # Get their last sent email
                email_client = self.db_manager.get_client("email")
                for person in people:
                    last_email = (
                        email_client.collection("email_search_index")
                        .where("fromPerson", "==", person.lower())
                        .order_by("sentAt", direction=firestore.Query.DESCENDING)
                        .limit(1)
                        .stream()
                    )

                    for email in last_email:
                        email_data = email.to_dict()
                        monitoring_results[f"{person}_last_email"] = {
                            "sentAt": email_data.get("sentAt"),
                            "subject": email_data.get("subject"),
                            "preview": email_data.get("bodyPreview"),
                        }

        return monitoring_results

    def run_daily_monitoring(self) -> str:
        """Run daily monitoring checks"""

        report = ["Daily Email Monitoring Report", "=" * 40]

        # Check unanswered emails
        unanswered = self.email_monitor.check_unanswered_emails(hours=48)
        if unanswered:
            report.append(f"\n📬 {len(unanswered)} emails need responses (>48 hours):")
            for email in unanswered[:5]:
                report.append(f"  - {email['subject']} from {email['from']}")

        # Check urgent unread
        urgent = self.email_monitor.find_urgent_unread()
        if urgent:
            report.append(f"\n⚠️ {len(urgent)} urgent unread emails:")
            for email in urgent[:5]:
                report.append(f"  - {email['subject']} from {email['from']}")

        if not unanswered and not urgent:
            report.append("\n✅ All caught up! No urgent items.")

        return "\n".join(report)

    def generate_cast_net_response(
        self,
        user_message: str,
        query_understanding: Dict,
        knowledge_context: Dict,
        email_results: Dict,
        monitoring_results: Dict,
    ) -> str:
        """Generate response using cast net search results and relevance scoring"""

        # Format context from cast net results
        messages_info = self._format_cast_net_messages(
            email_results.get("messages", [])
        )
        threads_info = self._format_cast_net_threads(email_results.get("threads", []))
        people_info = self._format_cast_net_people(
            email_results.get("people_directory", [])
        )
        search_stats = email_results.get("stats", {})

        # Handle monitoring results
        monitoring_info = ""
        if monitoring_results:
            monitoring_info = self._format_monitoring_results(monitoring_results)

        prompt = f"""
        You are AD3Gem, an intelligent email assistant with advanced cast net search capabilities.

        USER QUERY: "{user_message}"

        CAST NET SEARCH ANALYSIS:
        Intent: {query_understanding.get("intent")}
        Mode: {query_understanding.get("mode")}
        Confidence: {query_understanding.get("confidence", 0)}%
        Search Strategy Used:
        - Primary fields: {len(query_understanding.get("search_strategy", {}).get("primary_fields", []))}
        - Secondary fields: {len(query_understanding.get("search_strategy", {}).get("secondary_fields", []))}
        - Tertiary fields: {len(query_understanding.get("search_strategy", {}).get("tertiary_fields", []))}

        CAST NET SEARCH RESULTS:
        Total Queries Executed: {search_stats.get("total_queries", 0)}
        Messages Found: {search_stats.get("total_messages", 0)} (ranked by relevance)
        Threads Found: {search_stats.get("total_threads", 0)}
        People Found: {search_stats.get("total_people", 0)}

        TOP RANKED MESSAGES:
        {messages_info if messages_info else "No messages found with current search strategy"}

        RELATED THREADS:
        {threads_info if threads_info else "No threads found"}

        PEOPLE DIRECTORY MATCHES:
        {people_info if people_info else "No people matches found"}

        {("MONITORING RESULTS:" + chr(10) + monitoring_info) if monitoring_info else ""}

        KNOWLEDGE CONTEXT:
        {self._format_entities(knowledge_context.get("entities", {}))}

        INSTRUCTIONS:
        1. Answer the user's question using the cast net search results, prioritizing higher relevance scores
        2. Explain what the cast net search found and how it relates to their query
        3. Reference specific emails, people, or threads with their relevance scores when helpful
        4. If results seem incomplete, mention what additional searches could be tried
        5. Be conversational but precise about the search strategy used
        6. Keep response focused and under 200 words unless listing many items

        Generate a helpful response explaining the cast net search results:
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Cast net response generation failed: {e}")
            return f"I found {search_stats.get('total_messages', 0)} messages using a multi-field search strategy, but had trouble generating a detailed response. The search covered {search_stats.get('total_queries', 0)} different field combinations to ensure comprehensive results."

    def generate_intelligent_response(
        self,
        user_message: str,
        query_understanding: Dict,
        knowledge_context: Dict,
        email_results: Dict,
        monitoring_results: Dict,
    ) -> str:
        """Generate response using enhanced query understanding and all gathered context"""

        # Format context for prompt
        people_info = "\n".join(
            [
                f"- {p.get('name')} ({', '.join(p.get('email_addresses', []))}): {p.get('roles', ['Unknown role'])[0]}"
                for p in knowledge_context.get("entities", {}).values()
            ]
        )

        thread_info = "\n".join(
            [
                f"- {t.get('subject', 'No subject')} (Last: {t.get('lastActivity', 'Unknown')})"
                for t in email_results.get("threads", [])[:5]
            ]
        )

        message_info = "\n".join(
            [
                f"- From {m.get('fromEmail')} on {m.get('sentAt', 'Unknown')}: {m.get('subject', 'No subject')}"
                for m in email_results.get("messages", [])[:5]
            ]
        )

        # Handle monitoring results
        monitoring_info = ""
        if monitoring_results:
            if "unanswered" in monitoring_results:
                unanswered = monitoring_results["unanswered"]
                monitoring_info = (
                    f"Found {len(unanswered)} unanswered emails older than 48 hours"
                )
                if unanswered:
                    monitoring_info += ":\n" + "\n".join(
                        [f"  - {u['subject']} from {u['from']}" for u in unanswered[:3]]
                    )

            for key, value in monitoring_results.items():
                if "_last_email" in key:
                    person = key.replace("_last_email", "")
                    sent_at = value.get("sentAt", "Unknown")
                    if hasattr(sent_at, "strftime"):
                        sent_at = sent_at.strftime("%Y-%m-%d %H:%M")
                    monitoring_info += f"\n{person.title()} last sent an email on {sent_at}: {value.get('subject', 'No subject')}"

        prompt = f"""
        You are an intelligent email assistant. Answer the user's question using the provided context.

        USER QUERY: "{user_message}"

        QUERY UNDERSTANDING:
        Intent: {query_understanding.get("intent")}
        Mode: {query_understanding.get("mode")}
        Time range: {query_understanding.get("time_range", {}).get("description", "All time")}
        Confidence: {query_understanding.get("confidence", 0)}%

        PEOPLE CONTEXT:
        {people_info if people_info else "No specific people identified"}

        EMAIL SEARCH RESULTS:
        Found {email_results.get("stats", {}).get("total_messages", 0)} messages in {email_results.get("stats", {}).get("total_threads", 0)} threads

        Recent Threads:
        {thread_info if thread_info else "No threads found"}

        Recent Messages:
        {message_info if message_info else "No messages found"}

        {("MONITORING RESULTS:" + chr(10) + monitoring_info) if monitoring_info else ""}

        INSTRUCTIONS:
        1. Answer directly and conversationally
        2. Use people's names when you know them
        3. Mention specific emails/dates when relevant
        4. If no results, explain what you searched for
        5. Keep response under 150 words unless listing items
        6. If monitoring results exist, prioritize them in your response

        Generate a helpful response:
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Enhanced response generation failed: {e}")
            return "I encountered an error generating a response. Please try rephrasing your question."

    def _format_cast_net_messages(self, messages: List[Dict]) -> str:
        """Format cast net search messages with relevance scores"""
        if not messages:
            return "No messages found"

        formatted = []
        for msg in messages[:5]:  # Top 5 results
            relevance = msg.get("relevance_score", 0)
            from_email = msg.get("fromEmail", "Unknown")
            from_person = msg.get("fromPerson", "Unknown")
            subject = msg.get("subject", "No subject")
            sent_at = msg.get("sentAt", "Unknown")

            if hasattr(sent_at, "strftime"):
                sent_at = sent_at.strftime("%Y-%m-%d %H:%M")

            preview = msg.get("bodyPreview", "")[:100]

            formatted.append(
                f"- [Score: {relevance:.1f}] From {from_person} ({from_email})\n"
                f"  Subject: {subject} | {sent_at}\n"
                f"  Preview: {preview}..."
            )

        return "\n".join(formatted)

    def _format_cast_net_threads(self, threads: List[Dict]) -> str:
        """Format thread results with relevance scores"""
        if not threads:
            return "No threads found"

        formatted = []
        for thread in threads[:3]:  # Top 3 threads
            relevance = thread.get("relevance_score", 0)
            subject = thread.get("subject", "No subject")
            message_count = thread.get("messageCount", 0)
            last_activity = thread.get("lastActivity", "Unknown")

            if hasattr(last_activity, "strftime"):
                last_activity = last_activity.strftime("%Y-%m-%d %H:%M")

            formatted.append(
                f"- [Score: {relevance:.1f}] {subject}\n"
                f"  Messages: {message_count} | Last: {last_activity}"
            )

        return "\n".join(formatted)

    def _format_cast_net_people(self, people: List[Dict]) -> str:
        """Format people directory results with relevance scores"""
        if not people:
            return "No people found"

        formatted = []
        for person in people[:5]:  # Top 5 people
            relevance = person.get("relevance_score", 0)
            display_name = person.get("displayName", "Unknown")
            email = person.get("email", "Unknown")
            is_internal = "Internal" if person.get("isInternal") else "External"

            formatted.append(
                f"- [Score: {relevance:.1f}] {display_name} ({email}) - {is_internal}"
            )

        return "\n".join(formatted)

    def _format_monitoring_results(self, monitoring_results: Dict) -> str:
        """Format monitoring results for response"""
        formatted = []

        if "unanswered" in monitoring_results:
            unanswered = monitoring_results["unanswered"]
            formatted.append(f"Found {len(unanswered)} unanswered emails:")
            for email in unanswered[:3]:
                formatted.append(f"  - {email['subject']} from {email['from']}")

        for key, value in monitoring_results.items():
            if "_last_email" in key:
                person = key.replace("_last_email", "")
                sent_at = value.get("sentAt", "Unknown")
                if hasattr(sent_at, "strftime"):
                    sent_at = sent_at.strftime("%Y-%m-%d %H:%M")
                formatted.append(
                    f"{person.title()} last emailed on {sent_at}: {value.get('subject', 'No subject')}"
                )

        return "\n".join(formatted)

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
        """Process user message with cast net search strategy"""

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
            "Processing user message with cast net search",
            extra={
                "user_id": self.config.user_id,
                "session_id": self.session_id,
                "message_length": len(user_message),
            },
        )

        try:
            # Save user message
            self.save_conversation_message("user", user_message)

            # Enhanced query understanding with field selection for cast net approach
            query_understanding = self.understand_query(user_message)

            # Get knowledge context
            knowledge_context = self.get_knowledge_context(user_message)

            # Execute cast net email search using field strategy from LLM
            email_results = {}
            monitoring_results = {}
            web_data = None

            if query_understanding.get("requires_email_search", True):
                self.logger.info("Executing cast net email search with field strategy")
                email_results = self.search_emails_enhanced(
                    query_understanding, knowledge_context
                )

            # Perform monitoring if needed
            if query_understanding.get("monitoring_type"):
                monitoring_results = self.perform_monitoring(query_understanding)

            if query_understanding.get("mode") == "general" and query_understanding.get(
                "requires_web_search"
            ):
                web_data = self.web_search(user_message)

            # Generate intelligent response using cast net results
            if (
                query_understanding.get("mode") == "company"
                or email_results
                or monitoring_results
            ):
                response = self.generate_cast_net_response(
                    user_message,
                    query_understanding,
                    knowledge_context,
                    email_results,
                    monitoring_results,
                )
            else:
                # Fallback to legacy routing for general queries
                routing = self.route_query(user_message)
                email_data = (
                    email_results.get("threads", []),
                    email_results.get("messages", []),
                )
                response = self.generate_response(
                    user_message, routing, knowledge_context, email_data, web_data
                )

            # Save response with detailed metadata about cast net search
            response_metadata = {
                "query_understanding": {
                    "intent": query_understanding.get("intent"),
                    "mode": query_understanding.get("mode"),
                    "confidence": query_understanding.get("confidence"),
                },
                "search_strategy": query_understanding.get("search_strategy", {}),
                "cast_net_stats": email_results.get("stats", {}),
                "knowledge_entities": len(knowledge_context.get("entities", {})),
                "monitoring_results": len(monitoring_results),
                "processing_time": round(time.time() - start_time, 3),
            }

            self.save_conversation_message("assistant", response, response_metadata)

            # Update session context
            self.session_context.update(
                {
                    "last_mode": query_understanding.get("mode"),
                    "last_intent": query_understanding.get("intent"),
                    "last_entities": list(knowledge_context.get("entities", {}).keys()),
                    "last_people_mentioned": query_understanding.get(
                        "people_mentioned", []
                    ),
                    "last_search_strategy": query_understanding.get(
                        "search_strategy", {}
                    ),
                }
            )

            self.logger.info(
                "Message processed with cast net search",
                extra={
                    "mode": query_understanding.get("mode"),
                    "intent": query_understanding.get("intent"),
                    "processing_time": round(time.time() - start_time, 3),
                    "response_length": len(response),
                    "search_queries": email_results.get("stats", {}).get(
                        "total_queries", 0
                    ),
                    "messages_found": email_results.get("stats", {}).get(
                        "total_messages", 0
                    ),
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
                "cast_net_enabled": True,
                "total_fields_available": sum(
                    len(fields) for fields in EMAIL_FIELD_MAPPING.values()
                ),
            }

            return status

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"status": "error", "message": str(e)}

    def start_interactive_session(self):
        """Start interactive chat session with cast net search and enhanced commands"""

        print("\n" + "=" * 80)
        print("🚀 Intelligent AD3Gem Chatbot - Cast Net Search Enabled")
        print("=" * 80)
        print(f"Environment: {self.config.environment}")
        print(f"User: {self.config.user_id}")
        print(f"Session: {self.session_id}")
        print(
            f"Available Email Fields: {sum(len(fields) for fields in EMAIL_FIELD_MAPPING.values())}"
        )
        print("\n🎯 Cast Net Search Features:")
        print("  • Multi-field parallel search across all email collections")
        print("  • Relevance scoring and intelligent result ranking")
        print("  • Person name resolution and smart field selection")
        print("  • Advanced query understanding with field strategy")
        print("  • Email monitoring (unanswered, urgent, response times)")
        print("  • Knowledge base integration with caching")
        print("  • Performance tracking and comprehensive logging")
        print("  • Error recovery and fallback strategies")
        print("\n💬 Commands:")
        print("  'status' - Show system status and cast net info")
        print("  'metrics' - Show performance metrics")
        print("  'cache' - Show cache statistics")
        print("  'health' - Check database health")
        print("  'monitor' - Run daily email monitoring report")
        print("  'preferences' - Show learned user preferences")
        print("  'fields' - Show available email fields for cast net search")
        print("  'quit/exit/bye' - End session")
        print("-" * 80)

        while True:
            try:
                user_input = input("\n💭 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\n👋 Session ended. Cast net search data saved. Goodbye!")
                    break

                if user_input.lower() == "status":
                    status = self.get_system_status()
                    print("\n📊 System Status:")
                    print(f"  Uptime: {status['uptime_seconds']}s")
                    print(f"  Messages: {status['message_count']}")
                    print(f"  Environment: {status['environment']}")
                    print(f"  Cast Net Enabled: {status['cast_net_enabled']}")
                    print(f"  Available Fields: {status['total_fields_available']}")
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
                    print(f"\n📊 {report}")
                    continue

                if user_input.lower() == "preferences":
                    prefs = self.preference_tracker.preferences
                    print("\n📚 Learned Preferences:")
                    print(f"  Name mappings: {prefs.get('name_mappings', {})}")
                    print(
                        f"  Recent queries: {len(prefs.get('common_queries', []))} tracked"
                    )
                    if prefs.get("common_queries"):
                        print("  Recent query patterns:")
                        for query_info in prefs["common_queries"][-5:]:
                            print(f"    - {query_info['query'][:50]}...")
                    continue

                if user_input.lower() == "fields":
                    print("\n🎯 Cast Net Search Fields:")
                    for category, fields in EMAIL_FIELD_MAPPING.items():
                        print(
                            f"\n  {category.upper().replace('_', ' ')} ({len(fields)} fields):"
                        )
                        for field_name, field_info in list(fields.items())[:3]:
                            print(
                                f"    - {field_name} ({field_info['collection']}) weight={field_info['weight']}"
                            )
                        if len(fields) > 3:
                            print(f"    ... and {len(fields) - 3} more")
                    continue

                if not user_input:
                    continue

                # Process normal message with cast net search
                response = self.send_message(user_input)
                print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print(
                    "\n\n👋 Session interrupted. Cast net search data saved. Goodbye!"
                )
                break
            except Exception as e:
                self.logger.error(f"Interactive session error: {e}")
                print(f"\n⚠️ Error: {e}")
                print("Continuing session...")


def main():
    """Main function to initialize and run the cast net chatbot"""

    print("🔥 Initializing Enhanced AD3Gem Chatbot with Cast Net Search...")
    print("   Production-ready with parallel field queries and relevance scoring")
    print("   • Cast net email search • Advanced AI understanding • Email monitoring")

    try:
        # Load configuration
        config = Config()

        # Initialize chatbot with cast net search
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
        print("5. Firestore collections have required composite indexes")


if __name__ == "__main__":
    main()
