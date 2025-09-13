"""
Manual Database Search Tool for AD3Gem
=====================================

Provides comprehensive manual search capabilities across all AD3Gem databases:
- Main database (ad3gem-database)
- Conversation database (ad3gem-conversation)
- Memory database (ad3gem-memory)
- Email database (ad3sam-email)

Features:
- Flexible search by collection, field, value
- Advanced filtering and sorting options
- Export results to JSON/CSV
- Interactive CLI interface
- Performance metrics and logging
"""

import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter


class DatabaseType(Enum):
    """Database types available for searching"""

    MAIN = "main"
    CONVERSATION = "conversation"
    MEMORY = "memory"
    EMAIL = "email"


class SearchOperator(Enum):
    """Search operators for filtering"""

    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not-in"
    ARRAY_CONTAINS = "array-contains"
    ARRAY_CONTAINS_ANY = "array-contains-any"


@dataclass
class SearchCriteria:
    """Search criteria configuration"""

    collection: str
    field: Optional[str] = None
    value: Any = None
    operator: SearchOperator = SearchOperator.EQUAL
    limit: int = 50
    order_by: Optional[str] = None
    order_direction: str = "DESCENDING"


@dataclass
class SearchResult:
    """Search result container"""

    database: DatabaseType
    collection: str
    documents: List[Dict[str, Any]]
    total_count: int
    execution_time: float
    search_criteria: SearchCriteria


class ManualSearchTool:
    """
    Comprehensive manual search tool for AD3Gem databases
    """

    def __init__(self):
        """Initialize the search tool with database connections"""
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "ad3-sam")
        self.logger = self._setup_logging()

        # Initialize database clients
        self.clients = self._init_clients()

        # Search history for performance tracking
        self.search_history = []

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the search tool"""
        logger = logging.getLogger("ManualSearch")
        logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _init_clients(self) -> Dict[DatabaseType, firestore.Client]:
        """Initialize connections to all databases"""
        clients = {}

        try:
            # Main database
            clients[DatabaseType.MAIN] = firestore.Client(
                project=self.project_id,
                database=os.getenv("FIRESTORE_DATABASE", "ad3gem-database"),
            )

            # Conversation database
            clients[DatabaseType.CONVERSATION] = firestore.Client(
                project=self.project_id,
                database=os.getenv("FIRESTORE_CONVERSATION_DB", "ad3gem-conversation"),
            )

            # Memory database
            clients[DatabaseType.MEMORY] = firestore.Client(
                project=self.project_id,
                database=os.getenv("FIRESTORE_MEMORY_DB", "ad3gem-memory"),
            )

            # Email database
            clients[DatabaseType.EMAIL] = firestore.Client(
                project=self.project_id,
                database=os.getenv("FIRESTORE_EMAIL_DB", "ad3sam-email"),
            )

            self.logger.info("All database clients initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize database clients: {e}")
            raise

        return clients

    def search_documents(
        self, database: DatabaseType, criteria: SearchCriteria
    ) -> SearchResult:
        """
        Perform a search on the specified database and collection

        Args:
            database: Database type to search
            criteria: Search criteria

        Returns:
            SearchResult object with results and metadata
        """
        import time

        start_time = time.time()

        try:
            client = self.clients[database]

            # Build the query
            query = client.collection(criteria.collection)

            # Apply filters
            if criteria.field and criteria.value is not None:
                if criteria.operator == SearchOperator.EQUAL:
                    query = query.where(
                        filter=FieldFilter(criteria.field, "==", criteria.value)
                    )
                elif criteria.operator == SearchOperator.NOT_EQUAL:
                    # Firestore doesn't support != directly, need to handle differently
                    pass  # Will implement later
                elif criteria.operator == SearchOperator.GREATER_THAN:
                    query = query.where(
                        filter=FieldFilter(criteria.field, ">", criteria.value)
                    )
                elif criteria.operator == SearchOperator.GREATER_EQUAL:
                    query = query.where(
                        filter=FieldFilter(criteria.field, ">=", criteria.value)
                    )
                elif criteria.operator == SearchOperator.LESS_THAN:
                    query = query.where(
                        filter=FieldFilter(criteria.field, "<", criteria.value)
                    )
                elif criteria.operator == SearchOperator.LESS_EQUAL:
                    query = query.where(
                        filter=FieldFilter(criteria.field, "<=", criteria.value)
                    )
                elif criteria.operator == SearchOperator.ARRAY_CONTAINS:
                    query = query.where(
                        filter=FieldFilter(
                            criteria.field, "array-contains", criteria.value
                        )
                    )

            # Apply ordering
            if criteria.order_by:
                direction = (
                    firestore.Query.DESCENDING
                    if criteria.order_direction == "DESCENDING"
                    else firestore.Query.ASCENDING
                )
                query = query.order_by(criteria.order_by, direction=direction)

            # Apply limit
            query = query.limit(criteria.limit)

            # Execute query
            docs = query.stream()

            # Process results
            documents = []
            for doc in docs:
                doc_data = doc.to_dict()
                doc_data["_id"] = doc.id
                doc_data["_timestamp"] = datetime.now(timezone.utc).isoformat()
                documents.append(doc_data)

            execution_time = time.time() - start_time

            result = SearchResult(
                database=database,
                collection=criteria.collection,
                documents=documents,
                total_count=len(documents),
                execution_time=execution_time,
                search_criteria=criteria,
            )

            # Add to search history
            self.search_history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "database": database.value,
                    "collection": criteria.collection,
                    "result_count": len(documents),
                    "execution_time": execution_time,
                }
            )

            self.logger.info(
                f"Search completed: {database.value}.{criteria.collection} "
                f"-> {len(documents)} results in {execution_time:.3f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            execution_time = time.time() - start_time

            return SearchResult(
                database=database,
                collection=criteria.collection,
                documents=[],
                total_count=0,
                execution_time=execution_time,
                search_criteria=criteria,
            )

    def list_collections(self, database: DatabaseType) -> List[str]:
        """List all collections in a database"""
        try:
            client = self.clients[database]
            collections = client.collections()
            return [collection.id for collection in collections]
        except Exception as e:
            self.logger.error(f"Failed to list collections for {database.value}: {e}")
            return []

    def get_collection_stats(
        self, database: DatabaseType, collection: str
    ) -> Dict[str, Any]:
        """Get statistics about a collection"""
        try:
            client = self.clients[database]
            coll_ref = client.collection(collection)

            # Get sample documents to estimate size
            docs = list(coll_ref.limit(100).stream())
            sample_size = len(docs)

            stats = {
                "collection": collection,
                "database": database.value,
                "sample_size": sample_size,
                "estimated_total": "unknown",  # Firestore doesn't provide total count easily
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            if sample_size > 0:
                # Get field information from first document
                first_doc = docs[0].to_dict()
                stats["sample_fields"] = list(first_doc.keys())
                stats["sample_document_id"] = docs[0].id

            return stats

        except Exception as e:
            self.logger.error(
                f"Failed to get stats for {database.value}.{collection}: {e}"
            )
            return {"error": str(e)}

    def advanced_search(
        self,
        database: DatabaseType,
        collection: str,
        filters: List[Tuple[str, SearchOperator, Any]],
        limit: int = 50,
        order_by: Optional[str] = None,
    ) -> SearchResult:
        """
        Perform advanced search with multiple filters

        Args:
            database: Database type
            collection: Collection name
            filters: List of (field, operator, value) tuples
            limit: Maximum results to return
            order_by: Field to order by
        """
        import time

        start_time = time.time()

        try:
            client = self.clients[database]
            query = client.collection(collection)

            # Apply multiple filters
            for field, operator, value in filters:
                if operator == SearchOperator.EQUAL:
                    query = query.where(filter=FieldFilter(field, "==", value))
                elif operator == SearchOperator.GREATER_THAN:
                    query = query.where(filter=FieldFilter(field, ">", value))
                elif operator == SearchOperator.LESS_THAN:
                    query = query.where(filter=FieldFilter(field, "<", value))
                elif operator == SearchOperator.ARRAY_CONTAINS:
                    query = query.where(
                        filter=FieldFilter(field, "array-contains", value)
                    )

            # Apply ordering and limit
            if order_by:
                query = query.order_by(order_by, direction=firestore.Query.DESCENDING)

            query = query.limit(limit)

            # Execute query
            docs = query.stream()
            documents = []
            for doc in docs:
                doc_data = doc.to_dict()
                doc_data["_id"] = doc.id
                documents.append(doc_data)

            execution_time = time.time() - start_time

            criteria = SearchCriteria(
                collection=collection, limit=limit, order_by=order_by
            )

            result = SearchResult(
                database=database,
                collection=collection,
                documents=documents,
                total_count=len(documents),
                execution_time=execution_time,
                search_criteria=criteria,
            )

            self.logger.info(
                f"Advanced search completed: {len(filters)} filters, "
                f"{len(documents)} results in {execution_time:.3f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Advanced search failed: {e}")
            execution_time = time.time() - start_time

            return SearchResult(
                database=database,
                collection=collection,
                documents=[],
                total_count=0,
                execution_time=execution_time,
                search_criteria=SearchCriteria(collection=collection),
            )

    def export_results(
        self,
        result: SearchResult,
        format_type: str = "json",
        filename: Optional[str] = None,
    ) -> str:
        """
        Export search results to file

        Args:
            result: SearchResult to export
            format_type: "json" or "csv"
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_results_{result.database.value}_{result.collection}_{timestamp}.{format_type}"

        try:
            if format_type == "json":
                with open(filename, "w") as f:
                    json.dump(
                        {
                            "search_metadata": {
                                "database": result.database.value,
                                "collection": result.collection,
                                "total_results": result.total_count,
                                "execution_time": result.execution_time,
                                "export_timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                            "documents": result.documents,
                        },
                        f,
                        indent=2,
                        default=str,
                    )

            elif format_type == "csv":
                if result.documents:
                    fieldnames = set()
                    for doc in result.documents:
                        fieldnames.update(doc.keys())

                    with open(filename, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                        writer.writeheader()
                        for doc in result.documents:
                            writer.writerow(doc)

            self.logger.info(f"Results exported to {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return ""

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get search history with performance metrics"""
        return self.search_history.copy()

    def clear_search_history(self):
        """Clear the search history"""
        self.search_history.clear()
        self.logger.info("Search history cleared")


# ================================
# INTERACTIVE CLI INTERFACE
# ================================


class ManualSearchCLI:
    """Interactive command-line interface for manual database searches"""

    def __init__(self):
        self.search_tool = ManualSearchTool()
        self.current_results = None

    def run(self):
        """Run the interactive CLI"""
        print("\n" + "=" * 80)
        print("🔍 AD3Gem Manual Database Search Tool")
        print("=" * 80)
        print("Available databases:")
        print("  • main: ad3gem-database")
        print("  • conversation: ad3gem-conversation")
        print("  • memory: ad3gem-memory")
        print("  • email: ad3sam-email")
        print("\nCommands:")
        print("  search <db> <collection> [field] [value] - Basic search")
        print("  advanced <db> <collection> - Advanced multi-filter search")
        print("  list <db> - List collections in database")
        print("  stats <db> <collection> - Get collection statistics")
        print("  export [json|csv] [filename] - Export last search results")
        print("  history - Show search history")
        print("  help - Show this help")
        print("  quit - Exit")
        print("-" * 80)

        while True:
            try:
                command = input("\n🔍 Search> ").strip()

                if not command:
                    continue

                if command.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                self._process_command(command)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _process_command(self, command: str):
        """Process a command from the CLI"""
        parts = command.split()
        cmd = parts[0].lower()

        try:
            if cmd == "search":
                self._handle_search(parts[1:])
            elif cmd == "advanced":
                self._handle_advanced_search(parts[1:])
            elif cmd == "list":
                self._handle_list_collections(parts[1:])
            elif cmd == "stats":
                self._handle_collection_stats(parts[1:])
            elif cmd == "export":
                self._handle_export(parts[1:])
            elif cmd == "history":
                self._handle_history()
            elif cmd == "help":
                self._show_help()
            else:
                print(f"❌ Unknown command: {cmd}")

        except IndexError:
            print(f"❌ Missing arguments for command: {cmd}")
        except Exception as e:
            print(f"❌ Command failed: {e}")

    def _handle_search(self, args: List[str]):
        """Handle basic search command"""
        if len(args) < 2:
            print("❌ Usage: search <db> <collection> [field] [value]")
            return

        db_name = args[0].lower()
        collection = args[1]

        # Parse database type
        try:
            database = DatabaseType(db_name)
        except ValueError:
            print(f"❌ Invalid database: {db_name}")
            return

        # Create search criteria
        criteria = SearchCriteria(collection=collection)

        if len(args) >= 4:
            criteria.field = args[2]
            criteria.value = args[3]

        # Execute search
        print(f"🔎 Searching {database.value}.{collection}...")
        result = self.search_tool.search_documents(database, criteria)

        self._display_results(result)
        self.current_results = result

    def _handle_advanced_search(self, args: List[str]):
        """Handle advanced search with multiple filters"""
        if len(args) < 2:
            print("❌ Usage: advanced <db> <collection>")
            return

        db_name = args[0].lower()
        collection = args[1]

        try:
            database = DatabaseType(db_name)
        except ValueError:
            print(f"❌ Invalid database: {db_name}")
            return

        # Get filters from user
        filters = []
        print("Enter filters (press Enter when done):")
        print("Format: field operator value")
        print("Operators: ==, >, <, >=, <=, array-contains")
        print("Example: created_at > 2024-01-01")

        while True:
            filter_input = input("Filter> ").strip()
            if not filter_input:
                break

            try:
                field, op, value = filter_input.split(maxsplit=2)

                # Parse operator
                if op == "==":
                    operator = SearchOperator.EQUAL
                elif op == ">":
                    operator = SearchOperator.GREATER_THAN
                elif op == "<":
                    operator = SearchOperator.LESS_THAN
                elif op == ">=":
                    operator = SearchOperator.GREATER_EQUAL
                elif op == "<=":
                    operator = SearchOperator.LESS_EQUAL
                elif op == "array-contains":
                    operator = SearchOperator.ARRAY_CONTAINS
                else:
                    print(f"❌ Invalid operator: {op}")
                    continue

                filters.append((field, operator, value))
                print(f"✅ Added filter: {field} {op} {value}")

            except ValueError:
                print("❌ Invalid filter format. Use: field operator value")

        if not filters:
            print("❌ No filters specified")
            return

        # Execute advanced search
        print(f"🔎 Advanced search with {len(filters)} filters...")
        result = self.search_tool.advanced_search(database, collection, filters)

        self._display_results(result)
        self.current_results = result

    def _handle_list_collections(self, args: List[str]):
        """Handle list collections command"""
        if len(args) < 1:
            print("❌ Usage: list <db>")
            return

        db_name = args[0].lower()

        try:
            database = DatabaseType(db_name)
        except ValueError:
            print(f"❌ Invalid database: {db_name}")
            return

        collections = self.search_tool.list_collections(database)

        if collections:
            print(f"📂 Collections in {database.value}:")
            for collection in sorted(collections):
                print(f"  • {collection}")
        else:
            print(f"📂 No collections found in {database.value}")

    def _handle_collection_stats(self, args: List[str]):
        """Handle collection stats command"""
        if len(args) < 2:
            print("❌ Usage: stats <db> <collection>")
            return

        db_name = args[0].lower()
        collection = args[1]

        try:
            database = DatabaseType(db_name)
        except ValueError:
            print(f"❌ Invalid database: {db_name}")
            return

        stats = self.search_tool.get_collection_stats(database, collection)

        print(f"📊 Stats for {database.value}.{collection}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def _handle_export(self, args: List[str]):
        """Handle export command"""
        if not self.current_results:
            print("❌ No search results to export")
            return

        format_type = args[0] if args else "json"
        filename = args[1] if len(args) > 1 else None

        if format_type not in ["json", "csv"]:
            print("❌ Invalid format. Use 'json' or 'csv'")
            return

        exported_file = self.search_tool.export_results(
            self.current_results, format_type, filename
        )

        if exported_file:
            print(f"✅ Results exported to: {exported_file}")
        else:
            print("❌ Export failed")

    def _handle_history(self):
        """Handle history command"""
        history = self.search_tool.get_search_history()

        if not history:
            print("📝 No search history")
            return

        print("📝 Search History:")
        for i, entry in enumerate(history[-10:], 1):  # Show last 10
            print(
                f"  {i}. {entry['database']}.{entry['collection']} "
                f"-> {entry['result_count']} results "
                f"({entry['execution_time']:.3f}s)"
            )

    def _show_help(self):
        """Show help information"""
        print("\n" + "=" * 60)
        print("AD3Gem Manual Search Tool - Help")
        print("=" * 60)
        print("Available databases:")
        print("  • main: ad3gem-database (general data)")
        print("  • conversation: ad3gem-conversation (chat history)")
        print("  • memory: ad3gem-memory (knowledge/claims)")
        print("  • email: ad3sam-email (email data)")
        print("\nCommands:")
        print("  search <db> <collection> [field] [value]")
        print("    Basic search in a collection")
        print("    Example: search email reports subject 'meeting'")
        print("\n  advanced <db> <collection>")
        print("    Advanced search with multiple filters")
        print("    Example: advanced memory heads")
        print("\n  list <db>")
        print("    List all collections in a database")
        print("    Example: list main")
        print("\n  stats <db> <collection>")
        print("    Get statistics about a collection")
        print("    Example: stats email sample_emails")
        print("\n  export [json|csv] [filename]")
        print("    Export last search results")
        print("    Example: export csv results.csv")
        print("\n  history")
        print("    Show recent search history")
        print("\n  help")
        print("    Show this help message")
        print("\n  quit")
        print("    Exit the search tool")
        print("-" * 60)

    def _display_results(self, result: SearchResult):
        """Display search results in a formatted way"""
        print(f"\n{'=' * 60}")
        print(f"🔍 Search Results: {result.database.value}.{result.collection}")
        print(f"{'=' * 60}")
        print(f"📊 Found {result.total_count} documents")
        print(f"⏱️  Execution time: {result.execution_time:.3f} seconds")
        print(f"{'=' * 60}")

        if result.total_count == 0:
            print("❌ No documents found")
            return

        # Display up to 5 results
        for i, doc in enumerate(result.documents[:5], 1):
            print(f"\n📄 Document {i} (ID: {doc.get('_id', 'unknown')})")
            print("-" * 40)

            # Show key fields (exclude internal fields)
            for key, value in doc.items():
                if not key.startswith("_"):
                    if isinstance(value, (list, dict)):
                        print(f"  {key}: {type(value).__name__} ({len(value)} items)")
                    else:
                        # Truncate long values
                        str_value = str(value)
                        if len(str_value) > 100:
                            str_value = str_value[:100] + "..."
                        print(f"  {key}: {str_value}")

        if result.total_count > 5:
            print(f"\n... and {result.total_count - 5} more documents")
            print("💡 Use 'export' command to save all results")

        print(f"{'=' * 60}")


# ================================
# CONVENIENCE FUNCTIONS
# ================================


def quick_search(
    database: str,
    collection: str,
    field: Optional[str] = None,
    value: Any = None,
    limit: int = 50,
) -> SearchResult:
    """
    Quick search function for programmatic use

    Args:
        database: Database name ("main", "conversation", "memory", "email")
        collection: Collection name
        field: Optional field to filter by
        value: Optional value to filter for
        limit: Maximum results

    Returns:
        SearchResult object
    """
    tool = ManualSearchTool()

    try:
        db_type = DatabaseType(database.lower())
    except ValueError:
        raise ValueError(f"Invalid database: {database}")

    criteria = SearchCriteria(
        collection=collection, field=field, value=value, limit=limit
    )

    return tool.search_documents(db_type, criteria)


def list_all_collections(database: str) -> List[str]:
    """List all collections in a database"""
    tool = ManualSearchTool()

    try:
        db_type = DatabaseType(database.lower())
    except ValueError:
        raise ValueError(f"Invalid database: {database}")

    return tool.list_collections(db_type)


if __name__ == "__main__":
    # Run interactive CLI
    cli = ManualSearchCLI()
    cli.run()
