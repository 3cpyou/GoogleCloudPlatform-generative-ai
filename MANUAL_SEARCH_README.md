# AD3Gem Manual Database Search Tool

A comprehensive manual search tool for querying across all AD3Gem Firestore databases with both programmatic and interactive interfaces.

## Features

- 🔍 **Flexible Search**: Search across main, conversation, memory, and email databases
- 🏗️ **Multiple Interfaces**: Both programmatic API and interactive CLI
- 📊 **Advanced Filtering**: Support for complex queries with multiple filters
- 📈 **Performance Tracking**: Built-in metrics and search history
- 💾 **Export Capabilities**: Export results to JSON or CSV
- 📋 **Collection Discovery**: List and analyze database collections

## Quick Start

### Interactive CLI
```bash
# Activate the conda environment first
conda activate ad3gem

# Run the interactive search tool
python manual_search.py
```

### Programmatic Usage
```python
from manual_search import ManualSearchTool, quick_search

# Initialize the search tool
tool = ManualSearchTool()

# Quick search
result = quick_search("email", "reports", limit=10)

# Advanced search with filters
from manual_search import SearchCriteria, SearchOperator
criteria = SearchCriteria(
    collection="users",
    field="email",
    value="example@domain.com",
    operator=SearchOperator.EQUAL,
    limit=50
)
result = tool.search_documents("main", criteria)
```

## Database Structure

The tool searches across these databases:

| Database       | Purpose                   | Collections            |
| -------------- | ------------------------- | ---------------------- |
| `main`         | General application data  | users, projects        |
| `conversation` | Chat history and sessions | conversations, users   |
| `memory`       | Knowledge base and claims | heads, claims          |
| `email`        | Email data and reports    | reports, sample_emails |

## CLI Commands

### Basic Commands
- `search <db> <collection> [field] [value]` - Basic search in a collection
- `list <db>` - List all collections in a database
- `stats <db> <collection>` - Get collection statistics
- `export [json|csv] [filename]` - Export last search results
- `history` - Show search history

### Advanced Commands
- `advanced <db> <collection>` - Multi-filter search
- `help` - Show detailed help
- `quit` - Exit the tool

### Example CLI Usage
```bash
# Search email reports
search email reports

# Search users by email
search main users email user@example.com

# Advanced search with multiple filters
advanced memory heads
# Then enter filters like:
# created_at > 2024-01-01
# confidence > 0.8

# List collections
list main

# Get collection stats
stats email reports

# Export results
export json search_results.json
```

## Programmatic API

### SearchCriteria
```python
from manual_search import SearchCriteria, SearchOperator

criteria = SearchCriteria(
    collection="users",
    field="email",  # Optional: field to filter by
    value="user@example.com",  # Optional: value to match
    operator=SearchOperator.EQUAL,  # Filter operator
    limit=50,  # Maximum results
    order_by="created_at",  # Field to order by
    order_direction="DESCENDING"  # Sort direction
)
```

### Basic Search
```python
from manual_search import ManualSearchTool, DatabaseType

tool = ManualSearchTool()
result = tool.search_documents(DatabaseType.EMAIL, criteria)

print(f"Found {result.total_count} documents")
print(f"Execution time: {result.execution_time:.3f}s")

# Access results
for doc in result.documents:
    print(f"ID: {doc['_id']}")
    print(f"Data: {doc}")
```

### Advanced Search with Multiple Filters
```python
filters = [
    ("confidence", SearchOperator.GREATER_THAN, 0.7),
    ("created_at", SearchOperator.GREATER_THAN, "2024-01-01")
]

result = tool.advanced_search(
    DatabaseType.MEMORY,
    "heads",
    filters,
    limit=100,
    order_by="confidence"
)
```

### Quick Search Convenience Function
```python
from manual_search import quick_search

# Simple search without creating SearchCriteria
result = quick_search("email", "reports", field="processed_at", value="2024-01-01")
```

## Export Results

### CLI Export
```bash
# Export last search results as JSON
export json

# Export with custom filename
export json my_search_results.json

# Export as CSV
export csv results.csv
```

### Programmatic Export
```python
# Export search results
filename = tool.export_results(result, "json", "results.json")
print(f"Results exported to {filename}")
```

## Search Operators

| Operator         | Description          | Example                        |
| ---------------- | -------------------- | ------------------------------ |
| `==`             | Equal to             | `status == "active"`           |
| `>`              | Greater than         | `confidence > 0.8`             |
| `<`              | Less than            | `created_at < "2024-01-01"`    |
| `>=`             | Greater or equal     | `score >= 90`                  |
| `<=`             | Less or equal        | `count <= 10`                  |
| `array-contains` | Array contains value | `tags array-contains "urgent"` |

## Performance & Monitoring

### Search History
```python
# Get search history
history = tool.get_search_history()

for entry in history:
    print(f"{entry['database']}.{entry['collection']} -> {entry['result_count']} results ({entry['execution_time']:.3f}s)")
```

### Collection Statistics
```python
# Get collection stats
stats = tool.get_collection_stats(DatabaseType.EMAIL, "reports")
print(f"Collection: {stats['collection']}")
print(f"Sample size: {stats['sample_size']}")
print(f"Fields: {stats['sample_fields']}")
```

## Error Handling

The tool includes comprehensive error handling:

- Database connection failures
- Invalid search criteria
- Export failures
- Permission issues

All errors are logged with detailed information for debugging.

## Environment Setup

Make sure you have:

1. **Conda Environment**: `conda activate ad3gem`
2. **Environment Variables**:
   - `GOOGLE_CLOUD_PROJECT` - Your GCP project ID
   - `FIRESTORE_DATABASE` - Main database name
   - `FIRESTORE_CONVERSATION_DB` - Conversation database name
   - `FIRESTORE_MEMORY_DB` - Memory database name
   - `FIRESTORE_EMAIL_DB` - Email database name
3. **Permissions**: Proper Firestore access permissions

## Files

- `manual_search.py` - Main search tool and CLI
- `test_manual_search.py` - Comprehensive test suite
- `quick_search_example.py` - Simple usage examples
- `MANUAL_SEARCH_README.md` - This documentation

## Examples

### Search Recent Emails
```bash
python manual_search.py
# Then in CLI:
search email sample_emails
```

### Find High-Confidence Memory Claims
```python
result = quick_search("memory", "claims", field="confidence", value="0.9")
print(f"Found {result.total_count} high-confidence claims")
```

### Export User Data
```bash
# In CLI
search main users
export json users_backup.json
```

## Troubleshooting

**Common Issues:**

1. **Database Connection Failed**
   - Check environment variables
   - Verify GCP credentials
   - Ensure proper IAM permissions

2. **No Results Found**
   - Verify collection and field names
   - Check data exists in the database
   - Try broader search criteria

3. **Export Failed**
   - Check write permissions in current directory
   - Ensure sufficient disk space
   - Try different filename

For additional help, use the `help` command in the CLI or check the test files for examples.
