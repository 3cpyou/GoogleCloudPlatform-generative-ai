#!/usr/bin/env python3
"""
Test script for Manual Search Tool
================================

Demonstrates various search capabilities of the AD3Gem Manual Search Tool.
This script shows how to use the search functionality programmatically
and through the interactive CLI.
"""

import os
import sys
from datetime import datetime, timezone

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manual_search import (
    DatabaseType,
    ManualSearchTool,
    SearchCriteria,
    SearchOperator,
    list_all_collections,
    quick_search,
)


def test_basic_functionality():
    """Test basic search functionality"""
    print("\n" + "=" * 60)
    print("🧪 Testing Basic Search Functionality")
    print("=" * 60)

    try:
        tool = ManualSearchTool()

        # Test listing collections
        print("\n📂 Testing collection listing...")

        for db_type in DatabaseType:
            collections = tool.list_collections(db_type)
            print(f"  {db_type.value}: {len(collections)} collections found")
            if collections:
                print(f"    Sample: {collections[:3]}")

        # Test basic search on email database
        print("\n🔍 Testing basic search on email database...")
        criteria = SearchCriteria(collection="reports", limit=5)
        result = tool.search_documents(DatabaseType.EMAIL, criteria)

        print(f"  Found {result.total_count} reports")
        print(f"  Execution time: {result.execution_time:.3f}s")
        # Test search with filters if we have results
        if result.documents:
            print("\n🔍 Testing filtered search...")
            # Try to find sample emails
            email_criteria = SearchCriteria(collection="sample_emails", limit=3)
            email_result = tool.search_documents(DatabaseType.EMAIL, email_criteria)
            print(f"  Found {email_result.total_count} sample emails")
            print(f"  Execution time: {email_result.execution_time:.3f}s")
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")


def test_advanced_search():
    """Test advanced search with multiple filters"""
    print("\n" + "=" * 60)
    print("🧪 Testing Advanced Search Functionality")
    print("=" * 60)

    try:
        tool = ManualSearchTool()

        # Test advanced search on memory database
        print("\n🔍 Testing advanced search on memory database...")

        filters = [
            ("confidence", SearchOperator.GREATER_THAN, 0.5),
        ]

        result = tool.advanced_search(DatabaseType.MEMORY, "heads", filters, limit=10)

        print(f"  Found {result.total_count} memory heads with confidence > 0.5")
        print(f"  Execution time: {result.execution_time:.3f}s")
        if result.documents:
            print("  Sample results:")
            for i, doc in enumerate(result.documents[:3], 1):
                confidence = doc.get("confidence", "N/A")
                facet = doc.get("facet", "N/A")
                print(f"    {i}. {facet} (confidence: {confidence})")

    except Exception as e:
        print(f"❌ Advanced search test failed: {e}")


def test_convenience_functions():
    """Test convenience functions"""
    print("\n" + "=" * 60)
    print("🧪 Testing Convenience Functions")
    print("=" * 60)

    try:
        # Test quick search
        print("\n🔍 Testing quick_search function...")
        result = quick_search("email", "reports", limit=5)
        print(f"  Quick search found {result.total_count} reports")
        print(f"  Execution time: {result.execution_time:.3f}s")
        # Test list collections
        print("\n📂 Testing list_all_collections function...")
        collections = list_all_collections("main")
        print(f"  Main database has {len(collections)} collections")

    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")


def test_export_functionality():
    """Test data export functionality"""
    print("\n" + "=" * 60)
    print("🧪 Testing Export Functionality")
    print("=" * 60)

    try:
        tool = ManualSearchTool()

        # Perform a search to export
        print("\n🔍 Performing search for export test...")
        criteria = SearchCriteria(collection="reports", limit=3)
        result = tool.search_documents(DatabaseType.EMAIL, criteria)

        if result.total_count > 0:
            # Test JSON export
            print("\n💾 Testing JSON export...")
            json_file = tool.export_results(result, "json", "test_export.json")
            if json_file and os.path.exists(json_file):
                print(f"  ✅ JSON export successful: {json_file}")
                # Clean up
                os.remove(json_file)
            else:
                print("  ❌ JSON export failed")

            # Test CSV export
            print("\n💾 Testing CSV export...")
            csv_file = tool.export_results(result, "csv", "test_export.csv")
            if csv_file and os.path.exists(csv_file):
                print(f"  ✅ CSV export successful: {csv_file}")
                # Clean up
                os.remove(csv_file)
            else:
                print("  ❌ CSV export failed")
        else:
            print("  ⚠️ No data to export")

    except Exception as e:
        print(f"❌ Export functionality test failed: {e}")


def demonstrate_cli_usage():
    """Demonstrate CLI usage examples"""
    print("\n" + "=" * 60)
    print("📖 CLI Usage Examples")
    print("=" * 60)

    print("""
🔍 Example CLI Commands:

1. List collections in main database:
   list main

2. Search for documents in email reports:
   search email reports

3. Search emails by sender:
   search email sample_emails from julie

4. Advanced search with multiple filters:
   advanced memory heads
   (then enter filters like: confidence > 0.7)

5. Get collection statistics:
   stats email reports

6. Export last search results:
   export json search_results.json

7. View search history:
   history

8. Get help:
   help

💡 To run the interactive CLI:
   python manual_search.py
""")


def run_health_check():
    """Run a health check on all databases"""
    print("\n" + "=" * 60)
    print("🏥 Database Health Check")
    print("=" * 60)

    try:
        tool = ManualSearchTool()

        # Check search history functionality
        print("\n📊 Testing search history...")
        history = tool.get_search_history()
        print(f"  Search history has {len(history)} entries")

        # Test clearing history
        tool.clear_search_history()
        cleared_history = tool.get_search_history()
        print(f"  After clearing: {len(cleared_history)} entries")

        print("  ✅ Health check completed")

    except Exception as e:
        print(f"❌ Health check failed: {e}")


def main():
    """Run all tests"""
    print("🚀 Starting AD3Gem Manual Search Tool Tests")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python version: {sys.version}")

    # Run tests
    test_basic_functionality()
    test_advanced_search()
    test_convenience_functions()
    test_export_functionality()
    run_health_check()
    demonstrate_cli_usage()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\n💡 To use the interactive search tool:")
    print("   python manual_search.py")
    print("\n💡 To use programmatically:")
    print("   from manual_search import ManualSearchTool, quick_search")
    print("   tool = ManualSearchTool()")


if __name__ == "__main__":
    main()
