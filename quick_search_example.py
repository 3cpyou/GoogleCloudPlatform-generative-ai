#!/usr/bin/env python3
"""
Quick Search Example for AD3Gem Manual Search Tool
================================================

This script demonstrates how to use the manual search tool programmatically
to search across AD3Gem databases.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manual_search import DatabaseType, ManualSearchTool, SearchCriteria, quick_search


def main():
    """Demonstrate manual search usage"""
    print("🔍 AD3Gem Manual Search - Quick Example")
    print("=" * 50)

    try:
        # Initialize the search tool
        tool = ManualSearchTool()
        print("✅ Search tool initialized")

        # Example 1: Search email reports
        print("\n📧 Example 1: Search email reports")
        criteria = SearchCriteria(collection="reports", limit=5)
        result = tool.search_documents(DatabaseType.EMAIL, criteria)
        print(f"Found {result.total_count} email reports")
        print(f"Execution time: {result.execution_time:.3f}s")
        # Example 2: Quick search convenience function
        print("\n🚀 Example 2: Quick search for users")
        result = quick_search("main", "users", limit=3)
        print(f"Found {result.total_count} users")
        print(f"Execution time: {result.execution_time:.3f}s")
        # Example 3: List all collections
        print("\n📂 Example 3: List collections in conversation database")
        collections = tool.list_collections(DatabaseType.CONVERSATION)
        print(f"Conversation database collections: {collections}")

        # Example 4: Get collection stats
        print("\n📊 Example 4: Get collection statistics")
        if collections:
            stats = tool.get_collection_stats(DatabaseType.CONVERSATION, collections[0])
            print(f"Stats for {collections[0]}: {stats}")

        print("\n✅ All examples completed successfully!")
        print("\n💡 For interactive searching, run: python manual_search.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Make sure:")
        print("1. You're in the ad3gem conda environment")
        print("2. Environment variables are set")
        print("3. Firestore databases are accessible")


if __name__ == "__main__":
    main()
