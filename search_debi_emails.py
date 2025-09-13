#!/usr/bin/env python3
"""
AD3Gem Email Search Script - Search for "debi" in emails
===================================================

This script searches through the ad3gem-emails database for any word containing "debi".
It searches through all email threads and their messages, looking for matches in:
- Email body content
- Subject lines
- Sender/recipient names

Results are displayed with context and can be exported to JSON/CSV.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.cloud import firestore


class DebiEmailSearch:
    """Search tool for finding emails containing "debi" in the AD3Gem email database."""

    def __init__(self):
        """Initialize the search tool with database connection."""
        self.project_id = os.getenv("PROJECT_ID", "ad3-sam")

        # Initialize email database connection
        self.email_db = firestore.Client(
            project=self.project_id,
            database=os.getenv("FIRESTORE_EMAIL_DB", "ad3sam-email"),
        )

        # Setup logging
        self.logger = logging.getLogger("DebiEmailSearch")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.search_results = []

    def search_emails_for_debi(
        self, search_term: str = "debi", limit: int = 50, case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search through all email threads and messages for the specified term.

        Args:
            search_term: The term to search for (default: "debi")
            limit: Maximum number of results to return
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of matching email messages with context
        """
        self.logger.info(f"🔍 Starting search for '{search_term}' in email database...")

        results = []
        flags = 0 if case_sensitive else re.IGNORECASE
        search_pattern = re.compile(re.escape(search_term), flags)

        try:
            # Get all thread documents
            threads_ref = self.email_db.collection("threads")
            threads = threads_ref.limit(1000).stream()  # Limit for performance

            thread_count = 0
            message_count = 0

            for thread_doc in threads:
                thread_count += 1
                thread_data = thread_doc.to_dict()
                thread_id = thread_doc.id

                # Get messages in this thread
                messages_ref = threads_ref.document(thread_id).collection("messages")
                messages = messages_ref.limit(100).stream()  # Limit messages per thread

                for message_doc in messages:
                    message_count += 1
                    message_data = message_doc.to_dict()
                    message_id = message_doc.id

                    # Search in various fields
                    matches = self._find_matches_in_message(
                        message_data, search_pattern, search_term
                    )

                    if matches:
                        # Create result entry
                        result = {
                            "thread_id": thread_id,
                            "message_id": message_id,
                            "matches": matches,
                            "email_metadata": {
                                "subject": message_data.get("subject", ""),
                                "from": message_data.get("from", ""),
                                "to": message_data.get("to", []),
                                "sent_at": message_data.get("sentAt", ""),
                                "has_full_body": message_data.get("hasFullBody", False),
                            },
                            "thread_metadata": {
                                "participants": thread_data.get("participants", []),
                                "subject": thread_data.get("subject", ""),
                                "message_count": thread_data.get("messageCount", 0),
                                "last_message_at": thread_data.get("lastMessageAt", ""),
                            },
                            "search_timestamp": datetime.now(timezone.utc).isoformat(),
                        }

                        results.append(result)

                        if len(results) >= limit:
                            break

                if len(results) >= limit:
                    break

            self.logger.info(
                f"📊 Search completed: {thread_count} threads, "
                f"{message_count} messages searched, {len(results)} matches found"
            )

        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")

        self.search_results = results
        return results

    def _find_matches_in_message(
        self, message_data: Dict[str, Any], pattern: re.Pattern, search_term: str
    ) -> List[Dict[str, Any]]:
        """
        Find all matches of the search term in a message.

        Args:
            message_data: The message document data
            pattern: Compiled regex pattern for searching
            search_term: The original search term

        Returns:
            List of match details
        """
        matches = []

        # Search in email body
        body = message_data.get("body", "")
        if body:
            body_matches = self._find_matches_in_text(
                body, pattern, search_term, "body"
            )
            matches.extend(body_matches)

        # Search in subject
        subject = message_data.get("subject", "")
        if subject:
            subject_matches = self._find_matches_in_text(
                subject, pattern, search_term, "subject"
            )
            matches.extend(subject_matches)

        # Search in sender/recipient names
        from_field = message_data.get("from", {})
        if isinstance(from_field, dict):
            from_name = from_field.get("name", "")
            if from_name:
                name_matches = self._find_matches_in_text(
                    from_name, pattern, search_term, "sender_name"
                )
                matches.extend(name_matches)

        to_field = message_data.get("to", [])
        for recipient in to_field:
            if isinstance(recipient, dict):
                recipient_name = recipient.get("name", "")
                if recipient_name:
                    name_matches = self._find_matches_in_text(
                        recipient_name, pattern, search_term, "recipient_name"
                    )
                    matches.extend(name_matches)

        return matches

    def _find_matches_in_text(
        self, text: str, pattern: re.Pattern, search_term: str, field_type: str
    ) -> List[Dict[str, Any]]:
        """
        Find all matches in a text string.

        Args:
            text: The text to search in
            pattern: Compiled regex pattern
            search_term: Original search term
            field_type: Type of field being searched

        Returns:
            List of match details
        """
        matches = []

        # Find all matches with context
        for match in pattern.finditer(text):
            start_pos = max(0, match.start() - 50)
            end_pos = min(len(text), match.end() + 50)

            context = text[start_pos:end_pos]
            if start_pos > 0:
                context = "..." + context
            if end_pos < len(text):
                context = context + "..."

            match_detail = {
                "field": field_type,
                "matched_text": match.group(),
                "context": context,
                "position": match.start(),
                "search_term": search_term,
            }

            matches.append(match_detail)

        return matches

    def display_results(self, results: List[Dict[str, Any]], max_results: int = 10):
        """Display search results in a formatted way."""
        print("\n" + "=" * 80)
        print("🔍 AD3Gem Email Search Results")
        print("=" * 80)
        print(f"📊 Found {len(results)} matching emails")
        print(f"⏱️  Search completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        if not results:
            print("❌ No emails found containing 'debi'")
            return

        # Display results
        for i, result in enumerate(results[:max_results], 1):
            print(f"\n📄 Result {i}")
            print("-" * 60)

            # Email metadata
            email_meta = result["email_metadata"]
            print(f"📧 Subject: {email_meta['subject']}")
            print(f"👤 From: {email_meta['from']}")
            print(f"📅 Sent: {email_meta['sent_at']}")

            # Thread info
            thread_meta = result["thread_metadata"]
            print(
                f"👥 Thread participants: {', '.join(thread_meta['participants'][:3])}"
            )
            if len(thread_meta["participants"]) > 3:
                print(f"    ... and {len(thread_meta['participants']) - 3} more")

            # Matches
            matches = result["matches"]
            print(f"🎯 Found {len(matches)} match(es):")

            for j, match in enumerate(matches, 1):
                print(f"  {j}. {match['field'].upper()}: ...{match['context']}...")
                print(f"     → Matched: '{match['matched_text']}'")

        if len(results) > max_results:
            print(f"\n... and {len(results) - max_results} more results")
            print("💡 Use export_results() to save all results")

        print("=" * 80)

    def export_results(
        self,
        results: List[Dict[str, Any]],
        format_type: str = "json",
        filename: Optional[str] = None,
    ) -> str:
        """
        Export search results to file.

        Args:
            results: Search results to export
            format_type: "json" or "csv"
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debi_search_results_{timestamp}.{format_type}"

        try:
            if format_type == "json":
                # Prepare data for JSON export
                export_data = {
                    "search_metadata": {
                        "search_term": "debi",
                        "total_results": len(results),
                        "export_timestamp": datetime.now(timezone.utc).isoformat(),
                        "database": "ad3sam-email",
                    },
                    "results": results,
                }

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

            elif format_type == "csv":
                import csv

                if results:
                    # Flatten the data for CSV
                    flattened_results = []
                    for result in results:
                        base_row = {
                            "thread_id": result["thread_id"],
                            "message_id": result["message_id"],
                            "subject": result["email_metadata"]["subject"],
                            "from": str(result["email_metadata"]["from"]),
                            "sent_at": result["email_metadata"]["sent_at"],
                            "participants": ", ".join(
                                result["thread_metadata"]["participants"]
                            ),
                            "match_count": len(result["matches"]),
                        }

                        # Add first match details if available
                        if result["matches"]:
                            first_match = result["matches"][0]
                            base_row["first_match_field"] = first_match["field"]
                            base_row["first_match_text"] = first_match["matched_text"]
                            base_row["first_match_context"] = first_match["context"]

                        flattened_results.append(base_row)

                    # Write CSV
                    fieldnames = flattened_results[0].keys()
                    with open(filename, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(flattened_results)

            self.logger.info(f"✅ Results exported to {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}")
            return ""

    def get_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics about the search results."""
        if not results:
            return {"total_results": 0}

        stats = {
            "total_results": len(results),
            "unique_threads": len(set(r["thread_id"] for r in results)),
            "field_distribution": {},
            "participant_frequency": {},
            "temporal_distribution": {},
        }

        # Field distribution
        for result in results:
            for match in result["matches"]:
                field = match["field"]
                stats["field_distribution"][field] = (
                    stats["field_distribution"].get(field, 0) + 1
                )

        # Participant frequency
        for result in results:
            for participant in result["thread_metadata"]["participants"]:
                stats["participant_frequency"][participant] = (
                    stats["participant_frequency"].get(participant, 0) + 1
                )

        # Sort by frequency
        stats["participant_frequency"] = dict(
            sorted(
                stats["participant_frequency"].items(), key=lambda x: x[1], reverse=True
            )
        )

        return stats


def main():
    """Main function to run the search."""
    print("🔥 AD3Gem Email Search for 'debi'")
    print("=" * 50)

    # Initialize search tool
    search_tool = DebiEmailSearch()

    # Perform search
    print("🔍 Searching for 'debi' in all emails...")
    results = search_tool.search_emails_for_debi(search_term="debi", limit=100)

    # Display results
    search_tool.display_results(results, max_results=20)

    # Show summary stats
    if results:
        stats = search_tool.get_summary_stats(results)
        print("\n📊 Summary Statistics:")
        print(f"   Total matches: {stats['total_results']}")
        print(f"   Unique threads: {stats['unique_threads']}")
        print(f"   Field distribution: {stats['field_distribution']}")

        print("\n👥 Top participants:")
        for participant, count in list(stats["participant_frequency"].items())[:5]:
            print(f"   {participant}: {count} matches")

        # Export results
        print("\n💾 Exporting results...")
        json_file = search_tool.export_results(results, "json")
        csv_file = search_tool.export_results(results, "csv")

        if json_file:
            print(f"   📄 JSON export: {json_file}")
        if csv_file:
            print(f"   📊 CSV export: {csv_file}")

    print("\n✅ Search completed!")


if __name__ == "__main__":
    main()
