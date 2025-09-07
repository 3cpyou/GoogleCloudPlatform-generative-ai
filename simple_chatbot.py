"""
Enhanced AD3Gem Chatbot with Maximum Intelligence
Uses Gemini 1.5 Pro with advanced understanding and self-correction
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import google.generativeai as genai

from firestore_client import (
    FirestoreClient,
    append_to_conversation,
    get_memory_heads,
    get_recent_conversation,
    query_collection,
)


class EnhancedAD3GemChatbot:
    """
    Highly intelligent chatbot using Gemini 1.5 Pro with self-correction
    and advanced understanding capabilities.
    """

    def __init__(self, user_id: str = "default_user"):
        """Initialize the enhanced chatbot with maximum intelligence."""
        self.user_id = user_id
        self.conversation_id = f"chat_{user_id}_{datetime.now().strftime('%Y%m%d')}"

        # Configure Google Generative AI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        # Initialize Firestore client
        self.firestore_client = FirestoreClient()

        # Use Gemini 1.5 Pro for maximum intelligence
        self.model = genai.GenerativeModel(
            "gemini-1.5-pro",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
        )

        # Analytical model for structured extraction
        self.analytical_model = genai.GenerativeModel(
            "gemini-1.5-pro",
            generation_config={
                "temperature": 0.1,  # Low temperature for consistency
                "top_p": 0.95,
                "max_output_tokens": 1024,
            },
        )

        # Context memory for better understanding
        self.context_memory = {
            "last_query_type": None,
            "last_sender_mentioned": None,
            "last_time_period": None,
            "clarification_attempts": 0,
            "user_preferences": {},
        }

        print("🤖 Enhanced AD3Gem Chatbot initialized with Gemini 1.5 Pro")
        print(f"💬 Conversation ID: {self.conversation_id}")

    def _understand_query(self, user_message: str, context: List[Dict]) -> Dict:
        """
        Use Gemini to deeply understand the user's query with context.
        """
        context_str = ""
        if context:
            context_str = "Recent conversation:\n"
            for msg in context[-3:]:
                context_str += (
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}[:200]\n"
                )

        understanding_prompt = f"""
        You are an intelligent email system assistant. Analyze this query with extreme care.

        Context from conversation:
        {context_str}

        Previous query context:
        - Last sender mentioned: {self.context_memory.get("last_sender_mentioned", "none")}
        - Last time period: {self.context_memory.get("last_time_period", "none")}
        - Last query type: {self.context_memory.get("last_query_type", "none")}

        Current user query: "{user_message}"

        Provide a detailed analysis in JSON format:
        {{
            "query_type": "email_search/user_info/project_info/general_chat/unclear",
            "email_specific": {{
                "intent": "list/count/find_last/search_content/check_existence/get_details",
                "sender": "extracted sender name/email or null",
                "recipient": "extracted recipient or null",
                "keywords": ["relevant", "search", "terms"],
                "time_period": {{
                    "type": "specific/relative/none",
                    "value": "today/yesterday/last_week/last_month/custom",
                    "start_date": "ISO date or null",
                    "end_date": "ISO date or null"
                }},
                "classification_filter": "spam/not_spam/all",
                "refers_to_previous": true/false
            }},
            "confidence": 0-100,
            "needs_clarification": true/false,
            "clarification_reason": "why unclear if applicable",
            "suggested_clarification": "question to ask user",
            "interpretation": "natural language explanation of what user wants",
            "possible_typos": ["detected", "typos"],
            "corrected_query": "corrected version if typos detected"
        }}

        Be intelligent about understanding:
        - "Julie" could mean emails FROM Julie or ABOUT Julie
        - "yesterday's emails" means emails from yesterday
        - "that invoice" might refer to something mentioned earlier
        - "check Craig" probably means check emails from Craig
        - Common typos: "emial"→"email", "juile"→"julie", "criag"→"craig"

        Return ONLY valid JSON.
        """

        try:
            response = self.analytical_model.generate_content(understanding_prompt)
            # Clean response to ensure valid JSON
            json_str = response.text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            understanding = json.loads(json_str.strip())

            # Update context memory
            if understanding.get("email_specific", {}).get("sender"):
                self.context_memory["last_sender_mentioned"] = understanding[
                    "email_specific"
                ]["sender"]
            if (
                understanding.get("email_specific", {})
                .get("time_period", {})
                .get("value")
            ):
                self.context_memory["last_time_period"] = understanding[
                    "email_specific"
                ]["time_period"]["value"]
            self.context_memory["last_query_type"] = understanding.get("query_type")

            return understanding

        except Exception as e:
            print(f"Understanding error: {e}")
            # Return basic understanding as fallback
            return {
                "query_type": "email_search"
                if "email" in user_message.lower()
                else "unclear",
                "confidence": 30,
                "needs_clarification": True,
                "interpretation": user_message,
            }

    def _execute_email_query(self, understanding: Dict) -> tuple[List, str]:
        """
        Execute the email query based on understanding.
        Returns (emails, search_description)
        """
        email_spec = understanding.get("email_specific", {})
        sender = email_spec.get("sender")
        time_period = email_spec.get("time_period", {})
        intent = email_spec.get("intent", "list")

        # Build search parameters
        limit = 1 if intent == "find_last" else 20

        # Get emails based on parameters
        if sender:
            # Clean sender (could be partial name or full email)
            if "@" not in sender:
                # Try to match partial names
                possible_senders = [
                    f"{sender}@lineagecoffee.com",
                    f"{sender}@adddesigns.co.za",
                    sender,  # Try as-is too
                ]
                emails = []
                for possible_sender in possible_senders:
                    result = self.firestore_client.search_emails_by_sender(
                        possible_sender, limit=limit
                    )
                    if result:
                        emails.extend(result)
                        sender = possible_sender  # Update with found sender
                        break
                if not emails:
                    # Try partial match
                    emails = self.firestore_client.search_emails_by_sender(
                        sender, limit=limit
                    )
            else:
                emails = self.firestore_client.search_emails_by_sender(
                    sender, limit=limit
                )

            search_desc = f"emails from {sender}"
        else:
            emails = self.firestore_client.get_recent_emails(limit=limit)
            search_desc = "recent emails"

        # Filter by time period if specified
        if time_period.get("value") and time_period["value"] != "none":
            emails = self._filter_by_time(emails, time_period)
            search_desc += f" from {time_period['value']}"

        return emails, search_desc

    def _filter_by_time(self, emails: List, time_period: Dict) -> List:
        """Filter emails by time period."""
        if not emails or not time_period:
            return emails

        value = time_period.get("value")
        now = datetime.now()

        # Calculate cutoff time
        if value == "today":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif value == "yesterday":
            cutoff = (now - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end_cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif value == "last_week":
            cutoff = now - timedelta(days=7)
        elif value == "last_month":
            cutoff = now - timedelta(days=30)
        else:
            return emails

        # Filter emails
        filtered = []
        for email in emails:
            try:
                processed_at = email.get("processed_at", "")
                if processed_at:
                    email_time = datetime.fromisoformat(
                        processed_at.replace("Z", "+00:00")
                    )
                    if value == "yesterday":
                        if cutoff <= email_time < end_cutoff:
                            filtered.append(email)
                    elif email_time >= cutoff:
                        filtered.append(email)
            except:
                continue

        return filtered

    def _generate_intelligent_response(
        self, user_message: str, understanding: Dict, emails: List, search_desc: str
    ) -> str:
        """
        Generate an intelligent, natural response using Gemini.
        """
        # Format email data for prompt
        email_data = ""
        if emails:
            for i, email in enumerate(emails[:5], 1):
                from_addr = email.get("from", "Unknown")
                to_addr = email.get("to", "Unknown")
                subject = email.get("subject", "No subject")
                body_preview = email.get("body", "")[:200] if email.get("body") else ""
                processed_at = email.get("processed_at", "Unknown")
                confidence = email.get("confidence", 0)
                classification = email.get("classification", "unknown")

                # Format date
                try:
                    if processed_at != "Unknown":
                        dt = datetime.fromisoformat(processed_at.replace("Z", "+00:00"))
                        date_str = dt.strftime("%B %d at %I:%M %p")
                    else:
                        date_str = processed_at
                except:
                    date_str = processed_at

                email_data += f"""
                Email {i}:
                - From: {from_addr}
                - To: {to_addr}
                - Subject: {subject}
                - Date: {date_str}
                - Classification: {classification} (confidence: {confidence})
                - Preview: {body_preview}...
                """

        # Generate response
        response_prompt = f"""
        Generate a helpful, conversational response to the user's query.

        User asked: "{user_message}"

        What they want (our understanding): {understanding.get("interpretation", "Unknown")}
        Intent: {understanding.get("email_specific", {}).get("intent", "list")}

        Search performed: {search_desc}
        Results found: {len(emails)} emails

        Email data:
        {email_data if emails else "No emails found matching the criteria"}

        Instructions for response:
        1. Be conversational and natural
        2. Directly answer what they asked for
        3. If intent was "find_last", focus on the most recent email
        4. If intent was "count", emphasize the number
        5. If intent was "list", provide a summary
        6. If no results, suggest alternatives
        7. Mention if results might be incomplete
        8. Offer to help with follow-up questions
        9. If user made a typo, acknowledge it naturally without being condescending

        Corrected query (if typos detected): {understanding.get("corrected_query", "")}

        Keep response under 150 words unless showing a list.
        Be helpful and anticipate what they might want next.
        """

        response = self.model.generate_content(response_prompt)
        return response.text

    def _handle_clarification(self, understanding: Dict) -> str:
        """
        Handle cases where clarification is needed.
        """
        self.context_memory["clarification_attempts"] += 1

        if self.context_memory["clarification_attempts"] > 2:
            # Provide helpful examples after multiple attempts
            return (
                "I'm having trouble understanding. Here are some examples of what you can ask:\n\n"
                "📧 Email queries:\n"
                "  • 'Show emails from Julie'\n"
                "  • 'When did Craig last email?'\n"
                "  • 'Any emails about coffee orders'\n"
                "  • 'How many emails yesterday'\n\n"
                "👥 User queries:\n"
                "  • 'Show all users'\n"
                "  • 'Get user info for craig@lineagecoffee.com'\n\n"
                "📁 Project queries:\n"
                "  • 'List all projects'\n"
                "  • 'Show project details'\n\n"
                "What would you like to know?"
            )

        # Use suggested clarification or generate one
        if understanding.get("suggested_clarification"):
            return understanding["suggested_clarification"]

        # Generate intelligent clarification
        clarification_prompt = f"""
        The user's query is unclear. Generate a helpful clarification question.

        User said: "{understanding.get("interpretation", "unknown query")}"
        Confidence: {understanding.get("confidence", 0)}%
        Reason unclear: {understanding.get("clarification_reason", "ambiguous intent")}

        Generate a friendly, specific question to understand what they need.
        Keep it conversational and under 50 words.
        """

        response = self.model.generate_content(clarification_prompt)
        return response.text

    def send_message(self, user_message: str) -> str:
        """
        Main message processing with maximum intelligence.
        """
        print(f"\n💭 User: {user_message}")

        # Store user message
        append_to_conversation(self.conversation_id, "user", user_message)

        # Get conversation context
        context = get_recent_conversation(self.conversation_id, limit=5)

        # Understand the query deeply
        understanding = self._understand_query(user_message, context)
        print(f"🧠 Understanding confidence: {understanding.get('confidence', 0)}%")

        # Reset clarification attempts on confident understanding
        if understanding.get("confidence", 0) > 70:
            self.context_memory["clarification_attempts"] = 0

        # Handle based on query type and confidence
        query_type = understanding.get("query_type")

        if (
            understanding.get("needs_clarification")
            and understanding.get("confidence", 0) < 60
        ):
            response = self._handle_clarification(understanding)

        elif query_type == "email_search":
            # Execute email search
            emails, search_desc = self._execute_email_query(understanding)
            response = self._generate_intelligent_response(
                user_message, understanding, emails, search_desc
            )

        elif query_type == "user_info":
            response = self._handle_user_query(understanding)

        elif query_type == "project_info":
            response = self._handle_project_query(understanding)

        else:
            # General conversation with context
            response = self._handle_general_conversation(user_message, context)

        # Store response
        append_to_conversation(self.conversation_id, "assistant", response)
        print(f"🤖 Assistant: {response}")

        return response

    def _handle_user_query(self, understanding: Dict) -> str:
        """Handle user information queries intelligently."""
        users = query_collection("users", limit=20)

        if not users:
            return "No users found in the database. The system might be initializing."

        # Format user data
        user_list = []
        for user in users:
            name = user.get("name", user.get("display_name", "Unknown"))
            email = user.get("email", user.get("id", "No email"))
            role = user.get("role", "user")
            user_list.append(f"• {name} ({email}) - {role}")

        return f"Found {len(users)} users in the system:\n\n" + "\n".join(
            user_list[:10]
        )

    def _handle_project_query(self, understanding: Dict) -> str:
        """Handle project information queries intelligently."""
        projects = query_collection("projects", limit=20)

        if not projects:
            return "No projects found in the database."

        # Format project data
        project_list = []
        for project in projects:
            name = project.get("name", "Unknown")
            status = project.get("status", "Unknown")
            team_size = project.get("team_size", 0)
            project_list.append(f"• {name} - Status: {status}, Team: {team_size}")

        return f"Found {len(projects)} projects:\n\n" + "\n".join(project_list[:10])

    def _handle_general_conversation(self, user_message: str, context: List) -> str:
        """
        Handle general conversation with full context awareness.
        """
        # Build context
        context_str = self._assemble_context(user_message)

        # Generate response
        conversation_prompt = f"""
        You are AD3Gem, an intelligent assistant for email and data management.

        Context and memory:
        {context_str}

        User message: {user_message}

        Respond naturally and helpfully. If they're asking about capabilities, mention:
        - Email search and analysis
        - User and project information
        - Intelligent understanding of queries
        - Memory of conversation context

        Keep responses conversational and under 150 words.
        """

        response = self.model.generate_content(conversation_prompt)
        return response.text

    def _assemble_context(self, user_message: str) -> str:
        """Assemble comprehensive context."""
        context_parts = []

        # Recent conversation
        recent_messages = get_recent_conversation(self.conversation_id, limit=5)
        if recent_messages:
            context_parts.append("Recent conversation:")
            for msg in recent_messages[-3:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                context_parts.append(f"{role}: {content}")

        # Memory heads
        user_memory = get_memory_heads(scope="user")
        if user_memory:
            context_parts.append("\nUser memory:")
            for memory_id, memory_data in list(user_memory.items())[:5]:
                context_parts.append(f"- {memory_data.get('claim', '')}")

        # Context memory
        context_parts.append("\nRecent context:")
        context_parts.append(
            f"- Last sender: {self.context_memory.get('last_sender_mentioned', 'none')}"
        )
        context_parts.append(
            f"- Last time: {self.context_memory.get('last_time_period', 'none')}"
        )

        return "\n".join(context_parts)

    def start_interactive_session(self):
        """Start an enhanced interactive session."""
        print("\n" + "=" * 60)
        print("🚀 Enhanced AD3Gem Chatbot - Gemini 1.5 Pro")
        print("=" * 60)
        print(
            "I can understand natural language, correct typos, and learn from context."
        )
        print("Just talk naturally - I'll ask if I need clarification.")
        print("\nType 'quit', 'exit', or 'bye' to end")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n💭 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\n👋 Goodbye! Chat session ended.")
                    break

                if not user_input:
                    continue

                # Process the message
                response = self.send_message(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("I encountered an issue. Let me try to recover...")
                # Attempt recovery
                self.context_memory["clarification_attempts"] = 0


def main():
    """Main function to run the enhanced chatbot."""
    print("🔥 Initializing Enhanced AD3Gem Chatbot with Gemini 1.5 Pro...")

    # Get user ID
    user_id = os.getenv("USER_ID", "default_user")

    try:
        # Create and start chatbot
        chatbot = EnhancedAD3GemChatbot(user_id=user_id)
        chatbot.start_interactive_session()
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("Please check your GEMINI_API_KEY environment variable.")


if __name__ == "__main__":
    main()
