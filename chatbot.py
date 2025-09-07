"""
AD3Gem Chatbot - Main Interface

This is the main chatbot interface that integrates:
- Gemini models via Vertex AI
- Function calling for Firestore operations
- Conversation memory and context assembly
- Multi-database support (main, conversation, memory)

Based on patterns from the ad3gem repository and the ad3gem-CI.md PRD.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import vertexai
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)

from firestore_client import (
    FirestoreClient,
    add_document,
    append_to_conversation,
    delete_document,
    get_document,
    get_memory_heads,
    get_recent_conversation,
    query_collection,
    store_memory_head,
    update_document,
)


class AD3GemChatbot:
    """
    Main chatbot class that handles conversation flow, context assembly,
    and interaction with Gemini models.
    """

    def __init__(self, user_id: str = "default_user"):
        """Initialize the chatbot with Gemini and Firestore connections."""
        self.user_id = user_id
        self.conversation_id = f"chat_{user_id}_{datetime.now().strftime('%Y%m%d')}"

        # Initialize Vertex AI
        project_id = os.getenv("PROJECT_ID", "ad3-sam")
        region = os.getenv("REGION", "us-central1")
        vertexai.init(project=project_id, location=region)

        # Initialize Firestore client
        self.firestore_client = FirestoreClient()

        # Define function declarations for Gemini
        self.tools = self._create_tools()

        # Initialize Gemini model
        self.model = GenerativeModel(
            model_name="gemini-1.5-flash-001",
            tools=self.tools,
            system_instruction=self._get_system_instruction(),
        )

        # Start chat session
        self.chat = self.model.start_chat()

        print(f"🤖 AD3Gem Chatbot initialized for user: {user_id}")
        print(f"💬 Conversation ID: {self.conversation_id}")

    def _get_system_instruction(self) -> str:
        """Get the system instruction for the Gemini model."""
        return """You are AD3Gem, an intelligent assistant that can manage data in Firestore databases.

Your capabilities:
- Read, create, update, delete, and query documents in the main database
- Access conversation history to maintain context
- Store and retrieve memory/knowledge for personalized responses
- Provide helpful, contextual responses based on stored data

Key behaviors:
- Always check conversation history and memory before responding
- Use function calls to interact with the databases when needed
- Maintain conversation context across messages
- Be helpful, accurate, and conversational
- When asked about data, use the appropriate database functions
- Store important facts or preferences as memory heads for future reference

You have access to three databases:
1. Main database (ad3gem-database): General application data
2. Conversation database (ad3gem-conversation): Chat history
3. Memory database (ad3gem-memory): Persistent knowledge and facts
"""

    def _create_tools(self) -> List[Tool]:
        """Create function declarations for Gemini to call."""

        # Main database operations
        get_document_func = FunctionDeclaration(
            name="get_document",
            description="Get a specific document from the main database",
            parameters={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "The collection name (e.g., 'users', 'products')",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "The document ID to retrieve",
                    },
                },
                "required": ["collection", "document_id"],
            },
        )

        add_document_func = FunctionDeclaration(
            name="add_document",
            description="Add a new document to the main database",
            parameters={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "The collection name",
                    },
                    "data": {
                        "type": "object",
                        "description": "The document data as a JSON object",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Optional specific document ID",
                    },
                },
                "required": ["collection", "data"],
            },
        )

        update_document_func = FunctionDeclaration(
            name="update_document",
            description="Update an existing document in the main database",
            parameters={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "The collection name",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "The document ID to update",
                    },
                    "data": {
                        "type": "object",
                        "description": "The update data as a JSON object",
                    },
                },
                "required": ["collection", "document_id", "data"],
            },
        )

        delete_document_func = FunctionDeclaration(
            name="delete_document",
            description="Delete a document from the main database",
            parameters={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "The collection name",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "The document ID to delete",
                    },
                },
                "required": ["collection", "document_id"],
            },
        )

        query_collection_func = FunctionDeclaration(
            name="query_collection",
            description="Query documents from the main database",
            parameters={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "The collection name",
                    },
                    "field": {
                        "type": "string",
                        "description": "Optional field name to filter by",
                    },
                    "value": {
                        "type": "string",
                        "description": "Optional value to filter for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                    },
                },
                "required": ["collection"],
            },
        )

        # Memory operations
        store_memory_func = FunctionDeclaration(
            name="store_memory",
            description="Store a fact or preference in memory for future reference",
            parameters={
                "type": "object",
                "properties": {
                    "facet": {
                        "type": "string",
                        "description": "The category/topic (e.g., 'preferences', 'facts')",
                    },
                    "scope": {
                        "type": "string",
                        "description": "The scope (e.g., 'user', 'general')",
                    },
                    "claim": {
                        "type": "string",
                        "description": "The fact or preference to remember",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level (0.0-1.0, default: 1.0)",
                    },
                },
                "required": ["facet", "scope", "claim"],
            },
        )

        get_memory_func = FunctionDeclaration(
            name="get_memory",
            description="Retrieve stored facts or preferences from memory",
            parameters={
                "type": "object",
                "properties": {
                    "facet": {
                        "type": "string",
                        "description": "Optional facet to filter by",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Optional scope to filter by",
                    },
                },
                "required": [],
            },
        )

        # Group all functions into a tool
        firestore_tool = Tool(
            function_declarations=[
                get_document_func,
                add_document_func,
                update_document_func,
                delete_document_func,
                query_collection_func,
                store_memory_func,
                get_memory_func,
            ]
        )

        return [firestore_tool]

    def _assemble_context(self, user_message: str) -> str:
        """
        Assemble context from conversation history and memory before processing.
        This implements the context assembly pattern from the PRD.
        """
        context_parts = []

        # Get recent conversation history
        recent_messages = get_recent_conversation(self.conversation_id, limit=5)
        if recent_messages:
            context_parts.append("## Recent Conversation:")
            for msg in recent_messages[-3:]:  # Last 3 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_parts.append(f"{role}: {content}")

        # Get relevant memory
        user_memory = get_memory_heads(scope="user")
        general_memory = get_memory_heads(scope="general")

        if user_memory or general_memory:
            context_parts.append("\n## Stored Memory:")

        if user_memory:
            context_parts.append("User-specific:")
            for memory_id, memory_data in user_memory.items():
                claim = memory_data.get("claim", "")
                context_parts.append(f"- {claim}")

        if general_memory:
            context_parts.append("General knowledge:")
            for memory_id, memory_data in general_memory.items():
                claim = memory_data.get("claim", "")
                context_parts.append(f"- {claim}")

        # Add current user message
        context_parts.append(f"\n## Current User Message:\n{user_message}")

        return "\n".join(context_parts)

    def _handle_function_call(self, function_call) -> Dict[str, Any]:
        """Handle function calls from Gemini."""
        function_name = function_call.name
        function_args = dict(function_call.args)

        print(f"🔧 Calling function: {function_name}")
        print(f"📝 Arguments: {json.dumps(function_args, indent=2)}")

        try:
            # Map function names to actual functions
            if function_name == "get_document":
                result = get_document(
                    function_args["collection"], function_args["document_id"]
                )
            elif function_name == "add_document":
                result = add_document(
                    function_args["collection"],
                    function_args["data"],
                    function_args.get("document_id"),
                )
            elif function_name == "update_document":
                result = update_document(
                    function_args["collection"],
                    function_args["document_id"],
                    function_args["data"],
                )
            elif function_name == "delete_document":
                result = delete_document(
                    function_args["collection"], function_args["document_id"]
                )
            elif function_name == "query_collection":
                result = query_collection(
                    function_args["collection"],
                    function_args.get("field"),
                    function_args.get("value"),
                    function_args.get("limit", 10),
                )
            elif function_name == "store_memory":
                result = store_memory_head(
                    function_args["facet"],
                    function_args["scope"],
                    function_args["claim"],
                    function_args.get("confidence", 1.0),
                )
            elif function_name == "get_memory":
                result = get_memory_heads(
                    function_args.get("facet"), function_args.get("scope")
                )
            else:
                result = {"error": f"Unknown function: {function_name}"}

            print(f"✅ Function result: {result}")
            return {"result": result}

        except Exception as e:
            error_msg = f"Error executing {function_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

    def send_message(self, user_message: str) -> str:
        """
        Send a message to the chatbot and get a response.
        This implements the full conversation flow with context assembly.
        """
        print(f"\n💭 User: {user_message}")

        # Store user message in conversation history
        append_to_conversation(self.conversation_id, "user", user_message)

        # Assemble context from history and memory
        context = self._assemble_context(user_message)

        print(f"🧠 Assembled context length: {len(context)} characters")

        # Send to Gemini
        response = self.chat.send_message(context)

        # Check if the response contains function calls
        if (
            response.candidates
            and response.candidates[0].content.parts
            and response.candidates[0].content.parts[0].function_call
        ):
            print("🔄 Processing function calls...")

            # Handle all function calls in the response
            function_responses = []
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    func_result = self._handle_function_call(part.function_call)
                    function_responses.append(
                        Part.from_function_response(
                            name=part.function_call.name, response=func_result
                        )
                    )

            # Send function results back to get the final response
            if function_responses:
                final_response = self.chat.send_message(function_responses)
                bot_message = final_response.text
            else:
                bot_message = "I encountered an issue processing your request."
        else:
            # Direct text response
            bot_message = response.text

        # Store bot response in conversation history
        append_to_conversation(self.conversation_id, "assistant", bot_message)

        print(f"🤖 Assistant: {bot_message}")
        return bot_message

    def start_interactive_session(self):
        """Start an interactive chat session."""
        print("\n" + "=" * 60)
        print("🚀 AD3Gem Chatbot - Interactive Session")
        print("=" * 60)
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'help' for available commands")
        print("Type 'stats' to see database statistics")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n💭 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\n👋 Goodbye! Chat session ended.")
                    break

                if user_input.lower() == "help":
                    self._show_help()
                    continue

                if user_input.lower() == "stats":
                    self._show_stats()
                    continue

                if not user_input:
                    continue

                # Process the message
                response = self.send_message(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")

    def _show_help(self):
        """Show help information."""
        help_text = """
🔧 Available Commands:
- 'help' - Show this help message
- 'stats' - Show database statistics
- 'quit'/'exit'/'bye' - End the session

💡 Example queries:
- "Add a new user named John with email john@example.com"
- "Get user123 from the users collection"
- "Show me all users"
- "Remember that I prefer coffee over tea"
- "What do you remember about my preferences?"
- "Update user123's age to 25"
- "Delete the product with ID prod456"

🗃️ Available Collections:
- users (sample data available)
- Any custom collections you create

🧠 Memory Features:
- Store personal preferences and facts
- Retrieve context from previous conversations
- Learn from your interactions
"""
        print(help_text)

    def _show_stats(self):
        """Show database statistics."""
        print("\n📊 Database Statistics:")
        stats = self.firestore_client.get_database_stats()
        print(json.dumps(stats, indent=2))


def main():
    """Main function to run the chatbot."""
    print("🔥 Initializing AD3Gem Chatbot...")

    # Get user ID (could be from command line args, config, etc.)
    user_id = os.getenv("USER_ID", "default_user")

    # Create and start chatbot
    chatbot = AD3GemChatbot(user_id=user_id)
    chatbot.start_interactive_session()


if __name__ == "__main__":
    main()
