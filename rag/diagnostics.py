#!/usr/bin/env python3
"""
Comprehensive RAG Setup Diagnostic Script

This script runs a series of tests to validate the entire RAG pipeline,
from environment configuration to a full end-to-end query, to identify
the root cause of configuration, authentication, or indexing errors.

Usage:
  python /Users/craigcharity/dev/ad3gem/rag/diagnostics.py --db-read ad3gem-gmail-lineage
"""

import os
import sys
import argparse
from typing import Optional

# Suppress noisy gRPC logs
os.environ["GRPC_VERBOSITY"] = "ERROR"

import google.auth
from google.api_core.client_options import ClientOptions
from google.cloud import firestore
from google.cloud.firestore_admin_v1 import FirestoreAdminClient
from google.cloud.firestore_v1.base_client import BaseClient
from dotenv import load_dotenv

# --- Helper Functions & Classes ---

class Style:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def run_test(name, test_func, *args, **kwargs):
    """Utility to run a test function and print formatted output."""
    print(f"\n{Style.BLUE}Running Test: {Style.BOLD}{name}{Style.RESET}")
    print("-" * 50)
    try:
        result = test_func(*args, **kwargs)
        print(f"{Style.GREEN}✅ PASS: {name} completed successfully.{Style.RESET}")
        return result
    except Exception as e:
        print(f"{Style.RED}❌ FAIL: {name} failed.{Style.RESET}")
        print(f"{Style.YELLOW}  Error: {type(e).__name__}: {e}{Style.RESET}")
        print("-" * 50)
        sys.exit(1)

# --- Test Functions ---

def test_0_environment_variables():
    """Checks for required environment variables."""
    print("ℹ️  Loading environment variables from .env file...")
    load_dotenv()

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    print(f"  - GOOGLE_CLOUD_PROJECT: {project_id}")
    print(f"  - GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID is not set.")
    if not creds_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"Credential file not found at: {creds_path}")

    return {"project_id": project_id}

def test_1_gcp_authentication(project_id: str):
    """Verifies authentication with Google Cloud."""
    print(f"ℹ️  Authenticating with GCP for project '{project_id}'...")
    credentials, detected_project_id = google.auth.default()

    if not credentials:
        raise ConnectionError("Authentication failed. Could not get default credentials.")
    if detected_project_id != project_id:
        print(f"{Style.YELLOW}  Warning: Env project ID '{project_id}' differs from credential's project ID '{detected_project_id}'. Using env ID.{Style.RESET}")

    print("  - Successfully obtained GCP credentials.")
    return {"credentials": credentials}

def test_2_firestore_connection(project_id: str, database: str):
    """Connects to Firestore and performs a basic read operation."""
    print(f"ℹ️  Connecting to Firestore project '{project_id}', database '{database}'...")
    try:
        db = firestore.Client(project=project_id, database=database)
        # Test read
        db.collection("email_search_index").limit(1).get()
        print(f"  - Successfully connected and performed a test read.")
        return {"db_client": db}
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Firestore or read from 'email_search_index': {e}")


def test_3_firestore_index_inspection(project_id: str, database: str, collection: str):
    """Uses the Admin API to check the status of the required vector index."""
    print(f"ℹ️  Inspecting Firestore indexes for collection '{collection}' in database '{database}'...")

    # Format parent resource string correctly
    parent = f"projects/{project_id}/databases/{database}/collectionGroups/{collection}"

    try:
        # The admin client may need a specific endpoint if not using default regional instance.
        # For most cases, this is sufficient.
        admin_client = FirestoreAdminClient()
        indexes = admin_client.list_indexes(parent=parent)

        vector_index_found = False
        print("  - Found the following composite indexes:")
        for index in indexes:
            # Check if any field has vector configuration
            is_vector = False
            for field in index.fields:
                if hasattr(field, 'vector_config') and field.vector_config is not None:
                    is_vector = True
                    break
            state = index.state.name
            print(f"    - Index ID: {index.name.split('/')[-1]}, State: {state}, Is Vector: {is_vector}")

            if is_vector:
                vector_index_found = True
                if state != "READY":
                    raise ConnectionError(f"Vector index found but it is not READY. Current state: {state}. Please wait for it to finish building.")

        if not vector_index_found:
             raise FileNotFoundError(f"No vector index found for collection group '{collection}'. Please create it using the gcloud command.")

    except Exception as e:
        print(f"{Style.YELLOW}  Admin API Error: {e}{Style.RESET}")
        raise

def test_4_vertex_ai_initialization(project_id: str):
    """Initializes Vertex AI components (Embeddings and LLM)."""
    # Import here to avoid polluting global namespace and ensure other tests run first
    import vertexai
    from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI

    print(f"ℹ️  Initializing Vertex AI for project '{project_id}'...")
    vertexai.init(project=project_id, location="us-central1")
    print("  - Vertex AI SDK initialized.")

    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    print("  - Instantiated VertexAIEmbeddings.")

    llm = ChatVertexAI(model_name="chat-bison@001")
    print("  - Instantiated ChatVertexAI (PaLM).")

    return {"embeddings": embeddings, "llm": llm}

def test_5_vector_store_operation(db_client: BaseClient, embeddings, vector_collection: str):
    """Performs an isolated vector search operation."""
    from langchain_google_firestore import FirestoreVectorStore
    print(f"ℹ️  Performing isolated similarity search on '{vector_collection}'...")

    store = FirestoreVectorStore(
        client=db_client,
        collection=vector_collection,
        embedding_service=embeddings
    )
    print("  - Instantiated FirestoreVectorStore.")

    # This is the call that was previously failing
    results = store.similarity_search("test query", k=1)
    print(f"  - Similarity search returned {len(results)} document(s).")
    if results:
        print(f"    - Sample result: {results[0].page_content[:100]}...")


def test_6_full_rag_chain(llm, db_client: BaseClient, embeddings, vector_collection: str):
    """Runs an end-to-end query through the full RAG chain."""
    from langchain.chains import RetrievalQA
    from langchain_google_firestore import FirestoreVectorStore

    print(f"ℹ️  Running end-to-end RAG chain test...")

    store = FirestoreVectorStore(
        client=db_client,
        collection=vector_collection,
        embedding_service=embeddings
    )
    retriever = store.as_retriever(k=3)
    print("  - Created vector store retriever.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print("  - Assembled RetrievalQA chain.")

    result = qa_chain.invoke({"query": "What are the latest project updates?"})

    print(f"  - RAG chain executed successfully.")
    print(f"  - Answer: {result.get('result', 'N/A')}")
    print(f"  - Sources Found: {len(result.get('source_documents', []))}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Comprehensive RAG Setup Diagnostic Script")
    parser.add_argument("--db-read", required=True, help="Firestore database ID to read from (e.g., ad3gem-gmail-lineage)")
    parser.add_argument("--vector-collection", default="email_vectors", help="Name of the Firestore collection for vectors")
    args = parser.parse_args()

    print("\n" + "="*50)
    print(f"{Style.BOLD}Starting RAG Pipeline Diagnostics...{Style.RESET}")
    print("="*50)

    # --- Execute Tests Sequentially ---
    test0_result = run_test("Environment Variable Check", test_0_environment_variables)
    project_id = test0_result["project_id"]

    run_test("GCP Authentication", test_1_gcp_authentication, project_id=project_id)

    test2_result = run_test("Firestore Connection & Read", test_2_firestore_connection, project_id=project_id, database=args.db_read)
    db_client = test2_result["db_client"]

    run_test("Firestore Index Inspection", test_3_firestore_index_inspection, project_id=project_id, database=args.db_read, collection=args.vector_collection)

    test4_result = run_test("Vertex AI Initialization", test_4_vertex_ai_initialization, project_id=project_id)
    embeddings = test4_result["embeddings"]
    llm = test4_result["llm"]

    run_test("Vector Store Similarity Search", test_5_vector_store_operation, db_client=db_client, embeddings=embeddings, vector_collection=args.vector_collection)

    run_test("Full RAG Chain Execution", test_6_full_rag_chain, llm=llm, db_client=db_client, embeddings=embeddings, vector_collection=args.vector_collection)

    print("\n" + "="*50)
    print(f"{Style.GREEN}{Style.BOLD}🎉 All diagnostics passed! Your RAG setup appears to be correctly configured.{Style.RESET}")
    print("="*50)

if __name__ == "__main__":
    main()
