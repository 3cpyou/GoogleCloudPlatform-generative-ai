#!/usr/bin/env python3
"""
Firestore → LangChain RAG integration using Vertex AI

- Loads email docs from existing Firestore collection `email_search_index`
- Builds/updates a Firestore vector store (`email_vectors` by default)
- Exposes a simple `ask()` function powered by LangChain + Gemini

Usage examples:
  # Index (subject + snippet + bodyPreview)
  python /Users/craigcharity/dev/ad3gem/rag/firestore_rag.py --index --limit 2000

  # Ask a question
  python /Users/craigcharity/dev/ad3gem/rag/firestore_rag.py --ask "Summarize last week's deadlines"
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import vertexai
from google.cloud import firestore

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_firestore import FirestoreVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def initialize_gcp(database: Optional[str] = None) -> firestore.Client:
    """Load env, init Vertex AI, and return a Firestore client.

    If `database` is provided (or env GOOGLE_CLOUD_FIRESTORE_DB is set), the client will
    connect to that database; otherwise uses the default.
    """
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    database = database or os.getenv("GOOGLE_CLOUD_FIRESTORE_DB")

    if not project_id:
        print("Missing GOOGLE_CLOUD_PROJECT environment variable", file=sys.stderr)
        sys.exit(1)

    vertexai.init(project=project_id, location=region)
    if database:
        return firestore.Client(project=project_id, database=database)
    return firestore.Client(project=project_id)


def _compose_page_content(doc: Dict[str, Any]) -> str:
    """Create a retrieval-friendly text block from email_search_index doc."""
    subject = (doc.get("subject") or "").strip()
    sender = (doc.get("fromEmailLc") or "").strip()
    date = (doc.get("sentDate") or "").strip()
    snippet = (doc.get("snippet") or "").strip()
    body_preview = (doc.get("bodyPreview") or "").strip()

    lines: List[str] = []
    if subject:
        lines.append(f"Subject: {subject}")
    if sender:
        lines.append(f"From: {sender}")
    if date:
        lines.append(f"Date: {date}")
    if snippet:
        lines.append(f"Snippet: {snippet}")
    if body_preview and body_preview != snippet:
        lines.append(f"Body: {body_preview}")

    return "\n".join(lines).strip()


def load_documents_from_search_index(
    db: firestore.Client,
    collection_name: str = "email_search_index",
    limit: Optional[int] = None,
) -> List[Document]:
    """Read Firestore docs and convert to LangChain Documents.

    Uses subject + snippet + bodyPreview as page content and preserves useful metadata.
    """
    coll = db.collection(collection_name)
    stream = coll.stream()

    documents: List[Document] = []
    processed_count = 0

    print(f"🔍 Loading documents from '{collection_name}'...")

    for i, snap in enumerate(stream, start=1):
        data = snap.to_dict() or {}

        page_content = _compose_page_content(data)
        if not page_content:
            continue

        # Show progress every 100 documents
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"   📄 Processed {processed_count} documents...")

        if limit and processed_count >= limit:
            print(f"   ⏹️  Reached limit of {limit} documents")
            break

        metadata = {
            "messageId": data.get("messageId"),
            "threadId": data.get("threadId"),
            "fromEmail": data.get("fromEmailLc"),
            "sentDate": data.get("sentDate"),
            "labelNames": data.get("labelNames", [])[:10],
            "category": data.get("category"),
            "hasAttachments": bool(data.get("hasAttachments")),
            "attachmentTypes": data.get("attachmentTypes", [])[:5],
            "collection": collection_name,
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

        if limit and i >= limit:
            break

    print(f"✅ Loaded {len(documents)} valid documents from {processed_count} emails")
    return documents


def index_emails(
    collection_src: str = "email_search_index",
    vector_collection: str = "email_vectors",
    limit: Optional[int] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    database_read: Optional[str] = None,
) -> int:
    """Create/update Firestore vector store from existing search index."""
    db = initialize_gcp(database=database_read)

    # Load and prepare documents
    raw_docs = load_documents_from_search_index(db, collection_name=collection_src, limit=limit)
    if not raw_docs:
        print("No documents found to index.")
        return 0

    # Chunk the documents for better retrieval
    print(f"📝 Splitting {len(raw_docs)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"   Created {len(chunks)} chunks (avg {len(chunks)/len(raw_docs):.1f} chunks per document)")

    # Create vector store in Firestore
    print(f"🧠 Initializing Vertex AI embeddings...")
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    print(f"🚀 Indexing {len(chunks)} chunks into Firestore collection '{vector_collection}'...")
    print(f"   This may take several minutes for large batches...")

    FirestoreVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=vector_collection,
        client=db,
    )

    print(f"✅ Successfully indexed {len(chunks)} chunks from {len(raw_docs)} emails into '{vector_collection}'!")
    return len(chunks)


def _rag_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=(
            "You are an assistant answering questions about emails.\n"
            "Use only the provided context. If the answer is not present, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        input_variables=["context", "question"],
    )


def ask(
    question: str,
    vector_collection: str = "email_vectors",
    model_name: str = "gpt-3.5-turbo",
    k: int = 5,
    temperature: float = 0.7,
    database_read: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask a question over the Firestore vector store and return answer + sources."""
    initialize_gcp(database=database_read)

    llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=2048)
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    # The constructor requires `embedding_service` for query-time embedding.
    # Also need to pass the database parameter if it's specified
    db = initialize_gcp(database=database_read)
    store = FirestoreVectorStore(
        client=db,
        collection=vector_collection,
        embedding_service=embeddings
    )

    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": _rag_prompt()},
        return_source_documents=True,
    )

    result = qa.invoke({"query": question})
    # Normalize sources for readability
    sources = []
    for d in result.get("source_documents", []) or []:
        meta = d.metadata or {}
        sources.append({
            "messageId": meta.get("messageId"),
            "threadId": meta.get("threadId"),
            "fromEmail": meta.get("fromEmail"),
            "sentDate": meta.get("sentDate"),
            "preview": d.page_content[:200],
        })

    return {"answer": result.get("result"), "sources": sources}


def main():
    parser = argparse.ArgumentParser(description="Firestore → LangChain RAG integration")
    parser.add_argument("--index", action="store_true", help="Index emails into Firestore vector store")
    parser.add_argument("--ask", type=str, help="Ask a question against the vector store")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of emails to index")
    parser.add_argument("--src", type=str, default="email_search_index", help="Source collection name")
    parser.add_argument("--dst", type=str, default="email_vectors", help="Vector collection name")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents for retrieval")
    parser.add_argument("--db-read", type=str, default=None, help="Firestore database id to READ from (emails)")
    args = parser.parse_args()

    if args.index:
        index_emails(
            collection_src=args.src,
            vector_collection=args.dst,
            limit=args.limit,
            database_read=args.db_read,
        )

    if args.ask:
        res = ask(args.ask, vector_collection=args.dst, k=args.k, database_read=args.db_read)
        print("\nAnswer:\n" + (res.get("answer") or ""))
        print("\nSources:")
        for i, s in enumerate(res.get("sources") or [], 1):
            print(f"  {i}. {s.get('messageId')} | {s.get('fromEmail')} | {s.get('sentDate')}\n     {s.get('preview')}")


if __name__ == "__main__":
    main()


