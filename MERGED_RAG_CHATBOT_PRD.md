## Email RAG Chatbot — Consolidated PRD and Implementation Guide

This document merges and streamlines the best parts of three drafts into one practical, end-to-end guide for building a Firestore + LangChain + Vertex AI conversational chatbot for email data. It combines: a clear PRD (vision, goals, user stories), a precise implementation plan with tested dependencies, and runnable snippets for RAG, agents, and a minimal API.

---

## Executive Summary

- **Goal**: An intelligent chatbot that answers questions about your emails, grounded by Firestore data via Retrieval-Augmented Generation (RAG).
- **Stack**: Python 3.12, Conda env `ad3gem`, Firestore, LangChain, Vertex AI (Gemini + text-embedding-004), Firestore vector search.
- **Outcomes**: Fast, accurate, explainable answers with conversation memory; clean architecture for local dev and cloud deployment.

Architecture:
```
User → Chat Interface (Agent) → Vector Search (Firestore) → Context → Vertex AI LLM → Answer
```

---

## Product Requirements Document (PRD)

### Vision
Create an intelligent, conversational “Email Sage” that provides quick, contextually relevant answers sourced from a user’s Firestore email corpus. Eliminate manual search; enable natural-language insight discovery.

### Goals
- Accurate, grounded answers using RAG and Firestore vector search
- Natural, multi-turn chat with conversation memory persisted in Firestore
- Scalable, cloud-ready architecture using Vertex AI and GCP services

### User Stories
- As a user, I can ask questions about my past emails in plain English and get accurate answers with sources.
- As a user, I can ask follow-ups without repeating context; the bot remembers the conversation.
- As a developer, I can run locally, index emails, test responses, and deploy a minimal API.

### Non-Functional Requirements
- **Security**: Service account credentials via env vars; Firestore rules enforced.
- **Performance**: Typical response time under 5 seconds for k≤5 retrieved chunks.
- **Scalability**: Handle increasing email volume; efficient chunking and embedding pipeline.
- **Reliability**: Automated tests for connectivity, embeddings, vector search, and chat loop.

### Success Metrics
- ≥90% perceived relevance in user feedback; latency p95 < 5s; <1% error rate in steady state.

### Assumptions & Risks
- Emails are stored as Firestore documents with subject/body/metadata.
- Vertex AI quotas and costs monitored; authentication correctly configured.

---

## Environment & Dependencies

### Conda and Python
```bash
conda create -n ad3gem python=3.12 -y
conda activate ad3gem
python --version  # Expect 3.12.x
```

### Environment Variables (.env)
```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/key.json
```

### requirements.txt (tested versions)
```txt
# Core Google Cloud
google-cloud-aiplatform==1.90.0
google-cloud-firestore==2.19.0
google-cloud-storage==2.18.2

# LangChain and integrations
langchain==0.3.12
langchain-google-firestore==0.5.0
langchain-google-vertexai==2.0.0
langchain-google-community==2.0.0
langchain-community==0.3.12

# Utilities
python-dotenv==1.0.1
numpy==1.26.4
pandas==2.2.2
tiktoken==0.7.0
aiohttp==3.10.5
```

Install:
```bash
pip install -r requirements.txt
```

---

## Firestore & Vertex Initialization

```python
# file: init_gcp.py
import os
from dotenv import load_dotenv
import vertexai
from google.cloud import firestore

def init_gcp():
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

    vertexai.init(project=project_id, location=region)
    db = firestore.Client(project=project_id)

    return db

if __name__ == "__main__":
    db = init_gcp()
    print("GCP initialized.")
```

Connectivity smoke test:
```python
# file: test_firestore.py
from google.cloud import firestore

def test_firestore_connection():
    db = firestore.Client()
    list(db.collections())  # raises if misconfigured
    print("✅ Firestore connected")

if __name__ == "__main__":
    test_firestore_connection()
```

---

## Data Loading & Preparation

Recommended Firestore email document fields: `subject`, `body`, `sender`, `recipients`, `date`, optional `thread_id`, `labels`, `attachments`.

Load and normalize for better retrieval:
```python
# file: load_emails.py
import os
from langchain_google_firestore import FirestoreLoader

def load_email_documents(collection_name: str = "emails"):
    loader = FirestoreLoader(
        collection_name=collection_name,
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    documents = loader.load()

    # Promote subject/sender into the content for richer recall
    processed = []
    for doc in documents:
        content = (
            f"Subject: {doc.metadata.get('subject', '')}\n"
            f"From: {doc.metadata.get('sender', '')}\n"
            f"Body: {doc.page_content}"
        )
        doc.page_content = content
        processed.append(doc)

    return processed

if __name__ == "__main__":
    docs = load_email_documents()
    print(f"Loaded {len(docs)} emails")
```

---

## Embeddings & Vector Store (Firestore)

Chunk, embed, and index into Firestore vector store:
```python
# file: vector_store_setup.py
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_store(documents, collection_name: str = "email_vectors"):
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    store = FirestoreVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=collection_name
    )
    return store

if __name__ == "__main__":
    from load_emails import load_email_documents
    docs = load_email_documents()
    store = create_vector_store(docs)
    print("Vector store created")
```

Quick search helper:
```python
# file: vector_search.py
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore

def search_emails(query: str, k: int = 5):
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    store = FirestoreVectorStore(collection="email_vectors", embedding=embeddings)
    return store.similarity_search(query=query, k=k)
```

---

## RAG Chains

### Basic RetrievalQA
```python
# file: rag_chain.py
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def create_rag_chain():
    llm = ChatVertexAI(model_name="gemini-1.5-flash", temperature=0.7, max_tokens=2048)

    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    store = FirestoreVectorStore(collection="email_vectors", embedding=embeddings)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt = PromptTemplate(
        template=(
            "You are an assistant analyzing email data.\n"
            "Use the context to answer. If unknown, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        input_variables=["context", "question"],
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa
```

### Conversational RAG with Memory
```python
# file: advanced_rag.py
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore, FirestoreChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def create_conversational_rag(session_id: str):
    llm = ChatVertexAI(model_name="gemini-1.5-pro", temperature=0.7, max_tokens=4096)
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    store = FirestoreVectorStore(collection="email_vectors", embedding=embeddings)

    chat_history = FirestoreChatMessageHistory(session_id=session_id, collection="chat_histories")
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store.as_retriever(k=5),
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )
    return chain
```

---

## Agent with Tools and Persona

Wrap the RAG chain as a tool and give the agent a light persona.

```python
# file: chatbot_agent.py
import uuid
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore, FirestoreChatMessageHistory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

class EmailChatbot:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self._init_components()
        self._init_tools()
        self._init_agent()

    def _init_components(self):
        self.llm = ChatVertexAI(model_name="gemini-1.5-flash", temperature=0.7, max_tokens=2048)
        self.embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
        self.store = FirestoreVectorStore(collection="email_vectors", embedding=self.embeddings)
        self.chat_history = FirestoreChatMessageHistory(session_id=self.session_id, collection="chat_histories")
        self.memory = ConversationBufferMemory(chat_memory=self.chat_history, memory_key="chat_history", return_messages=True)

    def _init_tools(self):
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.store.as_retriever(k=5),
            return_source_documents=True,
        )
        self.tools = [
            Tool(
                name="Email_Search",
                func=lambda q: qa.invoke({"query": q})["result"],
                description="Search the user's emails for relevant information",
            )
        ]

    def _init_agent(self):
        # Persona via system prompt is supported in LangChain toolkits; here we keep it simple via agent type
        self.agent = initialize_agent(
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            tools=self.tools,
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
        )

    def chat(self, message: str) -> str:
        return self.agent.run(message)
```

---

## Minimal API for Local Use

```python
# file: api.py
from flask import Flask, request, jsonify
from chatbot_agent import EmailChatbot

app = Flask(__name__)
bot = EmailChatbot()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400
    response = bot.chat(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
```

---

## Testing (Essentials)

- Firestore connectivity: instantiate client, list collections, write/read a test doc
- Embeddings: call `VertexAIEmbeddings.embed_query` with a test string
- Vector search: `FirestoreVectorStore.similarity_search("query", k=5)` returns docs
- Chat loop: instantiate `EmailChatbot` and run a few prompts

Optional unit test scaffolding mirrors these checks with mocks for networked calls.

---

## Runbook & Commands

```bash
# 1) Environment
conda activate ad3gem
pip install -r requirements.txt

# 2) Smoke tests
python /Users/craigcharity/dev/ad3gem/init_gcp.py
python /Users/craigcharity/dev/ad3gem/test_firestore.py

# 3) Index data
python -c "from load_emails import load_email_documents; from vector_store_setup import create_vector_store; create_vector_store(load_email_documents())"

# 4) Quick search
python -c "from vector_search import search_emails; print(search_emails('project updates')[:1])"

# 5) Chatbot (CLI-style quick check)
python -c "from chatbot_agent import EmailChatbot; b=EmailChatbot(); print(b.chat('Summarize last week\'s emails'))"

# 6) API
python /Users/craigcharity/dev/ad3gem/api.py
```

---

## Best Practices & Tips

- Prefer smaller chunk sizes (500–1000) with modest overlap (150–200) for emails.
- Include `subject`, `sender`, and `date` in the chunk text to boost semantic recall.
- Start with `gemini-1.5-flash` for speed; use `-pro` for tougher queries.
- Log inputs/latency; capture failure cases and add guardrails (“I don’t know” when uncertain).
- Monitor Vertex AI quotas and costs; batch re-embedding if fields change.

---

## Common Issues

- Auth errors: ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service-account JSON.
- Firestore `PERMISSION_DENIED`: check project ID and IAM; enable Firestore and Vertex AI APIs.
- Module not found: ensure `conda activate ad3gem` before `pip install -r requirements.txt`.

---

## Roadmap (Next Iterations)

- Add advanced retrieval (metadata filters, recency bias, hybrid search)
- Improve persona and prompt templates with structured system prompts
- Add evaluation harness for answer faithfulness and groundedness
- Deploy to Cloud Run, add auth and rate limiting

---

Last updated: 2025-09 • Python: 3.12 • Env: ad3gem


