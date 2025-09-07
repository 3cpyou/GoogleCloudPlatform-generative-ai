# AD3Gem - Conversational Intelligence System

**A complete Retrieval-Augmented Generation (RAG) chatbot built with Gemini AI and Google Cloud Firestore, featuring intelligent email search and conversation memory.**

---

## 🚀 What is AD3Gem?

AD3Gem is a production-ready conversational AI system that connects to your organization's email data and provides intelligent, context-aware responses. Built for the Lineage Coffee business, it demonstrates how to create a comprehensive RAG system using Google Cloud technologies.

### ✨ Key Features

- **📧 Email Intelligence**: Search and analyze business emails with natural language queries
- **🧠 Conversation Memory**: Maintains context across chat sessions with persistent memory
- **🗄️ Multi-Database Architecture**: Connects to 4 specialized Firestore databases
- **🤖 Gemini AI Integration**: Powered by Google's advanced language models
- **🔍 Flexible Search**: Find emails by sender, content, date, or complex queries
- **💬 Natural Interaction**: Understands business context and provides helpful responses

## 🏗️ System Architecture

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Main Data     │  Conversations  │     Memory      │     Emails      │
│ ad3gem-database │ad3gem-conversation│ ad3gem-memory  │  ad3sam-email   │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ • users         │ • chat_history  │ • memory_heads  │ • reports       │
│ • projects      │ • sessions      │ • claims        │ • sample_emails │
│ • sample_data   │ • contexts      │ • beliefs       │ • top_senders   │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
                                    │
                            ┌───────▼───────┐
                            │ firestore_client.py │
                            │ (Database Layer)    │
                            └───────┬───────┘
                                    │
                            ┌───────▼───────┐
                            │ simple_chatbot.py │
                            │ (AI Layer)        │
                            └───────┬───────┘
                                    │
                            ┌───────▼───────┐
                            │ Gemini 1.5 Flash │
                            │ (Language Model)  │
                            └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ with Conda
- Google Cloud Project with Vertex AI and Firestore enabled
- Service Account with appropriate permissions

### Installation

1. **Clone and Setup Environment**
   ```bash
   conda create -n ad3gem python=3.12
   conda activate ad3gem
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   ```bash
   # Copy and modify the environment setup
   cp /path/to/environment-setup/templates/ad3gem-env-setup.sh ./
   source ad3gem-env-setup.sh
   ```

3. **Initialize Sample Data**
   ```bash
   python setup_firestore.py
   ```

4. **Start the Chatbot**
   ```bash
   python simple_chatbot.py
   ```

## 💬 Usage Examples

```
You: "when did julie last email?"
Bot: Found 3 recent emails from julie:
     📧 Re: Payment now due... (2025-09-04 13:20)
     📧 Re: Zapper refund request... (2025-09-04 13:19)
     📧 Re: POP Invoice... (2025-09-04 11:11)

You: "emails from craig@wmcahn.co.za"
Bot: Found 10 emails from Craig Wishart:
     📧 Invoice (2025-09-04 13:17)
     📧 Statement follow-up (2025-09-04 12:45)

You: "show recent emails"
Bot: Here are the 5 most recent emails:
     📧 From: Julie Pilbrough...
```

## 🗂️ Core Components

### `firestore_client.py`
- **Purpose**: Database abstraction layer
- **Functions**: Email search, conversation storage, memory management
- **Databases**: Manages connections to all 4 Firestore databases

### `simple_chatbot.py`
- **Purpose**: Main chatbot interface with Gemini integration
- **Features**: Natural language processing, email queries, context awareness
- **Intelligence**: 95% understanding confidence for complex queries

### `ad3gem-env-setup.sh`
- **Purpose**: Environment configuration
- **Variables**: Database names, API keys, service account settings

## 📊 Database Schema

### Email Database (`ad3sam-email`)
```
reports/
├── 20250904_110423/
│   └── sample_emails/
│       ├── {doc_id}/
│       │   ├── from: "Julie Pilbrough <julie@lineagecoffee.com>"
│       │   ├── to: "recipient@example.com"
│       │   ├── subject: "Email subject"
│       │   ├── processed_at: "2025-09-04T13:20:32Z"
│       │   └── body: "Email content..."
```

### Memory Database (`ad3gem-memory`)
```
memory_heads/
├── {memory_id}/
│   ├── facet: "user_preferences"
│   ├── scope: "email_queries"
│   ├── claim: "User prefers recent emails first"
│   ├── confidence: 0.95
│   └── timestamp: "2025-09-04T13:20:32Z"
```

## 🔧 Advanced Configuration

### Custom Email Queries
The system supports complex natural language queries:
- **Sender-specific**: "emails from julie@lineagecoffee.com"
- **Content search**: "emails about invoices"
- **Date ranges**: "emails from last week"
- **Recipient filtering**: "emails to craig@lineagecoffee.com"

### Memory Management
AD3Gem learns and remembers:
- User preferences
- Common query patterns
- Business relationships
- Email context and importance

---

# Original Google Cloud Generative AI Repository

> NOTE: [Gemini 2.0 Flash](https://cloud.google.com/vertex-ai/generative-ai/docs/gemini-v2) has been released! Here are the latest notebooks and demos using the new model:
>
> - [Intro to Gemini 2.0 Flash](gemini/getting-started/intro_gemini_2_0_flash.ipynb)
> - [Intro to Multimodal Live API with Gen AI SDK](gemini/multimodal-live-api/intro_multimodal_live_api_genai_sdk.ipynb)
> - [Intro to Gemini 2.0 Thinking Mode](gemini/getting-started/intro_gemini_2_0_flash_thinking_mode.ipynb)
> - [Intro to Code Execution](gemini/code-execution/intro_code_execution.ipynb)
> - [Multimodal Live API Demo App](gemini/multimodal-live-api/websocket-demo-app/)
> - [Intro to Google Gen AI SDK](gemini/getting-started/intro_genai_sdk.ipynb)
> - [Real-Time RAG with Multimodal Live API](gemini/multimodal-live-api/real_time_rag_retail_gemini_2_0.ipynb)
> - [Creating Marketing Assets using Gemini 2.0](gemini/use-cases/marketing/creating_marketing_assets_gemini_2_0.ipynb)
> - [Vertex AI Gemini Research Multi Agent Demo Research Agent for EV Industry](gemini/agents/research-multi-agents)
> - [Create a Multi-Speaker Podcast with Gemini 2.0 & Text-to-Speech](audio/speech/use-cases/podcast/multi-speaker-podcast.ipynb)
> - [Intro to Gemini 2.0 Flash REST API](gemini/getting-started/intro_gemini_2_0_flash_rest_api.ipynb)

<!-- markdownlint-disable MD033 -->

<a href="gemini"><img src="https://lh3.googleusercontent.com/eDr6pYKs1tT0iK0nt3pPhvVlP2Wn96fbGqbWgBAARRZ7isej037g_tWobjV8zQkxOsWzJuEH8p-fksczXUOeqxGZZIo_HUCdkn8q-a4fuwATD7Q9Xrs=w2456-l100-sg-rj-c0xffffff" style="width:35em" alt="Welcome to the Gemini era"></a>

This repository contains notebooks, code samples, sample apps, and other resources that demonstrate how to use, develop and manage generative AI workflows using [Generative AI on Google Cloud](https://cloud.google.com/ai/generative-ai), powered by [Vertex AI](https://cloud.google.com/vertex-ai).

For more Vertex AI samples, please visit the [Vertex AI samples GitHub repository](https://github.com/GoogleCloudPlatform/vertex-ai-samples/).

## Using this repository

[![Applied AI Summit: The cloud toolkit for generative AI](https://img.youtube.com/vi/xT7WW2SKLfE/hqdefault.jpg)](https://www.youtube.com/watch?v=xT7WW2SKLfE)

<table>
  <tr>
    <th></th>
    <th style="text-align: center;">Description</th>
  </tr>
  <tr>
    <td>
      <img src="https://storage.googleapis.com/github-repo/img/gemini/Spark__Gradient_Alpha_100px.gif" width="45px" alt="Gemini">
      <br>
      <a href="gemini/"><code>gemini/</code></a>
    </td>
    <td>
      Discover Gemini through starter notebooks, use cases, function calling, sample apps, and more.
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/service_discovery/v1/24px.svg" width="40px" alt="Search">
      <br>
      <a href="search/"><code>search/</code></a>
    </td>
    <td>Use this folder if you're interested in using <a href="https://cloud.google.com/enterprise-search">Vertex AI Search</a>, a Google-managed solution to help you rapidly build search engines for websites and across enterprise data. (Formerly known as Enterprise Search on Generative AI App Builder)</td>
  </tr>
  <tr>
    <td>
      <img src="https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/nature_people/default/40px.svg" alt="RAG Grounding">
      <br>
      <a href="rag-grounding/"><code>rag-grounding/</code></a>
    </td>
    <td>Use this folder for information on Retrieval Augmented Generation (RAG) and Grounding with Vertex AI. This is an index of notebooks and samples across other directories focused on this topic.</td>
  </tr>
  <tr>
    <td>
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/dialogflow_cx/v1/24px.svg" width="40px" alt="Conversation">
      <br>
      <a href="conversation/"><code>conversation/</code></a>
    </td>
    <td>Use this folder if you're interested in using <a href="https://cloud.google.com/generative-ai-app-builder">Vertex AI Conversation</a>, a Google-managed solution to help you rapidly build chat bots for websites and across enterprise data. (Formerly known as Chat Apps on Generative AI App Builder)</td>
  </tr>
  <tr>
    <td>
      <img src="https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/edit_note/default/40px.svg" alt="Language">
      <br>
      <a href="language/"><code>language/</code></a>
    </td>
    <td>
      Use this folder if you're interested in building your own solutions from scratch using Google's language foundation models (Vertex AI PaLM API).
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/image/default/40px.svg" alt="Vision">
      <br>
      <a href="vision/"><code>vision/</code></a>
    </td>
    <td>
      Use this folder if you're interested in building your own solutions from scratch using features from Imagen on Vertex AI (Vertex AI Imagen API).
      These are the features that Imagen on Vertex AI offers:
      <ul>
        <li>Image generation</li>
        <li>Image editing</li>
        <li>Visual captioning</li>
        <li>Visual question answering</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/mic/default/40px.svg" alt="Speech">
      <br>
      <a href="audio/"><code>audio/</code></a>
    </td>
    <td>
      Use this folder if you're interested in building your own solutions from scratch using features from Chirp, a version of Google's Universal Speech Model (USM) on Vertex AI (Vertex AI Chirp API).
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/build/default/40px.svg" alt="Setup Env">
      <br>
      <a href="setup-env/"><code>setup-env/</code></a>
    </td>
    <td>Instructions on how to set up Google Cloud, the Vertex AI Python SDK, and notebook environments on Google Colab and Vertex AI Workbench.</td>
  </tr>
  <tr>
    <td>
      <img src="https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/media_link/default/40px.svg" alt="Resources">
      <br>
      <a href="RESOURCES.md"><code>RESOURCES.md</code></a>
    </td>
    <td>Learning resources (e.g. blogs, YouTube playlists) about Generative AI on Google Cloud</td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

## Related Repositories

- [Gemini Cookbook](https://github.com/google-gemini/cookbook/)
- [Google Cloud Applied AI Engineering](https://github.com/GoogleCloudPlatform/applied-ai-engineering-samples)
- [Generative AI for Marketing using Google Cloud](https://github.com/GoogleCloudPlatform/genai-for-marketing)
- [Generative AI for Developer Productivity](https://github.com/GoogleCloudPlatform/genai-for-developers)
- Vertex AI Core
  - [Vertex AI Samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples)
  - [MLOps with Vertex AI](https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai)
  - [Developing NLP solutions with T5X and Vertex AI](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai)
  - [AlphaFold batch inference with Vertex AI Pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-alphafold-inference-pipeline)
  - [Serving Spark ML models using Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-spark-ml-serving)
  - [Sensitive Data Protection (Cloud DLP) for Vertex AI Generative Models (PaLM2)](https://github.com/GoogleCloudPlatform/Sensitive-Data-Protection-for-Vertex-AI-PaLM2)
- Conversational AI
  - [Contact Center AI Samples](https://github.com/GoogleCloudPlatform/contact-center-ai-samples)
  - [Reimagining Customer Experience 360](https://github.com/GoogleCloudPlatform/dialogflow-ccai-omnichannel)
- Document AI
  - [Document AI Samples](https://github.com/GoogleCloudPlatform/document-ai-samples)
- Duet AI
  - [Cymbal Superstore](https://github.com/GoogleCloudPlatform/cymbal-superstore)
- Cloud Databases
  - [Gen AI Databases Retrieval App](https://github.com/GoogleCloudPlatform/genai-databases-retrieval-app)
- Other
  - [ai-on-gke](https://github.com/GoogleCloudPlatform/ai-on-gke)
  - [ai-infra-cluster-provisioning](https://github.com/GoogleCloudPlatform/ai-infra-cluster-provisioning)
  - [solutions-genai-llm-workshop](https://github.com/GoogleCloudPlatform/solutions-genai-llm-workshop)
  - [terraform-genai-doc-summarization](https://github.com/GoogleCloudPlatform/terraform-genai-doc-summarization)
  - [terraform-genai-knowledge-base](https://github.com/GoogleCloudPlatform/terraform-genai-knowledge-base)
  - [genai-product-catalog](https://github.com/GoogleCloudPlatform/genai-product-catalog)
  - [solutionbuilder-terraform-genai-doc-summarization](https://github.com/GoogleCloudPlatform/solutionbuilder-terraform-genai-doc-summarization)
  - [solutions-viai-edge-provisioning-configuration](https://github.com/GoogleCloudPlatform/solutions-viai-edge-provisioning-configuration)
  - [mis-ai-accelerator](https://github.com/GoogleCloudPlatform/mis-ai-accelerator)
  - [dataflow-opinion-analysis](https://github.com/GoogleCloudPlatform/dataflow-opinion-analysis)
  - [genai-beyond-basics](https://github.com/meteatamel/genai-beyond-basics)

## Contributing

Contributions welcome! See the [Contributing Guide](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/CONTRIBUTING.md).

## Getting help

Please use the [issues page](https://github.com/GoogleCloudPlatform/generative-ai/issues) to provide suggestions, feedback or submit a bug report.

## Disclaimer

This repository itself is not an officially supported Google product. The code in this repository is for demonstrative purposes only.
