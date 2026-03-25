# 🤖 LicenseBot — AI-Powered Document Q&A Assistant

> Ask questions about any document in plain English. Get grounded, cited answers instantly.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-green)
![Supabase](https://img.shields.io/badge/Supabase-Auth%20%2B%20DB-darkgreen)
![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2-orange)

---

## What Is LicenseBot?

LicenseBot is a production-grade RAG (Retrieval-Augmented Generation) application that lets users upload PDF documents and ask questions about them in plain English. It retrieves the most relevant context, generates grounded answers using a local LLM, and always cites its sources — refusing to answer questions outside the uploaded content.

Built as a portfolio project to demonstrate end-to-end AI engineering skills — from document ingestion to cloud deployment.

---

## Live Demo

> Sign in with: `demo@licensebot.com` / `Demo1234!`

---

## Architecture
```
PDF / TXT Documents
        ↓
Document Loader (pdfplumber + LangChain)
        ↓
Text Chunker (RecursiveCharacterTextSplitter)
        ↓
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
        ↓
Vector Store (Pinecone — cloud, persistent)
        ↓
User Question → Embed → Semantic Search → Top 3 Chunks
        ↓
Prompt Engineering → Llama 3.2 (Ollama — local)
        ↓
Streamed Answer + Source Citations
        ↓
Conversation History (Supabase Postgres)
```

---

## Features

- **RAG Pipeline** — retrieves relevant document chunks before answering
- **Semantic Search** — finds meaning, not just keywords
- **Hallucination Guard** — refuses to answer outside uploaded content
- **Source Citations** — every answer links back to its source document
- **PDF Upload** — upload any PDF and query it instantly
- **Conversation Memory** — remembers context across questions
- **User Authentication** — email/password login via Supabase Auth
- **Persistent History** — conversations saved and resumable across sessions
- **Streaming Responses** — answers stream word by word like ChatGPT
- **100% Free Stack** — no paid APIs required for core functionality

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Chat interface |
| LLM | Llama 3.2 via Ollama | Answer generation |
| Embeddings | sentence-transformers | Text → vectors |
| Vector DB | Pinecone (free tier) | Semantic search |
| Database | Supabase Postgres | Conversation history |
| Auth | Supabase Auth | User management |
| PDF Parsing | pdfplumber | Document ingestion |
| Chunking | LangChain | Text splitting |

---

## Project Structure
```
licensebot/
├── docs/                          # Sample policy documents
│   ├── software_license_policy.txt
│   ├── compliance_rules.txt
│   └── renewal_terms.txt
├── src/
│   ├── loader.py                  # Document loading and chunking
│   ├── embedder.py                # Embedding generation
│   ├── vector_store.py            # FAISS local vector store
│   ├── pinecone_store.py          # Pinecone cloud vector store
│   ├── main.py                    # Core RAG logic + terminal loop
│   ├── database.py                # Supabase auth + conversation history
│   └── app.py                     # Streamlit web UI
├── .env                           # API keys (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed with Llama 3.2
- [Pinecone](https://pinecone.io) free account
- [Supabase](https://supabase.com) free account

### Installation
```bash
# Clone the repo
git clone https://github.com/Teja2205/licensebot.git
cd licensebot

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Llama 3.2
ollama pull llama3.2
```

### Configuration

Create a `.env` file in the project root:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=licensebot
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
SUPABASE_KEY=your_supabase_publishable_key
```

### Run
```bash
python -m streamlit run src/app.py
```

Open `http://localhost:8501` in your browser.

---

## How It Works

### 1. Document Ingestion
Documents are loaded, split into 500-character chunks with 50-character overlap using `RecursiveCharacterTextSplitter`, embedded using `all-MiniLM-L6-v2` (384 dimensions), and upserted into Pinecone.

### 2. Semantic Search
User questions are embedded using the same model and compared against stored vectors using cosine similarity. The top 3 chunks above a relevance threshold of 0.1 are retrieved.

### 3. Answer Generation
Retrieved chunks are injected into a carefully engineered prompt that instructs Llama 3.2 to answer only from context, cite sources, and refuse out-of-scope questions. Responses stream token by token.

### 4. Persistence
Every conversation is stored in Supabase Postgres tied to the authenticated user's ID, enabling full history retrieval across sessions.

---

## Key Engineering Decisions

**Why Pinecone over FAISS?**
FAISS is in-memory only — restarting the app loses all vectors. Pinecone persists vectors in the cloud, enabling production-grade reliability without managing infrastructure.

**Why Llama 3.2 over OpenAI?**
Zero cost, full privacy (data never leaves your machine), and no API rate limits. The quality is sufficient for domain-specific Q&A with good prompt engineering.

**Why RecursiveCharacterTextSplitter?**
It preserves semantic meaning by splitting on paragraph → sentence → word boundaries in order, never cutting mid-sentence. The 50-character overlap ensures context isn't lost at chunk boundaries.

**Why cosine similarity over L2 distance?**
Cosine measures the angle between vectors (semantic direction) rather than magnitude. For text search, two passages can have different lengths but identical meaning — cosine handles this correctly while L2 would penalize the longer one.

---

## Roadmap

- [ ] RAGAS evaluation framework
- [ ] Feedback loop (thumbs up/down)
- [ ] Multi-namespace Pinecone (per-user knowledge bases)
- [ ] Deploy to Render.com
- [ ] Google OAuth login
- [ ] Support for Word documents (.docx)

---

## Author

**Teja Guduguntla**
Full Stack AI Engineer
[LinkedIn](https://linkedin.com/in/tejag) | [GitHub](https://github.com/Teja2205)

---

## License

MIT License — free to use, modify, and distribute.


## Phase 5 Session 2 Complete 🎉
```
✅ 10-question evaluation dataset
✅ Faithfulness scoring — answer vs ground truth
✅ Relevancy scoring — answer vs question
✅ Pass/fail per question
✅ Summary report
✅ Results saved to evaluation_results.json
```

---

## What's Next — Phase 5 Session 3: Deploy to Render.com

This is the final piece. Instead of running locally, LicenseBot will have a **live public URL** anyone can access.
```
Local:  http://localhost:8501
Live:   https://licensebot.onrender.com