#  Medical RAG Chatbot using Endee Vector Database

##  Problem Statement

Medical professionals and students often need quick, accurate answers from large medical textbooks and documents. Manually searching through hundreds of PDF pages is inefficient and time-consuming.

This project solves that by building an AI-powered Medical Chatbot using **RAG (Retrieval Augmented Generation)** — where questions are answered strictly from verified medical PDF documents, reducing hallucinations and ensuring accuracy.

---

##  System Design
```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                                                             │
│  Medical PDFs → PyPDFLoader → RecursiveTextSplitter         │
│       → HuggingFace Embeddings → Endee Vector Database      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                           │
│                                                             │
│  User Query → Embed Query → Endee Similarity Search         │
│       → Top-K Chunks → Prompt Builder → Zephyr LLM          │
│       → Final Answer                                        │
└─────────────────────────────────────────────────────────────┘
```

### Flow Diagram
```
User
 │
 ▼
[Input Query]
 │
 ▼
[Embedding Model]  ←── sentence-transformers/all-MiniLM-L6-v2
 │
 ▼
[Endee Vector DB]  ←── Cosine Similarity Search (top_k=3)
 │
 ▼
[Context Chunks]
 │
 ▼
[Prompt Template]  ←── System prompt + Context + Question
 │
 ▼
[Zephyr-7B LLM]   ←── HuggingFace Inference API
 │
 ▼
[Answer to User]
```

---

##  Technical Approach

### 1. Document Ingestion (`medical_chatbot/ingest.py`)

| Step | Tool Used | Details |
|------|-----------|---------|
| PDF Loading | `PyPDFLoader` + `DirectoryLoader` | Loads all `.pdf` files from `medical_chatbot/data/` |
| Text Splitting | `RecursiveCharacterTextSplitter` | chunk_size=500, chunk_overlap=50 |
| Embedding | `all-MiniLM-L6-v2` | 384-dimension dense vectors |
| Vector Storage | **Endee** | cosine space, float32 precision |

**Why chunk_size=500?**
- Small enough to retrieve precise context
- Large enough to maintain sentence meaning
- overlap=50 ensures no context is lost at boundaries

**Why cosine similarity?**
- Best for semantic text search
- Normalized vectors → better accuracy

### 2. Query Pipeline (`medical_chatbot/chatbot.py`)

| Step | Tool Used | Details |
|------|-----------|---------|
| Query Embedding | `all-MiniLM-L6-v2` | Same model as ingestion |
| Vector Search | **Endee** | top_k=3 most similar chunks |
| LLM | `zephyr-7b-beta` | HuggingFace Inference API |
| Prompt Strategy | System + User message | Constrained to context only |

### 3. How Endee is Used
```python
# 1. Connect to Endee server (Docker)
client = Endee()  # localhost:8080

# 2. Create index
client.create_index(
    name="medical_chatbot",
    dimension=384,
    space_type="cosine",
    precision="float32"
)

# 3. Store vectors with metadata
index.upsert([{
    "id": "chunk_0",
    "vector": [0.1, 0.2, ...],
    "meta": {"text": "...", "source": "book.pdf"}
}])

# 4. Semantic search
results = index.query(vector=query_embedding, top_k=3)
```

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Database | [Endee](https://github.com/endee-io/endee) |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | HuggingFaceH4/zephyr-7b-beta |
| PDF Processing | LangChain + PyPDF |
| Inference API | HuggingFace Hub |
| Containerization | Docker |

---

##  Project Structure
```
endee/ (forked from endee-io/endee)
├── medical_chatbot/
│   ├── data/              # Medical PDF files
│   ├── ingest.py          # PDF ingestion pipeline
│   ├── chatbot.py         # RAG chatbot
│   ├── requirements.txt   # Python dependencies
│   └── .env               # API tokens (not in git)
├── src/                   # Endee C++ source (unchanged)
├── docker-compose.yml     # Endee server
└── README.md
```

---

##  Setup Instructions

### Prerequisites
- Python 3.8+
- Docker Desktop
- HuggingFace API token

### Step 1 — Clone Repository
```bash
git clone https://github.com/VishaSharmaPro/endee.git
cd endee
```

### Step 2 — Start Endee Server
```bash
docker compose up -d
```
Verify: http://localhost:8080

### Step 3 — Install Dependencies
```bash
pip install -r medical_chatbot/requirements.txt
```

### Step 4 — Configure Environment
Create `medical_chatbot/.env`:
```
HF_TOKEN=your_huggingface_token_here
```

### Step 5 — Add PDFs
Place medical PDFs in `medical_chatbot/data/` folder.

### Step 6 — Run Ingestion
```bash
python medical_chatbot/ingest.py
```

### Step 7 — Run Chatbot
```bash
python medical_chatbot/chatbot.py
```

---

##  Example
```
Aap: What are symptoms of fever?
Bot: Symptoms of fever include elevated body temperature (100-104°F),
     chills, sweating, headache, and muscle aches...

Aap: exit
```

---

##  Error Handling

| Scenario                 | Handling |
|--------------------------|----------|
| Existing index           | try/except deletes old index before creating new |
| Endee server not running | Ensure Docker is running first |
| Invalid HF token         | Check .env file |

---

##  Future Improvements

- [ ] Streamlit web UI
- [ ] Multi-turn conversation history  
- [ ] Source citation with page numbers
- [ ] Hybrid search (keyword + semantic)