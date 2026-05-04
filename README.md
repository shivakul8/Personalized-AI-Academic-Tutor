

# 📘 Personalized AI Academic Tutor

**Textbook-Aware RAG + Knowledge Graph + Conversational Memory**

An AI-powered academic tutor that answers student queries **strictly grounded in textbook content**, using an advanced **Retrieval-Augmented Generation (RAG)** pipeline combining **vector search, knowledge graphs, and long-term conversational memory**.

Built using **Python, LangChain, Google Gemini, Qdrant, Neo4j, Mem0, Docker, and Streamlit**.

---

## 🚀 Key Features

### ✅ Implemented

* **Textbook-specific RAG system**

  * PDF ingestion using LangChain loaders
  * Chunking via recursive text splitters
  * Dense embeddings using **Google Generative AI Embeddings**
  * Semantic retrieval via **Qdrant Vector Database**

* **Multi-Query Retrieval + Rank Fusion**

  * LLM-generated query rewrites for improved recall
  * **Reciprocal Rank Fusion (RRF)** for robust result aggregation

* **Knowledge Graph Construction**

  * Automatic entity & relationship extraction using `LLMGraphTransformer`
  * Graph storage in **Neo4j**
  * Enables future graph-based reasoning

* **Personalized Conversational Memory**

  * Long-term memory using **mem0**
  * Stores and retrieves past interactions per user
  * Enhances contextual continuity across sessions

* **Gemini-powered Answer Generation**

  * Context + memory injected into system prompt
  * Low-temperature responses for factual accuracy

* **Containerized Backend**

  * Docker-ready backend architecture
  * Compatible with local or cloud deployment

---

## 🧠 System Architecture

```
PDF (Textbook)
   │
   ├── Chunking & Embeddings ──► Qdrant (Vector Store)
   │
   ├── LLMGraphTransformer ───► Neo4j (Knowledge Graph)
   │
User Query
   │
   ├── Multi-Query Expansion (Gemini)
   │
   ├── Vector Retrieval (Qdrant)
   │
   ├── Rank Fusion (RRF)
   │
   ├── Memory Retrieval (mem0)
   │
   └── Answer Generation (Gemini)
```

---

## 🛠️ Tech Stack

| Component       | Technology              |
| --------------- | ----------------------- |
| Language        | Python                  |
| LLM             | Google Gemini 1.5 Flash |
| Framework       | LangChain               |
| Vector DB       | Qdrant                  |
| Knowledge Graph | Neo4j                   |
| Memory          | mem0                    |
| UI              | Streamlit               |
| Deployment      | Docker                  |

---

## 📂 Project Structure

```
├── advanced_rag.py        # Core RAG + KG + Memory pipeline
├── .env                  # API keys & credentials
├── mem0_history.db       # Conversational memory store
├── class9.pdf             # Example textbook (user-provided)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/personalized-ai-tutor.git
cd personalized-ai-tutor
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_genai_api_key
```

### 5️⃣ Start Required Services

```bash
# Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Neo4j
docker run -p 7474:7474 -p 7687:7687 \
-e NEO4J_AUTH=neo4j/password neo4j
```

### 6️⃣ Run the Tutor

```bash
python advanced_rag.py
```

---

## 💬 Example Interaction

```
User: Explain reflection of light in simple terms

Assistant:
Reflection of light is the process by which light bounces back
after hitting a smooth surface such as a mirror...
```

The response is:

* Grounded in textbook content
* Personalized using prior interactions
* Free from hallucinated explanations

---

## 🧪 Current Limitations (Honest Disclosure)

* Graph retrieval is **currently implicit** (graph stored but not queried directly)
* Single-PDF ingestion (multi-book support planned)
* CLI-based interaction (Streamlit UI under development)
* No authentication / role-based access yet

---

## 🛣️ Roadmap

### 🔜 Planned Enhancements

* Explicit **graph-based retrieval & reasoning**
* Multi-textbook + subject-wise indexing
* Streamlit dashboard for students & teachers
* Progress-aware tutoring (difficulty adaptation)
* Cloud deployment (GCP / Azure)

---

## 📌 Why This Project Matters

This project demonstrates:

* **Real-world RAG engineering**
* **Hybrid vector + graph AI systems**
* **Memory-aware LLM applications**
* **Production-oriented AI design**

It goes beyond “chat-with-PDF” by focusing on **retrieval quality, reasoning structure, and personalization**.

---

Just tell me what’s next 👌
