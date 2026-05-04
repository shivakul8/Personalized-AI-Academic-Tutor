import os
import sys
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for API Key
if not os.environ.get("GOOGLE_API_KEY"):
    print("[ERROR] GOOGLE_API_KEY not found in environment or .env file.")
    # We continue strictly for the module import check, but it will fail at runtime if not set.

# Database Configuration
QDRANT_HOST = "localhost"
QDRANT_URL = f"http://{QDRANT_HOST}:6333"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Mem0 Import
from mem0 import Memory

# --- Global Clients ---
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- Ingestion Pipeline ---

def ingest_data(pdf_path: str, collection_name: str = "rag_collection"):
    """
    Ingests a PDF into Qdrant (Vector) and Neo4j (Graph).
    """
    if not os.path.exists(pdf_path):
        print(f"[WARN] PDF not found at {pdf_path}. Skipping ingestion.")
        return

    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    # 1. Vector Store Injection (Qdrant)
    print("Ingesting into Qdrant...")
    try:
        QdrantVectorStore.from_documents(
            splits,
            embedding_model,
            url=QDRANT_URL,
            collection_name=collection_name,
            force_recreate=True # For demo purposes, we overwrite. In prod, check existence.
        )
        print("Vector ingestion complete.")
    except Exception as e:
        print(f"[ERROR] Qdrant ingestion failed: {e}")

    # 2. Graph Store Injection (Neo4j)
    # Note: LLMGraphTransformer might need adjustment for Gemini if it expects specific OpenAI structures,
    # but generally strict Pydantic structures work with LangChain chat models.
    print("Ingesting into Neo4j Graph...")
    try:
        graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        llm_transformer = LLMGraphTransformer(llm=llm)
        
        # Taking a subset for graph to save time/cost in this demo, or all if feasible
        graph_docs = llm_transformer.convert_to_graph_documents(splits)
        
        graph.add_graph_documents(graph_docs)
        print("Graph ingestion complete.")
    except Exception as e:
        print(f"[ERROR] Neo4j ingestion failed: {e}")

# --- Retrieval Components ---

def generate_multi_queries(question: str) -> List[str]:
    """
    Generates alternative versions of the user's question for better retrieval.
    """
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"question": question})
    queries = response.strip().split("\n")
    return queries + [question] # Include original

def reciprocal_rank_fusion(results: List[List[Document]], k=60):
    """
    Fused rank of documents from multiple retrievers/queries.
    """
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
            
    reranked_results = [
        (doc, score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results # Returns list of (content, score)

def hybrid_retrieval(question: str, collection_name: str = "rag_collection"):
    """
    Performs retrieval from Vector (Qdrant) and Graph (Neo4j) and fuses results.
    """
    # 1. Vector Retrieval with Multi-Query
    queries = generate_multi_queries(question)
    print(f"[DEBUG] Generated queries: {queries}")
    
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name=collection_name,
        url=QDRANT_URL
    )
    
    vector_results = []
    for q in queries:
        vector_results.append(vector_store.similarity_search(q, k=5))
    
    # Flatten vector results for RRF or simple set
    # For RRF we keep them as separate lists
    fused_vector = reciprocal_rank_fusion(vector_results)
    
    # 2. Graph Retrieval 
    # Placeholder for direct graph context retrieval
    # Return top 5 unique fused contents
    top_docs = [content for content, score in fused_vector[:5]]
    return "\n\n".join(top_docs)

# --- Memory Integration ---

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "google",
        "config": {"model": "models/embedding-001"},
    },
    "llm": {"provider": "google", "config": {"model": "gemini-1.5-flash", "temperature": 0}},
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": 6333,
        },
    },
    "history_db_path": "mem0_history.db"
}
mem_client = Memory.from_config(config)

# --- Main Interaction Loop ---

def main():
    print("Initializing Advanced RAG Pipeline (Google GenAI Edition)...")
    
    # 1. Ingest (Will skip if file missing)
    ingest_data("class9.pdf")
    
    user_id = "user_terminal_1"
    
    print("\n--- RAG Pipeline Ready. Type 'exit' to quit. ---\n")
    
    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        # 1. Retrieve Memory
        print("[Thinking] Checking memory...")
        mem_results = mem_client.search(query, user_id=user_id)
        past_interactions = ""
        if mem_results and "results" in mem_results:
             past_interactions = "\n".join([m["memory"] for m in mem_results["results"]])
        
        # 2. Retrieve Context (Hybrid)
        print("[Thinking] Retrieving context from Vector/Graph...")
        context = hybrid_retrieval(query)
        
        # 3. Generate Answer
        system_prompt = f"""You are a helpful assistant. Use the following context and memory to answer the user's question.
        
        MEMORY:
        {past_interactions}
        
        CONTEXT:
        {context}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = llm.invoke(messages)
        print(f"\nAssistant: {response.content}")
        
        # 4. Update Memory
        mem_client.add([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response.content}
        ], user_id=user_id)

if __name__ == "__main__":
    main()
