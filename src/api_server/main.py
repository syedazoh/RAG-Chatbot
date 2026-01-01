"""
FastAPI Backend Server for RAG Chatbot
Provides REST API endpoints for querying and document ingestion
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_pipeline import get_rag_chain, ingest_documents

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for contextual question-answering using RAG",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:8501",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG chain instance
rag_chain = None


# Request/Response Models
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


class IngestResponse(BaseModel):
    message: str
    status: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG chain on application startup"""
    global rag_chain
    print("🚀 Starting up RAG Chatbot API...")
    print("📚 Loading RAG chain...")
    
    try:
        rag_chain = get_rag_chain()
        if rag_chain:
            print("✅ RAG chain initialized successfully!")
        else:
            print("⚠️  RAG chain initialization failed. Please run ingestion first.")
    except Exception as e:
        print(f"❌ Error initializing RAG chain: {e}")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "RAG Chatbot API is running",
        "status": "healthy",
        "endpoints": {
            "chat": "/chat (POST)",
            "ingest": "/ingest (POST)"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global rag_chain
    
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain not initialized.")
    
    try:
        # 1. Invoke the chain
        # Note: If using a simple retriever, result might just be a list of docs.
        # If using a RetrievalQA chain, it's a dict.
        result = rag_chain.invoke({"input": request.query})
        
        # 2. Extract Answer safely
        # If result is a string (direct answer), use it. 
        # If it's a dict, look for 'answer' or 'result'.
        if isinstance(result, str):
            answer = result
        else:
            answer = result.get("answer") or result.get("result") or "I found some relevant info but couldn't summarize it."

        # 3. Extract Sources safely
        sources = []
        # Look for context in 'context' or 'source_documents'
        context_docs = result.get("context") or result.get("source_documents") or []
        
        if context_docs:
            for doc in context_docs:
                s = doc.metadata.get("source", "data.txt")
                s_name = os.path.basename(s)
                if s_name not in sources:
                    sources.append(s_name)

        # 4. Final Validation
        return ChatResponse(
            answer=answer,
            sources=sources if sources else ["Manual Reference"]
        )
    
    except Exception as e:
        print(f"❌ Chat Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the server is up and if the RAG database is loaded."""
    db_exists = os.path.exists("./chroma_db")
    return {
        "status": "online",
        "database_found": db_exists,
        "rag_ready": rag_chain is not None,
        "mode": "Retriever-Only"
    }


# Run with: uvicorn api_server.main:app --reload --host 0.0.0.0 --port 8000