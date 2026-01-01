"""
RAG Pipeline - Skincare Chatbot
"""
import os
import shutil
from dotenv import load_dotenv

# Document Loading & Splitting
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store & Embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LLM & Prompts
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- THE CRITICAL LEGACY IMPORTS ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMA_DB_DIR = "./chroma_db"
DOCUMENTS_DIR = "./documents"

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_documents():
    """Load documents, chunk them, and store in local vector store."""
    print("🔄 Starting document ingestion...")
    
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"📁 Created {DOCUMENTS_DIR} folder.")
        return

    try:
        txt_loader = DirectoryLoader(
            DOCUMENTS_DIR,
            glob="data.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = txt_loader.load()
        print(f"📄 Loaded {len(documents)} TXT files")
    except Exception as e:
        print(f"⚠️ Error loading documents: {e}")
        return

    if not documents:
        print("⚠️ No content found in data.txt.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks") 

    print("💾 Creating vector store...")
    try:
        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print(f"✅ Ingestion complete!")
        return vectorstore
    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}")
        return None

def get_rag_chain():
    """Initialize the RAG chain using OpenRouter."""
    if not os.path.exists(CHROMA_DB_DIR):
        print("⚠️ Database directory not found.")
        return None
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    
    # Line 92 - Indented exactly 4 spaces
    llm = ChatOllama(
    model="llama3",
    temperature=0
)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Skincare Assistant. Context: {context}"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(
        vectorstore.as_retriever(), 
        question_answer_chain
    )

if __name__ == "__main__":
    ingest_documents()