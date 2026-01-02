import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DB_DIR = "./chroma_db"
DOCUMENTS_DIR = "./documents"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_documents():
    """Load, chunk, and store documents."""
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        return

    txt_loader = DirectoryLoader(
        DOCUMENTS_DIR,
        glob="data.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = txt_loader.load()

    if not documents:
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    return vectorstore

def get_rag_chain():
    """Returns a FUNCTION that app.py can call directly."""
    if not os.path.exists(CHROMA_DB_DIR):
        ingest_documents()
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant", # <--- CHANGE THIS LINE
    temperature=0.2
)
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are a professional skincare assistant. Use the context below to answer the user's question.
    If the answer isn't in the context, say "I don't know based on the provided data."

    Context: {context}
    Question: {question}
    
    Answer:""")

    # This is the function app.py will actually use
    def rag_handler(user_query):
        # 1. Get documents
        docs = retriever.invoke(user_query)
        context = "\n\n".join([d.page_content for d in docs])
        
        # 2. Get answer from LLM
        chain = prompt_template | llm
        response = chain.invoke({"context": context, "question": user_query})
        
        # 3. Format for Streamlit
        return {
            "answer": response.content,
            "sources": list(set([os.path.basename(d.metadata.get("source", "data.txt")) for d in docs]))
        }

    return rag_handler

if __name__ == "__main__":
    ingest_documents()