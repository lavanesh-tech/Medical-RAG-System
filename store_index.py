import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone

# 1. Load Environment Variables with override to ensure .env takes priority
load_dotenv(override=True)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "medical-chatbot" 

# 2. Extract Data from PDF
def load_pdf(data_path):
    print(f"Reading PDFs from {data_path}...")
    loader = PyPDFDirectoryLoader(data_path)
    return loader.load()

# 3. Metadata Filtering (Stripping complex metadata to prevent upload errors)
def filter_to_minimal_docs(docs):
    minimal_docs = []
    for doc in docs:
        # We only keep the text and the source file name
        src = doc.metadata.get("source", "Unknown")
        minimal_docs.append(
            Document(
                page_content=doc.page_content, 
                metadata={"source": src}
            )
        )
    return minimal_docs

# 4. Chunking Logic
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20
    )
    return text_splitter.split_documents(extracted_data)

# 5. Download Embedding Model
def download_embeddings():
    print("Downloading HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

if __name__ == "__main__":
    print("--- Starting Ingestion Process ---")
    
    # Validation: Stop early if API Key is missing
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found! Check your .env file and ensure no quotes are used.")

    # Step 1: Load and Filter
    raw_docs = load_pdf("data/")
    print(f"Successfully loaded {len(raw_docs)} pages.")
    
    minimal_docs = filter_to_minimal_docs(raw_docs)
    
    # Step 2: Chunk
    chunks = text_split(minimal_docs)
    print(f"Created {len(chunks)} text chunks.")
    
    # Step 3: Embeddings
    embeddings = download_embeddings()
    
    # Step 4: Initialize Pinecone Client & Upload
    print(f"Connecting to Pinecone index: {index_name}...")
    
    try:
        # Standardize connection
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Uploading documents
        print("Uploading to Pinecone (this may take a few minutes for 5000+ chunks)...")
        docsearch = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        print("--- Ingestion Complete! ---")
        # Final Verification
        index = pc.Index(index_name)
        print("Index Stats:")
        print(index.describe_index_stats())
        
    except Exception as e:
        print(f"❌ ERROR DURING UPLOAD: {e}")
        print("Tip: If you see 'Unauthorized (401)', your API Key in .env is likely incorrect or contains quotes.")