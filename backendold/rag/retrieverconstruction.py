"""
This module is responsible for loading documents from a knowledge base directory and storing them in FAISS and BM25.
It will only load the documents if the database is not already initialized.
It supports PDF, Word, EPUB, and text files.
It uses the langchain_community library for document loading.
"""
import torch
import gc
import os
import pickle
import hashlib
from datetime import datetime
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader,
    TextLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import json
from langchain_community.document_loaders import WikipediaLoader
from datasets import load_dataset
from langchain_core.documents import Document
from backend.rag.scraper import crawl_website, save_to_json
from langchain.text_splitter import TokenTextSplitter
import csv
# Add imports for database access
from backend.db.connection import SessionLocal
from backend.db.models import Document as DB_Document, Chunk as DB_Chunk
from sqlalchemy.orm import Session, joinedload

import os
from pathlib import Path

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Goes up 3 levels from this file
#
splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=64)

csv_path = str(PROJECT_ROOT / "backend/rag/knowledge_base/links.csv")
VECSTORE_PATH = str(PROJECT_ROOT / "backend/rag/vector_stores/faiss_index")
config_path = str(PROJECT_ROOT / "config.json")
cache_file = str(PROJECT_ROOT / "backend/rag/cache/cached_kb.pkl")
RETRIEVER_STATE_FILE = str(PROJECT_ROOT / "backend/rag/cache/retriever_state.pkl")

with open(config_path, "r") as f:
    config = json.load(f)

folder_path = str(PROJECT_ROOT / "backend/rag/knowledge_base")
all_documents = []

# Initialize embeddings as None - will be loaded only when needed
embeddings = None
embedding_model_name = "intfloat/multilingual-e5-base"

# Global variables - these will be loaded from persistent storage
db_initialized = False
faiss_db = None
hybrid_retriever = None
faiss_retriever = None
bm25_retriever = None

# Keep track of visited URLs to avoid re-scraping
visited_urls = set()

def get_embeddings():
    """Lazy load embeddings to avoid loading them in every worker"""
    global embeddings
    if embeddings is None:
        print(f"🔧 Loading embedding model: {embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return embeddings

def save_retriever_state():
    """Save the current state of retrievers to disk"""
    try:
        state = {
            'db_initialized': db_initialized,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(RETRIEVER_STATE_FILE), exist_ok=True)
        with open(RETRIEVER_STATE_FILE, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 Retriever state saved to {RETRIEVER_STATE_FILE}")
    except Exception as e:
        print(f"⚠️ Failed to save retriever state: {e}")

def load_retriever_state():
    """Load the retriever state from disk"""
    global db_initialized
    try:
        if os.path.exists(RETRIEVER_STATE_FILE):
            with open(RETRIEVER_STATE_FILE, 'rb') as f:
                state = pickle.load(f)
            db_initialized = state.get('db_initialized', False)
            print(f"📁 Loaded retriever state: db_initialized={db_initialized}")
            return True
    except Exception as e:
        print(f"⚠️ Failed to load retriever state: {e}")
        db_initialized = False
    return False

def load_documents_from_postgres(db: Session = None, tenant_id: str = None):
    """
    Load document chunks from PostgreSQL database and append them to all_documents.
    
    Args:
        db: Database session (optional, will create new if not provided)
        tenant_id: Filter documents by tenant_id (optional, loads all if not provided)
    """
    global all_documents
    
    # Create database session if not provided
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    
    try:
        print("Loading documents from PostgreSQL database...")
        
        # Query to get all processed documents, optionally filtered by tenant_id
        query = db.query(DB_Document).filter(DB_Document.status == 'processed')
     
        documents = query.all()
        
        postgres_docs_count = 0
        postgres_chunks_count = 0
        
        for doc in documents:
            # Get chunks for this document using the foreign key relationship
            chunks = db.query(DB_Chunk).filter(DB_Chunk.doc_id == doc.id).all()
            for chunk in chunks:
                # Create metadata for the chunk
                metadata = {
                    "source": "user_uploaded",
                    "filename": doc.filename,
                    "filetype": doc.filetype,
                    "doc_id": str(doc.id),
                    "tenant_id": str(doc.tenant_id),
                    "user_id": str(doc.user_id),
                    "is_public": doc.is_public,
                    "upload_time": doc.upload_time.isoformat() if doc.upload_time else None,
                    "file_size": doc.size,
                    "chunk_id": str(chunk.id),
                    "language": chunk.language
                }
                # print(metadata)
                
                # Create langchain Document and append to all_documents
                lc_document = Document(
                    page_content=chunk.text,
                    metadata=metadata
                )
                all_documents.append(lc_document)
                postgres_chunks_count += 1
            
            postgres_docs_count += 1
        
        print(f"  -> Loaded {postgres_chunks_count} chunks from {postgres_docs_count} PostgreSQL documents")
        return postgres_chunks_count, postgres_docs_count
        
    except Exception as e:
        print(f"  -> Error loading documents from PostgreSQL: {e}")
        return 0, 0
    
    finally:
        if close_db:
            db.close()

def load_scraped_documents(start_url:str, excluded_sublinks:str, depth=-1, json_path =str(PROJECT_ROOT / "backend/rag/scraped_docs.json")):
    """
        Takes a url, crawls the website and saves all documents extracted from it to the json file
    """
    scraped_docs = crawl_website(start_url, excluded_sublinks, depth)
    if scraped_docs:
        save_to_json(scraped_docs, json_path)
        for entry in scraped_docs:
            chunks = splitter.split_text(entry["content"])
            all_documents.extend([
                Document(page_content=chunk, metadata={"source": "university_website", "url": entry["url"]})
                for chunk in chunks
            ])

        print(f"  -> Crawled and chunked {len(scraped_docs)} scraped web documents")

def scrape_links_from_csv(csv_path=csv_path, json_path = str(PROJECT_ROOT / "backend/rag/scraped_docs.json")):
    """
    for all the links in the CSV file, scrape the website and save the documents
    """
    global all_documents

    if os.path.exists(json_path):
        print(f"Found cached scraped data at '{json_path}'. Skipping scraping...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data:
            chunks = splitter.split_text(entry["content"])
            all_documents.extend([
                Document(page_content=chunk, metadata={"source": "url", "url": entry["url"]})
                for chunk in chunks
            ])

            
        print(f"  -> Loaded {len(data)} scraped web documents")
        return

    # if the json file does not exist, start crawling
    print("Scraped data not found. Starting crawl...")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_url = row["url"]
            if row["depth"]:
                d = int(row["depth"])

                load_scraped_documents(start_url=start_url, excluded_sublinks=row["excluded_sublinks"], depth=d)
            else:
                load_scraped_documents(start_url=start_url, excluded_sublinks=row["excluded_sublinks"])

def load_documents(include_postgres=True):
    global all_documents, db_initialized

   

    print("Loading documents from knowledge base...")

    # First try loading from cache
    if load_documents_cache():
        db_initialized = True
        # Still load PostgreSQL documents even if cache exists, as they might be new
        if include_postgres:
            load_documents_from_postgres()
        return

    # If cache is not available or invalid, load from files
    print("Cache not available or invalid. Loading documents from files...")
    all_documents = []

    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    if not os.listdir(folder_path):
        print(f"Folder '{folder_path}' is empty.")
        return

    print("Cache not available or invalid. Loading documents from files...")

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.lower().endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        elif filename.lower().endswith(".epub"):
            loader = UnstructuredEPubLoader(filepath)
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            continue

        try:
            documents = loader.load()
            n_chunks = len(documents)
            for doc in documents:
                chunks = splitter.split_text(doc.page_content)
                all_documents.extend([
                    Document(page_content=chunk, metadata=doc.metadata)
                    for chunk in chunks
                ])
                n_chunks = len(chunks)
            print(f"  -> Loaded {n_chunks} chunks from {filename}")
        except Exception as e:
            print(f"  -> Error loading {filename}: {e}")

    print(f"Total local documents loaded: {len(all_documents)}")
    
    scrape_links_from_csv()
    print(f"Total local documents loaded after scraping: {len(all_documents)}")

    # Load documents from PostgreSQL
    if include_postgres:
        load_documents_from_postgres()
        print(f"Total documents loaded after PostgreSQL: {len(all_documents)}")

    # Wikipedia documents
    try:
        wiki_loader = WikipediaLoader(query="FAST National University Pakistan", lang="en", load_max_docs=3)
        wiki_docs = wiki_loader.load()
        for doc in wiki_docs:
            doc.metadata["source"] = "wikipedia"
        all_documents.extend(wiki_docs)

        wiki_loader = WikipediaLoader(query="Higher Education Commission (Pakistan)", lang="en", load_max_docs=3)
        wiki_docs = wiki_loader.load()
        for doc in wiki_docs:
            doc.metadata["source"] = "wikipedia"
        all_documents.extend(wiki_docs)

        print(f"  -> Loaded {len(wiki_docs)} Wikipedia docs")
    except Exception as e:
        print(f"  -> Failed to load Wikipedia docs: {e}")

    # Urdu QA dataset
    try:
        dataset = load_dataset("tydiqa", "primary_task")
        urdu_qas = [x for x in dataset["train"] if x["language"] == "ur"]
        urdu_docs = [
            Document(page_content=qa["context"], metadata={"question": qa["question"], "source": "tydiqa"})
            for qa in urdu_qas[:1000]
        ]
        all_documents.extend(urdu_docs)
        print(f"  -> Loaded {len(urdu_docs)} Urdu QA docs")
    except Exception as e:
        print(f"  -> Failed to load Urdu QA dataset: {e}")
    
    try:
        dataset = load_dataset("tydiqa", "primary_task")
        english_qas = [x for x in dataset["train"] if x["language"] == "en"]
        english_docs = [
            Document(page_content=qa["context"], metadata={"question": qa["question"], "source": "tydiqa"})
            for qa in english_qas[:1000]
        ]
        all_documents.extend(english_docs)
        print(f"  -> Loaded {len(english_docs)} English QA docs")
    except Exception as e:
        print(f"  -> Failed to load English QA dataset: {e}")

    if all_documents:
        save_documents_cache()

    db_initialized = True
    print(f"Total documents loaded (all sources): {len(all_documents)}")


def load_vectorstore():
    if os.path.exists(os.path.join(VECSTORE_PATH, "index.faiss")):
        print("📂 Loading FAISS vector store from disk...")
        embeddings = get_embeddings()  # Add this line
        return FAISS.load_local(VECSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🚫 FAISS index not found, constructing a new one...")
        return None



def construct_retrievers(include_postgres=True, tenant_id=None, user_id=None):
    """
    Construct the retrievers for the loaded documents.
    This function initializes the FAISS and BM25 retrievers.
    
    Args:
        include_postgres: Whether to include documents from PostgreSQL
        tenant_id: Filter PostgreSQL documents by tenant_id (optional)
        user_id: Filter PostgreSQL documents by user_id (optional)
    """
    global db_initialized, faiss_db, hybrid_retriever, faiss_retriever, bm25_retriever

    faiss_db = load_vectorstore()
    # the folder hash must be checked to make sure the faiss index is still valid
    if faiss_db and load_documents_cache():
        print("FAISS vector store loaded from disk.")
        # Still load PostgreSQL documents even if cache exists
        if include_postgres:
            load_documents_from_postgres()
    else:
        if db_initialized and faiss_db is not None and hybrid_retriever is not None:
            print("Retrievers already constructed. Skipping.")
            return faiss_db, hybrid_retriever  # Fix: Use consistent return signature

        load_documents(include_postgres=include_postgres)
        
        if not all_documents:
            print("No documents loaded. Cannot construct retrievers.")
            return None, None  # Fix: Use consistent return signature (2 values)

        # Step 2: Create FAISS vector store
        print("Creating FAISS vector store...")
        embeddings = get_embeddings()
        faiss_db = FAISS.from_documents(all_documents, embeddings)
        print("FAISS vector store created successfully.")

    if not all_documents:
        load_documents(include_postgres=include_postgres, )
        if not all_documents:
            print("No documents loaded. Cannot construct retrievers.")
            return None, None  # Fix: Use consistent return signature

    # Step 2.1: Create FAISS retriever
    print("Creating FAISS retriever...")
    faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("FAISS retriever created successfully.")

    # Step 3: Create BM25 retriever
    print("Creating BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = 5
    print("BM25 retriever created successfully.")
    
    print("Creating hybrid retriever (EnsembleRetriever)...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[config["faissWeight"], config["bm25Weight"]]
    )
    print("Hybrid retriever created successfully.")

    save_vectorstore(faiss_db)

    db_initialized = True
    print("Retrievers constructed successfully.")
    unload_documents_and_embeddings(clear_faiss=False)
    
    return faiss_db, hybrid_retriever  # Fix: Consistent return signature

def save_vectorstore(faiss_db):
    if not os.path.exists(VECSTORE_PATH):
        os.makedirs(VECSTORE_PATH)
    faiss_db.save_local(VECSTORE_PATH)
    print(f"🔐 FAISS vector store saved to {VECSTORE_PATH}")

def get_folder_hash():
    """Get hash of all files in knowledge base to detect changes"""
    if not os.path.exists(folder_path):
        return "empty_folder"
    
    file_info = []
    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            file_info.append(f"{filename}_{stat.st_mtime}_{stat.st_size}")
    
    if not file_info:
        return "empty_folder"
    
    return hashlib.md5("_".join(file_info).encode()).hexdigest()


def save_documents_cache():
    """Save processed documents to cache"""
    cache_data = {
        "documents": all_documents,
        "folder_hash": get_folder_hash(),
        "timestamp": datetime.now().isoformat(),
        "total_docs": len(all_documents)
    }
  
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved {len(all_documents)} documents to cache")
    except Exception as e:
        print(f"Failed to save cache: {e}")


def load_documents_cache():
    """Load documents from cache if valid"""
  
    if not os.path.exists(cache_file):
        print("No cache file found")
        return False
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check if folder content changed
        current_hash = get_folder_hash()
        cached_hash = cache_data.get("folder_hash")
        
        if cached_hash != current_hash:
            print("Knowledge base changed, cache invalid")
            return False
        
        global all_documents
        all_documents = cache_data["documents"]
        print(f"Loaded {len(all_documents)} documents from cache (saved on {cache_data.get('timestamp', 'unknown')})")
        return True
        
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return False



def unload_documents_and_embeddings(clear_faiss=False):
    """Unload documents and embeddings after retrievers are created."""
    global all_documents, embeddings, faiss_db

    all_documents = []

    # if embeddings:
    #     del embeddings
    #     embeddings = None

    # if clear_faiss and faiss_db:
    #     del faiss_db
    #     faiss_db = None

    gc.collect()
    torch.cuda.empty_cache()
    print("Documents and embeddings unloaded and GPU cache cleared.")


def get_retrievers():
    """Get the constructed retrievers. Constructs them if not already done."""
    global faiss_db, hybrid_retriever, db_initialized
    
    # Load state from disk first
    load_retriever_state()
    
    # Check if FAISS index exists on disk
    faiss_exists = os.path.exists(os.path.join(VECSTORE_PATH, "index.faiss"))
    
    if not db_initialized or faiss_db is None or hybrid_retriever is None:
        if faiss_exists:
            print("📁 Loading existing FAISS index from disk...")
            try:
                # Load FAISS from disk
                faiss_db = load_vectorstore()
                if faiss_db is not None:
                    print("✅ FAISS vector store loaded from disk")
                    
                    # Create retrievers from loaded FAISS
                    faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                    
                    # For BM25, we need to check if we have documents
                    global bm25_retriever
                    if bm25_retriever is None:
                        # Load documents for BM25 (lighter operation)
                        load_documents_for_bm25()
                    
                    if bm25_retriever is not None:
                        hybrid_retriever = EnsembleRetriever(
                            retrievers=[faiss_retriever, bm25_retriever],
                            weights=[config["faissWeight"], config["bm25Weight"]]
                        )
                        print("✅ Hybrid retriever created from existing components")
                    else:
                        print("⚠️ BM25 not available, using FAISS only")
                        hybrid_retriever = faiss_retriever
                    
                    db_initialized = True
                    save_retriever_state()
                    return faiss_db, hybrid_retriever
            except Exception as e:
                print(f"❌ Failed to load existing FAISS: {e}")
        
        # If loading failed or nothing exists, construct from scratch
        print("🔧 Constructing retrievers from scratch...")
        faiss_db, hybrid_retriever = construct_retrievers()
        if faiss_db is not None:
            db_initialized = True
            save_retriever_state()
    
    return faiss_db, hybrid_retriever

def load_documents_for_bm25():
    """Load documents specifically for BM25 retriever creation"""
    global bm25_retriever, all_documents
    
    try:
        # Load from cache first
        if load_documents_cache():
            print(f"📁 Loaded {len(all_documents)} documents from cache for BM25")
        else:
            # Load fresh documents
            load_documents(include_postgres=True)
        
        if all_documents:
            print("🔧 Creating BM25 retriever...")
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            bm25_retriever.k = 5
            print("✅ BM25 retriever created successfully")
        else:
            print("⚠️ No documents available for BM25")
            bm25_retriever = None
            
    except Exception as e:
        print(f"❌ Failed to create BM25 retriever: {e}")
        bm25_retriever = None

def update_faiss_and_hybrid_retriever(faiss_db):
    """
    Update the hybrid retriever with a new FAISS store while keeping existing BM25.
    This is efficient because we don't recreate BM25.
    """
    global hybrid_retriever, faiss_retriever
    
    try:
        # Create new FAISS retriever from updated FAISS store
        faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        print("✅ FAISS retriever updated")
        
        # Create new hybrid retriever with updated FAISS + existing BM25
        if bm25_retriever is not None:
            hybrid_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[config["faissWeight"], config["bm25Weight"]]
            )
            print("✅ Hybrid retriever updated (new FAISS + existing BM25)")
        else:
            print("⚠️ BM25 not available, using FAISS only")
            hybrid_retriever = faiss_retriever
        
        return faiss_retriever, hybrid_retriever
        
    except Exception as e:
        print(f"❌ Error updating retrievers: {e}")
        return None, None
