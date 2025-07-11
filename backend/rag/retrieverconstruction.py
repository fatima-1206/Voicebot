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

splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=64)

csv_path = "backend/rag/knowledge_base/links.csv"
VECSTORE_PATH = "backend/rag/vector_stores/faiss_index"
config_path = "config.json"
cache_file = "backend/rag/cache/cached_kb.pkl"

with open(config_path, "r") as f:
    config = json.load(f)

folder_path = "backend/rag/knowledge_base"
all_documents = []

embedding_model_name = "intfloat/multilingual-e5-base"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Global variables
db_initialized = False
faiss_db = None
hybrid_retriever = None
faiss_retriever = None
bm25_retriever = None



# Keep track of visited URLs to avoid re-scraping
visited_urls = set()

def load_scraped_documents(start_url:str, excluded_sublinks:str, depth=-1, json_path = "backend/rag/scraped_docs.json"):
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

def scrape_links_from_csv(csv_path=csv_path, json_path = "backend/rag/scraped_docs.json"):
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

def load_documents():
    global all_documents, db_initialized

    if db_initialized:
        print("Database already initialized. Skipping document loading.")
        return

    print("Loading documents from knowledge base...")

    # First try loading from cache
    if load_documents_cache():
        db_initialized = True
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
        return FAISS.load_local(VECSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🚫 FAISS index not found, constructing a new one...")
        return None



def construct_retrievers():
    """
    Construct the retrievers for the loaded documents.
    This function initializes the FAISS and BM25 retrievers.
    """
    global db_initialized, faiss_db, hybrid_retriever

    faiss_db = load_vectorstore()
    # the folder hash must be checked to make sure the faiss index is still valid
    if faiss_db and load_documents_cache():
        print("FAISS vector store loaded from disk.")
    else:
        if db_initialized:
            print("Retrievers already constructed. Skipping.")
            return faiss_db, bm25_retriever, faiss_retriever

        load_documents()
        
        if not all_documents:
            print("No documents loaded. Cannot construct retrievers.")
            return None, None, None

        # Step 2: Create FAISS vector store
        print("Creating FAISS vector store...")
        faiss_db = FAISS.from_documents(all_documents, embeddings)
        print("FAISS vector store created successfully.")

    if not all_documents:
        load_documents()
        if not all_documents:
            print("No documents loaded. Cannot construct retrievers.")
            return None, None, None

    # Step 2.1: Create FAISS retriever
    print("Creating FAISS retriever...")
    faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("FAISS retriever created successfully.")

    # Step 3: Create BM25 retriever
    print("Creating BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = 5  # Set number of documents to retrieve
    print("BM25 retriever created successfully.")
    
    print("Creating hybrid retriever (EnsembleRetriever)...")
    hybrid_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[config["faissWeight"], config["bm25Weight"]]
    )
    print("Hybrid retriever created successfully.")

    save_vectorstore(faiss_db)  # Save the FAISS vector store to disk

    db_initialized = True
    print("Retrievers constructed successfully.")
    unload_documents_and_embeddings(clear_faiss=False)  # Clear documents and embeddings after retrievers are created
    
    return faiss_db, hybrid_retriever

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
    if not db_initialized:
        construct_retrievers()
    

    return faiss_db, hybrid_retriever


def clear_cache():
    """Clear the documents cache file"""
  
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("Cache cleared")
    else:
        print("No cache file to clear")
