"""
    this file contains the code to add a document to the pre existing vector database.
    It requires the faiss_db and hybrid_retriever to be constructed first.
    The chunks made from the document will be added to the vector dtabase and the postgres database.
    When the vector database is constructed, all chunks in the postgres database will be added to the vector database.
"""
from langchain_core.documents import Document as LC_Document
from langchain.text_splitter import TokenTextSplitter
from uuid import uuid4, UUID
import mimetypes
import os
from datetime import datetime
from backend.db.models import Document as DB_Document, Chunk as DB_Chunk
from sqlalchemy.orm import Session
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader,
    TextLoader
)
from langchain.retrievers import EnsembleRetriever
from io import BytesIO
import tempfile
import json

# Import config for weights
with open("config.json", "r") as f:
    config = json.load(f)

splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=64)


def process_uploaded_document(file_bytes, filename, doc_id, tenant_id, user_id, is_public, db: Session):
    """
    Process an uploaded document and add it to both the vector database and PostgreSQL.
    
    Args:
        file_bytes: The file content as bytes
        filename: Original filename
        doc_id: Document UUID (as string)
        tenant_id: Tenant UUID (as string) 
        user_id: User UUID (as string)
        is_public: Boolean indicating if document is public
        db: SQLAlchemy database session
    """
    global faiss_db, hybrid_retriever, bm25_retriever
    temp_path = None
    
    try:
        # Convert string UUIDs to UUID objects
        doc_id = UUID(doc_id) if isinstance(doc_id, str) else doc_id
        tenant_id = UUID(tenant_id) if isinstance(tenant_id, str) else tenant_id
        user_id = UUID(user_id) if isinstance(user_id, str) else user_id

        # Step 1: Save to temp file
        extension = os.path.splitext(filename)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        # Step 2: Pick loader
        if extension == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif extension == ".docx":
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif extension == ".epub":
            loader = UnstructuredEPubLoader(temp_path)
        elif extension == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: {extension}")

        # Step 3: Load and split
        docs = loader.load()
        chunks = []
        for doc in docs:
            for chunk in splitter.split_text(doc.page_content):
                chunks.append(
                    LC_Document(page_content=chunk, metadata={
                        "filename": filename,
                        "tenant_id": str(tenant_id),
                        "user_id": str(user_id),
                        "doc_id": str(doc_id),
                        "is_public": is_public,
                        **doc.metadata
                    })
                )

        # Step 4: Add to FAISS vectorstore
        if not faiss_db:
            raise ValueError("FAISS vector store is not initialized.")

        # Step 5: Re-create only the FAISS retriever to reflect the updated index
        faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Step 6: Re-create hybrid retriever with UPDATED faiss_retriever + STATIC bm25_retriever
        if bm25_retriever is not None:
            hybrid_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[config["faissWeight"], config["bm25Weight"]]
            )
            print("🔄 Updated hybrid retriever with new FAISS documents (BM25 kept static)")
        else:
            print("⚠️ Warning: BM25 retriever not available, using only FAISS")
            hybrid_retriever = faiss_retriever

        # Step 7: Store metadata in PostgreSQL
        db_doc = DB_Document(
            id=doc_id,
            filename=filename,
            path=temp_path,
            filetype=extension,
            tenant_id=tenant_id,
            user_id=user_id,
            is_public=is_public,
            size=len(file_bytes),
            status="processed",
            upload_time=datetime.utcnow()
        )
        db.add(db_doc)
        db.flush()

        # Store chunks in database
        for chunk in chunks:
            db_chunk = DB_Chunk(
                doc_id=db_doc.id,
                text=chunk.page_content,
            )
            db.add(db_chunk)

        db.commit()
        print(f"✅ Uploaded and processed document '{filename}' with {len(chunks)} chunks.")
        print(f"📊 FAISS updated with new documents, BM25 remains static (as intended)")
        
        return {
            "status": "success",
            "doc_id": str(doc_id),
            "chunks_created": len(chunks),
            "message": f"Document '{filename}' processed successfully"
        }

    except Exception as e:
        print(f"❌ Error processing document '{filename}': {str(e)}")
        db.rollback()
        raise e
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not delete temp file {temp_path}: {e}")