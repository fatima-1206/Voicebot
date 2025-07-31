"""
This file contains the code to add a document to the pre-existing vector database.
It requires the faiss_db and hybrid_retriever to be constructed first.
The chunks made from the document will be added to the vector database and the postgres database.
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
from celery import Celery
import torch
import gc

# Import config for weights
with open("config.json", "r") as f:
    config = json.load(f)

splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=64)

celery_app = Celery('tasks', broker='redis://localhost:6379/0')
db = None

@celery_app.task(name='process_uploaded_document')
def process_uploaded_document(file_bytes, filename, doc_id, tenant_id, user_id, is_public):
    """
    Process an uploaded document and add it to both the vector database and PostgreSQL.
    """
    from backend.db.connection import SessionLocal
    db = SessionLocal()  # Create session inside task
    temp_path = None

    try:
        print(f"🔄 Task started for document: {filename} (doc_id: {doc_id})")

        # Import and get fresh references to retrievers
        from backend.rag import retrieverconstruction

        # Get or initialize retrievers (this will use caching now)
        faiss_db, hybrid_retriever = retrieverconstruction.get_retrievers()

        if faiss_db is None:
            print("❌ Failed to initialize FAISS vector store")
            raise ValueError("Failed to initialize FAISS vector store")

        print(f"✅ Retrieved FAISS DB: {faiss_db is not None}")
        print(f"✅ Retrieved Hybrid Retriever: {hybrid_retriever is not None}")

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
            print(f"⚠️ Unsupported file extension: {extension}")
            raise ValueError(f"Unsupported file extension: {extension}")

        # Step 3: Load and split document
        loaded_docs = loader.load()
        chunks = []
        chunk_ids = []
        chunk_data_for_db = []

        for doc in loaded_docs:
            for chunk_text in splitter.split_text(doc.page_content):
                chunk_id = str(uuid4())
                chunk_ids.append(chunk_id)

                # Add chunk to FAISS list
                chunks.append(
                    LC_Document(
                        page_content=chunk_text,
                        metadata={
                            "chunk_id": chunk_id,
                            "filename": filename,
                            "tenant_id": str(tenant_id),
                            "user_id": str(user_id),
                            "doc_id": str(doc_id),
                            "filetype": extension,
                            "size": len(file_bytes),
                            "is_public": is_public,
                            **doc.metadata
                        }
                    )
                )

                # Store tuple for PostgreSQL insertion
                chunk_data_for_db.append((chunk_id, chunk_text))

        print(f"📝 Created {len(chunks)} chunks from document")

        # Step 4: Add to FAISS vectorstore and update retrievers
        if len(chunks) > 0:
            print(f"📝 Adding {len(chunks)} chunks to FAISS vector store...")
            faiss_db.add_documents(chunks, ids=chunk_ids)
            retrieverconstruction.save_vectorstore(faiss_db)
            print("✅ Chunks added to FAISS successfully")

            # Step 5: Update retrievers (new FAISS + existing BM25)
            faiss_retriever, hybrid_retriever = retrieverconstruction.update_faiss_and_hybrid_retriever(faiss_db)

            if faiss_retriever is not None and hybrid_retriever is not None:
                retrieverconstruction.faiss_db = faiss_db
                retrieverconstruction.hybrid_retriever = hybrid_retriever
                retrieverconstruction.faiss_retriever = faiss_retriever
                print("✅ Retrievers updated efficiently")
            else:
                print("❌ Failed to update retrievers")
        else:
            print("⚠️ Warning: No chunks created from document, skipping FAISS update")

        # Step 6: Update document metadata in PostgreSQL (don't create new, update existing)
        print(f"💾 Updating document metadata in PostgreSQL...")
        existing_doc = db.query(DB_Document).filter(DB_Document.id == doc_id).first()
        if existing_doc:
            existing_doc.path = temp_path
            existing_doc.status = "processed"
            print(f"✅ Updated existing document with ID: {existing_doc.id}")
        else:
            print(f"⚠️ Warning: Document with ID {doc_id} not found, creating new record")
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
            existing_doc = db_doc

        db.flush()

        # Step 7: Store chunks in database
        print(f"💾 Saving {len(chunks)} chunks to PostgreSQL...")
        chunks_saved = 0
        for chunk_id, chunk_text in chunk_data_for_db:
            try:
                db_chunk = DB_Chunk(
                    id=chunk_id,  # Set chunk_id explicitly
                    doc_id=existing_doc.id,
                    text=chunk_text,
                )
                db.add(db_chunk)
                chunks_saved += 1
            except Exception as chunk_error:
                print(f"❌ Error saving chunk {chunk_id}: {chunk_error}")


        # Commit all changes
        print("💾 Committing changes to database...")
        db.commit()
        print(f"✅ Successfully saved {chunks_saved} chunks to PostgreSQL")
        
        return {
            "status": "success",
            "doc_id": str(doc_id),
            "chunks_created": len(chunks),
            "chunks_saved": chunks_saved,
            "message": f"Document '{filename}' processed successfully"
        }

    except Exception as e:
        print(f"❌ Error processing document '{filename}': {str(e)}")
        db.rollback()
        raise e
    
    finally:

        db.close()  # Always close the session
        # Clean up temp file and free memory
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not delete temp file {temp_path}: {e}")
        
        # Clear GPU cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()



