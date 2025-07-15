# To run celery:
# celery -A backend.workers.celery_worker.celery_app worker --loglevel=info --concurrency=1 --hostname=voicebot@%h
from backend.workers.celery_worker import celery_app
import json
from backend.response_generation.generator import generate_response
from backend.db.connection import SessionLocal
from uuid import UUID
from datetime import datetime
import torch
from contextlib import contextmanager
from typing import List, Dict, Any
import gc

# Add connection pooling and memory management
@contextmanager
def get_db_session():
    """Context manager for database sessions with proper cleanup"""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

@contextmanager
def manage_gpu_memory():
    """Context manager for GPU memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@celery_app.task(name='process_uploaded_document')
def process_uploaded_document(file_bytes, filename, doc_id, tenant_id, user_id, is_public):
    """Process uploaded document - extract text, create chunks, etc."""
    print(f"🔄 Task started for document: {filename} (doc_id: {doc_id})")
    
    try:
        from backend.db.connection import SessionLocal
        from backend.rag.add_document import process_uploaded_document as process_doc
        from sqlalchemy import text
        
        print(f"📊 Processing document with {len(file_bytes)} bytes")
        
        # Create database session with error handling
        db = SessionLocal()
        
        try:
            # Test database connection before proceeding (fixed SQL)
            db.execute(text("SELECT 1")).fetchone()
            print("✅ Database connection verified in task")
            
            # Convert file_bytes back to bytes if it was encoded as string
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode("latin1")
                print(f"🔧 Converted string to bytes: {len(file_bytes)} bytes")
            
            # Process the document (INCLUDE db parameter!)
            result = process_doc(
                file_bytes=file_bytes,
                filename=filename,
                doc_id=doc_id,
                tenant_id=tenant_id,
                user_id=user_id,
                is_public=is_public
                # db=db  # Don't forget this!
            )
            print(f"✅ Document processing completed: {result}")
            return result
            
        except Exception as db_error:
            print(f"❌ Database error in task: {db_error}")
            db.rollback()
            raise db_error
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

@celery_app.task(name='update_document_visibility')
def update_document_visibility(doc_id: str, is_public: bool, tenant_id: str):
    """Update document visibility"""
    print(f"🔄 Updating visibility for document {doc_id} to {'public' if is_public else 'private'}")
    
    with get_db_session() as db:
        try:
            from backend.db.models import Document as DB_Document
            from uuid import UUID
            
            doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id
            tenant_uuid = UUID(tenant_id) if isinstance(tenant_id, str) else tenant_id
            
            # Update document metadata
            document = db.query(DB_Document).filter(DB_Document.id == doc_uuid).first()
            if not document:
                raise ValueError(f"Document with ID {doc_id} not found")
            
            document.is_public = is_public
            document.status = "processed"
            db.commit()
            
            # Rebuild retrievers to reflect visibility change
            try:
                from backend.rag import retrieverconstruction
                retrieverconstruction.clear_cache()
                faiss_db, hybrid_retriever = retrieverconstruction.construct_retrievers(
                    include_postgres=True,
                    force_rebuild=True
                )
                print("✅ Retrievers rebuilt successfully")
            except Exception as e:
                print(f"❌ Error rebuilding retrievers: {e}")
            
            return {
                "status": "success",
                "doc_id": str(doc_uuid),
                "filename": document.filename,
                "is_public": is_public,
                "message": f"Document visibility updated successfully"
            }
            
        except Exception as e:
            print(f"❌ Error updating document visibility: {e}")
            return {"status": "error", "error": str(e)}

@celery_app.task(name='process_query', bind=True) 
def process_query_task(self, query_id: str, query: str, tenant_id: str, user_id: str, 
                      max_tokens: int = 500, temperature: float = 0.7, session_id: str = None):
    """Process a user query using RAG pipeline asynchronously"""
    print(f"🔄 Processing query {query_id}: {query[:50]}...")
    
    with manage_gpu_memory():
        try:
            # Generate response
            from backend.response_generation.generator import generate_response
            response = generate_response(query)
            
            # Save to database efficiently
            with get_db_session() as db:
                from backend.db.models import QueryLog as DB_QueryLog
                
                query_log_record = DB_QueryLog(
                    id=UUID(query_id),
                    user_id=UUID(user_id),
                    session_id=UUID(session_id) if session_id and session_id != 'string' else None,
                    query=query,
                    response=response,
                    timestamp=datetime.utcnow()
                )
                db.add(query_log_record)
                db.commit()
            
            return {
                "status": "success",
                "query_id": query_id,
                "query": query,
                "response": response,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "worker": self.request.hostname
            }
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "query_id": query_id,
                "query": query,
                "error": str(e),
                "worker": self.request.hostname
            }
