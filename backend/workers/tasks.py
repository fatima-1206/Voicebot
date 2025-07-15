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

# Optimized batch document processing
@celery_app.task(name='batch_process_documents', bind=True)
def batch_process_documents(self, documents_data: List[Dict], tenant_id: str, user_id: str):
    """Process multiple documents in a single task for better efficiency"""
    print(f"🔄 Batch processing {len(documents_data)} documents")
    
    with manage_gpu_memory():
        with get_db_session() as db:
            try:
                from backend.rag.add_document import process_uploaded_document as process_doc
                
                results = []
                total_docs = len(documents_data)
                
                for i, doc_data in enumerate(documents_data):
                    self.update_state(
                        state='PROGRESS',
                        meta={'current': i + 1, 'total': total_docs, 'status': f'Processing {doc_data["filename"]}'}
                    )
                    
                    try:
                        result = process_doc(
                            file_bytes=doc_data["file_bytes"],
                            filename=doc_data["filename"],
                            doc_id=doc_data["doc_id"],
                            tenant_id=tenant_id,
                            user_id=user_id,
                            is_public=doc_data.get("is_public", False)
                        )
                        results.append({"doc_id": doc_data["doc_id"], "status": "success", "result": result})
                    except Exception as e:
                        results.append({"doc_id": doc_data["doc_id"], "status": "error", "error": str(e)})
                
                # Single retriever rebuild after all documents processed
                self.update_state(
                    state='PROGRESS',
                    meta={'current': total_docs, 'total': total_docs, 'status': 'Rebuilding search index...'}
                )
                
                # Only rebuild once at the end
                _rebuild_retrievers_once()
                
                return {
                    "status": "success",
                    "processed": len([r for r in results if r["status"] == "success"]),
                    "failed": len([r for r in results if r["status"] == "error"]),
                    "results": results
                }
                
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                return {"status": "error", "error": str(e)}

# Optimized single retriever rebuild function
def _rebuild_retrievers_once():
    """Rebuild retrievers only once - used by batch operations"""
    try:
        from backend.rag import retrieverconstruction
        
        # Clear cache efficiently
        retrieverconstruction.clear_cache()
        
        # Rebuild with optimized settings
        faiss_db, hybrid_retriever = retrieverconstruction.construct_retrievers(
            include_postgres=True,
            force_rebuild=True
        )
        
        if faiss_db is None or hybrid_retriever is None:
            raise ValueError("Failed to rebuild retrievers")
        
        print("✅ Retrievers rebuilt successfully")
        
    except Exception as e:
        print(f"❌ Error rebuilding retrievers: {e}")
        raise

# Optimized document visibility update (avoid full rebuild)
@celery_app.task(name='update_document_visibility_optimized')
def update_document_visibility_optimized(doc_id: str, is_public: bool, tenant_id: str):
    """Update document visibility without full retriever rebuild"""
    print(f"🔄 Updating visibility for document {doc_id} to {'public' if is_public else 'private'}")
    
    with get_db_session() as db:
        try:
            from backend.db.models import Document as DB_Document
            from uuid import UUID
            
            doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id
            tenant_uuid = UUID(tenant_id) if isinstance(tenant_id, str) else tenant_id
            
            # Update document metadata only
            document = db.query(DB_Document).filter(DB_Document.id == doc_uuid).first()
            if not document:
                raise ValueError(f"Document with ID {doc_id} not found")
            
            document.is_public = is_public
            document.status = "processed"
            db.commit()
            
            # Instead of full rebuild, just update metadata in memory
            from backend.rag import retrieverconstruction
            retrieverconstruction.update_document_metadata(doc_uuid, {"is_public": is_public})
            
            return {
                "status": "success",
                "doc_id": str(doc_uuid),
                "filename": document.filename,
                "is_public": is_public,
                "message": f"Document visibility updated efficiently"
            }
            
        except Exception as e:
            print(f"❌ Error updating document visibility: {e}")
            return {"status": "error", "error": str(e)}

# Optimized query processing with caching
@celery_app.task(name='process_query_optimized', bind=True)
def process_query_optimized(self, query_id: str, query: str, tenant_id: str, user_id: str, 
                           max_tokens: int = 500, temperature: float = 0.7, session_id: str = None):
    """Process query with optimized memory management and caching"""
    print(f"🔄 Processing query {query_id}: {query[:50]}...")
    
    with manage_gpu_memory():
        try:
            # Use cached response if available (implement query hash caching)
            query_hash = _get_query_hash(query, tenant_id)
            cached_response = _get_cached_response(query_hash)
            
            if cached_response:
                print(f"📋 Using cached response for query hash: {query_hash[:8]}")
                response = cached_response
            else:
                # Initialize components lazily
                _ensure_rag_initialized()
                
                # Generate response
                from backend.response_generation.generator import generate_response
                response = generate_response(query)
                
                # Cache the response
                _cache_response(query_hash, response)
            
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
                "worker": self.request.hostname,
                "cached": cached_response is not None
            }
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return {
                "status": "error",
                "query_id": query_id,
                "query": query,
                "error": str(e),
                "worker": self.request.hostname
            }

# Optimized batch query processing
@celery_app.task(name='batch_process_queries', bind=True)
def batch_process_queries(self, queries_data: List[Dict], tenant_id: str, user_id: str):
    """Process multiple queries efficiently in batch"""
    print(f"🔄 Batch processing {len(queries_data)} queries")
    
    with manage_gpu_memory():
        try:
            _ensure_rag_initialized()  # Initialize once for all queries
            
            from backend.response_generation.generator import generate_response
            
            results = []
            total_queries = len(queries_data)
            
            with get_db_session() as db:
                from backend.db.models import QueryLog as DB_QueryLog
                
                for i, query_data in enumerate(queries_data):
                    self.update_state(
                        state='PROGRESS',
                        meta={'current': i + 1, 'total': total_queries, 'status': f'Processing query {i+1}'}
                    )
                    
                    try:
                        query = query_data["query"]
                        query_id = query_data["query_id"]
                        
                        # Check cache first
                        query_hash = _get_query_hash(query, tenant_id)
                        cached_response = _get_cached_response(query_hash)
                        
                        if cached_response:
                            response = cached_response
                        else:
                            response = generate_response(query)
                            _cache_response(query_hash, response)
                        
                        # Save to database
                        query_log_record = DB_QueryLog(
                            id=UUID(query_id),
                            user_id=UUID(user_id),
                            query=query,
                            response=response,
                            timestamp=datetime.utcnow()
                        )
                        db.add(query_log_record)
                        
                        results.append({
                            "query_id": query_id,
                            "query": query,
                            "response": response,
                            "status": "success",
                            "cached": cached_response is not None
                        })
                        
                    except Exception as e:
                        results.append({
                            "query_id": query_data.get("query_id", "unknown"),
                            "query": query_data.get("query", ""),
                            "status": "error",
                            "error": str(e)
                        })
                
                # Commit all at once
                db.commit()
            
            return {
                "status": "success",
                "total_queries": total_queries,
                "successful": len([r for r in results if r["status"] == "success"]),
                "failed": len([r for r in results if r["status"] == "error"]),
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Error in batch query processing: {e}")
            return {"status": "error", "error": str(e)}

# Helper functions for caching and initialization
def _get_query_hash(query: str, tenant_id: str) -> str:
    """Generate hash for query caching"""
    import hashlib
    content = f"{query}:{tenant_id}"
    return hashlib.md5(content.encode()).hexdigest()

def _get_cached_response(query_hash: str) -> str:
    """Get cached response (implement with Redis or memory cache)"""
    # TODO: Implement with Redis for production
    return None

def _cache_response(query_hash: str, response: str):
    """Cache response (implement with Redis or memory cache)"""
    # TODO: Implement with Redis for production
    pass

def _ensure_rag_initialized():
    """Ensure RAG components are initialized only once"""
    try:
        from backend.response_generation.generator import initialize_rag_chain, initialize_model
        initialize_model()
        initialize_rag_chain()
    except Exception as e:
        print(f"❌ Failed to initialize RAG components: {e}")
        raise

# Keep original tasks for backward compatibility but mark as deprecated
@celery_app.task(name='process_uploaded_document', bind=True)
def process_uploaded_document(self, file_bytes, filename, doc_id, tenant_id, user_id, is_public):
    """Process a single uploaded document"""
    print(f"🔄 Processing single document: {filename}")
    
    with manage_gpu_memory():
        try:
            from backend.rag.add_document import process_uploaded_document as process_doc
            
            # Process the document
            result = process_doc(
                file_bytes=file_bytes,
                filename=filename,
                doc_id=doc_id,
                tenant_id=tenant_id,
                user_id=user_id,
                is_public=is_public
            )
            
            # Rebuild retrievers
            _rebuild_retrievers_once()
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "result": result
            }
            
        except Exception as e:
            print(f"❌ Error processing document {filename}: {e}")
            return {
                "status": "error",
                "doc_id": doc_id,
                "error": str(e)
            }

@celery_app.task(name='update_document_visibility')
def update_document_visibility(doc_id: str, is_public: bool, tenant_id: str):
    """DEPRECATED: Use update_document_visibility_optimized for better efficiency"""
    return update_document_visibility_optimized(doc_id, is_public, tenant_id)

# Keep only this definition (starting at line 352)
@celery_app.task(name='process_query', bind=True) 
def process_query_task(self, query_id: str, query: str, tenant_id: str, user_id: str, 
                      max_tokens: int = 500, temperature: float = 0.7, session_id: str = None):
    """Process a user query using RAG pipeline asynchronously"""
    print(f"🔄 Processing query {query_id}: {query[:50]}...")
    
    with manage_gpu_memory():
        try:
            # Use cached response if available
            query_hash = _get_query_hash(query, tenant_id)
            cached_response = _get_cached_response(query_hash)
            
            if cached_response:
                print(f"📋 Using cached response for query hash: {query_hash[:8]}")
                response = cached_response
            else:
                # Initialize components lazily
                _ensure_rag_initialized()
                
                # Generate response
                from backend.response_generation.generator import generate_response
                response = generate_response(query)
                
                # Cache the response
                _cache_response(query_hash, response)
            
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
                "worker": self.request.hostname,
                "cached": cached_response is not None
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
