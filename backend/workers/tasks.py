# To run celery:
# celery -A backend.workers.celery_worker.celery_app worker --loglevel=info --concurrency=1 --hostname=voicebot@%h
from backend.workers.celery_worker import celery_app

# Add this new task after your existing process_uploaded_document task
# Replace the update_document_visibility task with this corrected version:

@celery_app.task(name='update_document_visibility')
def update_document_visibility(doc_id: str, is_public: bool, tenant_id: str):
    """Update document visibility by rebuilding retrievers with updated metadata"""
    print(f"🔄 Updating visibility for document {doc_id} to {'public' if is_public else 'private'}")
    
    try:
        from backend.db.connection import SessionLocal
        from backend.rag import retrieverconstruction
        from backend.db.models import Document as DB_Document, Chunk as DB_Chunk
        from uuid import UUID
        from sqlalchemy import text
        
        # Create database session
        db = SessionLocal()
        
        try:
            # Verify database connection
            db.execute(text("SELECT 1")).fetchone()
            print("✅ Database connection verified for visibility update")
            
            # Convert string UUIDs to UUID objects
            doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id
            tenant_uuid = UUID(tenant_id) if isinstance(tenant_id, str) else tenant_id
            
            # Get the document
            document = db.query(DB_Document).filter(DB_Document.id == doc_uuid).first()
            if not document:
                raise ValueError(f"Document with ID {doc_id} not found")
            
            print(f"📄 Found document: {document.filename}")
            
            # Get all chunks for this document
            chunks = db.query(DB_Chunk).filter(DB_Chunk.doc_id == doc_uuid).all()
            print(f"📝 Found {len(chunks)} chunks to update")
            
            # Method: Force complete rebuild of retrievers
            print("🔧 Forcing complete rebuild of retrievers with updated metadata...")
            
            # Clear the cache to force fresh loading from database
            retrieverconstruction.clear_cache()
            
            # Reset global state
            retrieverconstruction.db_initialized = False
            retrieverconstruction.faiss_db = None
            retrieverconstruction.hybrid_retriever = None
            retrieverconstruction.faiss_retriever = None
            retrieverconstruction.bm25_retriever = None
            
            # Force reload of all documents from PostgreSQL (which now has updated metadata)
            retrieverconstruction.all_documents = []
            
            print("🔄 Reconstructing retrievers from scratch...")
            
            # Reconstruct with fresh data from database
            faiss_db, hybrid_retriever = retrieverconstruction.construct_retrievers(
                include_postgres=True,
                tenant_id=None  # Load all tenants to maintain complete index
            )
            
            if faiss_db is None or hybrid_retriever is None:
                raise ValueError("Failed to rebuild retrievers")
            
            print("✅ Retrievers rebuilt successfully with updated metadata")
            
            # Update the document status to indicate completion
            document.status = "processed"
            db.commit()
            
            return {
                "status": "success",
                "doc_id": str(doc_uuid),
                "filename": document.filename,
                "is_public": is_public,
                "chunks_count": len(chunks),
                "message": f"Document visibility updated to {'public' if is_public else 'private'}. Retrievers rebuilt with updated metadata."
            }
            
        except Exception as db_error:
            print(f"❌ Database error in visibility update: {db_error}")
            db.rollback()
            raise db_error
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error updating document visibility: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
    

    
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
                is_public=is_public,
                db=db  # Don't forget this!
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