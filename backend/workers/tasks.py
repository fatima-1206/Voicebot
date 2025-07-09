# backend/worker/tasks.py

from backend.workers.celery_worker import celery_app
@celery_app.task
def process_uploaded_document(file_bytes, filename, doc_id, tenant_id, user_id, is_public):
    """Process uploaded document - extract text, create chunks, etc."""
    try:
        from backend.db.connection import SessionLocal
        from backend.rag.add_document import process_uploaded_document as process_doc
        
        # Create database session
        db = SessionLocal()
        
        try:
            # Convert file_bytes back to bytes if it was encoded as string
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode("latin1")
            
            result = process_doc(file_bytes, filename, doc_id, tenant_id, user_id, is_public, db)
            return result
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error processing document: {e}")
        return {"status": "error", "error": str(e)}