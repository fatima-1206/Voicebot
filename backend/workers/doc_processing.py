"""
Document processing related Celery tasks
"""
from backend.workers.celery_worker import celery_app
from backend.workers.utilities import get_db_session


@celery_app.task(name='process_uploaded_document')
def process_uploaded_document(file_bytes, filename, doc_id, tenant_id, user_id, is_public):
    """Process uploaded document - extract text, create chunks, etc."""
    print(f"🔄 [DOC] Processing: {filename}")
    
    try:
        from backend.db.connection import SessionLocal
        from backend.rag.add_document import process_uploaded_document as process_doc
        from sqlalchemy import text
        
        db = SessionLocal()
        
        try:
            db.execute(text("SELECT 1")).fetchone()
            
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode("latin1")
            
            result = process_doc(
                file_bytes=file_bytes,
                filename=filename,
                doc_id=doc_id,
                tenant_id=tenant_id,
                user_id=user_id,
                is_public=is_public
            )
            
            return result
            
        except Exception as db_error:
            db.rollback()
            raise db_error
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ [DOC] Error: {e}")
        return {"status": "error", "error": str(e)}


@celery_app.task(name='update_document_visibility')
def update_document_visibility(doc_id: str, is_public: bool, tenant_id: str):
    """Update document visibility"""
    print(f"🔄 [DOC] Updating visibility for {doc_id}")
    
    with get_db_session() as db:
        try:
            from backend.db.models import Document as DB_Document
            from uuid import UUID
            
            doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id
            
            document = db.query(DB_Document).filter(DB_Document.id == doc_uuid).first()
            if not document:
                raise ValueError(f"Document with ID {doc_id} not found")
            
            document.is_public = is_public
            document.status = "processed"
            db.commit()
            
            # Rebuild retrievers
            try:
                from backend.rag import retrieverconstruction
                retrieverconstruction.clear_cache()
                retrieverconstruction.construct_retrievers(
                    include_postgres=True,
                    force_rebuild=True
                )
            except Exception as e:
                print(f"❌ Error rebuilding retrievers: {e}")
            
            return {
                "status": "success",
                "doc_id": str(doc_uuid),
                "is_public": is_public,
                "message": "Document visibility updated successfully"
            }
            
        except Exception as e:
            print(f"❌ [DOC] Error updating visibility: {e}")
            return {"status": "error", "error": str(e)}