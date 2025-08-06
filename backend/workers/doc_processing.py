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



@celery_app.task(name='delete_chunks_by_ids')
def delete_chunks_by_ids(chunk_ids: list[str]):
    """
    Celery task to delete chunks from the FAISS vector store using their chunk IDs.
    
    Args:
        chunk_ids (list[str]): A list of chunk UUIDs to delete from FAISS.
    """
    from backend.rag import retrieverconstruction

    try:
        # Load FAISS vectorstore
        faiss_db, _ = retrieverconstruction.get_retrievers()
        if faiss_db is None:
            print("❌ Could not load FAISS DB")
            return {"status": "error", "message": "FAISS DB not available"}

        print(f"🗑️ Deleting {len(chunk_ids)} chunks from FAISS...")

        # Perform the deletion
        faiss_db.delete(ids=chunk_ids)

        # Save the updated FAISS store
        retrieverconstruction.save_vectorstore(faiss_db)

        # Update retrievers
        faiss_retriever, hybrid_retriever = retrieverconstruction.update_faiss_and_hybrid_retriever(faiss_db)
        if faiss_retriever and hybrid_retriever:
            retrieverconstruction.faiss_db = faiss_db
            retrieverconstruction.faiss_retriever = faiss_retriever
            retrieverconstruction.hybrid_retriever = hybrid_retriever
            print("✅ FAISS and hybrid retrievers updated")
        else:
            print("⚠️ FAISS updated, but retrievers not refreshed")

        return {"status": "success", "deleted_chunk_count": len(chunk_ids)}

    except Exception as e:
        print(f"❌ Exception during FAISS deletion: {e}")
        return {"status": "error", "message": str(e)}

@celery_app.task(name='get_chunk_from_faiss_by_id')
def get_chunk_from_faiss_by_id(chunk_id: str):
    from backend.rag import retrieverconstruction

    try:
        faiss_db, _ = retrieverconstruction.get_retrievers()
        if faiss_db is None:
            return {"status": "error", "message": "FAISS DB not available"}

        for doc in faiss_db.docstore._dict.values():
            if doc.metadata.get("chunk_id") == chunk_id:
                return {
                    "status": "success",
                    "chunk_id": chunk_id,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }

        return {"status": "not_found", "chunk_id": chunk_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}



@celery_app.task(name='update_document_visibility')
def update_document_visibility(doc_id: str, is_public: bool, tenant_id: str, user_id: str):
    """
    Update document visibility in FAISS by:
    1. Fetching chunk IDs from Postgres for the given document.
    2. Retrieving corresponding chunks from FAISS.
    3. Deleting old chunks from FAISS.
    4. Re-inserting updated chunks with new `is_public` metadata.
    """
    from backend.db.connection import SessionLocal
    from backend.db.models import Chunk  # Assuming this is your SQLAlchemy model
    from backend.rag import retrieverconstruction
    from langchain_core.documents import Document as LC_Document

    db = SessionLocal()
    try:
        # Step 1: Get chunk IDs from Postgres
        chunks = db.query(Chunk).filter(
            Chunk.doc_id == doc_id
        ).all()

        if not chunks:
            return {"status": "error", "message": "No chunks found for the given document"}

        chunk_ids = [str(chunk.id) for chunk in chunks]

        # Step 2: Load FAISS and retrieve the chunks
        faiss_db, _ = retrieverconstruction.get_retrievers()
        if faiss_db is None:
            return {"status": "error", "message": "FAISS DB not available"}

        matched_chunks = []
        for doc in faiss_db.docstore._dict.values():
            if doc.metadata.get("chunk_id") in chunk_ids:
                doc.metadata["is_public"] = is_public  # Update visibility
                matched_chunks.append(
                    LC_Document(page_content=doc.page_content, metadata=doc.metadata)
                )

        if not matched_chunks:
            return {"status": "error", "message": "Chunks not found in FAISS"}

        # Step 3: Delete old chunks from FAISS
        faiss_db.delete(ids=chunk_ids)

        # Step 4: Reinsert updated chunks
        faiss_db.add_documents(matched_chunks)

        # Save FAISS store and refresh retrievers
        retrieverconstruction.save_vectorstore(faiss_db)      

        faiss_retriever, hybrid_retriever = retrieverconstruction.update_faiss_and_hybrid_retriever(faiss_db)

        if faiss_retriever is not None and hybrid_retriever is not None:
            retrieverconstruction.faiss_db = faiss_db
            retrieverconstruction.hybrid_retriever = hybrid_retriever
            retrieverconstruction.faiss_retriever = faiss_retriever
            print("✅ Retrievers updated efficiently")
        else:
            print("❌ Failed to update retrievers")

   
        return {
            "status": "success",
            "updated_chunk_count": len(matched_chunks),
            "is_public": is_public
        }

    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()
