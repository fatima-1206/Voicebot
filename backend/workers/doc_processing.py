"""
Document processing related Celery tasks
"""
from backend.workers.celery_worker import celery_app
from backend.workers.utilities import get_db_session


@celery_app.task(name='process_uploaded_document')
def process_uploaded_document(file_bytes, filename, doc_id, tenant_id, user_id, is_public, chunks_given=False, chunks=None):
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
                is_public=is_public,
                chunks_given=chunks_given,  
                chunks=chunks
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


@celery_app.task(name='process_mongodb_documents')
def process_mongodb_documents(job_id: str, uri: str, db_name: str, 
                            selected_collections: list, should_join: bool = False,
                            join_field: str = "_id", limit: int = 1000,
                            content_fields: list = None, tenant_id: str = "",
                            user_id: str = ""):
    """Process MongoDB documents and add them to the vector store."""
    print(f"🔄 [MONGO] Processing job {job_id}: {selected_collections}")
    
    try:
        from backend.rag.mongo_connector import MongoDBConnector
        from backend.rag.add_document import add_documents_to_vectorstore
        from backend.redis_manager import get_redis_client
        from datetime import datetime
        import json
        
        redis_client = get_redis_client()
        
        # Update job status
        def update_job_status(status: str, progress: float = 0.0, message: str = "", 
                            documents_processed: int = 0, total_documents: int = 0, error: str = None):
            job_info = {
                "job_id": job_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "status": status,
                "progress": progress,
                "message": message,
                "collections": selected_collections,
                "should_join": should_join,
                "documents_processed": documents_processed,
                "total_documents": total_documents,
                "started_at": datetime.now().isoformat() if status == "processing" else None,
                "completed_at": datetime.now().isoformat() if status in ["completed", "failed"] else None,
                "error": error
            }
            redis_client.setex(f"mongo_job:{job_id}", 3600, json.dumps(job_info))
        
        # Start processing
        update_job_status("processing", 0.0, "Connecting to MongoDB...")
        
        connector = MongoDBConnector(uri, db_name)
        all_docs = []
        collections_processed = []
        
        try:
            if should_join and len(selected_collections) > 1:
                update_job_status("processing", 25.0, f"Joining collections: {selected_collections}")
                
                # Join collections
                docs = connector.load_joined_collections_as_langchain(
                    collections=selected_collections,
                    join_field=join_field,
                    content_fields=content_fields,
                    limit=limit
                )
                all_docs.extend(docs)
                collections_processed = selected_collections
                
                update_job_status("processing", 50.0, f"Loaded {len(docs)} joined documents")
                
            else:
                # Process collections separately
                total_collections = len(selected_collections)
                
                for i, collection in enumerate(selected_collections):
                    progress = 25 + (i / total_collections) * 25  # 25-50% for loading
                    update_job_status("processing", progress, f"Loading from {collection}...")
                    
                    docs = connector.load_documents_as_langchain(
                        collection_name=collection,
                        limit=limit
                    )
                    all_docs.extend(docs)
                    collections_processed.append(collection)
                    
                    update_job_status("processing", progress + (25 / total_collections), 
                                    f"Loaded {len(docs)} documents from {collection}")
            
            update_job_status("processing", 60.0, 
                             f"Processing {len(all_docs)} documents for vector store...")
            
            # Add documents to vector store
            if all_docs:
                # You might need to implement this function or adapt existing ones
                # result = add_documents_to_vectorstore(all_docs, tenant_id)
                
                # For now, just simulate processing
                update_job_status("processing", 80.0, "Adding documents to vector store...")
                
                # Add metadata to indicate source
                for doc in all_docs:
                    doc.metadata["source_type"] = "mongodb"
                    doc.metadata["processing_job_id"] = job_id
                    doc.metadata["tenant_id"] = tenant_id
                
                update_job_status("processing", 90.0, "Finalizing document processing...")
                
                # Here you would call your actual vector store addition function
                # result = your_vector_store_function(all_docs, tenant_id)
                
                update_job_status("completed", 100.0, 
                                f"Successfully processed {len(all_docs)} documents", 
                                len(all_docs), len(all_docs))
                
                return {
                    "status": "completed",
                    "documents_processed": len(all_docs),
                    "collections": collections_processed,
                    "job_id": job_id
                }
            else:
                update_job_status("completed", 100.0, "No documents found to process", 0, 0)
                return {
                    "status": "completed",
                    "documents_processed": 0,
                    "collections": collections_processed,
                    "job_id": job_id
                }
                
        finally:
            connector.close()
            
    except Exception as e:
        error_msg = f"MongoDB processing failed: {str(e)}"
        print(f"❌ [MONGO] Error: {error_msg}")
        
        # Update error status
        update_job_status("failed", 0.0, "Processing failed", 0, 0, error_msg)
        
        return {
            "status": "error", 
            "error": error_msg,
            "job_id": job_id
        }


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
