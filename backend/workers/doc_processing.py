"""
Document processing related Celery tasks
"""
from backend.workers.celery_worker import celery_app
from backend.workers.utilities import get_db_session
import json
import logging
from typing import Dict, List, Any, Optional
import tempfile
import os

# Set up logging
logger = logging.getLogger(__name__)

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


@celery_app.task(name='process_graphql_to_rag')
def process_graphql_to_rag(
    api_url: str, 
    headers: Optional[Dict[str, str]] = None,
    source_name: str = "GraphQL API",
    max_depth: int = 3,
    tenant_id: str = "default",
    user_id: str = "system",
    device_id: str = "unknown",
    session_id: str = "unknown",
    is_public: bool = True
):
    """
    Celery task to extract data from a GraphQL API and process it into RAG documents.
    
    Args:
        api_url: The GraphQL endpoint URL
        headers: Optional headers for authentication (e.g., {"Authorization": "Bearer token"})
        source_name: Name of the data source for metadata
        max_depth: Maximum depth for nested GraphQL queries
        tenant_id: Tenant ID for document storage
        user_id: User ID for document storage
        is_public: Whether documents should be public
        
    Returns:
        Dict with status and processing results
    """
    print(f"🔄 [GRAPHQL-RAG] Starting GraphQL extraction and RAG processing")
    print(f"    API URL: {api_url}")
    print(f"    Source: {source_name}")
    print(f"    Max Depth: {max_depth}")
    print(f"    Tenant ID: {tenant_id}")
    print(f"    User ID: {user_id}")
    
    try:
        # Import the required classes
        import sys
        import os
        
        # Add the root directory to Python path to import our modules
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if root_dir not in sys.path:
            sys.path.append(root_dir)
        
        from backend.rag.gql_extractor import GraphQLExtractor
        from backend.rag.rag_document_processor import RAGDocumentProcessor
        
        # Step 1: Extract data from GraphQL API
        print(f"📡 [GRAPHQL-RAG] Extracting data from GraphQL API...")
        extractor = GraphQLExtractor(api_url, headers)
        
        # Fetch the schema first
        schema = extractor.fetch_schema()
        if not schema:
            return {
                "status": "error", 
                "error": "Failed to fetch GraphQL schema",
                "stage": "schema_fetch"
            }
        
        # Extract all available data
        extracted_data = extractor.extract_all_data(max_depth=max_depth)
        if not extracted_data:
            return {
                "status": "error", 
                "error": "No data extracted from GraphQL API",
                "stage": "data_extraction"
            }
        
        print(f"✅ [GRAPHQL-RAG] Extracted {len(extracted_data)} root fields from GraphQL API")
        
        # Step 2: Process the extracted data into RAG documents
        print(f"🔄 [GRAPHQL-RAG] Processing data into RAG documents...")
        processor = RAGDocumentProcessor(source_name)
        documents = processor.process_graphql_data(extracted_data)
        
        if not documents:
            return {
                "status": "error", 
                "error": "No documents generated from GraphQL data",
                "stage": "rag_processing"
            }
        
        # Create relationship documents
        relationship_docs = processor.create_relationship_documents()
        all_documents = documents + relationship_docs
        
        print(f"✅ [GRAPHQL-RAG] Generated {len(all_documents)} RAG documents")
        
        # Step 3: Add documents to the RAG system
        print(f"🔄 [GRAPHQL-RAG] Adding documents to RAG system...")
        
        from backend.rag.add_document import add_text_document_to_rag
        from backend.db.connection import SessionLocal
        import uuid
        
        db = SessionLocal()
        added_documents = []
        failed_documents = []
        
        try:
            for i, doc in enumerate(all_documents):
                try:
                    # Create a unique document ID
                    doc_id = str(uuid.uuid4())
                    
                    # Create document title and content
                    title = doc.get('title', f"GraphQL Document {i+1}")
                    content = doc.get('content', '')
                    
                    # Add metadata as part of the content
                    metadata = doc.get('metadata', {})
                    metadata_text = f"\n\n--- Document Metadata ---\n"
                    metadata_text += f"Source: {metadata.get('source', source_name)}\n"
                    metadata_text += f"Type: {metadata.get('type', 'unknown')}\n"
                    metadata_text += f"ID: {metadata.get('id', doc_id)}\n"
                    
                    if metadata.get('keywords'):
                        metadata_text += f"Keywords: {', '.join(metadata['keywords'])}\n"
                    
                    full_content = content + metadata_text
                    
                    print(f"🔄 Adding document {i+1}/{len(all_documents)}: {title}")
                    
                    # Add to RAG system using the new function
                    result = add_text_document_to_rag(
                        title=title,
                        content=full_content,
                        doc_id=doc_id,
                        tenant_id=tenant_id,  # This will be safely converted to UUID
                        user_id=user_id,      # This will be safely converted to UUID
                        is_public=is_public,
                        db=db
                    )
                    
                    if result.get('status') == 'success':
                        added_documents.append({
                            'doc_id': doc_id,
                            'title': title,
                            'type': doc.get('type'),
                            'chunk_count': result.get('chunk_count', 0)
                        })
                        print(f"✅ [GRAPHQL-RAG] Added document: {title}")
                    else:
                        failed_documents.append({
                            'title': title,
                            'error': result.get('error', 'Unknown error')
                        })
                        print(f"❌ [GRAPHQL-RAG] Failed to add document: {title} - {result.get('error')}")
                        
                except Exception as doc_error:
                    failed_documents.append({
                        'title': doc.get('title', f'Document {i+1}'),
                        'error': str(doc_error)
                    })
                    print(f"❌ [GRAPHQL-RAG] Error processing document {i+1}: {doc_error}")
                    
        except Exception as db_error:
            db.rollback()
            print(f"❌ [GRAPHQL-RAG] Database error: {db_error}")
            raise db_error
        finally:
            db.close()
        
        # Step 4: Save processed documents to files (optional)
        print(f"🔄 [GRAPHQL-RAG] Saving processed documents to files...")
        
        try:
            # Save to knowledge base directory
            knowledge_base_dir = "/home/genesys/voicebot/Apollo-Voicebot/backend/rag/knowledge_base"
            os.makedirs(knowledge_base_dir, exist_ok=True)
            
            # Save JSON version
            json_filename = f"graphql_{source_name.lower().replace(' ', '_')}_{len(all_documents)}_docs.json"
            json_path = os.path.join(knowledge_base_dir, json_filename)
            
            output_data = {
                "metadata": {
                    **processor.metadata, 
                    "entity_types": list(processor.metadata["entity_types"]),
                    "api_url": api_url,
                    "extraction_params": {
                        "max_depth": max_depth,
                        "source_name": source_name
                    }
                },
                "documents": all_documents,
                "total_documents": len(all_documents)
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ [GRAPHQL-RAG] Saved JSON to: {json_path}")
            
            # Save individual text files
            text_dir = os.path.join(knowledge_base_dir, f"graphql_{source_name.lower().replace(' ', '_')}_texts")
            os.makedirs(text_dir, exist_ok=True)
            
            for doc in all_documents:
                safe_id = "".join(c for c in doc['id'] if c.isalnum() or c in "_-").lower()
                filename = f"{safe_id}.txt"
                filepath = os.path.join(text_dir, filename)
                
                content_lines = [
                    f"Title: {doc['title']}",
                    f"Type: {doc['type']}",
                    f"Source: {doc['metadata']['source']}",
                    f"ID: {doc['id']}",
                    f"Keywords: {', '.join(doc['metadata'].get('keywords', []))}",
                    "=" * 50,
                    "",
                    doc['content']
                ]
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(content_lines))
            
            print(f"✅ [GRAPHQL-RAG] Saved {len(all_documents)} text files to: {text_dir}")
            
        except Exception as file_error:
            print(f"⚠️ [GRAPHQL-RAG] Warning: Could not save files: {file_error}")
        
        # Return comprehensive results
        total_chunks = sum(doc.get('chunk_count', 0) for doc in added_documents)
        
        result = {
            "status": "success",
            "summary": {
                "api_url": api_url,
                "source_name": source_name,
                "extracted_fields": len(extracted_data),
                "generated_documents": len(all_documents),
                "added_documents": len(added_documents),
                "failed_documents": len(failed_documents),
                "total_chunks": total_chunks,
                "entity_types": list(processor.metadata.get("entity_types", []))
            },
            "added_documents": added_documents,
            "failed_documents": failed_documents,
            "files_saved": {
                "json_path": json_path if 'json_path' in locals() else None,
                "text_dir": text_dir if 'text_dir' in locals() else None
            }
        }
        
        print(f"🎉 [GRAPHQL-RAG] Complete! Added {len(added_documents)} documents with {total_chunks} chunks")
        return result
        
    except Exception as e:
        error_msg = f"GraphQL to RAG processing failed: {str(e)}"
        print(f"❌ [GRAPHQL-RAG] {error_msg}")
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error", 
            "error": error_msg,
            "stage": "unknown"
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
def update_document_visibility(doc_id: str, is_public: bool, tenant_id: str, user_id: str, device_id: str = "unknown", session_id: str = "unknown"):
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