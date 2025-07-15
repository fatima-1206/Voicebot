# To run celery:
# celery -A backend.workers.celery_worker.celery_app worker --loglevel=info --concurrency=1 --hostname=voicebot@%h
from backend.workers.celery_worker import celery_app
import json
from backend.response_generation.generator import generate_response
from backend.db.connection import SessionLocal
# from backend.db.models import Query as DB_Query
from uuid import UUID
from datetime import datetime
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
    

# Add this new task after your existing tasks

@celery_app.task(name='scrape_link')
def scrape_link(link_id: str, url: str, depth: int, excluded_sublinks: str, tenant_id: str, user_id: str):
    """Scrape a website link and add documents to the knowledge base"""
    print(f"🔄 Starting scraping for link {link_id}: {url}")
    
    try:
        from backend.db.connection import SessionLocal
        from backend.db.models import Link, Document as DB_Document, Chunk as DB_Chunk
        from backend.rag.scraper import crawl_website
        from backend.rag import retrieverconstruction
        from langchain_core.documents import Document as LC_Document
        from langchain.text_splitter import TokenTextSplitter
        from uuid import UUID, uuid4
        from datetime import datetime
        
        db = SessionLocal()
        
        try:
            # Get the link record
            link_uuid = UUID(link_id)
            link = db.query(Link).filter(Link.id == link_uuid).first()
            
            if not link:
                raise ValueError(f"Link with ID {link_id} not found")
            
            print(f"📄 Scraping: {link.name or url}")
            
            # Perform the scraping
            scraped_docs = crawl_website(
                start_url=url,
                excluded_sublinks=excluded_sublinks,
                depth=depth
            )
            
            if not scraped_docs:
                link.status = "completed"
                link.last_scraped = datetime.utcnow()
                db.commit()
                return {
                    "status": "success",
                    "message": "No content found to scrape",
                    "docs_scraped": 0
                }
            
            print(f"📝 Scraped {len(scraped_docs)} pages")
            
            # Process scraped content into chunks
            splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=64)
            
            total_chunks = 0
            for doc_data in scraped_docs:
                # Create a document record for each scraped page
                doc_id = uuid4()
                
                # Create document in database
                document = DB_Document(
                    id=doc_id,
                    tenant_id=UUID(tenant_id),
                    user_id=UUID(user_id),
                    filename=f"scraped_{doc_data['url'].split('/')[-1] or 'page'}.html",
                    filetype=".html",
                    size=len(doc_data['content']),
                    is_public=False,  # Scraped content is private by default
                    status="processed",
                    upload_time=datetime.utcnow()
                )
                db.add(document)
                db.flush()  # Get the ID
                
                # Split content into chunks
                chunks = splitter.split_text(doc_data['content'])
                
                # Save chunks to database
                for chunk_text in chunks:
                    chunk = DB_Chunk(
                        doc_id=document.id,
                        text=chunk_text,
                        language="en"  # Assume English for web content
                    )
                    db.add(chunk)
                    total_chunks += 1
                
                print(f"📄 Created document {document.filename} with {len(chunks)} chunks")
            
            # Commit all database changes
            db.commit()
            print(f"💾 Saved {len(scraped_docs)} documents with {total_chunks} total chunks")
            
            # Update vector store
            print("🔧 Updating vector store with scraped content...")
            
            # Force rebuild of retrievers to include new content
            retrieverconstruction.clear_cache()
            retrieverconstruction.db_initialized = False
            retrieverconstruction.faiss_db = None
            retrieverconstruction.hybrid_retriever = None
            
            # Reconstruct with new data
            faiss_db, hybrid_retriever = retrieverconstruction.construct_retrievers(
                include_postgres=True
            )
            
            if faiss_db is None:
                print("⚠️ Warning: Failed to update vector store")
            else:
                print("✅ Vector store updated with scraped content")
            
            # Update link status
            link.status = "completed"
            link.last_scraped = datetime.utcnow()
            link.updated_at = datetime.utcnow()
            db.commit()
            
            return {
                "status": "success",
                "link_id": link_id,
                "url": url,
                "docs_scraped": len(scraped_docs),
                "total_chunks": total_chunks,
                "message": f"Successfully scraped {len(scraped_docs)} pages with {total_chunks} chunks"
            }
            
        except Exception as db_error:
            print(f"❌ Database error in scraping: {db_error}")
            
            # Update link status to error
            try:
                link = db.query(Link).filter(Link.id == UUID(link_id)).first()
                if link:
                    link.status = "error"
                    link.updated_at = datetime.utcnow()
                    db.commit()
            except:
                pass
            
            db.rollback()
            raise db_error
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error scraping link: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
    

# Add this task to your existing tasks.py file

@celery_app.task(name='process_query')
def process_query_task(query_id: str, query: str, tenant_id: str, user_id: str, 
                      max_tokens: int = 500, temperature: float = 0.7, session_id: str = None):
    """Process a user query using RAG pipeline asynchronously"""
    print(f"🔄 Processing query {query_id}: {query[:50]}...")
    
    try:
        # Generate the response using RAG
        print(f"🤖 Generating response for query: {query}")
        response = generate_response(query)
        
        if response is None:
            raise ValueError("generate_response returned None")
        
        print(f"✅ Generated response: {response[:100]}...")
        
        # Save query and response to database using query_logs table
        try:
            db = SessionLocal()
            
            # Import the correct model - QueryLog instead of Query
            from backend.db.models import QueryLog as DB_QueryLog  # Changed this line
            
            # Create query log record using the correct table structure
            query_log_record = DB_QueryLog(  # Changed this
                id=UUID(query_id),
                user_id=UUID(user_id),
                session_id=UUID(session_id) if session_id and session_id != 'string' else None,
                query=query,  # Changed from query_text to query
                response=response,  # Changed from response_text to response
                timestamp=datetime.utcnow()  # Using timestamp instead of created_at
                # Note: latency could be calculated if needed
            )
            db.add(query_log_record)
            db.commit()
            print(f"💾 Query and response saved to query_logs table")
            
        except Exception as db_error:
            print(f"⚠️ Warning: Failed to save to database: {db_error}")
            # Continue even if database save fails
        finally:
            if 'db' in locals():
                db.close()
        
        result = {
            "status": "success",
            "query_id": query_id,
            "query": query,
            "response": response,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id
        }
        
        print(f"✅ Query processing completed successfully")
        print(f"📝 Response: {response[:100]}...")
        return result
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "query_id": query_id,
            "query": query,
            "error": str(e)
        }

@celery_app.task(name='speech_to_text')
def speech_to_text_task(stt_id: str, audio_bytes: str, filename: str, language: str, tenant_id: str, user_id: str):
    """Convert speech to text using Whisper or other STT engine"""
    print(f"🎤 Processing STT request {stt_id}: {filename}")

@celery_app.task(name='generate_response_async', bind=True)
def generate_response_async(self, query: str, use_reranking: bool = False, top_k: int = 2):
    """
    Asynchronously generate a response using RAG.
    
    Args:
        query (str): The input query
        use_reranking (bool): Whether to use reranking
        top_k (int): Number of top documents for reranking
    
    Returns:
        dict: Response with generated text and metadata
    """
    try:
        print(f"🤖 Starting response generation for query: {query[:100]}...")
        
        # Update task state to indicate processing has started
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Initializing response generation...', 'progress': 10}
        )
        
        # Import here to avoid circular imports
        from backend.response_generation.generator import generate_response, generate_response_with_reranking
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Retrieving relevant documents...', 'progress': 30}
        )
        
        # Generate response based on reranking preference
        if use_reranking:
            response = generate_response_with_reranking(query, top_k=top_k)
        else:
            response = generate_response(query)
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Generating final response...', 'progress': 80}
        )
        
        result = {
            "status": "success",
            "query": query,
            "response": response,
            "use_reranking": use_reranking,
            "top_k": top_k if use_reranking else None,
            "task_id": self.request.id
        }
        
        print(f"✅ Response generated successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "error": str(e),
            "query": query,
            "task_id": self.request.id
        }

@celery_app.task(name='batch_generate_responses')
def batch_generate_responses(queries: list, use_reranking: bool = False, top_k: int = 2):
    """
    Generate responses for multiple queries in batch.
    
    Args:
        queries (list): List of query strings
        use_reranking (bool): Whether to use reranking
        top_k (int): Number of top documents for reranking
    
    Returns:
        dict: Batch results
    """
    try:
        print(f"🔄 Processing batch of {len(queries)} queries...")
        
        from backend.response_generation.generator import generate_response, generate_response_with_reranking
        
        results = []
        for i, query in enumerate(queries):
            try:
                if use_reranking:
                    response = generate_response_with_reranking(query, top_k=top_k)
                else:
                    response = generate_response(query)
                
                results.append({
                    "query": query,
                    "response": response,
                    "status": "success"
                })
                
            except Exception as query_error:
                results.append({
                    "query": query,
                    "response": None,
                    "status": "error",
                    "error": str(query_error)
                })
        
        return {
            "status": "completed",
            "total_queries": len(queries),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results
        }
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        return {
            "status": "error",
            "error": str(e),
            "total_queries": len(queries)
        }
