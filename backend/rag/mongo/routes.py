from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging
from datetime import datetime
import asyncio
from uuid import uuid4
import json

from backend.db.connection import SessionLocal
from backend.auth.jwt_ import get_current_user, UserTokenData
from backend.rag.mongo_connector import MongoDBConnector
from backend.workers.tasks import process_mongodb_documents
from .schemas import (
    MongoConnectionRequest, 
    MongoConnectionResponse, 
    CollectionInfo,
    CollectionSelectionRequest,
    DocumentLoadResponse,
    MongoDocumentProcessingRequest,
    ProcessingStatus
)
from backend.redis_manager import get_redis_client

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/test-connection", response_model=MongoConnectionResponse)
async def test_mongo_connection(
    connection_request: MongoConnectionRequest,
    current_user: UserTokenData = Depends(get_current_user)
):
    """Test MongoDB connection and list available collections."""
    try:
        connector = MongoDBConnector(
            uri=connection_request.uri,
            db_name=connection_request.db_name,
            timeout=connection_request.timeout
        )
        
        collections = connector.get_collections()
        collection_info_list = []
        
        for collection_name in collections:
            info = connector.get_collection_info(collection_name)
            collection_info_list.append(CollectionInfo(
                name=info.get('name', collection_name),
                document_count=info.get('document_count', 0),
                fields=info.get('fields', []),
                sample_document=info.get('sample_document')
            ))
        
        connector.close()
        
        return MongoConnectionResponse(
            success=True,
            message=f"Successfully connected to {connection_request.db_name}",
            collections=collection_info_list
        )
        
    except Exception as e:
        logger.error(f"MongoDB connection test failed: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to connect to MongoDB: {str(e)}"
        )

@router.post("/load-documents", response_model=DocumentLoadResponse)
async def load_mongodb_documents(
    selection_request: CollectionSelectionRequest,
    background_tasks: BackgroundTasks,
    current_user: UserTokenData = Depends(get_current_user)
):
    """Load documents from selected MongoDB collections."""
    try:
        start_time = datetime.now()
        
        # Generate job ID for tracking
        job_id = str(uuid4())
        
        # Store job info in Redis for tracking
        redis_client = get_redis_client()
        job_info = {
            "job_id": job_id,
            "user_id": str(current_user.user_id),
            "tenant_id": str(current_user.tenant_id),
            "status": "pending",
            "progress": 0.0,
            "message": "Job queued for processing",
            "started_at": start_time.isoformat(),
            "collections": selection_request.selected_collections,
            "should_join": selection_request.should_join,
            "total_documents": 0,
            "documents_processed": 0
        }
        
        redis_client.setex(
            f"mongo_job:{job_id}", 
            3600,  # 1 hour TTL
            json.dumps(job_info)  # Use JSON instead of str()
        )
        
        # Add background task
        background_tasks.add_task(
            process_mongodb_documents_task,
            job_id,
            selection_request,
            current_user
        )
        
        return DocumentLoadResponse(
            success=True,
            message="Document processing started",
            documents_loaded=0,
            collections_processed=[],
            processing_time=0.0,
            job_id=job_id
        )
        
    except Exception as e:
        logger.error(f"Failed to start MongoDB document loading: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start document loading: {str(e)}"
        )

@router.get("/processing-status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(
    job_id: str,
    current_user: UserTokenData = Depends(get_current_user)
):
    """Get the status of a MongoDB document processing job."""
    try:
        redis_client = get_redis_client()
        job_data = redis_client.get(f"mongo_job:{job_id}")
        
        if not job_data:
            raise HTTPException(
                status_code=404,
                detail="Job not found or expired"
            )
        
        # Parse job data using JSON
        if isinstance(job_data, bytes):
            job_data = job_data.decode('utf-8')
        
        job_info = json.loads(job_data)
        
        # Verify user can access this job
        if job_info.get("user_id") != str(current_user.user_id):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this job"
            )
        
        return ProcessingStatus(
            job_id=job_id,
            status=job_info.get("status", "unknown"),
            progress=job_info.get("progress", 0.0),
            message=job_info.get("message", ""),
            documents_processed=job_info.get("documents_processed", 0),
            total_documents=job_info.get("total_documents", 0),
            started_at=job_info.get("started_at"),
            completed_at=job_info.get("completed_at"),
            error=job_info.get("error")
        )
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse job data as JSON: {e}")
        raise HTTPException(
            status_code=500,
            detail="Invalid job data format"
        )
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve job status"
        )

async def process_mongodb_documents_sync(
    selection_request: CollectionSelectionRequest,
    current_user: UserTokenData
) -> DocumentLoadResponse:
    """Synchronous document processing (for small datasets)."""
    start_time = datetime.now()
    
    try:
        connector = MongoDBConnector(
            uri=selection_request.uri,
            db_name=selection_request.db_name
        )
        
        all_docs = []
        collections_processed = []
        
        try:
            if selection_request.should_join and len(selection_request.selected_collections) > 1:
                # Join collections
                docs = connector.load_joined_collections_as_langchain(
                    collections=selection_request.selected_collections,
                    join_field=selection_request.join_field,
                    content_fields=selection_request.content_fields,
                    limit=selection_request.limit
                )
                all_docs.extend(docs)
                collections_processed = selection_request.selected_collections
            else:
                # Process collections separately
                for collection in selection_request.selected_collections:
                    docs = connector.load_documents_as_langchain(
                        collection_name=collection,
                        limit=selection_request.limit
                    )
                    all_docs.extend(docs)
                    collections_processed.append(collection)
            # TODO
            # Here you would typically add documents to your vector store
            # from backend.rag.add_document import add_documents_to_vectorstore
            # await add_documents_to_vectorstore(all_docs, current_user.tenant_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DocumentLoadResponse(
                success=True,
                message=f"Successfully processed {len(all_docs)} documents",
                documents_loaded=len(all_docs),
                collections_processed=collections_processed,
                processing_time=processing_time
            )
            
        finally:
            connector.close()
            
    except Exception as e:
        logger.error(f"Synchronous processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

async def process_mongodb_documents_task(
    job_id: str,
    selection_request: CollectionSelectionRequest,
    current_user: UserTokenData
):
    """Background task for processing MongoDB documents."""
    redis_client = get_redis_client()
    
    def update_job_status(status: str, progress: float = 0.0, message: str = "", 
                         documents_processed: int = 0, total_documents: int = 0, error: str = None):
        job_info = {
            "job_id": job_id,
            "user_id": str(current_user.user_id),
            "tenant_id": str(current_user.tenant_id),
            "status": status,
            "progress": progress,
            "message": message,
            "collections": selection_request.selected_collections,
            "should_join": selection_request.should_join,
            "documents_processed": documents_processed,
            "total_documents": total_documents,
            "started_at": datetime.now().isoformat() if status == "processing" else None,
            "completed_at": datetime.now().isoformat() if status in ["completed", "failed"] else None,
            "error": error
        }
        redis_client.setex(f"mongo_job:{job_id}", 3600, json.dumps(job_info))
    
    try:
        # Update job status
        update_job_status("processing", 0.0, "Starting document processing...")
        
        # Process documents
        result = await process_mongodb_documents_sync(selection_request, current_user)
        
        # Update completion status
        update_job_status(
            "completed", 
            100.0, 
            f"Successfully processed {result.documents_loaded} documents", 
            result.documents_loaded, 
            result.documents_loaded
        )
        
    except Exception as e:
        logger.error(f"Background task failed for job {job_id}: {e}")
        
        # Update error status
        update_job_status("failed", 0.0, "Processing failed", 0, 0, str(e))

@router.get("/collections/{db_name}")
async def list_collections(
    db_name: str,
    uri: str,
    current_user: UserTokenData = Depends(get_current_user)
):
    """Quick endpoint to list collections for a given database."""
    try:
        connector = MongoDBConnector(uri, db_name)
        collections = connector.get_collections()
        
        collection_summaries = []
        for collection_name in collections:
            info = connector.get_collection_info(collection_name)
            collection_summaries.append({
                "name": collection_name,
                "document_count": info.get("document_count", 0),
                "field_count": len(info.get("fields", []))
            })
        
        connector.close()
        
        return {
            "database": db_name,
            "collections": collection_summaries
        }
        
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to list collections: {str(e)}"
        )