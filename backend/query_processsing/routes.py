from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
from backend.auth.jwt_ import get_current_user, UserTokenData
from pydantic import BaseModel
from typing import Optional, Dict, Any
from backend.workers.tasks import process_query_task, process_query_with_tts_task, audio_to_audio_pipeline
from backend.workers.celery_worker import celery_app
from celery.result import AsyncResult
from uuid import uuid4
import json

with open('config.json') as config_file:
    config = json.load(config_file)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = config.get("max_input_tokens", 100)
    temperature: Optional[float] = config.get("temperature", 0.7)
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    use_rag: Optional[bool] = True  # Enable RAG by default
    fallback_on_error: Optional[bool] = True  # Enable fallback by default

class QuerySubmissionResponse(BaseModel):
    query_id: str
    status: str
    message: str

class QueryWithTTSRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = config.get("max_input_tokens", 100)
    temperature: Optional[float] = config.get("temperature", 0.7)
    session_id: str
    device_id: str
    language: str = "en"  # Language for TTS
    voice: Optional[str] = ""  # Optional specific voice
    stream_audio: bool = True  # Whether to stream audio via WebSocket
    use_rag: Optional[bool] = True  # Enable RAG by default
    fallback_on_error: Optional[bool] = True  # Enable fallback by default

class QueryWithTTSResponse(BaseModel):
    query_id: str
    tts_task_id: str
    status: str
    message: str

# class GraphQLToRAGRequest(BaseModel):
#     api_url: str
#     headers: Optional[Dict[str, str]] = None
#     source_name: Optional[str] = None
#     max_depth: Optional[int] = 3
#     is_public: Optional[bool] = True

# class GraphQLToRAGResponse(BaseModel):
#     task_id: str
#     status: str
#     message: str

@router.post("/submit", response_model=QuerySubmissionResponse)
async def submit_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """Submit a query for asynchronous processing with optional RAG fallback"""
    try:
        query_id = str(uuid4())
        
        # Submit to Celery for processing
        process_query_task.delay(
            query_id=query_id,
            query=request.query,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.user_id),
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            session_id=request.session_id,
            device_id=request.device_id,
            use_rag=request.use_rag,
            fallback_on_error=request.fallback_on_error
        )
        
        # Create message based on RAG settings
        if request.use_rag:
            if request.fallback_on_error:
                message = "Query submitted for RAG processing with LLM fallback. Response will be sent via WebSocket."
            else:
                message = "Query submitted for RAG processing only. Response will be sent via WebSocket."
        else:
            message = "Query submitted for LLM processing (no RAG). Response will be sent via WebSocket."
        
        return QuerySubmissionResponse(
            query_id=query_id,
            status="submitted",
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query submission failed: {str(e)}")

@router.post("/submit-with-tts", response_model=QueryWithTTSResponse)
async def submit_query_with_tts(
    request: QueryWithTTSRequest,
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """Submit a query that will be processed and converted to speech with optional RAG fallback"""
    try:
        query_id = str(uuid4())
        tts_task_id = str(uuid4())
        
        # Submit the chained task to Celery
        process_query_with_tts_task.delay(
            query_id=query_id,
            tts_task_id=tts_task_id,
            query=request.query,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.user_id),
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            session_id=request.session_id,
            device_id=request.device_id,
            language=request.language,
            voice=request.voice,
            stream_audio=request.stream_audio,
            use_rag=request.use_rag,
            fallback_on_error=request.fallback_on_error
        )
        
        # Create message based on RAG settings
        if request.use_rag:
            if request.fallback_on_error:
                message = "Query and TTS submitted for RAG processing with LLM fallback. Audio will be streamed via WebSocket."
            else:
                message = "Query and TTS submitted for RAG processing only. Audio will be streamed via WebSocket."
        else:
            message = "Query and TTS submitted for LLM processing (no RAG). Audio will be streamed via WebSocket."
        
        return QueryWithTTSResponse(
            query_id=query_id,
            tts_task_id=tts_task_id,
            status="submitted",
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query with TTS submission failed: {str(e)}")

@router.post("/audio-to-audio")
async def process_audio_query(
    audio_file: UploadFile = File(...),
    language: str = "en",
    voice: Optional[str] = "",
    stream_audio: bool = True,
    use_rag: Optional[bool] = True,  # Add RAG options to audio processing too
    fallback_on_error: Optional[bool] = True,
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """Process audio input and return audio response with optional RAG fallback"""
    try:
        task_id = str(uuid4())
        
        # Read audio file
        audio_content = await audio_file.read()
        
        # Submit to Celery for processing
        audio_to_audio_pipeline.delay(
            task_id=task_id,
            audio_data=audio_content.decode('latin1'),  # Store as string for Celery
            filename=audio_file.filename,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.user_id),
            language=language,
            voice=voice,
            stream_audio=stream_audio,
            use_rag=use_rag,
            fallback_on_error=fallback_on_error
        )
        
        # Create message based on RAG settings
        if use_rag:
            if fallback_on_error:
                message = "Audio-to-audio processing started with RAG and LLM fallback."
            else:
                message = "Audio-to-audio processing started with RAG only."
        else:
            message = "Audio-to-audio processing started with LLM only (no RAG)."
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

# @router.post("/ingest-graphql", response_model=GraphQLToRAGResponse)
# async def ingest_graphql_data(
#     request: GraphQLToRAGRequest,
#     current_user: UserTokenData = Depends(get_current_user)
# ):
#     """
#     Ingest data from a GraphQL API into the RAG knowledge base.
    
#     Provide a GraphQL API URL and this endpoint will:
#     1. Extract all available data from the API
#     2. Process it into searchable documents  
#     3. Add it to your RAG knowledge base
#     4. Update the retrievers automatically
    
#     The data will then be available for querying through the regular query endpoints.
#     """
#     try:
#         # Validate URL
#         if not request.api_url.startswith(('http://', 'https://')):
#             raise HTTPException(status_code=400, detail="API URL must start with http:// or https://")
        
#         # Auto-generate source name if not provided
#         source_name = request.source_name
#         if not source_name:
#             # Extract domain name from URL as source name
#             from urllib.parse import urlparse
#             parsed = urlparse(request.api_url)
#             source_name = parsed.netloc.replace('www.', '') or "GraphQL API"
        
#         # Submit the Celery task
#         task = celery_app.send_task(
#             'process_graphql_to_rag',
#             kwargs={
#                 'api_url': request.api_url,
#                 'headers': request.headers,
#                 'source_name': source_name,
#                 'max_depth': request.max_depth,
#                 'tenant_id': str(current_user.tenant_id),
#                 'user_id': str(current_user.user_id),
#                 'is_public': request.is_public
#             }
#         )
        
#         return GraphQLToRAGResponse(
#             task_id=task.id,
#             status="processing",
#             message=f"GraphQL data ingestion started for {source_name}. Check status endpoint for progress."
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to start GraphQL ingestion: {str(e)}")

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a processing task"""
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        response = {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None
        }
        
        # Add summary info for completed GraphQL ingestion tasks
        if result.ready() and isinstance(result.result, dict) and "summary" in result.result:
            summary = result.result["summary"]
            response["summary"] = {
                "documents_added": summary.get("added_documents", 0),
                "documents_failed": summary.get("failed_documents", 0),
                "total_chunks": summary.get("total_chunks", 0),
                "source_name": summary.get("source_name", ""),
                "entity_types": summary.get("entity_types", [])
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")