from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
from backend.auth.jwt_ import get_current_user, UserTokenData
from pydantic import BaseModel
from typing import Optional
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

class QuerySubmissionResponse(BaseModel):
    query_id: str
    status: str
    message: str


class QueryWithTTSRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = config.get("max_input_tokens", 100)
    temperature: Optional[float] = config.get("temperature", 0.7)
    session_id: Optional[str] = None
    language: str = "en"  # Language for TTS
    voice: Optional[str] = ""  # Optional specific voice
    stream_audio: bool = True  # Whether to stream audio via WebSocket

class QueryWithTTSResponse(BaseModel):
    query_id: str
    tts_task_id: str
    status: str
    message: str

@router.post("/submit", response_model=QuerySubmissionResponse)
async def submit_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """Submit a query for asynchronous processing"""
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
            session_id=request.session_id
        )
        
        return QuerySubmissionResponse(
            query_id=query_id,
            status="submitted",
            message="Query submitted for processing. Response will be sent via WebSocket."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query submission failed: {str(e)}")


@router.post("/submit-with-tts", response_model=QueryWithTTSResponse)
async def submit_query_with_tts(
    request: QueryWithTTSRequest,
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """Submit a query that will be processed and converted to speech with optimized streaming"""
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
            language=request.language,
            voice=request.voice,
            stream_audio=request.stream_audio
        )
        
        return QueryWithTTSResponse(
            query_id=query_id,
            tts_task_id=tts_task_id,
            status="submitted",
            message="Query and TTS submitted for processing. Audio will be streamed via WebSocket."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query with TTS submission failed: {str(e)}")
