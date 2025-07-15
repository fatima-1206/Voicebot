from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
from backend.auth.jwt_ import get_current_user, UserTokenData
from pydantic import BaseModel
from typing import Optional
from backend.workers.tasks import process_query_task
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

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "query_processing"}