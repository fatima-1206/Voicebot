from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
from backend.auth.jwt_ import get_current_user, UserTokenData
from pydantic import BaseModel
from typing import Optional
from backend.workers.tasks import process_query_task
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

# @router.post("/generate-async")
# async def generate_response_async(request: dict):
#     """Start async response generation"""
#     query = request.get("query", "")
#     use_reranking = request.get("use_reranking", False)
#     top_k = request.get("top_k", 2)
    
#     if not query:
#         raise HTTPException(status_code=400, detail="Query is required")
    
#     # Submit task to Celery
#     task = celery_app.send_task(
#         'generate_response_async',
#         args=[query, use_reranking, top_k]
#     )
    
#     return {
#         "task_id": task.id,
#         "status": "submitted",
#         "message": "Response generation started"
#     }

@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of async task"""
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        if result.state == 'PENDING':
            response = {
                "task_id": task_id,
                "state": result.state,
                "message": "Task is waiting to be processed"
            }
        elif result.state == 'PROCESSING':
            response = {
                "task_id": task_id,
                "state": result.state,
                "message": result.info.get('message', ''),
                "progress": result.info.get('progress', 0)
            }
        elif result.state == 'SUCCESS':
            response = {
                "task_id": task_id,
                "state": result.state,
                "result": result.result
            }
        else:  # FAILURE
            response = {
                "task_id": task_id,
                "state": result.state,
                "error": str(result.info)
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking task status: {str(e)}")

# @router.post("/generate-batch")
# async def generate_batch_responses(request: dict):
#     """Generate responses for multiple queries"""
#     queries = request.get("queries", [])
#     use_reranking = request.get("use_reranking", False)
#     top_k = request.get("top_k", 2)
    
#     if not queries or not isinstance(queries, list):
#         raise HTTPException(status_code=400, detail="Queries list is required")
    
#     # Submit batch task to Celery
#     task = celery_app.send_task(
#         'batch_generate_responses',
#         args=[queries, use_reranking, top_k]
#     )
    
#     return {
#         "task_id": task.id,
#         "status": "submitted",
#         "total_queries": len(queries),
#         "message": "Batch processing started"
#     }