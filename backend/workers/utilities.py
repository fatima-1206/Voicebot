"""
Shared utilities, configurations and context managers for Celery tasks
"""
import torch
import gc
import redis
import os
from contextlib import contextmanager
from backend.db.connection import SessionLocal
from groq import Groq
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
import json
from datetime import datetime

# Global configurations
sample_rate = 16000

# Initialize models and clients globally
model = WhisperModel("medium", device="cuda", compute_type="float16")
vad_model = load_silero_vad()

# Initialize Groq client globally
groq_client = None
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("✅ Groq TTS client initialized")
except Exception as e:
    print(f"❌ Groq TTS client initialization failed: {e}")

# Redis connection for WebSocket messaging
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connection successful")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None


@contextmanager
def get_db_session():
    """Context manager for database sessions with proper cleanup"""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def manage_gpu_memory():
    """Context manager for GPU memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def notify_audio_status(tenant_id: str, user_id: str, task_id: str, status: str, data: dict):
    """Notify audio status via WebSocket"""
    if not redis_client:
        return
        
    try:
        message = {
            "type": "audio_status",
            "task_id": task_id,
            "status": status,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        redis_client.publish(f"ws:{tenant_id}:{user_id}", json.dumps(message))
        print(f"📡 [WS] Audio status sent: {status} for task {task_id[:8]}")
    except Exception as e:
        print(f"❌ Failed to notify audio status: {e}")


def notify_text_response_ready(tenant_id: str, user_id: str, query_id: str, query: str, response: str):
    """Notify that text response is ready"""
    if not redis_client:
        return
        
    try:
        message = {
            "type": "text_response",
            "query_id": query_id,
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        redis_client.publish(f"ws:{tenant_id}:{user_id}", json.dumps(message))
        print(f"📡 [WS] Text response sent for query {query_id[:8]}")
    except Exception as e:
        print(f"❌ Failed to notify text response: {e}")


def notify_error(tenant_id: str, user_id: str, query_id: str, error: str):
    """Notify error via WebSocket"""
    if not redis_client:
        return
        
    try:
        message = {
            "type": "error",
            "query_id": query_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        redis_client.publish(f"ws:{tenant_id}:{user_id}", json.dumps(message))
        print(f"📡 [WS] Error sent for query {query_id[:8]}: {error}")
    except Exception as e:
        print(f"❌ Failed to notify error: {e}")


def save_query_to_database(query_id: str, user_id: str, session_id: str, query: str, response: str):
    """Save query and response to database"""
    try:
        with get_db_session() as db:
            from backend.db.models import QueryLog
            from uuid import uuid4, UUID
            
            # Handle session_id validation
            processed_session_id = None
            if session_id:
                try:
                    # Try to parse as UUID - if it's a valid UUID string, use it
                    processed_session_id = UUID(session_id) if isinstance(session_id, str) else session_id
                except ValueError:
                    # If it's not a valid UUID (like "string"), set to None
                    print(f"⚠️ [DB] Invalid session_id '{session_id}', setting to None")
                    processed_session_id = None
            
            # Create new query record
            db_query = QueryLog(
                id=query_id,
                user_id=user_id,
                session_id=processed_session_id,  # Use processed session_id
                query=query,
                response=response,
                timestamp=datetime.now()
            )
            
            db.add(db_query)
            db.commit()
            
            print(f"💾 [DB] Query saved: {query_id[:8]}")
            
    except Exception as e:
        print(f"❌ Failed to save query to database: {e}")
        # Don't raise exception - this shouldn't break the main workflow