"""
Shared utilities, configurations and context managers for Celery tasks with improved error handling
"""
import torch
import gc
import redis
import os
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
from backend.db.connection import SessionLocal
from groq import Groq
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Global configurations
sample_rate = 16000

# Initialize models and clients globally with error handling
model = None
vad_model = None
groq_client = None

def initialize_models():
    """Initialize AI models with proper error handling"""
    global model, vad_model, groq_client
    
    try:
        # Initialize Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("medium", device=device, compute_type=compute_type)
        logger.info(f"✅ Whisper model initialized on {device}")
    except Exception as e:
        logger.error(f"❌ Whisper model initialization failed: {e}")
    
    try:
        # Initialize VAD model
        vad_model = load_silero_vad()
        logger.info("✅ VAD model initialized")
    except Exception as e:
        logger.error(f"❌ VAD model initialization failed: {e}")
    
    try:
        # Initialize Groq client
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            groq_client = Groq(api_key=api_key)
            logger.info("✅ Groq client initialized")
        else:
            logger.warning("⚠️ GROQ_API_KEY not found")
    except Exception as e:
        logger.error(f"❌ Groq client initialization failed: {e}")

# Initialize models
initialize_models()

class ImprovedRedisClient:
    """Improved Redis client with connection pooling, retry logic, and health monitoring"""
    
    def __init__(self, host='localhost', port=6379, db=2, max_retries=3, retry_delay=1):
        self.host = host
        self.port = port
        self.db = db
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_pool = None
        self.client = None
        self.is_healthy = False
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Redis connection with connection pooling"""
        try:
            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                max_connections=20,
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            self.client = redis.Redis(
                connection_pool=self.connection_pool,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.client.ping()
            self.is_healthy = True
            self.last_health_check = time.time()
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.is_healthy = False
            self.client = None
    
    def _health_check(self) -> bool:
        """Perform health check if needed"""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return self.is_healthy
        
        try:
            if self.client:
                self.client.ping()
                self.is_healthy = True
            else:
                self.is_healthy = False
        except Exception:
            self.is_healthy = False
            logger.warning("⚠️ Redis health check failed")
        
        self.last_health_check = current_time
        return self.is_healthy
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to Redis"""
        logger.info("🔄 Attempting Redis reconnection...")
        self._initialize_connection()
        return self.is_healthy
    
    def publish(self, channel: str, message: str, max_retries: Optional[int] = None) -> bool:
        """Publish message with retry logic"""
        if max_retries is None:
            max_retries = self.max_retries
        
        for attempt in range(max_retries + 1):
            # Check health and reconnect if needed
            if not self._health_check() and not self._reconnect():
                if attempt < max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return False
            
            try:
                result = self.client.publish(channel, message)
                return True
            except Exception as e:
                logger.warning(f"❌ Redis publish attempt {attempt + 1} failed: {e}")
                self.is_healthy = False
                
                if attempt < max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
        
        return False
    
    def get_client(self) -> Optional[redis.Redis]:
        """Get Redis client if healthy"""
        if self._health_check():
            return self.client
        return None

# Initialize improved Redis client
redis_client = ImprovedRedisClient()

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

def safe_redis_publish(channel: str, message: dict, max_retries: int = 3) -> bool:
    """Safely publish message to Redis with error handling"""
    try:
        message_str = json.dumps(message)
        # message_str = message
        return redis_client.publish(channel, message_str, max_retries=max_retries)
    except Exception as e:
        logger.error(f"❌ Failed to publish to Redis channel {channel}: {e}")
        return False

def notify_audio_status(tenant_id: str, user_id: str, task_id: str, number_of_scentences:int, status: str, data: dict) -> bool:
    """Notify audio status via WebSocket with improved error handling"""
    try:
        message = {
            "type": "audio_status",
            "task_id": task_id,
            "status": status,
            "data": data,
            "scentences": number_of_scentences,
            "timestamp": datetime.now().isoformat()
        }
        
        success = safe_redis_publish(f"ws:{tenant_id}:{user_id}", message)
        if success:
            logger.info(f"📡 [WS] Audio status sent: {status} for task {task_id[:8]}")
        else:
            logger.warning(f"⚠️ [WS] Failed to send audio status for task {task_id[:8]}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Failed to notify audio status: {e}")
        return False

def notify_text_response_ready(tenant_id: str, user_id: str, query_id: str, query: str, response: str, use_rag: bool = True) -> bool:
    """Notify that text response is ready with improved error handling"""
    try:
        message = {
            "type": "text_response",
            "query_id": query_id,
            "query": query,
            "response": response,
            "use_rag": use_rag,
            "timestamp": datetime.now().isoformat()
        }
        # Serialize message to JSON string
        # message_str = json.dumps(message)
        message_str = message
        success = safe_redis_publish(f"ws:{tenant_id}:{user_id}", message_str)
        if success:
            logger.info(f"📡 [WS] Text response sent for query {query_id[:8]}")
        else:
            logger.warning(f"⚠️ [WS] Failed to send text response for query {query_id[:8]}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Failed to notify text response: {e}")
        return False
    
def notify_error(tenant_id: str, user_id: str, query_id: str, error: str) -> bool:
    """Notify error via WebSocket with improved error handling"""
    try:
        message = {
            "type": "error",
            "query_id": query_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        success = safe_redis_publish(f"ws:{tenant_id}:{user_id}", message)
        if success:
            logger.info(f"📡 [WS] Error notification sent for query {query_id[:8]}")
        else:
            logger.warning(f"⚠️ [WS] Failed to send error notification for query {query_id[:8]}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Failed to notify error: {e}")
        return False

def save_query_to_database(query_id: str, user_id: str, session_id: str, query: str, response: str) -> bool:
    """Save query and response to database with improved error handling"""
    try:
        import uuid
        
        with get_db_session() as db:
            from backend.db.models import QueryLog
            
            # Handle session_id conversion
            processed_session_id = None
            if session_id and session_id != "string" and session_id.lower() != "none":
                try:
                    # Try to parse as UUID if it's a valid UUID string
                    processed_session_id = uuid.UUID(session_id)
                except ValueError:
                    # If not a valid UUID, generate a new one
                    processed_session_id = uuid.uuid4()
                    logger.warning(f"⚠️ Invalid session_id '{session_id}', generated new UUID: {processed_session_id}")
            else:
                # Generate a new session ID if None or invalid
                processed_session_id = uuid.uuid4()
                logger.info(f"📝 Generated new session_id: {processed_session_id}")
            
            query_record = QueryLog(
                id=uuid.UUID(query_id),
                user_id=uuid.UUID(user_id),
                session_id=processed_session_id,
                query=query,
                response=response,
                timestamp=datetime.now()
            )
            
            db.add(query_record)
            db.commit()
            
            logger.info(f"💾 Query saved to database: {query_id[:8]}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to save query to database: {e}")
        return False
    
def get_redis_health() -> Dict[str, Any]:
    """Get Redis health information"""
    return {
        "is_healthy": redis_client.is_healthy,
        "last_health_check": datetime.fromtimestamp(redis_client.last_health_check).isoformat(),
        "host": redis_client.host,
        "port": redis_client.port,
        "db": redis_client.db
    }

def get_model_status() -> Dict[str, Any]:
    """Get status of all AI models"""
    return {
        "whisper_model": model is not None,
        "vad_model": vad_model is not None,
        "groq_client": groq_client is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.debug("🧹 GPU memory cleaned")
    except Exception as e:
        logger.warning(f"⚠️ GPU memory cleanup failed: {e}")

def validate_audio_data(audio_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> tuple[bool, str]:
    """Validate audio data"""
    if not audio_bytes:
        return False, "No audio data provided"
    
    if len(audio_bytes) > max_size:
        return False, f"Audio data too large: {len(audio_bytes)} bytes > {max_size} bytes"
    
    if len(audio_bytes) < 1000:  # Very small audio unlikely to be useful
        return False, "Audio data too small"
    
    return True, ""

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

# Backward compatibility aliases
def notify_websocket_message(tenant_id: str, user_id: str, message: dict) -> bool:
    """Legacy function for backward compatibility"""
    return safe_redis_publish(f"ws:{tenant_id}:{user_id}", message)