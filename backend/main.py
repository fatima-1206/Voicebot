from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
import backend.db.models  # makes sure models are registered
from backend.db.setup import db_has_tables, run_sql_file  
import os
from backend.auth.routes import router as auth_router
from backend.documents.routes import router as documents_router
from backend.query_processsing.routes import router as query_router
from backend.tts.routes import router as tts_router
from backend.ws_manager import manager, connection_monitor, health_monitor
from fastapi import WebSocket, WebSocketDisconnect
import json
import base64
import tempfile
from backend.workers.tasks import transcribe_chunk
from uuid import uuid4
from collections import defaultdict
import time
import redis
import asyncio
import threading
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Voicebot API", version="1.0.0")
app.include_router(auth_router, prefix="/auth")
app.include_router(documents_router, prefix="/documents")
app.include_router(query_router, prefix="/query")
app.include_router(tts_router, prefix="/tts")

# Audio buffer management with size limits and proper cleanup
class AudioBufferManager:
    """Improved audio buffer management with size limits and cleanup"""
    
    def __init__(self, max_buffer_size: int = 1024 * 1024):  # Reduce to 1MB
        self.user_audio_buffers: Dict[str, bytearray] = defaultdict(bytearray)
        self.user_last_activity: Dict[str, float] = defaultdict(float)
        self.max_buffer_size = max_buffer_size  # 1MB default
        self.max_idle_time = 300  # 5 minutes default
        self._lock = threading.RLock()
        
        # Add automatic cleanup timer
        asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        while True:
            await asyncio.sleep(30)  # Clean up every 30 seconds
            self.cleanup_stale_buffers()
    
    def get_buffer(self, key: str) -> bytearray:
        """Get or create buffer for a user"""
        with self._lock:
            self.user_last_activity[key] = time.time()
            return self.user_audio_buffers[key]
    
    def append_audio(self, key: str, audio_bytes: bytes) -> bool:
        """Append audio data with size checking"""
        with self._lock:
            current_buffer = self.user_audio_buffers[key]
            
            # Check if adding this data would exceed the limit
            if len(current_buffer) + len(audio_bytes) > self.max_buffer_size:
                # Trim the buffer to half size to make room
                trim_size = self.max_buffer_size // 2
                current_buffer[:] = current_buffer[-trim_size:]
                logger.warning(f"⚠️ Buffer trimmed for {key}, was too large")
            
            current_buffer.extend(audio_bytes)
            self.user_last_activity[key] = time.time()
            return True
    
    def get_and_clear_buffer(self, key: str) -> bytes:
        """Get buffer contents and clear it"""
        with self._lock:
            buffer_data = bytes(self.user_audio_buffers[key])
            self.user_audio_buffers[key].clear()
            self.user_last_activity[key] = time.time()
            return buffer_data
    
    def clear_buffer(self, key: str):
        """Clear a specific buffer"""
        with self._lock:
            if key in self.user_audio_buffers:
                del self.user_audio_buffers[key]
            if key in self.user_last_activity:
                del self.user_last_activity[key]
    
    def cleanup_stale_buffers(self) -> int:
        """Clean up buffers that have been idle too long"""
        current_time = time.time()
        stale_keys = []
        
        with self._lock:
            for key, last_activity in self.user_last_activity.items():
                if current_time - last_activity > self.max_idle_time:
                    stale_keys.append(key)
        
        # Clean up stale buffers
        for key in stale_keys:
            self.clear_buffer(key)
        
        if stale_keys:
            logger.info(f"🧹 Cleaned up {len(stale_keys)} stale audio buffers")
        
        return len(stale_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            total_buffers = len(self.user_audio_buffers)
            total_size = sum(len(buffer) for buffer in self.user_audio_buffers.values())
            avg_size = total_size / total_buffers if total_buffers > 0 else 0
            
            return {
                "total_buffers": total_buffers,
                "total_size_bytes": total_size,
                "average_size_bytes": avg_size,
                "max_buffer_size": self.max_buffer_size,
                "max_idle_time": self.max_idle_time
            }

# Initialize improved buffer manager
audio_buffer_manager = AudioBufferManager()

# Redis configuration with connection pooling and retry logic
class RedisManager:
    """Improved Redis manager with connection pooling and health monitoring"""
    
    def __init__(self, host='localhost', port=6379, db=2, max_connections=10):
        self.host = host
        self.port = port
        self.db = db
        self.max_connections = max_connections
        self.client = None
        self.connection_pool = None
        self.is_healthy = False
        self.circuit_breaker_open = False
        self.failure_count = 0
        self.last_failure_time = 0
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection with error handling"""
        try:
            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                max_connections=self.max_connections,
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self.client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.client.ping()
            self.is_healthy = True
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.is_healthy = False
    
    def publish(self, channel: str, message: str) -> bool:
        """Publish message with error handling"""
        if not self.is_healthy:
            return False
        
        try:
            self.client.publish(channel, message)
            return True
        except Exception as e:
            logger.error(f"❌ Redis publish failed: {e}")
            self.is_healthy = False
            return False
    
    def get_pubsub(self):
        """Get pubsub client with error handling"""
        if not self.is_healthy:
            return None
        
        try:
            return self.client.pubsub()
        except Exception as e:
            logger.error(f"❌ Redis pubsub failed: {e}")
            self.is_healthy = False
            return None
    
    async def health_check(self) -> bool:
        """Check Redis health and reconnect if needed"""
        try:
            if self.client:
                self.client.ping()
                self.is_healthy = True
                return True
        except Exception:
            self.is_healthy = False
            logger.warning("⚠️ Redis health check failed, attempting reconnection...")
            self._init_redis()
            return self.is_healthy
    
    async def publish_with_fallback(self, channel: str, message: str):
        if self.circuit_breaker_open:
            # Use local fallback or queue messages
            return await self._local_fallback(channel, message)
        
        try:
            return await self.publish(channel, message)
        except Exception:
            self._handle_failure()
            return await self._local_fallback(channel, message)

# Initialize Redis manager
redis_manager = RedisManager()

# Improved Redis listener with health monitoring and error recovery
class RedisListener:
    """Improved Redis listener with health monitoring and reconnection"""
    
    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager
        self.is_running = False
        self.pubsub = None
        self.message_count = 0
        self.error_count = 0
        self.last_message_time = time.time()
        
    async def start(self):
        """Start the Redis listener with automatic reconnection"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🔄 Starting Redis WebSocket listener")
        
        while self.is_running:
            try:
                await self._listen_loop()
            except Exception as e:
                self.error_count += 1
                logger.error(f"❌ Redis listener error: {e}")
                await asyncio.sleep(5)  # Wait before retry
                
                # Check Redis health and reconnect if needed
                await self.redis_manager.health_check()
    
    async def _listen_loop(self):
        """Main listening loop"""
        self.pubsub = self.redis_manager.get_pubsub()
        if not self.pubsub:
            logger.error("❌ Could not get Redis pubsub client")
            await asyncio.sleep(10)
            return
        
        try:
            # Subscribe to all WebSocket channels
            self.pubsub.psubscribe("ws:*")
            logger.info("🔄 Redis listener subscribed to ws:* channels")
            
            while self.is_running and self.redis_manager.is_healthy:
                try:
                    message = self.pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'pmessage':
                        await self._process_message(message)
                    
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    
                except Exception as e:
                    logger.error(f"❌ Error processing Redis message: {e}")
                    self.error_count += 1
                    
        finally:
            if self.pubsub:
                try:
                    self.pubsub.close()
                except Exception:
                    pass
                self.pubsub = None
    
    async def _process_message(self, message):
        """Process a single Redis message"""
        try:
            # Parse channel to extract tenant_id and user_id
            channel = message['channel']
            parts = channel.split(':')
            if len(parts) >= 3:
                tenant_id = parts[1]
                user_id = parts[2]
                
                # Check if user is still connected before processing
                if manager.is_connected(tenant_id, user_id):
                    # Parse message data
                    data = json.loads(message['data'])
                    
                    # Forward to WebSocket client
                    success = await manager.send_json_to_user(tenant_id, user_id, data)
                    if success:
                        self.message_count += 1
                        self.last_message_time = time.time()
                        
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in Redis message: {e}")
            self.error_count += 1
        except Exception as e:
            logger.error(f"❌ Error processing Redis message: {e}")
            self.error_count += 1
    
    def stop(self):
        """Stop the Redis listener"""
        self.is_running = False
        logger.info("🛑 Stopping Redis listener")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get listener statistics"""
        return {
            "is_running": self.is_running,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_message_time": datetime.fromtimestamp(self.last_message_time).isoformat(),
            "redis_healthy": self.redis_manager.is_healthy
        }

# Initialize Redis listener
redis_listener = RedisListener(redis_manager)

# Background cleanup task
async def cleanup_manager():
    """Background task to clean up stale data"""
    while True:
        try:
            # Clean up audio buffers
            audio_buffer_manager.cleanup_stale_buffers()
            
            # Check Redis health
            await redis_manager.health_check()
            
            await asyncio.sleep(60)  # Run every minute
            
        except Exception as e:
            logger.error(f"❌ Cleanup manager error: {e}")
            await asyncio.sleep(30)  # Retry after 30 seconds on error

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "Voicebot API is running!", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    websocket_health = await manager.health_check()
    audio_buffer_stats = audio_buffer_manager.get_stats()
    redis_stats = redis_listener.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "websockets": websocket_health,
        "audio_buffers": audio_buffer_stats,
        "redis": redis_stats
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "websockets": {
            "active_connections": manager.get_connection_count(),
            "total_created": manager.metrics.connections_created,
            "total_closed": manager.metrics.connections_closed,
            "successful_sends": manager.metrics.successful_sends,
            "failed_sends": manager.metrics.failed_sends
        },
        "audio_buffers": audio_buffer_manager.get_stats(),
        "redis": redis_listener.get_stats()
    }

# Combined startup event handler with improved error handling
@app.on_event("startup")
async def startup_event():
    """Initialize all services with proper error handling"""
    try:
        # Initialize database
        if not db_has_tables():
            schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
            run_sql_file(schema_path)
            logger.info("✅ Database initialized")
        
        # Start background tasks
        asyncio.create_task(connection_monitor())
        asyncio.create_task(health_monitor())
        asyncio.create_task(cleanup_manager())
        asyncio.create_task(redis_listener.start())
        
        # Add connection monitoring task
        asyncio.create_task(monitor_connection_health())
        
        logger.info("🚀 All services started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

async def monitor_connection_health():
    while True:
        await asyncio.sleep(60)  # Check every minute
        health = await manager.health_check()
        if health.get("dead_connections_cleaned", 0) > 0:
            logger.warning(f"Cleaned up {health['dead_connections_cleaned']} dead connections")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of all services"""
    try:
        redis_listener.stop()
        logger.info("✅ Services shut down cleanly")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Input validation helper
def validate_websocket_message(data: dict) -> tuple[bool, str]:
    """Validate WebSocket message format and size"""
    try:
        # Check message size
        message_str = json.dumps(data)
        if len(message_str) > 10 * 1024 * 1024:  # 10MB limit
            return False, "Message too large"
        
        # Check required fields
        if "type" not in data:
            return False, "Missing 'type' field"
        
        # Validate audio data if present
        if "audio" in data:
            audio_data = data["audio"]
            if not isinstance(audio_data, str):
                return False, "Audio data must be base64 string"
            
            # Check audio size
            try:
                decoded_size = len(base64.b64decode(audio_data))
                if decoded_size > 5 * 1024 * 1024:  # 5MB limit for audio
                    return False, "Audio chunk too large"
            except Exception:
                return False, "Invalid base64 audio data"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

@app.websocket("/ws/{tenant_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, tenant_id: str, user_id: str):
    """Enhanced WebSocket endpoint with improved error handling and validation"""
    
    # Connect with improved connection manager
    if not await manager.connect(tenant_id, user_id, websocket):
        return
    
    try:
        while True:
            # Receive messages from client with timeout
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_json({"type": "ping", "timestamp": time.time()})
                continue
            except Exception as e:
                logger.error(f"❌ Error receiving WebSocket message: {e}")
                break
            
            # Validate message
            is_valid, error_msg = validate_websocket_message(data)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid message: {error_msg}"
                })
                continue
            
            # Process message based on type
            try:
                await process_websocket_message(websocket, tenant_id, user_id, data)
            except Exception as e:
                logger.error(f"❌ Error processing WebSocket message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {tenant_id}:{user_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        # Clean up resources
        key = f"{tenant_id}_{user_id}"
        audio_buffer_manager.clear_buffer(key)
        manager.disconnect(tenant_id, user_id)

async def process_websocket_message(websocket: WebSocket, tenant_id: str, user_id: str, data: dict):
    """Process individual WebSocket messages with improved error handling"""
    
    message_type = data.get("type")
    key = f"{tenant_id}_{user_id}"
    
    if message_type == "ping":
        await websocket.send_json({"type": "pong", "timestamp": time.time()})
    
    elif message_type == "continuous_audio_stream":
        await handle_continuous_audio_stream(websocket, tenant_id, user_id, data, key)
    
    elif message_type == "start_continuous_conversation":
        await handle_start_continuous_conversation(websocket, tenant_id, user_id, key)
    
    elif message_type == "stop_continuous_conversation":
        await handle_stop_continuous_conversation(websocket, tenant_id, user_id, key)
    
    elif message_type == "query":
        await handle_query(websocket, tenant_id, user_id, data)
    
    elif message_type == "query_with_tts":
        await handle_query_with_tts(websocket, tenant_id, user_id, data)
    
    elif message_type == "audio_stream":
        await handle_audio_stream(websocket, tenant_id, user_id, data, key)
    
    elif message_type == "audio_stream_end":
        await handle_audio_stream_end(websocket, tenant_id, user_id, data, key)
    
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })

async def handle_continuous_audio_stream(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str):
    """Handle continuous audio stream with improved buffer management"""
    audio_chunk_base64 = data.get("audio")
    if not audio_chunk_base64:
        return
    
    try:
        audio_bytes = base64.b64decode(audio_chunk_base64)
        audio_buffer_manager.append_audio(key, audio_bytes)
        
        # Only process if we have enough audio data
        current_buffer = audio_buffer_manager.get_buffer(key)
        min_process_size = 16000  # 1 second at 16kHz
        
        if len(current_buffer) >= min_process_size:
            # Get buffer contents and clear it
            buffer_data = audio_buffer_manager.get_and_clear_buffer(key)
            
            # Submit to Celery for processing
            from backend.workers.stt_tasks import process_continuous_audio_task
            process_continuous_audio_task.delay(
                audio_bytes=buffer_data,
                user_id=user_id,
                tenant_id=tenant_id,
                buffer_key=key,
                tts_language=data.get("tts_language", "en"),
                voice=data.get("voice", "en-US-JennyNeural"),
                auto_respond=data.get("auto_respond", True)
            )
    
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error processing continuous audio: {str(e)}"
        })

async def handle_start_continuous_conversation(websocket: WebSocket, tenant_id: str, user_id: str, key: str):
    """Initialize continuous conversation mode"""
    audio_buffer_manager.clear_buffer(key)
    await websocket.send_json({
        "type": "continuous_conversation_started",
        "message": "Ready for continuous conversation"
    })

async def handle_stop_continuous_conversation(websocket: WebSocket, tenant_id: str, user_id: str, key: str):
    """Stop continuous conversation mode"""
    audio_buffer_manager.clear_buffer(key)
    await websocket.send_json({
        "type": "continuous_conversation_stopped",
        "message": "Continuous conversation stopped"
    })

async def handle_query(websocket: WebSocket, tenant_id: str, user_id: str, data: dict):
    """Handle regular query submission"""
    from backend.workers.tasks import process_query_task
    
    query_id = str(uuid4())
    query = data.get("message", "").strip()
    
    if not query:
        await websocket.send_json({
            "type": "error",
            "message": "Empty query received"
        })
        return
    
    # Submit to Celery
    process_query_task.delay(
        query_id=query_id,
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        max_tokens=data.get("max_tokens", 500),
        temperature=data.get("temperature", 0.7),
        session_id=data.get("session_id")
    )
    
    await websocket.send_json({
        "type": "query_submitted",
        "query_id": query_id,
        "message": "Query received and being processed..."
    })

async def handle_query_with_tts(websocket: WebSocket, tenant_id: str, user_id: str, data: dict):
    """Handle query with TTS"""
    from backend.workers.tasks import process_query_with_tts_task
    
    query_id = str(uuid4())
    tts_task_id = str(uuid4())
    query = data.get("message", "").strip()
    
    if not query:
        await websocket.send_json({
            "type": "error",
            "message": "Empty query received"
        })
        return
    
    # Submit to Celery
    process_query_with_tts_task.delay(
        query_id=query_id,
        tts_task_id=tts_task_id,
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        max_tokens=data.get("max_tokens", 500),
        temperature=data.get("temperature", 0.7),
        session_id=data.get("session_id"),
        language=data.get("language", "en"),
        voice=data.get("voice", ""),
        stream_audio=data.get("stream_audio", True)
    )
    
    await websocket.send_json({
        "type": "query_with_tts_submitted",
        "query_id": query_id,
        "tts_task_id": tts_task_id,
        "message": "Query with TTS received and being processed..."
    })

async def handle_audio_stream(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str):
    """Handle audio stream with rolling buffer"""
    audio_chunk_base64 = data.get("audio")
    if not audio_chunk_base64:
        return
    
    conversation_mode = data.get("conversation_mode", False)
    
    try:
        audio_bytes = base64.b64decode(audio_chunk_base64)
        audio_buffer_manager.append_audio(key, audio_bytes)
        
        # Rolling buffer configuration
        if conversation_mode:
            buffer_size = data.get("buffer_size", 16000)
            overlap_size = data.get("overlap_size", 4000)
        else:
            buffer_size = data.get("buffer_size", 32000)
            overlap_size = data.get("overlap_size", 8000)
        
        current_buffer = audio_buffer_manager.get_buffer(key)
        if len(current_buffer) >= buffer_size:
            transcription_id = str(uuid4())
            
            # Get buffer data for processing
            buffer_data = bytes(current_buffer)
            
            # Choose task based on conversation mode
            if conversation_mode:
                from backend.workers.tasks import transcribe_and_respond_task
                transcribe_and_respond_task.delay(
                    audio_bytes=buffer_data,
                    user_id=user_id,
                    chunk_id=transcription_id,
                    tenant_id=tenant_id,
                    conversation_mode=True,
                    tts_language=data.get("tts_language", "en"),
                    voice=data.get("voice", ""),
                    auto_respond=data.get("auto_respond", True)
                )
            else:
                transcribe_chunk.delay(
                    audio_bytes=buffer_data,
                    user_id=user_id,
                    chunk_id=transcription_id,
                    tenant_id=tenant_id
                )
            
            # Keep overlap for next chunk
            current_buffer[:] = current_buffer[-overlap_size:]
            
            await websocket.send_json({
                "type": "transcription_submitted",
                "transcription_id": transcription_id,
                "buffer_size": len(current_buffer),
                "conversation_mode": conversation_mode
            })
    
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error processing audio chunk: {str(e)}"
        })

async def handle_audio_stream_end(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str):
    """Handle end of audio stream"""
    conversation_mode = data.get("conversation_mode", False)
    
    current_buffer = audio_buffer_manager.get_buffer(key)
    if len(current_buffer) > 0:
        try:
            transcription_id = str(uuid4())
            buffer_data = audio_buffer_manager.get_and_clear_buffer(key)
            
            if conversation_mode:
                from backend.workers.tasks import transcribe_and_respond_task
                transcribe_and_respond_task.delay(
                    audio_bytes=buffer_data,
                    user_id=user_id,
                    chunk_id=transcription_id,
                    tenant_id=tenant_id,
                    conversation_mode=True,
                    tts_language=data.get("tts_language", "en"),
                    voice=data.get("voice", ""),
                    auto_respond=data.get("auto_respond", True)
                )
            else:
                transcribe_chunk.delay(
                    audio_bytes=buffer_data,
                    user_id=user_id,
                    chunk_id=transcription_id,
                    tenant_id=tenant_id
                )
            
            await websocket.send_json({
                "type": "transcription_submitted",
                "transcription_id": transcription_id,
                "final": True,
                "conversation_mode": conversation_mode
            })
            
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Error processing final audio chunk: {str(e)}"
            })

@app.get("/health/websockets")
async def websocket_health():
    """WebSocket-specific health check"""
    return await manager.health_check()

@app.get("/health/detailed")
async def detailed_health():
    """Detailed system health report"""
    from backend.monitoring import get_monitoring_report
    return get_monitoring_report()

@app.get("/health/status")
async def health_status():
    """Current system status"""
    from backend.monitoring import get_health_status
    return get_health_status()

@app.post("/admin/cleanup")
async def force_cleanup():
    """Force cleanup of stale connections and buffers"""
    # Clean up WebSocket connections
    ws_health = await manager.cleanup_stale_connections()
    
    # Clean up audio buffers
    buffer_cleanup = audio_buffer_manager.cleanup_stale_buffers()
    
    return {
        "status": "cleanup_completed",
        "websocket_cleanup": "completed",
        "audio_buffers_cleaned": buffer_cleanup,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/admin/ping-connections")
async def ping_all_connections():
    """Ping all active WebSocket connections"""
    results = await manager.ping_all_connections()
    return {
        "status": "ping_completed",
        "results": results,
        "timestamp": datetime.now().isoformat()
    }