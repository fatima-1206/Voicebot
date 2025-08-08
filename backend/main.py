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
import wave
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import Query
import json
import torch
from backend.workers.stt_tasks import bytes_to_numpy_audio
from backend.workers.stt_tasks import vad_model, sample_rate
import base64
import tempfile
from backend.workers.tasks import transcribe_chunk
from uuid import uuid4
from collections import defaultdict
import time
import redis
import asyncio
import threading
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from backend.audio_manager import AudioBufferManager
from backend.redis_manager import redis_manager, redis_listener
from backend.auth.utils import verify_websocket_token
from backend.auth.jwt_ import get_current_user, UserTokenData
import base64


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Voicebot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["*"] to allow all origins (not recommended in production)
    allow_credentials=True,
    allow_methods=["*"],    # Or restrict to specific methods like ["GET", "POST"]
    allow_headers=["*"],    # Or restrict headers if needed
)
app.include_router(auth_router, prefix="/auth")
app.include_router(documents_router, prefix="/documents")
app.include_router(query_router, prefix="/query")
app.include_router(tts_router, prefix="/tts")

# Initialize improved buffer manager
audio_buffer_manager = AudioBufferManager()


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
async def websocket_endpoint(websocket: WebSocket, tenant_id: str, user_id: str, device_id: Optional[str] = Query(None)):
    await websocket.accept()
    
    if not device_id:
        device_id = str(uuid4())
    
    # Generate unique session ID for this connection
    session_id = str(uuid4())
    
    # Send session info to client
    await websocket.send_json({
        "type": "session_info", 
        "session_id": session_id,
        "device_id": device_id,
        "tenant_id": tenant_id,
        "user_id": user_id
    })
    
    # Connect using the new session-based structure
    if not await manager.connect(tenant_id, user_id, websocket, session_id, device_id):
        return

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping", "timestamp": time.time()})
                continue
            except Exception as e:
                logger.error(f"❌ Error receiving WebSocket message: {e}")
                break

            is_valid, error_msg = validate_websocket_message(data)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid message: {error_msg}"
                })
                continue

            try:
                await process_websocket_message(websocket, tenant_id, user_id, data, session_id, device_id)
            except Exception as e:
                logger.error(f"❌ Error processing WebSocket message: {e}\n{data}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {tenant_id}|{user_id}|{device_id}|{session_id}")
    finally:
        # Cleanup using session-based keys
        session_buffer_key = f"{tenant_id}_{user_id}_{device_id}_{session_id}"
        audio_buffer_manager.clear_buffer(session_buffer_key)
        
        # Use proper session key format for Redis manager
        session_key = f"{tenant_id}|{user_id}|{device_id}|{session_id}"
        redis_manager.revoke_all_tasks(session_key)
        
        manager.disconnect(tenant_id, user_id, session_id, device_id)


async def process_websocket_message(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, session_id: str, device_id: str = None):
    """Process individual WebSocket messages with session-specific isolation"""
    if "type" in data:
        message_type = data.get("type")
    else:
        message_type = "unknown"
        print(f"❓ Unknown message type received: {data}")
    
    # Use session-specific key for complete isolation
    key = f"{tenant_id}_{user_id}_{device_id}_{session_id}"

    if message_type == "ping":
        await websocket.send_json({
            "type": "pong", 
            "timestamp": time.time(), 
            "session_id": session_id,
            "device_id": device_id
        })

    elif message_type == "audio_chunk":
        await handle_audio_chunk(websocket, tenant_id, user_id, data, key, session_id, device_id)

    elif message_type == "continuous_audio_stream":
        await handle_continuous_audio_stream(websocket, tenant_id, user_id, data, key, session_id, device_id)

    elif message_type == "start_continuous_conversation":
        await handle_start_continuous_conversation(websocket, tenant_id, user_id, key, session_id, device_id)

    elif message_type == "stop_continuous_conversation":
        await handle_stop_continuous_conversation(websocket, tenant_id, user_id, key, session_id, device_id)

    elif message_type == "query":
        await handle_query(websocket, tenant_id, user_id, data, session_id, device_id)

    elif message_type == "query_with_tts":
        await handle_query_with_tts(websocket, tenant_id, user_id, data, session_id, device_id)

    elif message_type == "audio_stream":
        await handle_audio_stream(websocket, tenant_id, user_id, data, key, session_id, device_id)

    elif message_type == "audio_stream_end":
        await handle_audio_stream_end(websocket, tenant_id, user_id, data, key, session_id, device_id)

    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "session_id": session_id,
            "device_id": device_id
        })



async def handle_audio_stream(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str, session_id: str, device_id: str = None):
  print(data)


# Update all handler functions to include device_id parameter
async def handle_continuous_audio_stream(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str, session_id: str, device_id: str = None):
    """Handle continuous audio stream - process each chunk independently, no buffer logic"""
    audio_chunk_base64 = data.get("audio")
    if not audio_chunk_base64:
        return

    try:
        audio_bytes = base64.b64decode(audio_chunk_base64)
        
        # Convert bytes to numpy audio
        audio_data = bytes_to_numpy_audio(audio_bytes)
        if audio_data is None or len(audio_data) == 0:
            await websocket.send_json({
                "type": "error",
                "message": "No valid audio data received",
                "session_id": session_id,
                "device_id": device_id
            })
            return

        # Convert to tensor for VAD
        audio_tensor = torch.from_numpy(audio_data).float()
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)

        # Apply VAD to detect speech segments
        from backend.workers.stt_tasks import get_speech_timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            vad_model, 
            sampling_rate=sample_rate,
            threshold=0.2,
            min_speech_duration_ms=200,
            min_silence_duration_ms=100,
            speech_pad_ms=50
        )

        # Convert audio to int16 bytes
        int16_audio = np.clip(audio_tensor.numpy(), -1.0, 1.0)
        int16_audio = (int16_audio * 32767).astype(np.int16)
        processed_audio_bytes = int16_audio.tobytes()

        # Submit chunk for processing (no buffer involved)
        from backend.workers.stt_tasks import process_continuous_audio_task
        process_continuous_audio_task.delay(
            speech_bytes=processed_audio_bytes,
            speech_timestamps=speech_timestamps,
            user_id=user_id,
            tenant_id=tenant_id,
            device_id=device_id,
            session_id=session_id,
            buffer_key=key,
            tts_language=data.get("tts_language", "en"),
            voice=data.get("voice", "Fritz-PlayAI"),
            auto_respond=data.get("auto_respond", True),
            chunk_id=data.get("chunk_id", str(uuid4())),
            use_rag=data.get("rag", True)
        )

        # Optionally send acknowledgment
        await websocket.send_json({
            "type": "continuous_audio_chunk_processed",
            "chunk_id": data.get("chunk_id", ""),
            "session_id": session_id,
            "device_id": device_id,
            "speech_segments_processed": len(speech_timestamps)
        })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error processing continuous audio: {str(e)}",
            "session_id": session_id,
            "device_id": device_id
        })

async def handle_start_continuous_conversation(websocket: WebSocket, tenant_id: str, user_id: str, key: str, session_id: str, device_id: str = None):
    """Initialize continuous conversation mode with session isolation"""
    audio_buffer_manager.clear_buffer(key)
    await websocket.send_json({
        "type": "continuous_conversation_started",
        "message": "Ready for continuous conversation",
        "session_id": session_id,
        "device_id": device_id
    })

async def handle_stop_continuous_conversation(websocket: WebSocket, tenant_id: str, user_id: str, key: str, session_id: str, device_id: str = None):
    """Stop continuous conversation mode with session cleanup"""
    audio_buffer_manager.clear_buffer(key)
    await websocket.send_json({
        "type": "continuous_conversation_stopped",
        "message": "Continuous conversation stopped",
        "session_id": session_id,
        "device_id": device_id
    })

async def handle_query(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, session_id: str, device_id: str = None):
    """Handle regular query submission with session tracking"""
    from backend.workers.tasks import process_query_task
    
    query_id = str(uuid4())
    query = data.get("message", "").strip()
    
    if not query:
        await websocket.send_json({
            "type": "error",
            "message": "Empty query received",
            "session_id": session_id,
            "device_id": device_id
        })
        return
    
    # Submit to Celery with session information
    process_query_task.delay(
        query_id=query_id,
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        device_id=device_id,
        max_tokens=data.get("max_tokens", 500),
        temperature=data.get("temperature", 0.7)
    )
    
    await websocket.send_json({
        "type": "query_submitted",
        "query_id": query_id,
        "session_id": session_id,
        "device_id": device_id,
        "message": "Query received and being processed..."
    })

async def handle_query_with_tts(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, session_id: str, device_id: str = None):
    """Handle query with TTS with session tracking"""
    from backend.workers.tasks import process_query_with_tts_task
    
    query_id = str(uuid4())
    tts_task_id = str(uuid4())
    query = data.get("message", "").strip()
    
    if not query:
        await websocket.send_json({
            "type": "error",
            "message": "Empty query received",
            "session_id": session_id,
            "device_id": device_id
        })
        return
    
    # Submit to Celery with session information
    process_query_with_tts_task.delay(
        query_id=query_id,
        tts_task_id=tts_task_id,
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        device_id=device_id,
        max_tokens=data.get("max_tokens", 500),
        temperature=data.get("temperature", 0.7),
        language=data.get("language", "en"),
        voice=data.get("voice", ""),
        stream_audio=data.get("stream_audio", True)
    )
    
    await websocket.send_json({
        "type": "query_with_tts_submitted",
        "query_id": query_id,
        "tts_task_id": tts_task_id,
        "session_id": session_id,
        "device_id": device_id,
        "message": "Query with TTS received and being processed..."
    })

# NEW HANDLER FUNCTION
async def handle_audio_chunk(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str, session_id: str, device_id: str = None):
    """Handle tiny audio chunks with session-specific processing"""
    audio_chunk_base64 = data.get("audio")
    if not audio_chunk_base64:
        return
    
    try:
        audio_bytes = base64.b64decode(audio_chunk_base64)
        
        # Just append to buffer - session-specific buffer management
        result = audio_buffer_manager.append_audio(key, audio_bytes)
        current_buffer_size = result.get("buffer_size", 0)
        
        # Be less aggressive - wait for more audio before processing
        min_processing_threshold = 32000  # 2 seconds at 16kHz
        
        if current_buffer_size >= min_processing_threshold:
            # Get buffer for VAD processing
            current_buffer = audio_buffer_manager.get_buffer(key)
            
            if len(current_buffer) > 0:
                # Convert buffer to proper format for VAD
                buffer_audio = bytes_to_numpy_audio(bytes(current_buffer))
                if buffer_audio is not None and len(buffer_audio) > 0:
                    buffer_tensor = torch.from_numpy(buffer_audio).float()
                    buffer_tensor = torch.clamp(buffer_tensor, -1.0, 1.0)
                    
                    # Apply VAD to detect speech segments
                    from backend.workers.stt_tasks import get_speech_timestamps
                    speech_timestamps = get_speech_timestamps(
                        buffer_tensor, 
                        vad_model, 
                        sampling_rate=sample_rate,
                        threshold=0.2,
                        min_speech_duration_ms=500,
                        min_silence_duration_ms=200,
                        speech_pad_ms=100
                    )
                    
                    # Check if we have a clear end of speech
                    has_clear_speech_end = False
                    if speech_timestamps:
                        last_segment = speech_timestamps[-1]
                        last_speech_end = last_segment['end']
                        buffer_end = len(buffer_tensor)
                        silence_after_speech = (buffer_end - last_speech_end) / sample_rate
                        has_clear_speech_end = silence_after_speech > 1.5
                    
                    # Determine if we should process
                    should_process = False
                    process_all_segments = False
                    
                    if speech_timestamps and len(speech_timestamps) > 1 and has_clear_speech_end:
                        should_process = True
                        process_all_segments = True
                    elif speech_timestamps and len(speech_timestamps) >= 1 and has_clear_speech_end:
                        should_process = True
                        process_all_segments = True
                    elif current_buffer_size >= 160000:  # 10 seconds - force processing
                        should_process = True
                        process_all_segments = True
                        
                    if should_process and process_all_segments:
                        # Process entire buffer - complete utterance
                        int16_audio = np.clip(buffer_tensor.numpy(), -1.0, 1.0)
                        int16_audio = (int16_audio * 32767).astype(np.int16)
                        audio_bytes_processed = int16_audio.tobytes()
                        
                        # Clear the session-specific buffer
                        audio_buffer_manager.clear_buffer(key)
                        
                        # Send processed audio for transcription with session info
                        from backend.workers.stt_tasks import process_continuous_audio_task
                        chunk_id = data.get("chunk_id", str(uuid4()))
                        
                        process_continuous_audio_task.delay(
                            speech_bytes=audio_bytes_processed,
                            speech_timestamps=speech_timestamps,
                            user_id=user_id,
                            tenant_id=tenant_id,
                            device_id=device_id,
                            session_id=session_id,  # Complete session isolation
                            buffer_key=key,
                            tts_language=data.get("tts_language", "en"),
                            voice=data.get("voice", "Fritz-PlayAI"),
                            auto_respond=data.get("auto_respond", True),
                            chunk_id=chunk_id,
                            use_rag=data.get("rag", True)
                        )
                        
                        logger.info(f"✅ Complete utterance processed for session {session_id}: {len(speech_timestamps)} segments, chunk_id: {chunk_id}")
                        
                        # Send acknowledgment back to client with session info
                        await websocket.send_json({
                            "type": "audio_chunk_processed",
                            "chunk_id": chunk_id,
                            "session_id": session_id,
                            "device_id": device_id,
                            "reason": "complete_utterance_processed",
                            "speech_segments_processed": len(speech_timestamps),
                            "speech_segments_remaining": 0,
                            "buffer_duration_ms": len(buffer_tensor) * 1000 / sample_rate
                        })
                    
                    # If buffer is getting extremely large without speech, trim it
                    elif current_buffer_size > 320000:  # 20 seconds
                        audio_buffer_manager.trim_buffer_to_last_n_bytes(key, 32000)  # Keep last 2 seconds
                        logger.debug(f"🔇 Trimmed very large non-speech buffer for session {session_id}")
        
        # Send status update less frequently
        if current_buffer_size > 0 and current_buffer_size % 64000 == 0:  # Every ~4 seconds
            await websocket.send_json({
                "type": "audio_chunk_accumulating",
                "session_id": session_id,
                "device_id": device_id,
                "buffer_duration_ms": current_buffer_size * 1000 / (sample_rate * 2)
            })
    
    except Exception as e:
        logger.error(f"❌ Error processing audio chunk for session {session_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Error processing audio chunk: {str(e)}",
            "session_id": session_id,
            "device_id": device_id
        })

async def handle_audio_stream_end(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str, session_id: str, device_id: str = None):
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

    """Ping all active WebSocket connections"""
    results = await manager.ping_all_connections()
    return {
        "status": "ping_completed",
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
