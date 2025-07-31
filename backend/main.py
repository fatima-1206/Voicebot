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
async def websocket_endpoint(websocket: WebSocket, tenant_id: str, user_id: str):
    await websocket.accept()  # <-- Add this line
    session_id = str(uuid4())
    await websocket.send_json({"type": "session_id", "session_id": session_id})

    if not await manager.connect(tenant_id, user_id, websocket):
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
                await process_websocket_message(websocket, tenant_id, user_id, data, session_id)
            except Exception as e:
                logger.error(f"❌ Error processing WebSocket message: {e}\n{data}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {tenant_id}:{user_id}:{session_id}")
    finally:
        audio_buffer_manager.clear_buffer(session_id)
        redis_manager.revoke_all_tasks(session_id)
        manager.disconnect(tenant_id, user_id)


async def process_websocket_message(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, session_id: str):
    """Process individual WebSocket messages with improved error handling"""
    if "type" in data:
        message_type = data.get("type")
    else:
        message_type = "unknown"
        print(f"❓ Unknown message type received: {data}")
    key = f"{tenant_id}_{user_id}"

    if message_type == "ping":
        await websocket.send_json({"type": "pong", "timestamp": time.time()})

    elif message_type == "continuous_audio_stream":
        await handle_continuous_audio_stream(websocket, tenant_id, user_id, data, key, session_id)

    elif message_type == "start_continuous_conversation":
        await handle_start_continuous_conversation(websocket, tenant_id, user_id, key, session_id)

    elif message_type == "stop_continuous_conversation":
        await handle_stop_continuous_conversation(websocket, tenant_id, user_id, key, session_id)

    elif message_type == "query":
        await handle_query(websocket, tenant_id, user_id, data, session_id)

    elif message_type == "query_with_tts":
        await handle_query_with_tts(websocket, tenant_id, user_id, data, session_id)

    elif message_type == "audio_stream":
        await handle_audio_stream(websocket, tenant_id, user_id, data, key, session_id)

    elif message_type == "audio_stream_end":
        await handle_audio_stream_end(websocket, tenant_id, user_id, data, key, session_id)

    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })

async def handle_continuous_audio_stream(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str, session_id: str):
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
            

            
            # Submit to Celery for processing only if there is speech detected
            # Apply VAD to detect speech segments
            from backend.workers.stt_tasks import get_speech_timestamps
            audio_data = bytes_to_numpy_audio(audio_bytes)
            if audio_data is None or len(audio_data) == 0:
                return {"status": "no_audio"}
            
            # Convert to tensor for VAD
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
            
            speech_timestamps = get_speech_timestamps(
                audio_tensor, 
                vad_model, 
                sampling_rate=sample_rate,
                threshold=0.2,
                min_speech_duration_ms=200,  # Minimum 0.5s speech
                min_silence_duration_ms=100,  # 0.3s silence to separate utterances
                speech_pad_ms=50  # Add padding around speech
            )
        #    convert tensor back to bytes
            audio_bytes = audio_tensor.numpy().astype(np.float32).tobytes()

            int16_audio = np.clip(audio_tensor.numpy(), -1.0, 1.0)
            int16_audio = (int16_audio * 32767).astype(np.int16)
            audio_bytes = int16_audio.tobytes()
            from backend.workers.stt_tasks import process_continuous_audio_task
            process_continuous_audio_task.delay(
                speech_bytes=audio_bytes,
                speech_timestamps=speech_timestamps,
                user_id=user_id,
                tenant_id=tenant_id,
                buffer_key=key,
                tts_language=data.get("tts_language", "en"),
                voice=data.get("voice", "Fritz-PlayAI"),
                auto_respond=data.get("auto_respond", True),
                chunk_id= data.get("chunk_id", str(uuid4())),
                use_rag=data.get("rag", True) # Enable RAG processing if not specified
            )
    
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error processing continuous audio: {str(e)}"
        })

async def handle_start_continuous_conversation(websocket: WebSocket, tenant_id: str, user_id: str, key: str, session_id: str):
    """Initialize continuous conversation mode"""
    audio_buffer_manager.clear_buffer(key)
    await websocket.send_json({
        "type": "continuous_conversation_started",
        "message": "Ready for continuous conversation"
    })

async def handle_stop_continuous_conversation(websocket: WebSocket, tenant_id: str, user_id: str, key: str, session_id: str):
    """Stop continuous conversation mode"""
    audio_buffer_manager.clear_buffer(key)
    await websocket.send_json({
        "type": "continuous_conversation_stopped",
        "message": "Continuous conversation stopped"
    })

async def handle_query(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, session_id: str):
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

async def handle_query_with_tts(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, session_id: str):
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
        session_id=data.get("session_id", None),
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

async def handle_audio_stream(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str, session_id: str):
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

async def handle_audio_stream_end(websocket: WebSocket, tenant_id: str, user_id: str, data: dict, key: str, session_id: str):
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

