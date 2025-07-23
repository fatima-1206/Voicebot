from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
import backend.db.models  # makes sure models are registered
from backend.db.setup import db_has_tables, run_sql_file  
import os
from backend.auth.routes import router as auth_router
from backend.documents.routes import router as documents_router
from backend.query_processsing.routes import router as query_router
from backend.tts.routes import router as tts_router  # Add this
from backend.ws_manager import ConnectionManager
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

# In-memory buffers
app = FastAPI()
app.include_router(auth_router, prefix="/auth")
app.include_router(documents_router, prefix="/documents")
app.include_router(query_router, prefix="/query")
app.include_router(tts_router, prefix="/tts")  # Add this

manager = ConnectionManager()

user_audio_buffers = defaultdict(bytearray)
user_last_activity = defaultdict(float)  # Track last activity for cleanup

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "Voicebot API is running!"}

redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)

# Add this Redis listener function
async def redis_listener():
    """Listen for Redis messages and forward to WebSocket clients"""
    pubsub = redis_client.pubsub()
    
    # Subscribe to all WebSocket channels
    pubsub.psubscribe("ws:*")
    
    print("🔄 Redis WebSocket listener started")
    
    try:
        while True:
            message = pubsub.get_message(timeout=1.0)
            if message and message['type'] == 'pmessage':
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
                            await manager.send_json_to_user(tenant_id, user_id, data)
                        else:
                            # User is no longer connected, skip message
                            pass
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON in Redis message: {e}")
                except Exception as e:
                    print(f"❌ Error processing Redis message: {e}")
            
            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
            
    except Exception as e:
        print(f"❌ Redis listener error: {e}")
        # Restart the listener after a delay
        await asyncio.sleep(5)
        asyncio.create_task(redis_listener())
    finally:
        try:
            pubsub.close()
        except Exception:
            pass

# Combined startup event handler
@app.on_event("startup")
async def startup_event():
    # Initialize database
    if not db_has_tables():
        schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
        run_sql_file(schema_path)
        print("✅ Database initialized")
    
    # Start Redis listener in background
    asyncio.create_task(redis_listener())
    print("🚀 Redis WebSocket listener started")

@app.websocket("/ws/{tenant_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, tenant_id: str, user_id: str):
    await manager.connect(tenant_id, user_id, websocket)
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif data.get("type") == "continuous_audio_stream":
                # New handler for continuous audio with server-side VAD
                from uuid import uuid4
                audio_chunk_base64 = data.get("audio", None)
                
                if audio_chunk_base64:
                    key = f"{tenant_id}_{user_id}"
                    
                    try:
                        audio_bytes = base64.b64decode(audio_chunk_base64)
                        user_audio_buffers[key] += audio_bytes
                        user_last_activity[key] = time.time()
                        
                        # Only process if we have enough new audio data
                        min_process_size = 16000  # 1 second at 16kHz
                        if len(user_audio_buffers[key]) >= min_process_size:
                            
                            # Send ONLY the new chunk for VAD processing, not the entire buffer
                            from backend.workers.stt_tasks import process_continuous_audio_task
                            process_continuous_audio_task.delay(
                                audio_bytes=bytes(user_audio_buffers[key]),  # Send the buffer
                                user_id=user_id,
                                tenant_id=tenant_id,
                                buffer_key=key,
                                tts_language=data.get("tts_language", "en"),
                                voice=data.get("voice", "en-US-JennyNeural"),
                                auto_respond=data.get("auto_respond", True)
                            )
                            
                            # Clear the buffer after processing to prevent reprocessing
                            user_audio_buffers[key] = bytearray()
                    
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error processing continuous audio: {str(e)}"
                        })

            elif data.get("type") == "start_continuous_conversation":
                # Initialize continuous conversation mode
                key = f"{tenant_id}_{user_id}"
                user_audio_buffers[key] = bytearray()  # Reset buffer
                user_last_activity[key] = time.time()
                
                await websocket.send_json({
                    "type": "continuous_conversation_started",
                    "message": "Ready for continuous conversation"
                })

            elif data.get("type") == "stop_continuous_conversation":
                # Stop continuous conversation mode
                key = f"{tenant_id}_{user_id}"
                if key in user_audio_buffers:
                    del user_audio_buffers[key]
                if key in user_last_activity:
                    del user_last_activity[key]
                
                await websocket.send_json({
                    "type": "continuous_conversation_stopped",
                    "message": "Continuous conversation stopped"
                })
            elif data.get("type") == "query":
                # Handle regular query submission via WebSocket
                from backend.workers.tasks import process_query_task
                from uuid import uuid4
                
                query_id = str(uuid4())
                query = data.get("message", "")
                
                if query.strip():
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
                    
                    # Acknowledge receipt
                    await websocket.send_json({
                        "type": "query_submitted",
                        "query_id": query_id,
                        "message": "Query received and being processed..."
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Empty query received"
                    })
            elif data.get("type") == "query_with_tts":
                # Handle query with TTS via WebSocket
                from backend.workers.tasks import process_query_with_tts_task
                from uuid import uuid4
                
                query_id = str(uuid4())
                tts_task_id = str(uuid4())
                query = data.get("message", "")
                
                if query.strip():
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
                    
                    # Acknowledge receipt
                    await websocket.send_json({
                        "type": "query_with_tts_submitted",
                        "query_id": query_id,
                        "tts_task_id": tts_task_id,
                        "message": "Query with TTS received and being processed..."
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Empty query received"
                    })
            elif data.get("type") == "audio_stream":
                from uuid import uuid4
                audio_chunk_base64 = data.get("audio", None)
                conversation_mode = data.get("conversation_mode", False)  # New parameter
                
                if audio_chunk_base64:
                    key = f"{tenant_id}_{user_id}"
                    
                    try:
                        audio_bytes = base64.b64decode(audio_chunk_base64)
                        user_audio_buffers[key] += audio_bytes
                        user_last_activity[key] = time.time()

                        # Rolling buffer configuration
                        if conversation_mode:
                            # Use smaller buffers for more responsive conversation
                            buffer_size = data.get("buffer_size", 16000)  # Reduced from 32000
                            overlap_size = data.get("overlap_size", 4000)  # Reduced from 8000
                        else:
                            # Use larger buffers for batch processing
                            buffer_size = data.get("buffer_size", 32000)
                            overlap_size = data.get("overlap_size", 8000)
                        
                        if len(user_audio_buffers[key]) >= buffer_size:
                            transcription_id = str(uuid4())
                            
                            # Choose task based on conversation mode
                            if conversation_mode:
                                # Use the new real-time conversation task
                                from backend.workers.tasks import transcribe_and_respond_task
                                transcribe_and_respond_task.delay(
                                    audio_bytes=bytes(user_audio_buffers[key]),
                                    user_id=user_id,
                                    chunk_id=transcription_id,
                                    tenant_id=tenant_id,
                                    conversation_mode=True,
                                    tts_language=data.get("tts_language", "en"),
                                    voice=data.get("voice", ""),
                                    auto_respond=data.get("auto_respond", True)
                                )
                            else:
                                # Use regular transcription only
                                transcribe_chunk.delay(
                                    audio_bytes=bytes(user_audio_buffers[key]),
                                    user_id=user_id,
                                    chunk_id=transcription_id,
                                    tenant_id=tenant_id
                                )

                            # Rolling buffer: keep overlap for next chunk
                            user_audio_buffers[key] = user_audio_buffers[key][-overlap_size:]

                            await websocket.send_json({
                                "type": "transcription_submitted",
                                "transcription_id": transcription_id,
                                "buffer_size": len(user_audio_buffers[key]),
                                "conversation_mode": conversation_mode
                            })
                            
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error processing audio chunk: {str(e)}"
                        })

            elif data.get("type") == "audio_stream_end":
                from uuid import uuid4
                key = f"{tenant_id}_{user_id}"
                conversation_mode = data.get("conversation_mode", False)
                
                if key in user_audio_buffers and len(user_audio_buffers[key]) > 0:
                    try:
                        transcription_id = str(uuid4())
                        
                        if conversation_mode:
                            from backend.workers.tasks import transcribe_and_respond_task
                            transcribe_and_respond_task.delay(
                                audio_bytes=bytes(user_audio_buffers[key]),
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
                                audio_bytes=bytes(user_audio_buffers[key]),
                                user_id=user_id,
                                chunk_id=transcription_id,
                                tenant_id=tenant_id
                            )

                        # Clean up buffer
                        user_audio_buffers[key] = bytearray()
                        if key in user_last_activity:
                            del user_last_activity[key]

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

                        
    except WebSocketDisconnect:
        # Clean up user buffers on disconnect
        key = f"{tenant_id}_{user_id}"
        if key in user_audio_buffers:
            del user_audio_buffers[key]
        if key in user_last_activity:
            del user_last_activity[key]
        manager.disconnect(tenant_id, user_id)