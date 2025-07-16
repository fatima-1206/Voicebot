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

app = FastAPI()
app.include_router(auth_router, prefix="/auth")
app.include_router(documents_router, prefix="/documents")
app.include_router(query_router, prefix="/query")
app.include_router(tts_router, prefix="/tts")  # Add this

manager = ConnectionManager()

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

@app.on_event("startup")
def init_db():
    if not db_has_tables():
        schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
        run_sql_file(schema_path)
        print("✅ Database initialized")

@app.websocket("/ws/{tenant_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, tenant_id: str, user_id: str):
    await manager.connect(tenant_id, user_id, websocket)
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
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
                    
    except WebSocketDisconnect:
        manager.disconnect(tenant_id, user_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(tenant_id, user_id)
