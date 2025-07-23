# To run celery:
# celery -A backend.workers.celery_worker.celery_app worker --loglevel=info --concurrency=1 --hostname=voicebot@%h
from backend.workers.celery_worker import celery_app
import json
from backend.response_generation.generator import generate_response
from backend.db.connection import SessionLocal
from uuid import UUID, uuid4
from datetime import datetime
import torch
from contextlib import contextmanager
from typing import List, Dict, Any
import gc
import asyncio
import io
import wave
from faster_whisper import WhisperModel
import tempfile
import os
import redis
from groq import Groq  # Add Groq import

model = WhisperModel("medium", device="cuda", compute_type="float16")
from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks
import numpy as np
import time
import traceback

# Load VAD model (you might want to load this once globally)
vad_model = load_silero_vad()
sample_rate = 16000  # Assuming 16kHz audio

# Initialize Groq client globally
groq_client = None
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("✅ Groq TTS client initialized")
except Exception as e:
    print(f"❌ Groq TTS client initialization failed: {e}")

# Add Redis connection for WebSocket messaging
# Add error handling for Redis operations
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
    # Test the connection
    redis_client.ping()
    print("✅ Redis connection successful")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    # You might want to use a fallback or exit


# Add connection pooling and memory management
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


# OPTIMIZED WEBSOCKET-INTEGRATED STT TASKS------------------------------------

@celery_app.task(name='transcribe_chunk', bind=True)
def transcribe_chunk(self, audio_bytes: bytes, user_id: str, chunk_id: str, tenant_id: str = None):
    """
    Transcribe audio chunk using direct numpy array processing (no temp files)
    This is called directly from WebSocket audio streams
    """
    print(f"🔄 [STT] Transcribing chunk {chunk_id} for user {user_id}")
    
    try:
        import numpy as np
        import torch
        import io
        import wave
        from datetime import datetime
        
        # Clear GPU cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Convert bytes to numpy array
        audio_data = bytes_to_numpy_audio(audio_bytes)
        
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Could not convert audio bytes to numpy array")
        
        # Convert to tensor and clamp values
        audio_tensor = torch.from_numpy(audio_data).float()
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
        
        # Apply VAD if available
        try:
            speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sample_rate)
            
            if speech_timestamps:
                speech_audio = collect_chunks(speech_timestamps, audio_tensor)
                audio_data = speech_audio.numpy().astype(np.float32)
                print(f"   ✅ VAD applied: {len(speech_timestamps)} speech segments")
            else:
                audio_data = audio_tensor.numpy().astype(np.float32)
                
        except Exception as vad_error:
            print(f"   ⚠️ VAD error: {vad_error}, processing full audio chunk")
            audio_data = audio_tensor.numpy().astype(np.float32)
        
        # Transcribe using Whisper model
        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        language = "en"  # You can make this configurable
        
        if device == 'cpu':
            segments, info = model.transcribe(audio_data, beam_size=1, language=language, task="transcribe")
        else:
            segments, info = model.transcribe(audio_data, beam_size=5, language=language, task="transcribe", vad_filter=True)
        
        transcription_duration = time.time() - start_time
        
        # Collect transcript segments
        transcript_parts = []
        segment_details = []
        
        for segment in segments:
            if segment.text.strip():
                transcript_parts.append(segment.text.strip())
                segment_details.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
        
        full_transcript = " ".join(transcript_parts)
        confidence = getattr(info, 'language_probability', 0.0)
        detected_language = getattr(info, 'language', 'unknown')
        
        print(f"   ✅ Transcribed: '{full_transcript[:50]}...'")
        
        # Clear GPU cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Send results back via WebSocket if tenant_id is provided
        if tenant_id:
            send_transcription_result_via_websocket(
                tenant_id=tenant_id,
                user_id=user_id,
                chunk_id=chunk_id,
                transcript=full_transcript,
                confidence=confidence,
                detected_language=detected_language,
                segments=segment_details,
                processing_time=transcription_duration
            )

        return {
            "status": "success",
            "user_id": user_id,
            "chunk_id": chunk_id,
            "transcript": full_transcript,
            "segments": segment_details,
            "confidence": confidence,
            "detected_language": detected_language,
            "processing_time": transcription_duration,
            "audio_duration": len(audio_data) / 16000,
            "worker": self.request.hostname
        }
        
    except Exception as e:
        print(f"❌ [STT] Error transcribing chunk {chunk_id}: {e}")
        
        # Clear GPU cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Send error back via WebSocket
        if tenant_id:
            send_transcription_error_via_websocket(
                tenant_id=tenant_id,
                user_id=user_id,
                chunk_id=chunk_id,
                error=str(e)
            )
        
        return {
            "status": "error",
            "user_id": user_id,
            "chunk_id": chunk_id,
            "error": str(e),
            "worker": self.request.hostname
        }

@celery_app.task(name='transcribe_and_respond', bind=True)
def transcribe_and_respond_task(self, audio_bytes: bytes, user_id: str, chunk_id: str, 
                               tenant_id: str = None, conversation_mode: bool = True,
                               tts_language: str = "en", voice: str = "", 
                               auto_respond: bool = True):
    """
    Real-time conversation task: STT → (optionally RAG → TTS)
    This is called from WebSocket when conversation_mode=True
    """
    print(f"🔄 [CONVERSATION] Processing audio chunk {chunk_id}")
    
    try:
        # Step 1: Do STT first
        stt_result = transcribe_chunk(audio_bytes, user_id, chunk_id, tenant_id)
        
        if stt_result['status'] != 'success':
            return stt_result
            
        transcript = stt_result['transcript']
        
        # Step 2: Check if we should auto-respond
        if auto_respond and conversation_mode and should_respond_to_transcript(transcript):
            print(f"🤖 [CONVERSATION] Auto-responding to: '{transcript}'")
            
            # Immediately trigger RAG + TTS response in background
            response_id = str(uuid4())
            
            # Use the optimized query+TTS task for immediate response
            process_query_with_tts_task.delay(
                query_id=response_id,
                tts_task_id=str(uuid4()),
                query=transcript,
                tenant_id=tenant_id,
                user_id=user_id,
                language=tts_language,
                voice=voice,
                stream_audio=True,
                session_id=None  # Could be passed from WebSocket
            )
            
            # Update STT result to indicate response is being generated
            stt_result['auto_response_triggered'] = True
            stt_result['response_id'] = response_id
        
        return stt_result
        
    except Exception as e:
        print(f"❌ [CONVERSATION] Error: {e}")
        return {
            "status": "error",
            "chunk_id": chunk_id,
            "error": str(e),
            "worker": self.request.hostname
        }

def should_respond_to_transcript(transcript: str) -> bool:
    """
    Determine if the transcript warrants an immediate response
    Enhanced with better conversation detection
    """
    if not transcript or len(transcript.strip()) < 3:
        return False
        
    transcript = transcript.strip().lower()
    
    # Respond to questions
    if transcript.endswith('?'):
        return True
        
    # Respond to common conversation starters/phrases
    conversation_triggers = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'what is', 'tell me', 'how do', 'can you', 'please', 'help me',
        'i need', 'i want', 'explain', 'show me', 'where is', 'when is',
        'who is', 'why', 'how much', 'how many', 'thank you', 'thanks'
    ]
    
    for trigger in conversation_triggers:
        if trigger in transcript:
            return True
    
    # Respond to complete sentences (basic check)
    if len(transcript.split()) >= 4 and (transcript.endswith('.') or transcript.endswith('!')):
        return True
        
    return False

def bytes_to_numpy_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Convert audio bytes to numpy array - handles multiple formats
    """
    try:
        # Try to parse as WAV first
        try:
            with io.BytesIO(audio_bytes) as audio_io:
                with wave.open(audio_io, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                    return audio_data
        except Exception:
            pass
        
        # Fallback: assume raw 16-bit PCM
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            return audio_data
        except Exception:
            pass
            
        # Last resort: try as float32
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            return audio_data
        except Exception:
            pass
            
        return None
        
    except Exception as e:
        print(f"❌ Error converting audio bytes: {e}")
        return None

# WebSocket messaging functions (synchronous for Redis)
def send_transcription_result_via_websocket(tenant_id: str, user_id: str, chunk_id: str, 
                                          transcript: str, confidence: float, detected_language: str,
                                          segments: list, processing_time: float):
    """Send transcription result via Redis for WebSocket delivery"""
    try:
        message = {
            "type": "transcription_result",
            "chunk_id": chunk_id,
            "transcript": transcript,
            "segments": segments,
            "confidence": confidence,
            "detected_language": detected_language,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Publish message to Redis channel
        channel = f"ws:{tenant_id}:{user_id}"
        redis_client.publish(channel, json.dumps(message))
        print(f"✅ Transcription sent via Redis: {transcript[:30]}...")
        
    except Exception as e:
        print(f"❌ Error sending transcription via Redis: {e}")

def send_transcription_error_via_websocket(tenant_id: str, user_id: str, chunk_id: str, error: str):
    """Send transcription error via Redis for WebSocket delivery"""
    try:
        message = {
            "type": "transcription_error",
            "chunk_id": chunk_id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        channel = f"ws:{tenant_id}:{user_id}"
        redis_client.publish(channel, json.dumps(message))
        
    except Exception as e:
        print(f"❌ Error sending transcription error via Redis: {e}")


# OPTIMIZED QUERY + TTS PIPELINE----------------------------------------------

@celery_app.task(name='process_query_with_tts', bind=True)
def process_query_with_tts_task(self, query_id: str, tts_task_id: str, query: str, tenant_id: str, 
                            user_id: str, max_tokens: int = 500, temperature: float = 0.7, 
                            session_id: str = None, language: str = "en", voice: str = "",
                            stream_audio: bool = True):
    """
    Optimized task: Query → RAG → TTS → Stream Audio
    This is the main pipeline for conversation responses
    """
    print(f"🔄 [QUERY+TTS] Processing: {query[:50]}...")
    
    with manage_gpu_memory():
        try:
            # Step 1: Generate text response
            from backend.response_generation.generator import generate_response
            response_text = generate_response(query)
            print(f"✅ [RAG] Generated: {response_text[:50]}...")
            
            # Step 2: Immediately notify WebSocket that text is ready
            notify_text_response_ready(tenant_id, user_id, query_id, query, response_text)
            
            # Step 3: Process TTS with streaming
            if stream_audio:
                audio_result = generate_and_stream_tts(
                    response_text, tenant_id, user_id, tts_task_id, language, voice
                )
            else:
                audio_result = generate_tts_file(response_text, tts_task_id, language, voice)
            
            # Step 4: Save to database
            save_query_to_database(query_id, user_id, session_id, query, response_text)
            
            return {
                "status": "success",
                "query_id": query_id,
                "tts_task_id": tts_task_id,
                "query": query,
                "response": response_text,
                "audio_result": audio_result,
                "worker": self.request.hostname
            }
            
        except Exception as e:
            print(f"❌ [QUERY+TTS] Error: {e}")
            
            # Notify WebSocket of error
            notify_error(tenant_id, user_id, query_id, str(e))
            
            return {
                "status": "error",
                "query_id": query_id,
                "error": str(e),
                "worker": self.request.hostname
            }

def generate_and_stream_tts(text: str, tenant_id: str, user_id: str, tts_task_id: str, 
                           language: str, voice: str) -> dict:
    """Generate TTS using Groq and stream via WebSocket"""
    global groq_client
    
    try:
        # Initialize Groq client if not already done
        if groq_client is None:
            groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Set voice based on language - Groq has limited voice options
        if not voice:
            voice = "Fritz-PlayAI"  # Default Groq voice for all languages
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{tts_task_id[:8]}.wav"
        output_dir = "backend/rag/temp_audioFiles"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Notify start
        notify_audio_status(tenant_id, user_id, tts_task_id, "generating", 
                          {"message": "Audio generation started"})
        
        print(f"🔄 [TTS] Generating audio with Groq: {text[:50]}...")
        
        # Generate audio using Groq
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        # Write to file
        response.write_to_file(output_path)
        
        print(f"✅ [TTS] Audio generated: {filename}")
        
        # Verify file was created and get size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            
            # Notify ready
            notify_audio_status(tenant_id, user_id, tts_task_id, "ready", {
                "filename": filename,
                "file_size": file_size,
                "audio_url": f"/tts/download/{filename}",
                "voice": voice,
                "language": language,
                "provider": "Groq PlayAI"
            })
            
            # Stream audio file via WebSocket
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(stream_audio_file(tenant_id, user_id, output_path, tts_task_id))
                finally:
                    loop.close()
            except Exception as stream_error:
                print(f"⚠️ [TTS] Audio streaming failed: {stream_error}")
                # Continue anyway, file is still available for download
            
            return {"filename": filename, "file_size": file_size, "status": "success"}
        else:
            raise Exception("Audio file was not created")
        
    except Exception as e:
        print(f"❌ [TTS] Groq generation failed: {e}")
        notify_audio_status(tenant_id, user_id, tts_task_id, "error", {
            "error": str(e),
            "provider": "Groq PlayAI"
        })
        raise e

def generate_tts_file(text: str, tts_task_id: str, language: str, voice: str) -> dict:
    """Generate TTS file using Groq without streaming"""
    global groq_client
    
    try:
        # Initialize Groq client if not already done
        if groq_client is None:
            groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Set voice - Groq has limited options
        if not voice:
            voice = "Fritz-PlayAI"  # Default Groq voice
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{tts_task_id[:8]}.wav"
        output_dir = "backend/rag/temp_audioFiles"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        print(f"🔄 [TTS] Generating audio file with Groq: {text[:50]}...")
        
        # Generate audio using Groq
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        # Write to file
        response.write_to_file(output_path)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ [TTS] File generated: {filename} ({file_size} bytes)")
            return {
                "filename": filename, 
                "file_size": file_size,
                "status": "success",
                "provider": "Groq PlayAI"
            }
        else:
            raise Exception("Failed to generate TTS file")
            
    except Exception as e:
        print(f"❌ [TTS] Groq file generation failed: {e}")
        raise Exception(f"Groq TTS generation failed: {str(e)}")

@celery_app.task(name='process_tts', bind=True)
def process_tts_task(self, text: str, language: str, voice: str = "", output_filename: str = None, play_audio: bool = False):
    """
    Process Text-to-Speech conversion using Groq (standalone TTS)
    """
    print(f"🔄 [TTS] Processing: {text[:50]}...")
    
    try:
        # Use the tts_task_id from the task itself
        task_id = self.request.id or str(uuid4())
        
        result = generate_tts_file(text, task_id, language, voice)
        
        return {
            "status": "success",
            "task_id": task_id,
            "text": text,
            "language": language,
            "voice": voice or "Fritz-PlayAI",
            "output_file": result["filename"],
            "file_size": result["file_size"],
            "provider": "Groq PlayAI",
            "worker": self.request.hostname
        }
        
    except Exception as e:
        print(f"❌ [TTS] Error: {e}")
        return {
            "status": "error",
            "task_id": self.request.id,
            "text": text,
            "error": str(e),
            "provider": "Groq PlayAI",
            "worker": self.request.hostname
        }


# REGULAR PROCESSING TASKS (for non-conversation mode)------------------------

@celery_app.task(name='process_query', bind=True)
def process_query_task(self, query_id: str, query: str, tenant_id: str, user_id: str, 
                    max_tokens: int = 500, temperature: float = 0.7, session_id: str = None):
    """Process a user query using RAG pipeline (text only)"""
    print(f"🔄 [QUERY] Processing: {query[:50]}...")
    
    with manage_gpu_memory():
        try:
            # Generate response
            response = generate_response(query)
            
            # Save to database
            save_query_to_database(query_id, user_id, session_id, query, response)
            
            # Notify via WebSocket
            notify_text_response_ready(tenant_id, user_id, query_id, query, response)
            
            return {
                "status": "success",
                "query_id": query_id,
                "query": query,
                "response": response,
                "worker": self.request.hostname
            }
            
        except Exception as e:
            print(f"❌ [QUERY] Error: {e}")
            notify_error(tenant_id, user_id, query_id, str(e))
            
            return {
                "status": "error",
                "query_id": query_id,
                "error": str(e),
                "worker": self.request.hostname
            }


# DOCUMENT PROCESSING TASKS (unchanged)--------------------------------------

@celery_app.task(name='process_uploaded_document')
def process_uploaded_document(file_bytes, filename, doc_id, tenant_id, user_id, is_public):
    """Process uploaded document - extract text, create chunks, etc."""
    print(f"🔄 [DOC] Processing: {filename}")
    
    try:
        from backend.db.connection import SessionLocal
        from backend.rag.add_document import process_uploaded_document as process_doc
        from sqlalchemy import text
        
        db = SessionLocal()
        
        try:
            db.execute(text("SELECT 1")).fetchone()
            
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode("latin1")
            
            result = process_doc(
                file_bytes=file_bytes,
                filename=filename,
                doc_id=doc_id,
                tenant_id=tenant_id,
                user_id=user_id,
                is_public=is_public
            )
            
            return result
            
        except Exception as db_error:
            db.rollback()
            raise db_error
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ [DOC] Error: {e}")
        return {"status": "error", "error": str(e)}

@celery_app.task(name='update_document_visibility')
def update_document_visibility(doc_id: str, is_public: bool, tenant_id: str):
    """Update document visibility"""
    print(f"🔄 [DOC] Updating visibility for {doc_id}")
    
    with get_db_session() as db:
        try:
            from backend.db.models import Document as DB_Document
            from uuid import UUID
            
            doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id
            
            document = db.query(DB_Document).filter(DB_Document.id == doc_uuid).first()
            if not document:
                raise ValueError(f"Document with ID {doc_id} not found")
            
            document.is_public = is_public
            document.status = "processed"
            db.commit()
            
            # Rebuild retrievers
            try:
                from backend.rag import retrieverconstruction
                retrieverconstruction.clear_cache()
                retrieverconstruction.construct_retrievers(
                    include_postgres=True,
                    force_rebuild=True
                )
            except Exception as e:
                print(f"❌ Error rebuilding retrievers: {e}")
            
            return {
                "status": "success",
                "doc_id": str(doc_uuid),
                "is_public": is_public,
                "message": "Document visibility updated successfully"
            }
            
        except Exception as e:
            print(f"❌ [DOC] Error updating visibility: {e}")
            return {"status": "error", "error": str(e)}


# LEGACY PIPELINE TASKS (for backward compatibility)-------------------------
# Keep the existing pipeline tasks for any legacy integrations

from celery import chain

@celery_app.task(name='audio_to_audio_pipeline')
def audio_to_audio_pipeline(pipeline_id: str, audio_bytes: bytes, tenant_id: str, 
                          user_id: str, stt_language: str = "en", tts_language: str = "en",
                          voice: str = "", session_id: str = None):
    """
    Legacy pipeline: STT → RAG → TTS
    Use transcribe_and_respond_task instead for better performance
    """
    print(f"🚀 [LEGACY-PIPELINE] Starting {pipeline_id}")
    
    try:
        # Just use the new optimized task
        chunk_id = str(uuid4())
        result = transcribe_and_respond_task(
            audio_bytes=audio_bytes,
            user_id=user_id,
            chunk_id=chunk_id,
            tenant_id=tenant_id,
            conversation_mode=True,
            tts_language=tts_language,
            voice=voice,
            auto_respond=True
        )
        
        return {
            "status": "pipeline_completed",
            "pipeline_id": pipeline_id,
            "result": result
        }
        
    except Exception as e:
        print(f"❌ [LEGACY-PIPELINE] Error: {e}")
        return {
            "status": "error",
            "pipeline_id": pipeline_id,
            "error": str(e)
        }

@celery_app.task(name='streaming_transcribe', bind=True)
def streaming_transcribe_task(self, audio_bytes: bytes, user_id: str, chunk_id: str, 
                            tenant_id: str = None, is_final: bool = False):
    """
    Streaming transcription - provides intermediate results
    """
    try:
        audio_data = bytes_to_numpy_audio(audio_bytes)
        
        # Use Whisper with streaming mode
        segments, info = model.transcribe(
            audio_data,
            beam_size=1,  # Faster decoding
            language="en",
            condition_on_previous_text=False,  # More responsive
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)  # Quick silence detection
        )
        
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())
        
        transcript = " ".join(transcript_parts).strip()
        
        # Send partial result immediately
        start_time = time.time()
        result = {
            "type": "transcription_result",
            "status": "success",
            "transcript": transcript,
            "chunk_id": chunk_id,
            "user_id": user_id,
            "is_partial": not is_final,
            "confidence": getattr(info, 'confidence', 0.0),
            "processing_time": time.time() - start_time
        }
        
        # Send via Redis for immediate WebSocket delivery
        redis_client.publish(f"ws:{tenant_id}:{user_id}", json.dumps(result))
        
        # If final and should respond, trigger response generation
        if is_final and should_respond_to_transcript(transcript):
            process_query_with_tts_task.delay(
                query_id=str(uuid4()),
                tts_task_id=str(uuid4()),
                query=transcript,
                tenant_id=tenant_id,
                user_id=user_id,
                language="en",
                voice="en-US-JennyNeural",
                stream_audio=True
            )
        
        return result
        
    except Exception as e:
        # Handle error
        pass

@celery_app.task(name='process_continuous_audio', bind=True)
def process_continuous_audio_task(self, audio_bytes: bytes, user_id: str, tenant_id: str, 
                                 buffer_key: str, tts_language: str = "en", 
                                 voice: str = "en-US-JennyNeural", auto_respond: bool = True):
    """
    Process continuous audio stream with server-side VAD
    Detects speech segments and triggers STT+RAG+TTS pipeline
    """
    print(f"🔄 [CONTINUOUS] Processing audio buffer for user {user_id}")
    
    try:
        import numpy as np
        import torch
        from silero_vad import get_speech_timestamps, collect_chunks
        
        # Convert bytes to numpy array
        audio_data = bytes_to_numpy_audio(audio_bytes)
        if audio_data is None or len(audio_data) == 0:
            return {"status": "no_audio"}
        
        # Convert to tensor for VAD
        audio_tensor = torch.from_numpy(audio_data).float()
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
        
        # Apply VAD to detect speech segments
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            vad_model, 
            sampling_rate=sample_rate,
            threshold=0.2,
            min_speech_duration_ms=200,  # Minimum 0.5s speech
            min_silence_duration_ms=100,  # 0.3s silence to separate utterances
            speech_pad_ms=50  # Add padding around speech
        )
        
        print(f"   🔍 Found {len(speech_timestamps)} speech segments")
        
        if not speech_timestamps:
            return {"status": "no_speech"}
        
        # Process each speech segment
        processed_segments = []
        for i, speech_segment in enumerate(speech_timestamps):
            start_sample = speech_segment['start']
            end_sample = speech_segment['end']
            
            # Extract speech segment
            speech_audio = audio_tensor[start_sample:end_sample]
            
            # Skip very short segments
            duration = len(speech_audio) / sample_rate
            if duration < 0.5:
                continue
            
            print(f"   🗣️ Processing speech segment {i+1}: {duration:.1f}s")
            
            # Convert back to bytes for STT
            speech_bytes = (speech_audio.numpy() * 32767).astype(np.int16).tobytes()
            
            # Create unique chunk ID for this segment
            chunk_id = f"{user_id}_{int(time.time() * 1000)}_{i}"
            
            # Process this speech segment immediately
            segment_result = process_speech_segment(
                speech_bytes, user_id, chunk_id, tenant_id,
                tts_language, voice, auto_respond
            )
            
            processed_segments.append(segment_result)
        
        return {
            "status": "success",
            "segments_processed": len(processed_segments),
            "results": processed_segments
        }
        
    except Exception as e:
        print(f"❌ [CONTINUOUS] Error: {e}")
        return {"status": "error", "error": str(e)}

def process_speech_segment(speech_bytes: bytes, user_id: str, chunk_id: str, 
                         tenant_id: str, tts_language: str, voice: str, auto_respond: bool):
    """
    Process a single detected speech segment through STT+RAG+TTS
    """
    try:
        print(f"🎯 [SPEECH] Processing segment {chunk_id}")
        
        # Step 1: STT
        audio_data = bytes_to_numpy_audio(speech_bytes)
        segments, info = model.transcribe(
            audio_data,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            vad_filter=False,  # We already did VAD
            initial_prompt=None
        )
        
        # Extract transcript
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())
        
        transcript = " ".join(transcript_parts).strip()
        
        if not transcript:
            print(f"   ❌ No transcript generated")
            return {"status": "no_transcript", "chunk_id": chunk_id}
        
        print(f"   📝 Transcript: '{transcript}'")
        
        # Send STT result immediately
        stt_result = {
            "type": "speech_detected",
            "transcript": transcript,
            "chunk_id": chunk_id,
            "user_id": user_id,
            "confidence": getattr(info, 'confidence', 0.0),
            "duration": len(audio_data) / sample_rate
        }
        
        # Send via Redis for immediate WebSocket delivery
        redis_client.publish(f"ws:{tenant_id}:{user_id}", json.dumps(stt_result))
        
        # Step 2: Check if we should respond
        if auto_respond and should_respond_to_transcript(transcript):
            print(f"   🤖 Triggering response for: '{transcript}'")
            
            # Generate response immediately (don't delay this)
            response_id = str(uuid4())
            tts_task_id = str(uuid4())
            
            # Trigger RAG + TTS response
            process_query_with_tts_task.delay(
                query_id=response_id,
                tts_task_id=tts_task_id,
                query=transcript,
                tenant_id=tenant_id,
                user_id=user_id,
                language=tts_language,
                voice=voice,
                stream_audio=True,
                max_tokens=300,  # Shorter responses for conversation
                temperature=0.7
            )
            
            return {
                "status": "responded",
                "transcript": transcript,
                "response_id": response_id,
                "chunk_id": chunk_id
            }
        else:
            return {
                "status": "transcribed_only", 
                "transcript": transcript,
                "chunk_id": chunk_id
            }
            
    except Exception as e:
        print(f"❌ [SPEECH] Error processing segment: {e}")
        return {"status": "error", "chunk_id": chunk_id, "error": str(e)}


# Add these missing helper functions at the end of the file:

def notify_audio_status(tenant_id: str, user_id: str, task_id: str, status: str, data: dict):
    """Notify audio status via WebSocket"""
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

async def stream_audio_file(tenant_id: str, user_id: str, file_path: str, task_id: str):
    """Stream audio file content via WebSocket in chunks"""
    try:
        import base64
        
        # Read and encode audio file
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        
        # Encode to base64 for transmission
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Send audio stream message
        message = {
            "type": "audio_stream",
            "task_id": task_id,
            "audio_data": audio_base64,
            "format": "wav",
            "encoding": "base64",
            "timestamp": datetime.now().isoformat()
        }
        
        redis_client.publish(f"ws:{tenant_id}:{user_id}", json.dumps(message))
        print(f"📡 [WS] Audio stream sent for task {task_id[:8]} ({len(audio_data)} bytes)")
        
    except Exception as e:
        print(f"❌ Failed to stream audio file: {e}")
        # Send error notification
        error_message = {
            "type": "audio_stream_error",
            "task_id": task_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        redis_client.publish(f"ws:{tenant_id}:{user_id}", json.dumps(error_message))

def save_query_to_database(query_id: str, user_id: str, session_id: str, query: str, response: str):
    """Save query and response to database"""
    try:
        with get_db_session() as db:
            from backend.db.models import Query
            
            # Create new query record
            db_query = Query(
                id=query_id,
                user_id=user_id,
                session_id=session_id,
                query_text=query,
                response_text=response,
                created_at=datetime.now()
            )
            
            db.add(db_query)
            db.commit()
            
            print(f"💾 [DB] Query saved: {query_id[:8]}")
            
    except Exception as e:
        print(f"❌ Failed to save query to database: {e}")
        # Don't raise exception - this shouldn't break the main workflow

def should_respond_to_transcript(transcript: str) -> bool:
    """Determine if we should generate a response to this transcript"""
    if not transcript or len(transcript.strip()) < 3:
        return False
    
    # Add your logic here - for now, respond to everything
    # You might want to filter out background noise, incomplete sentences, etc.
    
    # Don't respond to very short utterances
    if len(transcript.split()) < 2:
        return False
    
    # Don't respond to common filler words
    filler_words = {"um", "uh", "hmm", "okay", "yeah", "yes", "no"}
    if transcript.lower().strip() in filler_words:
        return False
    
    return True