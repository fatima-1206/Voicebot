# To run celery:
# celery -A backend.workers.celery_worker.celery_app worker --loglevel=info --concurrency=1 --hostname=voicebot@%h
from backend.workers.celery_worker import celery_app
import json
from backend.response_generation.generator import generate_response
from backend.db.connection import SessionLocal
from uuid import UUID
from datetime import datetime
import torch
from contextlib import contextmanager
from typing import List, Dict, Any
import gc
import asyncio
from faster_whisper import WhisperModel
import tempfile
model = WhisperModel("large-v2", device="cuda", compute_type="float16")
from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks
import numpy as np
import time

# Load VAD model (you might want to load this once globally)
vad_model = load_silero_vad()
sample_rate = 16000  # Assuming 16kHz audio


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


# SPEECH-TO-TEXT TASKS---------------------------------------------------------
# Replace the existing transcribe_chunk function with this:

@celery_app.task(bind=True)
def transcribe_chunk(self, audio_bytes: bytes, user_id: str, chunk_id: str, tenant_id: str = None):
    """
    Transcribe audio chunk using direct numpy array processing (no temp files)
    
    Args:
        audio_bytes: The audio data as bytes
        user_id: User ID for routing the response
        chunk_id: Unique identifier for this transcription chunk
        tenant_id: Tenant ID for WebSocket routing (optional for backwards compatibility)
    """
    print(f"🔄 Transcribing audio chunk {chunk_id} for user {user_id}")
    
    try:
        import numpy as np
        import torch
        import io
        import wave
        import asyncio
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
        
        # Apply VAD if available (optional enhancement)
        try:
    
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sample_rate)
            
            if speech_timestamps:
                # Extract only speech portions
                speech_audio = collect_chunks(speech_timestamps, audio_tensor)
                audio_data = speech_audio.numpy().astype(np.float32)
                print(f"✅ VAD applied: {len(speech_timestamps)} speech segments found")
            else:
                print("⚠️ No speech detected in audio chunk")
                # Still process the audio even if VAD doesn't detect speech
                audio_data = audio_tensor.numpy().astype(np.float32)
                
        except ImportError:
            print("⚠️ VAD not available, processing full audio chunk")
            audio_data = audio_tensor.numpy().astype(np.float32)
        except Exception as vad_error:
            print(f"⚠️ VAD error: {vad_error}, processing full audio chunk")
            audio_data = audio_tensor.numpy().astype(np.float32)
        
        # Transcribe using Whisper model directly on numpy array
        print(f"📝 Transcribing at {datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        
        # Use different settings based on device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        language = "en"  # You can make this configurable
        
        if device == 'cpu':
            segments, info = model.transcribe(audio_data, beam_size=1, language=language, task="transcribe")
        else:
            segments, info = model.transcribe(audio_data, beam_size=5, language=language, task="transcribe", vad_filter=True)
        
        end_time = time.time()
        transcription_duration = end_time - start_time
        
        # Collect all transcript segments
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
        
        # Get transcription info
        confidence = getattr(info, 'language_probability', 0.0)
        detected_language = getattr(info, 'language', 'unknown')
        
        print(f"✅ Transcription completed in {transcription_duration:.2f}s: {full_transcript[:50]}...")
        
        # Clear GPU cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Send results back via WebSocket if tenant_id is provided
        if tenant_id and full_transcript:
            asyncio.run(send_transcription_result_via_websocket(
                tenant_id=tenant_id,
                user_id=user_id,
                chunk_id=chunk_id,
                transcript=full_transcript,
                confidence=confidence,
                detected_language=detected_language,
                segments=segment_details,
                processing_time=transcription_duration
            ))
        
        return {
            "status": "success",
            "user_id": user_id,
            "chunk_id": chunk_id,
            "transcript": full_transcript,
            "segments": segment_details,
            "confidence": confidence,
            "detected_language": detected_language,
            "processing_time": transcription_duration,
            "audio_duration": len(audio_data) / 16000,  # Approximate duration assuming 16kHz
            "worker": self.request.hostname
        }
        
    except Exception as e:
        print(f"❌ Error transcribing audio chunk {chunk_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Clear GPU cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Send error back via WebSocket if tenant_id is provided
        if tenant_id:
            asyncio.run(send_transcription_error_via_websocket(
                tenant_id=tenant_id,
                user_id=user_id,
                chunk_id=chunk_id,
                error=str(e)
            ))
        
        return {
            "status": "error",
            "user_id": user_id,
            "chunk_id": chunk_id,
            "error": str(e),
            "worker": self.request.hostname
        }

def bytes_to_numpy_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Convert audio bytes to numpy array
    Handles both raw PCM and WAV format
    """
    try:
        import numpy as np
        import io
        import wave
        
        # Try to parse as WAV first
        try:
            with io.BytesIO(audio_bytes) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    # Get audio parameters
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    sample_width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()
                    
                    # Read audio data
                    audio_data = wav_file.readframes(frames)
                    
                    # Convert to numpy array
                    if sample_width == 1:
                        audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                        audio_array = (audio_array.astype(np.float32) - 128) / 128.0
                    elif sample_width == 2:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        audio_array = audio_array.astype(np.float32) / 32768.0
                    elif sample_width == 4:
                        audio_array = np.frombuffer(audio_data, dtype=np.int32)
                        audio_array = audio_array.astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")
                    
                    # Convert to mono if stereo
                    if channels == 2:
                        audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                    
                    print(f"✅ Parsed WAV: {sample_rate}Hz, {channels}ch, {sample_width}B, {len(audio_array)} samples")
                    return audio_array
                    
        except Exception as wav_error:
            print(f"⚠️ WAV parsing failed: {wav_error}")
        
        # Fallback: assume raw 16-bit PCM
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            print(f"✅ Parsed as raw PCM: {len(audio_array)} samples")
            return audio_array
        except Exception as pcm_error:
            print(f"❌ PCM parsing failed: {pcm_error}")
            
        # Last resort: try as float32
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            print(f"✅ Parsed as float32: {len(audio_array)} samples")
            return audio_array
        except Exception as float_error:
            print(f"❌ Float32 parsing failed: {float_error}")
            
        return None
        
    except Exception as e:
        print(f"❌ Error converting audio bytes to numpy: {e}")
        return None

# Update the WebSocket helper function to include more details
async def send_transcription_result_via_websocket(tenant_id: str, user_id: str, chunk_id: str, 
                                                transcript: str, confidence: float, detected_language: str,
                                                segments: list, processing_time: float):
    """Send transcription result via WebSocket with enhanced details"""
    try:
        from backend.main import manager
        
        await manager.send_json_to_user(tenant_id, user_id, {
            "type": "transcription_result",
            "chunk_id": chunk_id,
            "transcript": transcript,
            "segments": segments,
            "confidence": confidence,
            "detected_language": detected_language,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        })
        print(f"✅ Transcription result sent via WebSocket: {transcript[:30]}...")
        
    except Exception as e:
        print(f"❌ Error sending transcription result via WebSocket: {e}")

async def send_transcription_error_via_websocket(tenant_id: str, user_id: str, chunk_id: str, error: str):
    """Send transcription error via WebSocket"""
    try:
        from backend.main import manager
        
        await manager.send_json_to_user(tenant_id, user_id, {
            "type": "transcription_error",
            "chunk_id": chunk_id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        })
        print(f"❌ Transcription error sent via WebSocket: {error}")
        
    except Exception as e:
        print(f"❌ Error sending transcription error via WebSocket: {e}")


# DOCUMENTS RELATED TASKS---------------------------------------------------------
@celery_app.task(name='process_uploaded_document')
def process_uploaded_document(file_bytes, filename, doc_id, tenant_id, user_id, is_public):
    """Process uploaded document - extract text, create chunks, etc."""
    print(f"🔄 Task started for document: {filename} (doc_id: {doc_id})")
    
    try:
        from backend.db.connection import SessionLocal
        from backend.rag.add_document import process_uploaded_document as process_doc
        from sqlalchemy import text
        
        print(f"📊 Processing document with {len(file_bytes)} bytes")
        
        # Create database session with error handling
        db = SessionLocal()
        
        try:
            # Test database connection before proceeding (fixed SQL)
            db.execute(text("SELECT 1")).fetchone()
            print("✅ Database connection verified in task")
            
            # Convert file_bytes back to bytes if it was encoded as string
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode("latin1")
                print(f"🔧 Converted string to bytes: {len(file_bytes)} bytes")
            
            # Process the document (INCLUDE db parameter!)
            result = process_doc(
                file_bytes=file_bytes,
                filename=filename,
                doc_id=doc_id,
                tenant_id=tenant_id,
                user_id=user_id,
                is_public=is_public
                # db=db  # Don't forget this!
            )
            print(f"✅ Document processing completed: {result}")
            return result
            
        except Exception as db_error:
            print(f"❌ Database error in task: {db_error}")
            db.rollback()
            raise db_error
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

@celery_app.task(name='update_document_visibility')
def update_document_visibility(doc_id: str, is_public: bool, tenant_id: str):
    """Update document visibility"""
    print(f"🔄 Updating visibility for document {doc_id} to {'public' if is_public else 'private'}")
    
    with get_db_session() as db:
        try:
            from backend.db.models import Document as DB_Document
            from uuid import UUID
            
            doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id
            tenant_uuid = UUID(tenant_id) if isinstance(tenant_id, str) else tenant_id
            
            # Update document metadata
            document = db.query(DB_Document).filter(DB_Document.id == doc_uuid).first()
            if not document:
                raise ValueError(f"Document with ID {doc_id} not found")
            
            document.is_public = is_public
            document.status = "processed"
            db.commit()
            
            # Rebuild retrievers to reflect visibility change
            try:
                from backend.rag import retrieverconstruction
                retrieverconstruction.clear_cache()
                faiss_db, hybrid_retriever = retrieverconstruction.construct_retrievers(
                    include_postgres=True,
                    force_rebuild=True
                )
                print("✅ Retrievers rebuilt successfully")
            except Exception as e:
                print(f"❌ Error rebuilding retrievers: {e}")
            
            return {
                "status": "success",
                "doc_id": str(doc_uuid),
                "filename": document.filename,
                "is_public": is_public,
                "message": f"Document visibility updated successfully"
            }
            
        except Exception as e:
            print(f"❌ Error updating document visibility: {e}")
            return {"status": "error", "error": str(e)}


# QUERY HANDLING------------------------------------------------------------------
@celery_app.task(name='process_query', bind=True) 
def process_query_task(self, query_id: str, query: str, tenant_id: str, user_id: str, 
                    max_tokens: int = 500, temperature: float = 0.7, session_id: str = None):
    """Process a user query using RAG pipeline asynchronously"""
    print(f"🔄 Processing query {query_id}: {query[:50]}...")
    
    with manage_gpu_memory():
        try:
            # Generate response
            from backend.response_generation.generator import generate_response
            response = generate_response(query)
            
            # Save to database efficiently
            with get_db_session() as db:
                from backend.db.models import QueryLog as DB_QueryLog
                
                query_log_record = DB_QueryLog(
                    id=UUID(query_id),
                    user_id=UUID(user_id),
                    session_id=UUID(session_id) if session_id and session_id != 'string' else None,
                    query=query,
                    response=response,
                    timestamp=datetime.utcnow()
                )
                db.add(query_log_record)
                db.commit()
            
            return {
                "status": "success",
                "query_id": query_id,
                "query": query,
                "response": response,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "worker": self.request.hostname
            }
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "query_id": query_id,
                "query": query,
                "error": str(e),
                "worker": self.request.hostname
            }

@celery_app.task(name='process_query_with_tts', bind=True)
def process_query_with_tts_task(self, query_id: str, tts_task_id: str, query: str, tenant_id: str, 
                            user_id: str, max_tokens: int = 500, temperature: float = 0.7, 
                            session_id: str = None, language: str = "en", voice: str = "",
                            stream_audio: bool = True):
    """
    Optimized task that processes query → generates TTS → streams audio via WebSocket
    This reduces latency by chaining operations and streaming immediately
    """
    import asyncio
    print(f"🔄 Processing query with TTS: {query[:50]}...")
    
    with manage_gpu_memory():
        try:
            # Step 1: Generate text response
            print("📝 Generating text response...")
            from backend.response_generation.generator import generate_response
            response_text = generate_response(query)
            print(f"✅ Text response generated: {response_text[:50]}...")
            
            # Step 2: Immediately notify WebSocket that text is ready
            asyncio.run(send_text_response_via_websocket(
                tenant_id, user_id, query_id, query, response_text
            ))
            
            # Step 3: Process TTS with optimized settings for streaming
            print("🎵 Starting TTS processing...")
            import asyncio
            import edge_tts
            import os
            import uuid
            from datetime import datetime
            import base64
            
            # Set voice based on language
            if language == "ur" and not voice:
                voice = "ur-PK-UzmaNeural"
            elif language == "en" and not voice:
                voice = "en-US-JennyNeural"
            elif not voice:
                voice = "en-US-JennyNeural"
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{timestamp}_{tts_task_id[:8]}.wav"
            output_dir = "backend/rag/temp_audioFiles"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            # Step 4: Generate audio with streaming capability
            async def generate_and_stream_audio():
                """Generate TTS and stream chunks via WebSocket for minimal latency"""
                try:
                    # Notify WebSocket that audio generation started
                    await send_audio_status_via_websocket(
                        tenant_id, user_id, tts_task_id, "generating", 
                        {"message": "Audio generation started"}
                    )
                    
                    # Generate audio
                    communicate = edge_tts.Communicate(response_text, voice)
                    audio_chunks = []
                    
                    with open(output_path, "wb") as f:
                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                f.write(chunk["data"])
                                audio_chunks.append(chunk["data"])
                    
                    # Step 5: Stream audio immediately via WebSocket
                    if stream_audio and os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        
                        # Send audio file info first
                        await send_audio_status_via_websocket(
                            tenant_id, user_id, tts_task_id, "ready",
                            {
                                "filename": filename,
                                "file_size": file_size,
                                "audio_url": f"/tts/download/{filename}",
                                "voice": voice,
                                "language": language
                            }
                        )
                        
                        # Stream audio data in chunks for immediate playback
                        await stream_audio_file_via_websocket(
                            tenant_id, user_id, output_path, tts_task_id
                        )
                        
                        print(f"✅ Audio streamed successfully: {filename}")
                    
                    return {
                        "output_path": output_path,
                        "filename": filename,
                        "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    }
                    
                except Exception as e:
                    print(f"❌ Error in audio generation/streaming: {e}")
                    await send_audio_status_via_websocket(
                        tenant_id, user_id, tts_task_id, "error",
                        {"error": str(e)}
                    )
                    raise e
            
            # Run the async audio generation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio_result = loop.run_until_complete(generate_and_stream_audio())
            finally:
                loop.close()
            
            # Step 6: Save to database
            with get_db_session() as db:
                from backend.db.models import QueryLog as DB_QueryLog
                
                query_log_record = DB_QueryLog(
                    id=UUID(query_id),
                    user_id=UUID(user_id),
                    session_id=UUID(session_id) if session_id and session_id != 'string' else None,
                    query=query,
                    response=response_text,
                    timestamp=datetime.utcnow()
                )
                db.add(query_log_record)
                db.commit()
            
            print(f"✅ Complete pipeline finished: {query_id}")
            
            return {
                "status": "success",
                "query_id": query_id,
                "tts_task_id": tts_task_id,
                "query": query,
                "response": response_text,
                "audio_file": audio_result["filename"],
                "file_size": audio_result["file_size"],
                "language": language,
                "voice": voice,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "worker": self.request.hostname
            }
            
        except Exception as e:
            print(f"❌ Error in query with TTS pipeline: {e}")
            import traceback
            traceback.print_exc()
            
            # Notify WebSocket of error
            asyncio.run(send_error_via_websocket(
                tenant_id, user_id, query_id, str(e)
            ))
            
            return {
                "status": "error",
                "query_id": query_id,
                "tts_task_id": tts_task_id,
                "error": str(e),
                "worker": self.request.hostname
            }



# TEXT-TO-SPEECH TASKS-------------------------------------------------------------
@celery_app.task(name='process_tts', bind=True)
def process_tts_task(self, text: str, language: str, voice: str = "", output_filename: str = None, play_audio: bool = False):
    """
    Process Text-to-Speech conversion asynchronously
    
    Args:
        text (str): The text to convert to speech
        language (str): The language of the text ('ur' for Urdu, 'en' for English)
        voice (str): Optional specific voice to use
        output_filename (str): Optional custom filename for output
        play_audio (bool): Whether to play audio (only works if worker has audio output)
    
    Returns:
        dict: Status and file path information
    """
    print(f"🔄 Processing TTS task: {text[:50]}...")
    
    try:
        import asyncio
        import edge_tts
        import soundfile as sf
        import sounddevice as sd
        import os
        import uuid
        from datetime import datetime
        
        # Set default voices based on language
        if language == "ur" and not voice:
            voice = "ur-PK-UzmaNeural"
        elif language == "en" and not voice:
            voice = "en-US-JennyNeural"
        elif not voice:
            voice = "en-US-JennyNeural"  # Default fallback
        
        # Generate unique filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            output_filename = f"tts_{timestamp}_{unique_id}.wav"
        
        # Ensure the filename has .wav extension
        if not output_filename.endswith('.wav'):
            output_filename += '.wav'
        
        # Ensure output directory exists
        output_dir = "backend/rag/temp_audioFiles"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        async def synthesize_text_async(text, voice, output_file):
            """Async function to synthesize text to speech"""
            communicate = edge_tts.Communicate(text, voice)
            with open(output_file, "wb") as f:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
        
        # Run the async synthesis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(synthesize_text_async(text, voice, output_path))
        finally:
            loop.close()
        
        # Verify file was created and get file size
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"TTS output file was not created: {output_path}")
        
        file_size = os.path.getsize(output_path)
        
        # Optional: Play audio (only if requested and audio output is available)
        if play_audio:
            try:
                data, samplerate = sf.read(output_path, dtype='float32')
                sd.play(data, samplerate)
                sd.wait()
                print(f"✅ Audio played successfully")
            except Exception as audio_error:
                print(f"⚠️ Could not play audio: {audio_error}")
        
        print(f"✅ TTS task completed: {output_filename} ({file_size} bytes)")
        
        return {
            "status": "success",
            "task_id": self.request.id,
            "text": text,
            "language": language,
            "voice": voice,
            "output_file": output_path,
            "filename": output_filename,
            "file_size": file_size,
            "audio_played": play_audio,
            "worker": self.request.hostname
        }
        
    except Exception as e:
        print(f"❌ Error in TTS task: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "task_id": self.request.id,
            "text": text,
            "error": str(e),
            "worker": self.request.hostname
        }


# QUERY HANDLING HELPER FUNCTIONS-----------------------------------------------
# Helper functions for WebSocket communication
async def send_text_response_via_websocket(tenant_id: str, user_id: str, query_id: str, query: str, response: str):
    """Send text response via WebSocket"""
    try:
        from backend.main import manager  # Import the WebSocket manager
        
        await manager.send_json_to_user(tenant_id, user_id, {
            "type": "query_response",
            "query_id": query_id,
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"❌ Error sending text response via WebSocket: {e}")

async def send_audio_status_via_websocket(tenant_id: str, user_id: str, tts_task_id: str, status: str, data: dict):
    """Send audio status updates via WebSocket"""
    try:
        from backend.main import manager
        
        await manager.send_json_to_user(tenant_id, user_id, {
            "type": "audio_status",
            "tts_task_id": tts_task_id,
            "status": status,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"❌ Error sending audio status via WebSocket: {e}")

async def stream_audio_file_via_websocket(tenant_id: str, user_id: str, file_path: str, tts_task_id: str):
    """Stream audio file in chunks via WebSocket for immediate playback"""
    try:
        from backend.main import manager
        import base64
        
        chunk_size = 4096  # 4KB chunks for smooth streaming
        
        with open(file_path, "rb") as audio_file:
            chunk_number = 0
            while True:
                chunk = audio_file.read(chunk_size)
                if not chunk:
                    break
                
                # Encode chunk as base64 for WebSocket transmission
                chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                
                await manager.send_json_to_user(tenant_id, user_id, {
                    "type": "audio_chunk",
                    "tts_task_id": tts_task_id,
                    "chunk_number": chunk_number,
                    "chunk_data": chunk_b64,
                    "is_final": False
                })
                
                chunk_number += 1
                
                # Small delay to prevent overwhelming the WebSocket
                await asyncio.sleep(0.01)
        
        # Send final chunk marker
        await manager.send_json_to_user(tenant_id, user_id, {
            "type": "audio_chunk",
            "tts_task_id": tts_task_id,
            "chunk_number": chunk_number,
            "chunk_data": "",
            "is_final": True
        })
        
        print(f"✅ Audio file streamed in {chunk_number} chunks")
        
    except Exception as e:
        print(f"❌ Error streaming audio file via WebSocket: {e}")

async def send_error_via_websocket(tenant_id: str, user_id: str, query_id: str, error: str):
    """Send error message via WebSocket"""
    try:
        from backend.main import manager
        
        await manager.send_json_to_user(tenant_id, user_id, {
            "type": "error",
            "query_id": query_id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"❌ Error sending error via WebSocket: {e}")


