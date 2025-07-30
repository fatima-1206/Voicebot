"""
Speech-to-Text (STT) related Celery tasks
"""
from backend.workers.celery_worker import celery_app
from backend.workers.utilities import (
    model, vad_model, sample_rate, redis_client, manage_gpu_memory
)
import numpy as np
import torch
import io
import wave
import json
import time
from datetime import datetime
from uuid import uuid4
from silero_vad import get_speech_timestamps, collect_chunks
from better_profanity import profanity

profanity.load_censor_words()


def bytes_to_numpy_audio(audio_bytes: bytes) -> np.ndarray:
    """Convert audio bytes to numpy array - handles multiple formats"""
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



def send_transcription_result_via_websocket(tenant_id: str, user_id: str, chunk_id: str, 
                                          transcript: str, confidence: float, detected_language: str,
                                          segments: list, processing_time: float):
    """Send transcription result via Redis for WebSocket delivery"""
    if not redis_client:
        return
        
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
    if not redis_client:
        return
        
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


@celery_app.task(name='transcribe_chunk', bind=True)
def transcribe_chunk(self, audio_bytes: bytes, user_id: str, chunk_id: str, tenant_id: str = None):
    """
    Transcribe audio chunk using direct numpy array processing (no temp files)
    This is called directly from WebSocket audio streams
    """
    print(f"🔄 [STT] Transcribing chunk {chunk_id} for user {user_id}")
    try:
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
            segments, info = model.transcribe(audio_data, beam_size=5, language=language, task="transcribe", vad_filter=False)
        
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
        # apply the profanity filter
        if full_transcript:
            full_transcript = profanity.censor(full_transcript)
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


@celery_app.task(name='streaming_transcribe', bind=True)
def streaming_transcribe_task(self, audio_bytes: bytes, user_id: str, chunk_id: str, 
                            tenant_id: str = None, is_final: bool = False):
    """
    Streaming transcription - provides intermediate results
    """
    try:
        audio_data = bytes_to_numpy_audio(audio_bytes)
        # store the audio data in a temporary file if needed


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
        
        # apply the profanity filter
        if transcript:
            transcript = profanity.censor(transcript)

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
        if redis_client and tenant_id:
            redis_client.publish(f"ws:{tenant_id}:{user_id}", json.dumps(result))
        
        # If final and should respond, trigger response generation
        if is_final:
            from backend.workers.query_tasks import process_query_with_tts_task
            process_query_with_tts_task.delay(
                query_id=str(uuid4()),
                tts_task_id=str(uuid4()),
                query=transcript,
                tenant_id=tenant_id,
                user_id=user_id,
                language="en",
                voice="Fritz-PlayAI",  # Default voice
                stream_audio=True
            )
        
        return result
        
    except Exception as e:
        print(f"❌ [STT] Streaming transcribe error: {e}")
        return {
            "status": "error",
            "chunk_id": chunk_id,
            "error": str(e)
        }


@celery_app.task(name='process_continuous_audio', bind=True)
def process_continuous_audio_task(self,speech_bytes, speech_timestamps, user_id: str, tenant_id: str, 
                                 buffer_key: str, tts_language: str = "en", 
                                 voice: str = "Fritz-PlayAI",chunk_id=None, auto_respond: bool = True, use_rag: bool = True):
    """
    Process continuous audio stream with server-side VAD
    Detects speech segments and triggers STT+RAG+TTS pipeline
    """
    print(f"🔄 [CONTINUOUS] Processing audio buffer for user {user_id}")
    
    try: 
        print(f"   🔍 Found {len(speech_timestamps)} speech segments")
        if not speech_timestamps:
            return { "type": "status",
                "status": "no_speech"}
        

        if not chunk_id:   
            # Create unique chunk ID for this segment
            chunk_id = f"{user_id}_{int(time.time() * 1000)}"
            
        segment_result = process_speech_segment(
            speech_bytes, user_id, chunk_id, tenant_id,
            tts_language, voice, auto_respond, use_rag=use_rag
        )
        
        
        return {
            "status": "success",
            "segments_processed": len(segment_result),
            "chunk_id": chunk_id,
            "results": segment_result
        }
        
    except Exception as e:
        print(f"❌ [CONTINUOUS] Error: {e}")
        return {"status": "error", "error": str(e)}


def process_speech_segment(speech_bytes: bytes, user_id: str, chunk_id: str, 
                         tenant_id: str, tts_language: str, voice: str, auto_respond: bool, use_rag: bool = True):
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
            vad_filter=True,  # We already did VAD
            initial_prompt=None
        )
        
        # Extract transcript
        transcript_parts = []
        for segment in segments:
            segment = profanity.censor(segment.text.strip())
            transcript_parts.append(segment.strip())
        
        transcript = " ".join(transcript_parts).strip()
        
        if not transcript:
            print(f"   ❌ No transcript generated")
            return {"status": "no_transcript", "chunk_id": chunk_id}
        else:
            transcript = profanity.censor(transcript)

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
        if redis_client:
            redis_client.publish(f"ws:{tenant_id}:{user_id}", json.dumps(stt_result))
        
        # Step 2: Check if we should respond
        if auto_respond:
            print(f"   🤖 Triggering response for: '{transcript}'")
            
            # Generate response immediately (don't delay this)
            response_id = str(uuid4())
            tts_task_id = str(uuid4())
            
            # Trigger RAG + TTS response
            from backend.workers.query_tasks import process_query_with_tts_task
            process_query_with_tts_task.delay(
                chunk_id=chunk_id,
                query_id=response_id,
                tts_task_id=tts_task_id,
                query=transcript,
                tenant_id=tenant_id,
                user_id=user_id,
                language=tts_language,
                voice=voice,
                stream_audio=True,
                max_tokens=300,  # Shorter responses for conversation
                temperature=0.7,
                use_rag=use_rag
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