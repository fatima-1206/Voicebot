"""
Text-to-Speech (TTS) related Celery tasks
"""
from backend.workers.celery_worker import celery_app
from backend.workers.utilities import groq_client, notify_audio_status, redis_client
import os
import json
import asyncio
import base64
from datetime import datetime
from uuid import uuid4


async def stream_audio_file(tenant_id: str, user_id: str, file_path: str, task_id: str):
    """Stream audio file content via WebSocket in chunks"""
    if not redis_client:
        return
        
    try:
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


def generate_and_stream_tts(text: str, tenant_id: str, user_id: str, tts_task_id: str, 
                           language: str, voice: str) -> dict:
    """Generate TTS using Groq and stream via WebSocket"""
    global groq_client
    
    try:
        # Initialize Groq client if not already done
        if groq_client is None:
            from groq import Groq
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
            from groq import Groq
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