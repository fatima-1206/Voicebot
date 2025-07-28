"""
Text-to-Speech (TTS) related Celery tasks with Groq and gTTS fallback
"""
from backend.workers.celery_worker import celery_app
from backend.workers.utilities import groq_client, notify_audio_status, redis_client
import os
import json
import asyncio
import base64
import subprocess
from datetime import datetime
from uuid import uuid4
from gtts import gTTS


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


def generate_tts_with_groq(text: str, voice: str, output_path: str) -> tuple[bool, str]:
    """Generate TTS using Groq API"""
    global groq_client
    
    try:
        # Initialize Groq client if not already done
        if groq_client is None:
            from groq import Groq
            groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        print(f"🔄 [TTS] Attempting Groq generation...")
        if voice not in ["Aaliyah-PlayAI",
            "Adelaide-PlayAI",
            "Angelo-PlayAI",
            "Arista-PlayAI",
            "Atlas-PlayAI",
            "Basil-PlayAI",
            "Briggs-PlayAI",
            "Calum-PlayAI",
            "Celeste-PlayAI",
            "Cheyenne-PlayAI",
            "Chip-PlayAI",
            "Cillian-PlayAI",
            "Deedee-PlayAI",
            "Eleanor-PlayAI",
            "Fritz-PlayAI",
            "Gail-PlayAI",
            "Indigo-PlayAI",
            "Jennifer-PlayAI",
            "Judy-PlayAI",
            "Mamaw-PlayAI",
            "Mason-PlayAI",
            "Mikail-PlayAI",
            "Mitch-PlayAI",
            "Nia-PlayAI",
            "Quinn-PlayAI",
            "Ruby-PlayAI",
            "Thunder-PlayAI"
        ]:
            voice = "Fritz-PlayAI"  # Default voice if not recognized

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
            print(f"✅ [TTS] Groq generation successful")
            return True, "Groq PlayAI"
        else:
            return False, "Groq PlayAI - File not created"
            
    except Exception as e:
        print(f"❌ [TTS] Groq generation failed: {e}")
        return False, f"Groq PlayAI - {str(e)}"


def generate_tts_with_gtts(text: str, language: str, output_path: str) -> tuple[bool, str]:
    """Generate TTS using Google TTS (gTTS) as fallback"""
    try:
        # Map language codes for gTTS
        gtts_language_map = {
            "ur": "ur",  # Urdu
            "en": "en",  # English
            "es": "es",  # Spanish
            "fr": "fr",  # French
            "de": "de",  # German
            "it": "it",  # Italian
            "pt": "pt",  # Portuguese
            "ru": "ru",  # Russian
            "ja": "ja",  # Japanese
            "ko": "ko",  # Korean
            "zh": "zh",  # Chinese
            "ar": "ar",  # Arabic
            "hi": "hi",  # Hindi
        }
        
        # Use mapped language or default to English
        gtts_lang = gtts_language_map.get(language, "en")
        
        print(f"🔄 [TTS] Attempting gTTS generation (language: {gtts_lang})...")
        
        # Create gTTS object
        tts_obj = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # Save to temporary file first (gTTS saves as MP3)
        temp_mp3_file = output_path.replace('.wav', '_temp.mp3')
        tts_obj.save(temp_mp3_file)
        
        # Convert MP3 to WAV using ffmpeg
        try:
            subprocess.run([
                'ffmpeg', '-i', temp_mp3_file, '-y', output_path
            ], check=True, capture_output=True)
            
            # Clean up temporary MP3 file
            os.remove(temp_mp3_file)
            print(f"✅ [TTS] gTTS generation successful (converted to WAV)")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg is not available, rename MP3 to WAV
            os.rename(temp_mp3_file, output_path)
            print("⚠️ [TTS] gTTS successful (saved as MP3 with .wav extension - ffmpeg not available)")
        
        return True, "Google TTS (gTTS)"
        
    except Exception as e:
        print(f"❌ [TTS] gTTS generation failed: {e}")
        return False, f"Google TTS - {str(e)}"


def generate_tts_with_fallback(text: str, language: str, voice: str, output_path: str) -> dict:
    """Generate TTS with Groq as primary and gTTS as fallback"""
    
    # Set default voice for Groq if not provided
    if not voice:
        voice = "Fritz-PlayAI"
    
    # Try Groq first
    success, provider = generate_tts_with_groq(text, voice, output_path)
    
    if success:
        file_size = os.path.getsize(output_path)
        return {
            "success": True,
            "provider": provider,
            "file_size": file_size,
            "method": "primary"
        }
    
    # Fallback to gTTS
    print("⚠️ [TTS] Groq failed, falling back to gTTS...")
    success, provider = generate_tts_with_gtts(text, language, output_path)
    
    if success:
        file_size = os.path.getsize(output_path)
        return {
            "success": True,
            "provider": provider,
            "file_size": file_size,
            "method": "fallback"
        }
    
    # Both failed
    return {
        "success": False,
        "provider": "None - Both Groq and gTTS failed",
        "file_size": 0,
        "method": "failed"
    }


def generate_and_stream_tts(text: str, tenant_id: str, user_id: str, tts_task_id: str, 
                           language: str, voice: str) -> dict:
    """Generate TTS using Groq with gTTS fallback and stream via WebSocket"""
    
    try:
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{tts_task_id[:8]}.wav"
        output_dir = "backend/rag/temp_audioFiles"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Notify start
        notify_audio_status(tenant_id, user_id, tts_task_id, "generating", 
                          {"message": "Audio generation started"})
        
        print(f"🔄 [TTS] Generating audio with fallback: {text[:50]}...")
        
        # Generate audio with fallback
        result = generate_tts_with_fallback(text, language, voice, output_path)
        
        if result["success"]:
            print(f"✅ [TTS] Audio generated: {filename} using {result['provider']}")
            
            # Notify ready
            notify_audio_status(tenant_id, user_id, tts_task_id, "ready", {
                "filename": filename,
                "file_size": result["file_size"],
                "audio_url": f"/tts/download/{filename}",
                "voice": voice,
                "language": language,
                "provider": result["provider"],
                "method": result["method"]
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
            
            return {
                "filename": filename, 
                "file_size": result["file_size"], 
                "status": "success",
                "provider": result["provider"],
                "method": result["method"]
            }
        else:
            raise Exception(f"Both TTS providers failed: {result['provider']}")
        
    except Exception as e:
        print(f"❌ [TTS] Generation failed: {e}")
        notify_audio_status(tenant_id, user_id, tts_task_id, "error", {
            "error": str(e),
            "provider": "TTS with fallback"
        })
        raise e


def generate_tts_file(text: str, tts_task_id: str, language: str, voice: str) -> dict:
    """Generate TTS file using Groq with gTTS fallback without streaming"""
    
    try:
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{tts_task_id[:8]}.wav"
        output_dir = "backend/rag/temp_audioFiles"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        print(f"🔄 [TTS] Generating audio file with fallback: {text[:50]}...")
        
        # Generate audio with fallback
        result = generate_tts_with_fallback(text, language, voice, output_path)
        
        if result["success"]:
            print(f"✅ [TTS] File generated: {filename} ({result['file_size']} bytes) using {result['provider']}")
            return {
                "filename": filename,
                "file_size": result["file_size"],
                "status": "success",
                "provider": result["provider"],
                "method": result["method"]
            }
        else:
            raise Exception(f"Both TTS providers failed: {result['provider']}")
            
    except Exception as e:
        print(f"❌ [TTS] File generation failed: {e}")
        raise Exception(f"TTS generation failed: {str(e)}")


@celery_app.task(name='process_tts', bind=True)
def process_tts_task(self, text: str, language: str, voice: str = "", output_filename: str = None, play_audio: bool = False):
    """
    Process Text-to-Speech conversion using Groq with gTTS fallback (standalone TTS)
    """
    print(f"🔄 [TTS] Processing with fallback: {text[:50]}...")
    
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
            "provider": result["provider"],
            "method": result["method"],
            "worker": self.request.hostname
        }
        
    except Exception as e:
        print(f"❌ [TTS] Error: {e}")
        return {
            "status": "error",
            "task_id": self.request.id,
            "text": text,
            "error": str(e),
            "provider": "TTS with fallback",
            "worker": self.request.hostname
        }