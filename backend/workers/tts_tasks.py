"""
Text-to-Speech (TTS) related Celery tasks with Groq and gTTS fallback
"""
from backend.workers.celery_worker import celery_app
from backend.workers.utilities import groq_client, notify_audio_status, redis_client
from backend.redis_manager import redis_manager
import os
import json
import asyncio
import base64
import subprocess
from datetime import datetime
from uuid import uuid4
from gtts import gTTS
from nltk.tokenize import sent_tokenize
import edge_tts 
from backend.db.models import Tenant, BillingPlan
from backend.db.connection import SessionLocal
from backend.auth.jwt_ import get_current_user

async def stream_audio_file(tenant_id: str, user_id: str, file_path: str, task_id: str, 
                          device_id: str = None, session_id: str = None):
    """Stream audio file content via WebSocket with session support"""
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
        
        # Use Redis manager for session-aware publishing
        from backend.redis_manager import redis_manager
        if device_id and session_id:
            redis_manager.publish_to_session(tenant_id, user_id, device_id, session_id, message)
        else:
            redis_manager.publish_to_user_sessions(tenant_id, user_id, message)
        
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
        
        from backend.redis_manager import redis_manager
        if device_id and session_id:
            redis_manager.publish_to_session(tenant_id, user_id, device_id, session_id, error_message)
        else:
            redis_manager.publish_to_user_sessions(tenant_id, user_id, error_message)

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
            "Gemma-PlayAI",
            "Haven-PlayAI",
            "Iris-PlayAI",
            "Javi-PlayAI",
            "Lara-PlayAI",
            "Malcolm-PlayAI",
            "Marin-PlayAI",
            "Moira-PlayAI",
            "Odette-PlayAI",
            "Regina-PlayAI",
            "Seth-PlayAI",
            "Todd-PlayAI",
            "Zara-PlayAI"]:
            voice = "Fritz-PlayAI"  # Default fallback
        # Create the speech
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        # Save the audio file
        with open(output_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1024):
                f.write(chunk)
        
        print(f"✅ [TTS] Groq generation successful")
        return True, "Groq PlayAI"
        
    except Exception as e:
        print(f"❌ [TTS] Groq generation failed: {e}")
        return False, str(e)

def generate_tts_with_gtts(text: str, language: str, output_path: str) -> tuple[bool, str]:
    """Generate TTS using gTTS as fallback"""
    try:
        print(f"🔄 [TTS] Attempting gTTS generation...")
        
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to temporary mp3 file first
        temp_mp3 = output_path.replace('.wav', '_temp.mp3')
        tts.save(temp_mp3)
        
        # Convert mp3 to wav using ffmpeg
        result = subprocess.run([
            'ffmpeg', '-i', temp_mp3, '-acodec', 'pcm_s16le', 
            '-ar', '22050', '-ac', '1', '-y', output_path
        ], capture_output=True, text=True)
        
        # Clean up temp file
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        
        if result.returncode == 0:
            print(f"✅ [TTS] gTTS generation successful")
            return True, "gTTS"
        else:
            print(f"❌ [TTS] FFmpeg conversion failed: {result.stderr}")
            return False, f"FFmpeg error: {result.stderr}"
            
    except Exception as e:
        print(f"❌ [TTS] gTTS generation failed: {e}")
        return False, str(e)

async def generate_tts_with_edge(text: str, voice: str, output_path: str) -> tuple[bool, str]:
    """Generate TTS using Edge TTS as fallback"""
    try:
        print(f"🔄 [TTS] Attempting Edge TTS generation...")
        
        # Voice mapping for Edge TTS
        edge_voices = {
            "en": "en-US-AriaNeural",
            "ur": "ur-PK-AsadNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
            "it": "it-IT-ElsaNeural",
            "pt": "pt-BR-FranciscaNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
            "zh": "zh-CN-XiaoxiaoNeural"
        }
        
        # Default to English if language not supported
        edge_voice = edge_voices.get("en", "en-US-AriaNeural")
        
        # Generate speech
        communicate = edge_tts.Communicate(text, edge_voice)
        await communicate.save(output_path)
        
        print(f"✅ [TTS] Edge TTS generation successful")
        return True, "Edge TTS"
        
    except Exception as e:
        print(f"❌ [TTS] Edge TTS generation failed: {e}")
        return False, f"Edge TTS - {str(e)}"
    
# ...existing code...
def get_user_billing_plan(tenant_id: str) -> str:
    """Get user's billing plan from the database"""
    db = SessionLocal()
    billing_plan = "Free"  # Default plan

    try:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if tenant and tenant.billing_plan_id:
            billing_plan_id = tenant.billing_plan_id
            if billing_plan_id:
                billing_plan = db.query(BillingPlan).filter(BillingPlan.id == billing_plan_id).first()
                if billing_plan:
                    return billing_plan.name
    except Exception as e:
        print(f"⚠️ Could not get billing plan: {e}")
        return "Free"

def generate_tts_with_fallback(tenant_id: str, text: str, language: str, voice: str, output_path: str) -> dict:
    """Generate TTS with intelligent fallback based on billing plan"""
    
    # Get user's billing plan
    billing_plan = get_user_billing_plan(tenant_id)
    print(f"\nTHE USERS BILLING PLAN ISSSS {billing_plan}")
    
    # Try methods in order based on plan
    if billing_plan in ["Pro", "Enterprise"]:
        # Premium users: Try Groq first, then Edge TTS, then gTTS
        methods = [
            ("groq", lambda: generate_tts_with_groq(text, voice, output_path)),
            ("edge", lambda: asyncio.run(generate_tts_with_edge(text, language, output_path))),
            ("gtts", lambda: generate_tts_with_gtts(text, language, output_path))
        ]
    else:
        # Free users: Try Edge TTS first, then gTTS, then Groq (if available)
        methods = [
            ("edge", lambda: asyncio.run(generate_tts_with_edge(text, language, output_path))),
            ("gtts", lambda: generate_tts_with_gtts(text, language, output_path)),
            ("groq", lambda: generate_tts_with_groq(text, voice, output_path))
        ]
    
    for method_name, method_func in methods:
        try:
            success, provider = method_func()
            if success:
                return {
                    "success": True,
                    "provider": provider,
                    "method": method_name,
                    "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0
                }
        except Exception as e:
            print(f"❌ [TTS] {method_name} method failed: {e}")
            continue
    
    # All methods failed
    return {
        "success": False,
        "provider": "none",
        "error": "All TTS methods failed",
        "file_size": 0,
        "method": "failed"
    }

def generate_and_stream_tts(text: str, tenant_id: str, user_id: str, tts_task_id: str, 
                           language: str, voice: str, device_id: str = None, session_id: str = None, 
                           chunk_id: str = None) -> dict:
    """Generate TTS using Groq with Edge TTS fallback and stream via WebSocket"""

    try:
        # Sentence tokenize the text response
        print("\nGENERATING AND STREAMING TTS")
        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            # Notify start for first sentence
            if i == 0:
                redis_manager.publish_to_session(
                    tenant_id, user_id, device_id, session_id, message={
                        "type": "tts_start",
                        "task_id": tts_task_id,
                        "sentence_count": len(sentences),
                        "chunk_id": chunk_id,
                        "device_id": device_id,
                        "session_id": session_id
                    }
                )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{timestamp}_{tts_task_id[:8]}_{i}.wav"
            output_dir = "backend/rag/temp_audioFiles"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            print(f"🔄 [TTS] Generating audio for sentence: {sentence[:50]}...")
            
            # Generate audio with fallback
            result = generate_tts_with_fallback(tenant_id, sentence, language, voice, output_path)
            
            if result["success"]:
                print(f"✅ [TTS] Audio generated: {filename} using {result['provider']}")

                # Stream audio file via WebSocket
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(stream_audio_file(
                        tenant_id, user_id, output_path, f"{tts_task_id}_{i}", 
                        device_id, session_id
                    ))
                finally:
                    loop.close()

        return {
            "status": "success",
            "task_id": tts_task_id,
            "sentences_processed": len(sentences)
        }

    except Exception as e:
        print(f"❌ [TTS] Stream error: {e}")
        return {
            "status": "error",
            "task_id": tts_task_id,
            "error": str(e)
        }

def generate_tts_file(text: str, tenant_id: str, user_id: str, tts_task_id: str, 
                     language: str, voice: str, device_id: str = None, session_id: str = None, 
                     chunk_id: str = None) -> dict:
    """Generate single TTS file (non-streaming)"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{tts_task_id[:8]}.wav"
        output_dir = "backend/rag/temp_audioFiles"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        print(f"🔄 [TTS] Generating single file for: {text[:50]}...")
        
        # Generate audio with fallback
        result = generate_tts_with_fallback(tenant_id, text, language, voice, output_path)
        
        if result["success"]:
            print(f"✅ [TTS] Audio file generated: {filename} using {result['provider']}")
            
            # Stream the single file
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(stream_audio_file(
                    tenant_id, user_id, output_path, tts_task_id, 
                    device_id, session_id
                ))
            finally:
                loop.close()
            
            return {
                "status": "success",
                "task_id": tts_task_id,
                "file_path": output_path
            }
        else:
            return {
                "status": "error",
                "task_id": tts_task_id,
                "error": result.get("error", "Unknown error")
            }

    except Exception as e:
        print(f"❌ [TTS] File generation error: {e}")
        return {
            "status": "error",
            "task_id": tts_task_id,
            "error": str(e)
        }

@celery_app.task(name='process_tts', bind=True)
def process_tts_task(self, text: str, language: str = "en", voice: str = "", 
                    output_filename: str = None, play_audio: bool = False):
    """Process TTS request (legacy task for backward compatibility)"""
    
    try:
        # Generate output filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            output_filename = f"tts_{timestamp}_{self.request.id[:8]}.wav"
        
        output_dir = "backend/rag/temp_audioFiles"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        # Use fallback generation (defaults to Free tier)
        result = generate_tts_with_fallback("default_tenant", text, language, voice, output_path)
        
        if result["success"]:
            file_size = os.path.getsize(output_path)
            
            return {
                "status": "success",
                "file_path": output_path,
                "file_size": file_size,
                "provider": result["provider"],
                "task_id": self.request.id
            }
        else:
            return {
                "status": "error",
                "error": result.get("error", "TTS generation failed"),
                "task_id": self.request.id
            }
            
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "task_id": self.request.id
        }