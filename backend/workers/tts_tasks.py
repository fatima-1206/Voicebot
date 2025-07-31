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
from nltk.tokenize import sent_tokenize
import edge_tts 

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


async def generate_tts_with_ftts(text: str, language: str, output_path: str, voice='en-US-AndrewMultilingualNeural') -> tuple[bool, str]:
    """Generate TTS using Edge TTS as fallback"""
    try:
        # Edge TTS voice mapping (add more as needed)
        edge_voice_map = {
            "en": "en-US-AriaNeural",
            "ur": "ur-PK-UzmaNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
            "it": "it-IT-ElsaNeural",
            "pt": "pt-BR-FranciscaNeural",
            "ru": "ru-RU-SvetlanaNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
            "ar": "ar-EG-SalmaNeural",
            "hi": "hi-IN-SwaraNeural",
        }
        edge_voice = edge_voice_map.get(language, "en-US-AriaNeural")
        if voice:
            edge_voice = voice  # Use provided voice if specified

        print(f"🔄 [TTS] Attempting Edge TTS generation (voice: {edge_voice})...")

        communicate = edge_tts.Communicate(text, voice=edge_voice)
        with open(output_path, "wb") as out_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    out_file.write(chunk["data"])

        if os.path.exists(output_path):
            print(f"✅ [TTS] Edge TTS generation successful")
            return True, "Edge TTS"
        else:
            return False, "Edge TTS - File not created"
    except Exception as e:
        print(f"❌ [TTS] Edge TTS generation failed: {e}")
        return False, f"Edge TTS - {str(e)}"
    
# ...existing code...

def generate_tts_with_fallback(text: str, language: str, voice: str, output_path: str) -> dict:
    """Generate TTS with Groq as primary and Edge TTS as fallback"""

    # Set default voice for Groq if not provided
    if not voice:
        voice = "Fritz-PlayAI"

    success, provider = generate_tts_with_groq(text, voice, output_path)

    if success:
        file_size = os.path.getsize(output_path)
        return {
            "success": True,
            "provider": provider,
            "file_size": file_size,
            "method": "primary"
        }

    # Fallback to Edge TTS
    print("⚠️ [TTS] Groq failed, falling back to Edge TTS...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success, provider = loop.run_until_complete(generate_tts_with_ftts(text, language, output_path))
    finally:
        loop.close()

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
        "provider": "None - Both Groq and Edge TTS failed",
        "file_size": 0,
        "method": "failed"
    }

def generate_and_stream_tts(text: str, tenant_id: str, user_id: str, tts_task_id: str, 
                           language: str, voice: str, chunk_id: str=None) -> dict:
    """Generate TTS using Groq with Edge TTS fallback and stream via WebSocket"""

    try:
        # scentence tokenize the text response
        print("\nGENERATING AND STREAMING TTS")
        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            # Notify start for first sentence
            if i == 0:
                notify_audio_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    task_id=tts_task_id,
                    number_of_scentences=len(sentences),
                    status="generating",
                    data={
                        "message": "Audio generation started",
                        "number_of_sentences": len(sentences)
                    }
                )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{timestamp}_{tts_task_id[:8]}_{i}.wav"
            output_dir = "backend/rag/temp_audioFiles"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            print(f"🔄 [TTS] Generating audio for sentence: {sentence[:50]}...")
            
            # Generate audio with fallback
            result = generate_tts_with_fallback(sentence, language, voice, output_path)
            
            if result["success"]:
                print(f"✅ [TTS] Audio generated: {filename} using {result['provider']}")

               
                # Stream audio file via WebSocket
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(stream_audio_file(tenant_id, user_id, output_path, f"{tts_task_id}_{i}"))
                finally:
                    loop.close()

        return {
            "status": "success",
            "message": f"Successfully generated and streamed {len(sentences)} sentence(s)."
        }
    except Exception as e:
        print(f"❌ [TTS] Generation failed: {e}")
        notify_audio_status(tenant_id=tenant_id, user_id=user_id, number_of_scentences=len(sentences),task_id= tts_task_id, status="error", data={
            "error": str(e),
            "provider": "TTS with fallback"
        })
        raise e


def generate_tts_file(text: str, tts_task_id: str, language: str, voice: str, chunk_id: str=None, tenant_id=None, user_id=None) -> dict:
    """Generate TTS file using Groq with gTTS fallback without streaming"""
    
    try:
        print("\nGENERATING TTS FILE")
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
                "chunk_id": chunk_id,
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