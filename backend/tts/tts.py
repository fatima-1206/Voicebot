import asyncio
import os
import uuid
from datetime import datetime
from typing import Optional, Union
import soundfile as sf
import sounddevice as sd
from groq import Groq

# TODO: Add a fallback in case the limit of play tts is reached

# Ensure output directory exists
os.makedirs("backend/rag/temp_audioFiles", exist_ok=True)

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def synthesize_text_groq(text, voice="Fritz-PlayAI", output_file="backend/rag/temp_audioFiles/output.wav"):
    """Synchronous function to synthesize text to speech using Groq API"""
    try:
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        response.write_to_file(output_file)
        return True
    except Exception as e:
        print(f"Error in Groq TTS synthesis: {e}")
        return False

async def synthesize_text(text, voice="Fritz-PlayAI", output_file="backend/rag/temp_audioFiles/output.wav"):
    """Async wrapper for Groq TTS synthesis"""
    # Run the synchronous Groq call in a thread pool
    import asyncio
    loop = asyncio.get_event_loop()
    
    def sync_synthesize():
        return synthesize_text_groq(text, voice, output_file)
    
    return await loop.run_in_executor(None, sync_synthesize)

def play_audio(file_path):
    """Play audio file using sounddevice"""
    try:
        data, samplerate = sf.read(file_path, dtype='float32')
        sd.play(data, samplerate)
        sd.wait()
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def get_groq_voice_for_language(language: str) -> str:
    """Get appropriate Groq voice for the given language"""
    # Load from config or use defaults
    try:
        import json
        with open("config.json", "r") as f:
            config = json.load(f)
        voices = config.get("groq_tts", {}).get("voices", {})
        return voices.get(language, "Fritz-PlayAI")
    except:
        # Fallback to default voice for all languages since Groq has limited voice options
        return "Fritz-PlayAI"

def tts(text: str, language: str, voice: str = "", output_filename: str = None, play_audio_flag: bool = True) -> str:
    """
    Synchronous Text-to-Speech function to convert text to audio using Groq.
    
    Args:
        text (str): The text to convert to speech.
        language (str): The language of the text ('ur' for Urdu, 'en' for English).
        voice (str): Optional voice parameter for TTS.
        output_filename (str): Optional custom filename for output.
        play_audio_flag (bool): Whether to play the audio after generation.
    
    Returns:
        str: Path to the saved audio file.
    """
    # Set voice based on language or use provided voice
    if not voice:
        voice = get_groq_voice_for_language(language)

    # Generate unique filename if not provided
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"tts_{timestamp}_{unique_id}.wav"
    
    # Ensure the filename has .wav extension
    if not output_filename.endswith('.wav'):
        output_filename += '.wav'
    
    output_file = f"backend/rag/temp_audioFiles/{output_filename}"
    
    # Generate audio using Groq
    success = synthesize_text_groq(text, voice, output_file)
    
    if not success:
        raise Exception("Failed to generate audio using Groq TTS")
    
    # Play audio if requested
    if play_audio_flag:
        play_audio(output_file)
    
    return output_file

def tts_async(text: str, language: str, voice: str = "", output_filename: str = None, play_audio_flag: bool = False) -> str:
    """
    Submit TTS task to Celery queue for asynchronous processing.
    
    Args:
        text (str): The text to convert to speech.
        language (str): The language of the text ('ur' for Urdu, 'en' for English).
        voice (str): Optional voice parameter for TTS.
        output_filename (str): Optional custom filename for output.
        play_audio_flag (bool): Whether to play the audio (only works if worker has audio output).
    
    Returns:
        str: Celery task ID for tracking the task status.
    """
    try:
        from backend.workers.tasks import process_tts_task
        
        # Submit task to Celery
        task = process_tts_task.delay(
            text=text,
            language=language,
            voice=voice,
            output_filename=output_filename,
            play_audio=play_audio_flag
        )
        
        return task.id
    except ImportError:
        print("❌ Celery not available, falling back to synchronous TTS")
        return tts(text, language, voice, output_filename, play_audio_flag)

def get_tts_task_status(task_id: str) -> dict:
    """
    Get the status and result of a TTS Celery task.
    
    Args:
        task_id (str): The Celery task ID.
    
    Returns:
        dict: Task status and result information.
    """
    try:
        from backend.workers.celery_worker import celery_app
        from celery.result import AsyncResult
        
        task_result = AsyncResult(task_id, app=celery_app)
        
        return {
            "task_id": task_id,
            "status": task_result.status,
            "ready": task_result.ready(),
            "successful": task_result.successful() if task_result.ready() else None,
            "result": task_result.result if task_result.ready() else None,
            "info": task_result.info
        }
    except ImportError:
        return {
            "task_id": task_id,
            "status": "ERROR",
            "error": "Celery not available"
        }

def test_tts():
    """Test function for TTS functionality"""
    urdu_text = "Hello, how are you? How can I help you today?"
    
    print("Testing synchronous Groq TTS...")
    output_file = tts(urdu_text, language="ur", voice="Fritz-PlayAI")
    print(f"Synchronous TTS completed: {output_file}")
    
    print("\nTesting asynchronous TTS...")
    task_id = tts_async(urdu_text, language="ur", voice="Fritz-PlayAI")
    print(f"Async TTS task submitted: {task_id}")
    
    # Check task status
    import time
    for i in range(10):  # Check for up to 10 seconds
        status = get_tts_task_status(task_id)
        print(f"Task status: {status['status']}")
        if status['ready']:
            print(f"Task result: {status['result']}")
            break
        time.sleep(1)

# Uncomment to test
# test_tts()