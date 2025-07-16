import asyncio
import edge_tts
import soundfile as sf
import sounddevice as sd
import os
import uuid
from datetime import datetime
from typing import Optional, Union

# Ensure output directory exists
os.makedirs("backend/rag/temp_audioFiles", exist_ok=True)

async def synthesize_text(text, voice="ur-PK-UzmaNeural", output_file="backend/rag/temp_audioFiles/output.wav"):
    """Async function to synthesize text to speech using edge-tts"""
    communicate = edge_tts.Communicate(text, voice)
    with open(output_file, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

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

def tts(text: str, language: str, voice: str = "", output_filename: str = None, play_audio_flag: bool = True) -> str:
    """
    Synchronous Text-to-Speech function to convert text to audio.
    
    Args:
        text (str): The text to convert to speech.
        language (str): The language of the text ('ur' for Urdu, 'en' for English).
        voice (str): Optional voice parameter for TTS.
        output_filename (str): Optional custom filename for output.
        play_audio_flag (bool): Whether to play the audio after generation.
    
    Returns:
        str: Path to the saved audio file.
    """
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
    
    output_file = f"backend/rag/temp_audioFiles/{output_filename}"
    
    # Run async synthesis
    asyncio.run(synthesize_text(text, voice, output_file))
    
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
    urdu_text = "ہیلو، آپ کیسے ہیں؟ میں آپ کی کس طرح مدد کر سکتی ہوں؟"
    
    print("Testing synchronous TTS...")
    output_file = tts(urdu_text, language="ur", voice="ur-PK-UzmaNeural")
    print(f"Synchronous TTS completed: {output_file}")
    
    print("\nTesting asynchronous TTS...")
    task_id = tts_async(urdu_text, language="ur", voice="ur-PK-UzmaNeural")
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