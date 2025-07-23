from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Union
import os

router = APIRouter()

class TTSRequest(BaseModel):
    text: str
    language: str  # 'ur' or 'en'
    voice: Optional[str] = ""
    output_filename: Optional[str] = None
    async_mode: bool = True  # Default to async mode
    play_audio: bool = False

class TTSSyncResponse(BaseModel):
    status: str
    text: str
    language: str
    voice: str
    output_file: str
    filename: str
    file_size: int

class TTSAsyncResponse(BaseModel):
    status: str
    task_id: str
    text: str
    language: str
    voice: str
    message: str

class TTSTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    ready: bool
    successful: Optional[bool]
    result: Optional[dict]
    info: Optional[dict]

@router.post("/convert", response_model=Union[TTSSyncResponse, TTSAsyncResponse])
async def convert_text_to_speech(request: TTSRequest):
    """
    Convert text to speech either synchronously or asynchronously
    """
    try:
        if request.async_mode:
            # Use async Celery task
            from backend.tts.tts import tts_async
            
            task_id = tts_async(
                text=request.text,
                language=request.language,
                voice=request.voice,
                output_filename=request.output_filename,
                play_audio_flag=request.play_audio
            )
            
            return TTSAsyncResponse(
                status="queued",
                task_id=task_id,
                text=request.text,
                language=request.language,
                voice=request.voice or ("ur-PK-UzmaNeural" if request.language == "ur" else "en-US-JennyNeural"),
                message="TTS task submitted to queue"
            )
        else:
            # Use synchronous processing
            from backend.tts.tts import tts
            
            output_file = tts(
                text=request.text,
                language=request.language,
                voice=request.voice,
                output_filename=request.output_filename,
                play_audio_flag=request.play_audio
            )
            
            # Get file info
            file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
            filename = os.path.basename(output_file)
            
            return TTSSyncResponse(
                status="completed",
                text=request.text,
                language=request.language,
                voice=request.voice or ("ur-PK-UzmaNeural" if request.language == "ur" else "en-US-JennyNeural"),
                output_file=output_file,
                filename=filename,
                file_size=file_size
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")

@router.get("/task/{task_id}", response_model=TTSTaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of an asynchronous TTS task
    """
    try:
        from backend.tts.tts import get_tts_task_status
        
        status_info = get_tts_task_status(task_id)
        
        return TTSTaskStatusResponse(
            task_id=status_info["task_id"],
            status=status_info["status"],
            ready=status_info["ready"],
            successful=status_info.get("successful"),
            result=status_info.get("result"),
            info=status_info.get("info")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.get("/download/{filename}")
async def download_audio_file(filename: str):
    """
    Download generated audio file
    """
    from fastapi.responses import FileResponse
    
    file_path = f"backend/rag/temp_audioFiles/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )

@router.get("/voices")
async def get_available_voices():
    """
    Get list of available voices for different languages
    """
    voices = {
        "english": [
            "Fritz-PlayAI",
            "Angelo-PlayAI",
            "Thunder-PlayAI"
        ],

        "default": [
            "Fritz-PlayAI"
        ]
    }
    
    return {
        "status": "success",
        "voices": voices,
        "default_voices": {
            "ur": "Fritz-PlayAI",
            "en": "Fritz-PlayAI",
            "default": "Fritz-PlayAI"
        },
        "provider": "Groq PlayAI",
        "model": "playai-tts"
    }