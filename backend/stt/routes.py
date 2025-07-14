from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
from backend.auth.jwt_ import get_current_user, UserTokenData
from pydantic import BaseModel
from typing import Optional
from backend.workers.tasks import speech_to_text_task
from uuid import uuid4
import os

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class STTSubmissionResponse(BaseModel):
    stt_id: str
    status: str
    message: str
    filename: str

class STTResponse(BaseModel):
    stt_id: str
    text: str
    language: str
    confidence: Optional[float] = None
    duration: Optional[float] = None

@router.post("/convert", response_model=STTSubmissionResponse)
async def convert_speech_to_text(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(default="auto"),  # auto-detect or specify language
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Convert speech to text asynchronously
    
    - **audio_file**: Audio file (wav, mp3, m4a, ogg, flac)
    - **language**: Language code (e.g., 'en', 'es', 'fr') or 'auto' for auto-detection
    """
    
    # Validate file type
    allowed_extensions = [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]
    file_extension = os.path.splitext(audio_file.filename)[-1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (limit to 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    audio_content = await audio_file.read()
    
    if len(audio_content) > max_size:
        raise HTTPException(
            status_code=400,
            detail="Audio file too large. Maximum size is 25MB."
        )
    
    try:
        stt_id = str(uuid4())
        
        # Submit to Celery for processing
        speech_to_text_task.delay(
            stt_id=stt_id,
            audio_bytes=audio_content.decode("latin1"),  # Encode as string for Celery
            filename=audio_file.filename,
            language=language,
            tenant_id=str(current_user.tenant_id),
            user_id=str(current_user.user_id)
        )
        
        return STTSubmissionResponse(
            stt_id=stt_id,
            status="submitted",
            message="Audio file submitted for speech-to-text conversion. Result will be sent via WebSocket.",
            filename=audio_file.filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT submission failed: {str(e)}")

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages for STT"""
    # Common language codes supported by most STT systems
    languages = {
        "auto": "Auto-detect",
        "en": "English",
        "ur": "Urdu",
        "es": "Spanish", 
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese (Mandarin)",
        "ar": "Arabic",
        "hi": "Hindi",
        "tr": "Turkish",
        "pl": "Polish",
        "nl": "Dutch",
        "sv": "Swedish",
        "da": "Danish",
        "no": "Norwegian",
        "fi": "Finnish"
    }
    return {"supported_languages": languages}

@router.get("/health")
async def health_check():
    """Health check endpoint for STT service"""
    return {"status": "healthy", "service": "speech_to_text"}