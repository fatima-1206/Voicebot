"""
Conversation and pipeline-related Celery tasks
"""
from backend.workers.celery_worker import celery_app
from backend.workers.stt_tasks import transcribe_chunk
from backend.workers.query_tasks import process_query_with_tts_task
from uuid import uuid4


@celery_app.task(name='transcribe_and_respond', bind=True)
def transcribe_and_respond_task(self, audio_bytes: bytes, user_id: str, chunk_id: str, 
                               tenant_id: str = None, conversation_mode: bool = True,
                               tts_language: str = "en", voice: str = "", 
                               auto_respond: bool = True, use_rag: bool = True,
                               fallback_on_error: bool = True):
    """
    Real-time conversation task: STT → (optionally RAG → TTS) with fallback options
    This is called from WebSocket when conversation_mode=True
    """
    print(f"🔄 [CONVERSATION] Processing audio chunk {chunk_id} (RAG: {use_rag}, Fallback: {fallback_on_error})")
    
    try:
        # Step 1: Do STT first
        stt_result = transcribe_chunk(audio_bytes, user_id, chunk_id, tenant_id)
        
        if stt_result['status'] != 'success':
            return stt_result
            
        transcript = stt_result['transcript']
        
        # Step 2: Check if we should auto-respond
        if auto_respond and conversation_mode:
            print(f"🤖 [CONVERSATION] Auto-responding to: '{transcript}' (RAG: {use_rag})")
            
            # Immediately trigger RAG + TTS response in background
            response_id = str(uuid4())
            
            # Use the optimized query+TTS task for immediate response with RAG options
            process_query_with_tts_task.delay(
                query_id=response_id,
                tts_task_id=str(uuid4()),
                query=transcript,
                tenant_id=tenant_id,
                user_id=user_id,
                language=tts_language,
                voice=voice,
                stream_audio=True,
                session_id=None,  # Could be passed from WebSocket
                use_rag=use_rag,
                fallback_on_error=fallback_on_error
            )
            
            # Update STT result to indicate response is being generated
            stt_result['auto_response_triggered'] = True
            stt_result['response_id'] = response_id
            stt_result['rag_enabled'] = use_rag
            stt_result['fallback_enabled'] = fallback_on_error
        
        return stt_result
        
    except Exception as e:
        print(f"❌ [CONVERSATION] Error: {e}")
        return {
            "status": "error",
            "chunk_id": chunk_id,
            "error": str(e),
            "worker": self.request.hostname
        }


@celery_app.task(name='audio_to_audio_pipeline')
def audio_to_audio_pipeline(task_id: str, audio_data: str, filename: str, tenant_id: str, 
                          user_id: str, language: str = "en", voice: str = "",
                          stream_audio: bool = True, use_rag: bool = True,
                          fallback_on_error: bool = True):
    """
    Enhanced audio-to-audio pipeline: STT → RAG/LLM → TTS with fallback options
    """
    print(f"🚀 [AUDIO-PIPELINE] Starting {task_id} (RAG: {use_rag}, Fallback: {fallback_on_error})")
    
    try:
        # Convert string back to bytes for processing
        audio_bytes = audio_data.encode('latin1')
        
        # Use the enhanced conversation task
        chunk_id = str(uuid4())
        result = transcribe_and_respond_task(
            audio_bytes=audio_bytes,
            user_id=user_id,
            chunk_id=chunk_id,
            tenant_id=tenant_id,
            conversation_mode=True,
            tts_language=language,
            voice=voice,
            auto_respond=True,
            use_rag=use_rag,
            fallback_on_error=fallback_on_error
        )
        
        return {
            "status": "pipeline_completed",
            "task_id": task_id,
            "filename": filename,
            "rag_used": use_rag,
            "fallback_enabled": fallback_on_error,
            "result": result
        }
        
    except Exception as e:
        print(f"❌ [AUDIO-PIPELINE] Error: {e}")
        return {
            "status": "error",
            "task_id": task_id,
            "error": str(e)
        }