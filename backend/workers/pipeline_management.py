"""
Conversation and pipeline-related Celery tasks
"""
from backend.workers.celery_worker import celery_app
from backend.workers.stt_tasks import transcribe_chunk, should_respond_to_transcript
from backend.workers.query_tasks import process_query_with_tts_task
from uuid import uuid4


@celery_app.task(name='transcribe_and_respond', bind=True)
def transcribe_and_respond_task(self, audio_bytes: bytes, user_id: str, chunk_id: str, 
                               tenant_id: str = None, conversation_mode: bool = True,
                               tts_language: str = "en", voice: str = "", 
                               auto_respond: bool = True):
    """
    Real-time conversation task: STT → (optionally RAG → TTS)
    This is called from WebSocket when conversation_mode=True
    """
    print(f"🔄 [CONVERSATION] Processing audio chunk {chunk_id}")
    
    try:
        # Step 1: Do STT first
        stt_result = transcribe_chunk(audio_bytes, user_id, chunk_id, tenant_id)
        
        if stt_result['status'] != 'success':
            return stt_result
            
        transcript = stt_result['transcript']
        
        # Step 2: Check if we should auto-respond
        if auto_respond and conversation_mode and should_respond_to_transcript(transcript):
            print(f"🤖 [CONVERSATION] Auto-responding to: '{transcript}'")
            
            # Immediately trigger RAG + TTS response in background
            response_id = str(uuid4())
            
            # Use the optimized query+TTS task for immediate response
            process_query_with_tts_task.delay(
                query_id=response_id,
                tts_task_id=str(uuid4()),
                query=transcript,
                tenant_id=tenant_id,
                user_id=user_id,
                language=tts_language,
                voice=voice,
                stream_audio=True,
                session_id=None  # Could be passed from WebSocket
            )
            
            # Update STT result to indicate response is being generated
            stt_result['auto_response_triggered'] = True
            stt_result['response_id'] = response_id
        
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
def audio_to_audio_pipeline(pipeline_id: str, audio_bytes: bytes, tenant_id: str, 
                          user_id: str, stt_language: str = "en", tts_language: str = "en",
                          voice: str = "", session_id: str = None):
    """
    Legacy pipeline: STT → RAG → TTS
    Use transcribe_and_respond_task instead for better performance
    """
    print(f"🚀 [LEGACY-PIPELINE] Starting {pipeline_id}")
    
    try:
        # Just use the new optimized task
        chunk_id = str(uuid4())
        result = transcribe_and_respond_task(
            audio_bytes=audio_bytes,
            user_id=user_id,
            chunk_id=chunk_id,
            tenant_id=tenant_id,
            conversation_mode=True,
            tts_language=tts_language,
            voice=voice,
            auto_respond=True
        )
        
        return {
            "status": "pipeline_completed",
            "pipeline_id": pipeline_id,
            "result": result
        }
        
    except Exception as e:
        print(f"❌ [LEGACY-PIPELINE] Error: {e}")
        return {
            "status": "error",
            "pipeline_id": pipeline_id,
            "error": str(e)
        }