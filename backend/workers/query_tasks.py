"""
Query processing and RAG-related Celery tasks
"""
from backend.workers.celery_worker import celery_app
from backend.workers.utilities import (
    manage_gpu_memory, notify_text_response_ready, notify_error, save_query_to_database
)
from backend.workers.tts_tasks import generate_and_stream_tts, generate_tts_file
from backend.response_generation.generator import generate_response
from better_profanity import profanity

profanity.load_censor_words()
# TODO: Add a fall back to no rag, only simple rag if the user passes a param or if the response generation from rag fails 
@celery_app.task(name='process_query', bind=True)
def process_query_task(self, query_id: str, query: str, tenant_id: str, user_id: str, 
                    max_tokens: int = 500, temperature: float = 0.7, session_id: str = None):
    """Process a user query using RAG pipeline (text only)"""
    print(f"🔄 [QUERY] Processing: {query[:50]}...")
    
    with manage_gpu_memory():
        try:
            # Generate response
            response = generate_response(query)
            if response:
                response = profanity.censor(response)  # Apply profanity filter
            # Save to database
            save_query_to_database(query_id, user_id, session_id, query, response)
            
            # Notify via WebSocket
            notify_text_response_ready(tenant_id, user_id, query_id, query, response)
            
            return {
                "status": "success",
                "query_id": query_id,
                "query": query,
                "response": response,
                "worker": self.request.hostname
            }
            
        except Exception as e:
            print(f"❌ [QUERY] Error: {e}")
            notify_error(tenant_id, user_id, query_id, str(e))
            
            return {
                "status": "error",
                "query_id": query_id,
                "error": str(e),
                "worker": self.request.hostname
            }


@celery_app.task(name='process_query_with_tts', bind=True)
def process_query_with_tts_task(self, query_id: str, tts_task_id: str, query: str, tenant_id: str, 
                            user_id: str, max_tokens: int = 500, temperature: float = 0.7, 
                            session_id: str = None, language: str = "en", voice: str = "",
                            stream_audio: bool = True):
    """
    Optimized task: Query → RAG → TTS → Stream Audio
    This is the main pipeline for conversation responses
    """
    print(f"🔄 [QUERY+TTS] Processing: {query[:50]}...")
    
    with manage_gpu_memory():
        try:
            # Step 1: Generate text response
            response_text = generate_response(query)
            if response_text:
                response_text = profanity.censor(response_text)  # Apply profanity filter
            else:
                print("   ❌ No response generated")
                return {"status": "no_response", "query_id": query_id}
            print(f"✅ [RAG] Generated: {response_text[:50]}...")
            
            # Step 2: Immediately notify WebSocket that text is ready
            notify_text_response_ready(tenant_id, user_id, query_id, query, response_text)
            
            # Step 3: Process TTS with streaming
            if stream_audio:
                audio_result = generate_and_stream_tts(
                    response_text, tenant_id, user_id, tts_task_id, language, voice
                )
            else:
                audio_result = generate_tts_file(response_text, tts_task_id, language, voice)
            
            # Step 4: Save to database
            save_query_to_database(query_id, user_id, session_id, query, response_text)
            
            return {
                "status": "success",
                "query_id": query_id,
                "tts_task_id": tts_task_id,
                "query": query,
                "response": response_text,
                "audio_result": audio_result,
                "worker": self.request.hostname
            }
            
        except Exception as e:
            print(f"❌ [QUERY+TTS] Error: {e}")
            
            # Notify WebSocket of error
            notify_error(tenant_id, user_id, query_id, str(e))
            
            return {
                "status": "error",
                "query_id": query_id,
                "error": str(e),
                "worker": self.request.hostname
            }