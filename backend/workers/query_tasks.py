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

@celery_app.task(name='process_query', bind=True)
def process_query_task(self, query_id: str, query: str, tenant_id: str, user_id: str, 
                    max_tokens: int = 500, temperature: float = 0.7, session_id: str = None,
                    use_rag: bool = True, fallback_on_error: bool = True):
    """Process a user query with optional RAG fallback (text only)"""
    print(f"🔄 [QUERY] Processing: {query[:50]}... (RAG: {use_rag}, Fallback: {fallback_on_error})")
    
    with manage_gpu_memory():
        try:
            # Generate response with RAG options
            response = generate_response(query, use_rag=use_rag, fallback_on_error=fallback_on_error,user_id=user_id, tenant_id=tenant_id)
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
                "rag_used": use_rag,
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
                            stream_audio: bool = True, use_rag: bool = True, chunk_id=None,fallback_on_error: bool = True):
    """
    Optimized task: Query → RAG → TTS → Stream Audio with fallback options
    This is the main pipeline for conversation responses
    """
    print(f"🔄 [QUERY+TTS] Processing: {query[:50]}... (RAG: {use_rag}, Fallback: {fallback_on_error})")
    
    with manage_gpu_memory():
        try:
            # Step 1: Generate text response with RAG options
            response_text = generate_response(query, use_rag=use_rag, fallback_on_error=fallback_on_error)
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
                print("\n BOUT TO CALL GENERATE AND STREAM TTS")
                audio_result = generate_and_stream_tts(
                    response_text, chunk_id=chunk_id, tenant_id=tenant_id, user_id=user_id, tts_task_id=tts_task_id, language=language, voice=voice
                )
            else:
                print("\n BOUT TO CALL GENERATE TTS FILE")
                audio_result = generate_tts_file(
                    response_text, chunk_id=chunk_id, tenant_id=tenant_id, user_id=user_id, tts_task_id=tts_task_id, language=language, voice=voice
                )

            # Step 4: Save query and response to database
            save_query_to_database(query_id, user_id, session_id, query, response_text)

            return {
                "status": "success",
                "query_id": query_id,
                "chunk_id": chunk_id,
                "tts_task_id": tts_task_id,
                "query": query,
                "response": response_text,
                "audio_status": audio_result.get("status", "unknown"),
                "rag_used": use_rag,
                "worker": self.request.hostname
            }

        except Exception as e:
            print(f"❌ [QUERY+TTS] Error: {e}")
            notify_error(tenant_id, user_id, query_id, str(e))
            
            return {
                "status": "error",
                "query_id": query_id,
                "tts_task_id": tts_task_id,
                "error": str(e),
                "worker": self.request.hostname
            }