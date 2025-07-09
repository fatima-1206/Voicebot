# backend/worker/tasks.py

from backend.workers.celery_worker import celery_app
from stt import transcribe_audio
from tts import synthesize_speech
from reranker import rerank
from rag_module import get_rag_response  # hypothetical

@celery_app.task
def stt_task(audio_path):
    return transcribe_audio(audio_path)

@celery_app.task
def rag_task(question):
    return get_rag_response(question)

@celery_app.task
def rerank_task(docs, query):
    return rerank(docs, query)

@celery_app.task
def tts_task(text, voice="default"):
    return synthesize_speech(text, voice=voice)

@celery_app.task
def process_uploaded_document(file_bytes, filename, doc_id, tenant_id):
    """Process uploaded document - extract text, create chunks, etc."""
    try:
        # Add your document processing logic here
        # For now, just simulate processing
        print(f"Processing document {filename} for tenant {tenant_id}")
        
        # You would typically:
        # 1. Extract text from the file
        # 2. Split into chunks
        # 3. Generate embeddings
        # 4. Store in vector database
        # 5. Update document status in DB
        
        return {"status": "processed", "doc_id": doc_id}
    except Exception as e:
        print(f"Error processing document: {e}")
        return {"status": "error", "error": str(e)}

# to run ---------------------------
# celery -A backend.workers.celery_worker.celery_app worker --loglevel=info --concurrency=1 --hostname=voicebot@%h


# to call tasks---------------------------------------
# cli_app/main.py

# from backend.workers.tasks import stt_task, rag_task, rerank_task, tts_task

# # Submit tasks
# audio_result = stt_task.delay("audio.wav")
# print("STT submitted")

# # Wait for result
# text = audio_result.get(timeout=60)
# print("Transcribed:", text)

# # Now do RAG
# rag_result = rag_task.delay(text)
# answer = rag_result.get(timeout=60)

# # Rerank
# reranked = rerank_task.delay([answer], text).get(timeout=60)

# # Convert to speech
# tts_task.delay(reranked[0])
