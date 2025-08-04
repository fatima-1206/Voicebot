"""
Main tasks module - imports all task modules for Celery discovery
This file serves as the entry point for all Celery 

To run celery:
celery -A backend.workers.celery_worker.celery_app worker --loglevel=info --concurrency=1 --hostname=voicebot@%h
"""

# Import all task modules so Celery can discover them
from backend.workers.stt_tasks import *
from backend.workers.tts_tasks import *
from backend.workers.query_tasks import *
from backend.workers.pipeline_management import *
from backend.workers.doc_processing import *
from backend.rag.add_document import process_loaded_chunks
# Re-export commonly used functions for backward compatibility
from backend.workers.utilities import (
    manage_gpu_memory,
    get_db_session,
    notify_audio_status,
    notify_text_response_ready,
    notify_error,
    save_query_to_database
)

from backend.workers.stt_tasks import (
    bytes_to_numpy_audio
)

from backend.workers.tts_tasks import (

    generate_tts_file
)

print("✅ All Celery tasks loaded and registered")