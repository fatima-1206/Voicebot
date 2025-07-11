# # To run celery:
# celery -A backend.workers.celery_worker.celery_app worker --loglevel=info --concurrency=1 --hostname=voicebot@%h
from celery import Celery
import json
import os

config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

celery_app = Celery(
    "voicebot",
    broker=config.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=config.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    include=['backend.workers.tasks']  # This tells Celery to import tasks
)

# Fix the task route
celery_app.conf.task_routes = {
    "backend.workers.tasks.*": {"queue": "default"},
}

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# IMPORTANT: Import the tasks module here to register them
from backend.workers import tasks  # Add this line