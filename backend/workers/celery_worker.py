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
)

celery_app.conf.task_routes = {
    "backend.worker.tasks.*": {"queue": "default"},
}
