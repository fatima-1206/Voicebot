# # To run celery:
# celery -A backend.workers.celery_worker.celery_app worker --loglevel=info --concurrency=1 --hostname=voicebot@%h
from celery import Celery
from celery.signals import worker_ready
import json
import os

config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

celery_app = Celery(
    "voicebot",
    broker=config.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=config.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    include=['backend.workers.tasks']
)

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

# Initialize everything when worker starts
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    print("🚀 Celery worker ready, initializing database and retrievers...")
    
    # Step 1: Initialize Database
    print("🔧 Initializing database connection and schema...")
    try:
        # Import database modules
        import backend.db.models  # Register models
        from backend.db.setup import db_has_tables, run_sql_file
        from backend.db.connection import SessionLocal, engine
        from sqlalchemy import text  # Import text for raw SQL
        
        # Test database connection
        db = SessionLocal()
        try:
            # Test basic connectivity (fixed SQL)
            result = db.execute(text("SELECT 1")).fetchone()
            print(f"✅ Database connection successful: {result}")
            
            # Check if tables exist
            if db_has_tables():
                print("✅ Database tables already exist")
            else:
                print("⏳ Creating database tables...")
                schema_path = os.path.join(os.path.dirname(__file__), "../db/schema.sql")
                run_sql_file(schema_path)
                print("✅ Database tables created")
                
        except Exception as db_error:
            print(f"❌ Database initialization failed: {db_error}")
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Database setup error: {e}")
    
    # Step 2: Force CPU usage for embeddings in worker process
    print("🔧 Configuring CPU-only mode for ML models...")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU from worker
    
    # Also set torch to CPU mode
    try:
        import torch
        torch.cuda.is_available = lambda: False
        print("✅ PyTorch configured for CPU-only mode")
    except ImportError:
        print("⚠️ PyTorch not available, skipping GPU configuration")
    
    # Step 3: Initialize RAG retrievers
    print("🔧 Initializing RAG retrievers...")
    try:
        from backend.rag.retrieverconstruction import construct_retrievers, get_retrievers
        
        # Try to get existing retrievers first
        faiss_db, hybrid_retriever = get_retrievers()
        
        if faiss_db is None:
            print("🔄 No existing retrievers found, constructing new ones...")
            faiss_db, hybrid_retriever = construct_retrievers()
        else:
            print("✅ Loaded existing retrievers from cache")
        
        if faiss_db is not None:
            print("✅ Retrievers initialized in Celery worker (CPU mode)")
            print(f"✅ FAISS DB available: {faiss_db is not None}")
            print(f"✅ Hybrid retriever available: {hybrid_retriever is not None}")
        else:
            print("❌ Failed to initialize retrievers in worker")
            
    except Exception as retriever_error:
        print(f"❌ RAG initialization failed: {retriever_error}")
        import traceback
        traceback.print_exc()
    
    print("🎉 Celery worker initialization complete!")
    print("=" * 50)

# Import tasks to register them
from backend.workers import tasks