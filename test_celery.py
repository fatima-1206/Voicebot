from backend.workers.tasks import transcribe_chunk
from backend.workers.celery_worker import celery_app
import time

def test_celery_connection():
    """Test if Celery is working at all"""
    print("🧪 Testing Celery connection...")
    
    try:
        # Test transcribe_chunk task with dummy data
        print("🧪 Testing transcribe_chunk task...")
        dummy_audio = b'\x00' * 1024  # 1KB of silence
        result = transcribe_chunk.delay(
            audio_bytes=dummy_audio,
            user_id="test_user",
            chunk_id="test_chunk_123",
            tenant_id="test_tenant"
        )
        print(f"✅ Transcribe task submitted: {result.id}")
        
        # Wait longer and check for result
        print("⏳ Waiting for transcribe task result...")
        try:
            output = result.get(timeout=60)  # Wait 60 seconds
            print(f"✅ Transcribe task result: {output}")
        except Exception as e:
            print(f"❌ Transcribe task failed: {e}")
            print(f"   Task state: {result.state}")
            print(f"   Task info: {result.info}")
            
    except Exception as e:
        print(f"❌ Celery connection test failed: {e}")

if __name__ == "__main__":
    test_celery_connection()