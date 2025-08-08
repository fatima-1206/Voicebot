"""
Test client for the GraphQL to RAG Celery task
"""
import json
from celery import Celery
import time

# Initialize Celery client
config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

celery_client = Celery(
    "voicebot_client",
    broker=config.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=config.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)

def test_rick_and_morty_api():
    """Test with Rick and Morty GraphQL API"""
    print("🚀 Testing GraphQL to RAG task with Rick and Morty API...")
    
    # Task parameters
    api_url = "https://rickandmortyapi.com/graphql"
    headers = {}  # No auth needed for this API
    source_name = "Rick and Morty API"
    max_depth = 2  # Limit depth to avoid huge documents
    
    # Submit the task
    task = celery_client.send_task(
        'process_graphql_to_rag',
        args=[],
        kwargs={
            'api_url': api_url,
            'headers': headers,
            'source_name': source_name,
            'max_depth': max_depth,
            'tenant_id': 'default',
            'user_id': 'system',
            'is_public': True
        }
    )
    
    print(f"📋 Task submitted with ID: {task.id}")
    print("⏳ Waiting for task to complete...")
    
    # Wait for result
    try:
        result = task.get(timeout=300)  # 5 minute timeout
        print("🎉 Task completed successfully!")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"❌ Task failed: {e}")
        return None

def test_countries_api():
    """Test with Countries GraphQL API"""
    print("🚀 Testing GraphQL to RAG task with Countries API...")
    
    # Task parameters
    api_url = "https://countries.trevorblades.com/"
    headers = {}  # No auth needed for this API
    source_name = "Countries API"
    max_depth = 2
    
    # Submit the task
    task = celery_client.send_task(
        'process_graphql_to_rag',
        args=[],
        kwargs={
            'api_url': api_url,
            'headers': headers,
            'source_name': source_name,
            'max_depth': max_depth,
            'tenant_id': 'default',
            'user_id': 'system',
            'is_public': True
        }
    )
    
    print(f"📋 Task submitted with ID: {task.id}")
    print("⏳ Waiting for task to complete...")
    
    # Wait for result
    try:
        result = task.get(timeout=300)  # 5 minute timeout
        print("🎉 Task completed successfully!")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"❌ Task failed: {e}")
        return None

def check_task_status(task_id: str):
    """Check the status of a task"""
    task = celery_client.AsyncResult(task_id)
    print(f"Task {task_id} status: {task.status}")
    
    if task.ready():
        if task.successful():
            print("✅ Task completed successfully")
            result = task.result
            print(json.dumps(result, indent=2))
            return result
        else:
            print("❌ Task failed")
            print(f"Error: {task.result}")
            return None
    else:
        print("⏳ Task still running...")
        return None

def main():
    """Main test function"""
    print("=== GraphQL to RAG Celery Task Test ===")
    
    # Test Rick and Morty API
    print("\n--- Testing Rick and Morty API ---")
    test_rick_and_morty_api()
    
    # Test Countries API
    print("\n--- Testing Countries API ---")
    test_countries_api()

if __name__ == "__main__":
    main()