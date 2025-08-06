import redis
import logging
from typing import Dict, Any
from datetime import datetime
import logging
import asyncio
import json
from collections import defaultdict
import time
from backend.ws_manager import manager
from backend.workers.celery_worker import celery_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration with connection pooling and retry logic
class RedisManager:
    """Improved Redis manager with connection pooling and health monitoring"""
    
    def __init__(self, host='localhost', port=6379, db=2, max_connections=10):
        self.host = host
        self.port = port
        self.db = db
        self.max_connections = max_connections
        self.client = None
        self.connection_pool = None
        self.is_healthy = False
        self.circuit_breaker_open = False
        self.failure_count = 0
        self.last_failure_time = 0
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection with error handling"""
        try:
            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                max_connections=self.max_connections,
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self.client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.client.ping()
            self.is_healthy = True
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.is_healthy = False
    
    def publish(self, channel: str, message: str) -> bool:
        """Publish message with error handling"""
        if not self.is_healthy:
            return False
        
        try:
            self.client.publish(channel, message)
            return True
        except Exception as e:
            logger.error(f"❌ Redis publish failed: {e}")
            self.is_healthy = False
            return False
    
    def get_pubsub(self):
        """Get pubsub client with error handling"""
        if not self.is_healthy:
            return None
        
        try:
            return self.client.pubsub()
        except Exception as e:
            logger.error(f"❌ Redis pubsub failed: {e}")
            self.is_healthy = False
            return None
    
    async def health_check(self) -> bool:
        """Check Redis health and reconnect if needed"""
        try:
            if self.client:
                self.client.ping()
                self.is_healthy = True
                return True
        except Exception:
            self.is_healthy = False
            logger.warning("⚠️ Redis health check failed, attempting reconnection...")
            self._init_redis()
            return self.is_healthy
    
    async def publish_with_fallback(self, channel: str, message: str):
        if self.circuit_breaker_open:
            # Use local fallback or queue messages
            return await self._local_fallback(channel, message)
        
        try:
            return await self.publish(channel, message)
        except Exception:
            self._handle_failure()
            return await self._local_fallback(channel, message)


    def register_task(self, session_key: str, task_id: str):
        """Track task_id for a given user (tenant_user_key)"""
        try:
            self.client.sadd(f"tasks:{session_key}", task_id)
            self.client.expire(f"tasks:{session_key}", 3600)  # Optional: Auto-expire in 1 hour
            logger.debug(f"📌 Task {task_id} registered for {session_key}")
        except Exception as e:
            logger.error(f"❌ Failed to register task {task_id} for {session_key}: {e}")

    def get_user_tasks(self, session_key: str):
        """Return all task_ids for a user"""
        try:
            return self.client.smembers(f"tasks:{session_key}")
        except Exception as e:
            logger.error(f"❌ Failed to get tasks for {session_key}: {e}")
            return set()

    def clear_user_tasks(self, session_key: str):
        """Clear task tracking for a user"""
        try:
            self.client.delete(f"tasks:{session_key}")
        except Exception as e:
            logger.error(f"❌ Failed to clear tasks for {session_key}: {e}")

    def revoke_all_tasks(self, session_key: str):
        """Revoke all tracked tasks for a user"""
        try:
            task_ids = self.get_user_tasks(session_key)
            control = celery_app.control
            for task_id in task_ids:
                control.revoke(task_id, terminate=True)
                logger.info(f"🚫 Revoked task {task_id} for {session_key}")
            self.clear_user_tasks(session_key)
        except Exception as e:
            logger.error(f"❌ Failed to revoke tasks for {session_key}: {e}")

    def publish_to_user_device(self, tenant_id: str, user_id: str, message: dict, device_id: str = None):
        """Send message to specific device or all user devices"""
        message['tenant_id'] = tenant_id
        message['user_id'] = user_id
        if device_id:
            message['device_id'] = device_id
            channel = f"voicebot_responses_{tenant_id}_{user_id}_{device_id}"
        else:
            channel = f"voicebot_responses_{tenant_id}_{user_id}"
        
        self.publish(channel, json.dumps(message))
    
    def revoke_device_tasks(self, session_id: str, device_id: str):
        """Revoke tasks for a specific device"""
        try:
            # Get all active tasks for this session/device
            active_tasks = celery_app.control.inspect().active()
            
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    # Check if task belongs to this session and device
                    if (task.get('kwargs', {}).get('session_id') == session_id and 
                        task.get('kwargs', {}).get('device_id') == device_id):
                        celery_app.control.revoke(task['id'], terminate=True)
                        logger.info(f"🚫 Revoked task {task['id']} for device {device_id}")
                        
        except Exception as e:
            logger.error(f"❌ Error revoking device tasks: {e}")





# Improved Redis listener with health monitoring and error recovery

class RedisListener:
    """Improved Redis listener with health monitoring and reconnection"""
    
    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager
        self.is_running = False
        self.pubsub = None
        self.message_count = 0
        self.error_count = 0
        self.last_message_time = time.time()
        
    async def start(self):
        """Start the Redis listener with automatic reconnection"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🔄 Starting Redis WebSocket listener")
        
        while self.is_running:
            try:
                await self._listen_loop()
            except Exception as e:
                self.error_count += 1
                logger.error(f"❌ Redis listener error: {e}")
                await asyncio.sleep(5)  # Wait before retry
                
                # Check Redis health and reconnect if needed
                await self.redis_manager.health_check()
    
    async def _listen_loop(self):
        """Main listening loop"""
        self.pubsub = self.redis_manager.get_pubsub()
        if not self.pubsub:
            logger.error("❌ Could not get Redis pubsub client")
            await asyncio.sleep(10)
            return
        
        try:
            # Subscribe to all WebSocket channels
            self.pubsub.psubscribe("ws:*")
            logger.info("🔄 Redis listener subscribed to ws:* channels")
            
            while self.is_running and self.redis_manager.is_healthy:
                try:
                    message = self.pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'pmessage':
                        await self._process_message(message)
                    
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    
                except Exception as e:
                    logger.error(f"❌ Error processing Redis message: {e}")
                    self.error_count += 1
                    
        finally:
            if self.pubsub:
                try:
                    self.pubsub.close()
                except Exception:
                    pass
                self.pubsub = None
    
    async def _process_message(self, message):
        """Process a single Redis message"""
        try:
            # Parse channel to extract tenant_id and user_id
            channel = message['channel']
            parts = channel.split(':')
            if len(parts) >= 3:
                tenant_id = parts[1]
                user_id = parts[2]
                device_id = parts[3] if len(parts) > 3 else None
                
                # Check if user has any connected devices before processing
                if manager.get_user_device_count(user_id) > 0:
                    # Parse message data
                    data = json.loads(message['data'])
                    
                    # Forward to WebSocket client - send to all user devices
                    if device_id:
                        success = await manager.send_json_to_user_device(tenant_id, user_id, data, device_id)
                    else:
                        results = await manager.send_json_to_all_user_devices(tenant_id, user_id, data)
                        success = any(results.values()) if results else False
                    
                    if success:
                        self.message_count += 1
                        self.last_message_time = time.time()
                        
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in Redis message: {e}")
            self.error_count += 1
        except Exception as e:
            logger.error(f"❌ Error processing Redis message: {e}")
            self.error_count += 1
    
    def stop(self):
        """Stop the Redis listener"""
        self.is_running = False
        logger.info("🛑 Stopping Redis listener")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get listener statistics"""
        return {
            "is_running": self.is_running,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_message_time": datetime.fromtimestamp(self.last_message_time).isoformat(),
            "redis_healthy": self.redis_manager.is_healthy
        }

# Initialize Redis manager
redis_manager = RedisManager()
# Initialize Redis listener
redis_listener = RedisListener(redis_manager)