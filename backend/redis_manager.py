
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
                
                # Check if user is still connected before processing
                if manager.is_connected(tenant_id, user_id):
                    # Parse message data
                    data = json.loads(message['data'])
                    
                    # Forward to WebSocket client
                    success = await manager.send_json_to_user(tenant_id, user_id, data)
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