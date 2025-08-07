import redis
import logging
from typing import Dict, Any, Optional
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

    def _create_session_key(self, tenant_id: str, user_id: str, device_id: str, session_id: str) -> str:
        """Create standardized session key for task tracking"""
        return f"{tenant_id}|{user_id}|{device_id}|{session_id}"

    def register_task(self, session_key: str, task_id: str):
        """Track task_id for a given session (tenant|user|device|session)"""
        try:
            self.client.sadd(f"tasks:{session_key}", task_id)
            self.client.expire(f"tasks:{session_key}", 3600)  # Auto-expire in 1 hour
            logger.debug(f"📌 Task {task_id} registered for session {session_key}")
        except Exception as e:
            logger.error(f"❌ Failed to register task {task_id} for session {session_key}: {e}")

    def register_task_for_session(self, tenant_id: str, user_id: str, device_id: str, session_id: str, task_id: str):
        """Register task with full session hierarchy"""
        session_key = self._create_session_key(tenant_id, user_id, device_id, session_id)
        self.register_task(session_key, task_id)
        
        # Also track at device level for broader cleanup if needed
        device_key = f"{tenant_id}|{user_id}|{device_id}"
        try:
            self.client.sadd(f"device_tasks:{device_key}", task_id)
            self.client.expire(f"device_tasks:{device_key}", 3600)
        except Exception as e:
            logger.error(f"❌ Failed to register device task {task_id}: {e}")

    def get_session_tasks(self, session_key: str):
        """Return all task_ids for a session"""
        try:
            return self.client.smembers(f"tasks:{session_key}")
        except Exception as e:
            logger.error(f"❌ Failed to get tasks for session {session_key}: {e}")
            return set()

    def get_device_tasks(self, tenant_id: str, user_id: str, device_id: str):
        """Return all task_ids for a device"""
        device_key = f"{tenant_id}|{user_id}|{device_id}"
        try:
            return self.client.smembers(f"device_tasks:{device_key}")
        except Exception as e:
            logger.error(f"❌ Failed to get device tasks for {device_key}: {e}")
            return set()

    def clear_session_tasks(self, session_key: str):
        """Clear task tracking for a session"""
        try:
            self.client.delete(f"tasks:{session_key}")
            logger.debug(f"🧹 Cleared tasks for session {session_key}")
        except Exception as e:
            logger.error(f"❌ Failed to clear tasks for session {session_key}: {e}")

    def clear_device_tasks(self, tenant_id: str, user_id: str, device_id: str):
        """Clear all task tracking for a device"""
        device_key = f"{tenant_id}|{user_id}|{device_id}"
        try:
            self.client.delete(f"device_tasks:{device_key}")
            logger.debug(f"🧹 Cleared device tasks for {device_key}")
        except Exception as e:
            logger.error(f"❌ Failed to clear device tasks for {device_key}: {e}")

    def revoke_all_tasks(self, session_key: str):
        """Revoke all tracked tasks for a session"""
        try:
            task_ids = self.get_session_tasks(session_key)
            control = celery_app.control
            for task_id in task_ids:
                control.revoke(task_id, terminate=True)
                logger.info(f"🚫 Revoked task {task_id} for session {session_key}")
            self.clear_session_tasks(session_key)
        except Exception as e:
            logger.error(f"❌ Failed to revoke tasks for session {session_key}: {e}")

    def revoke_session_tasks(self, tenant_id: str, user_id: str, device_id: str, session_id: str):
        """Revoke all tasks for a specific session"""
        session_key = self._create_session_key(tenant_id, user_id, device_id, session_id)
        self.revoke_all_tasks(session_key)

    def revoke_device_tasks(self, tenant_id: str, user_id: str, device_id: str):
        """Revoke all tasks for a specific device (all sessions)"""
        try:
            # Get all tasks for this device
            task_ids = self.get_device_tasks(tenant_id, user_id, device_id)
            control = celery_app.control
            
            for task_id in task_ids:
                control.revoke(task_id, terminate=True)
                logger.info(f"🚫 Revoked device task {task_id} for {tenant_id}|{user_id}|{device_id}")
            
            # Clear device task tracking
            self.clear_device_tasks(tenant_id, user_id, device_id)
            
            # Also clear any session-specific tasks for this device
            device_sessions = manager.get_device_sessions(tenant_id, user_id, device_id)
            for session_id in device_sessions:
                session_key = self._create_session_key(tenant_id, user_id, device_id, session_id)
                self.clear_session_tasks(session_key)
                
        except Exception as e:
            logger.error(f"❌ Error revoking device tasks: {e}")

    def revoke_user_tasks(self, tenant_id: str, user_id: str):
        """Revoke all tasks for a user (all devices and sessions)"""
        try:
            # Get all devices for this user
            user_devices = manager.get_user_devices(tenant_id, user_id)
            
            for device_id, sessions in user_devices.items():
                # Revoke device-level tasks
                self.revoke_device_tasks(tenant_id, user_id, device_id)
                
                # Revoke session-level tasks
                for session_id in sessions:
                    self.revoke_session_tasks(tenant_id, user_id, device_id, session_id)
                    
        except Exception as e:
            logger.error(f"❌ Error revoking user tasks: {e}")

    def publish_to_session(self, tenant_id: str, user_id: str, device_id: str, session_id: str, message: dict):
        """Send message to a specific session"""
        message.update({
            'tenant_id': tenant_id,
            'user_id': user_id,
            'device_id': device_id,
            'session_id': session_id
        })
        
        channel = f"ws:{tenant_id}:{user_id}:{device_id}:{session_id}"
        self.publish(channel, json.dumps(message))

    def publish_to_device_sessions(self, tenant_id: str, user_id: str, device_id: str, message: dict):
        """Send message to all sessions on a device"""
        message.update({
            'tenant_id': tenant_id,
            'user_id': user_id,
            'device_id': device_id
        })
        
        # Get all sessions for this device
        device_sessions = manager.get_device_sessions(tenant_id, user_id, device_id)
        for session_id in device_sessions:
            channel = f"ws:{tenant_id}:{user_id}:{device_id}:{session_id}"
            session_message = message.copy()
            session_message['session_id'] = session_id
            self.publish(channel, json.dumps(session_message))

    def publish_to_user_sessions(self, tenant_id: str, user_id: str, message: dict):
        """Send message to all sessions for a user (all devices)"""
        message.update({
            'tenant_id': tenant_id,
            'user_id': user_id
        })
        
        # Get all devices and sessions for this user
        user_devices = manager.get_user_devices(tenant_id, user_id)
        for device_id, sessions in user_devices.items():
            for session_id in sessions:
                channel = f"ws:{tenant_id}:{user_id}:{device_id}:{session_id}"
                session_message = message.copy()
                session_message.update({
                    'device_id': device_id,
                    'session_id': session_id
                })
                self.publish(channel, json.dumps(session_message))

    def publish_to_tenant_sessions(self, tenant_id: str, message: dict):
        """Send message to all sessions in a tenant"""
        message['tenant_id'] = tenant_id
        
        # Use WebSocket manager to get all tenant connections
        connections = manager.get_connection_hierarchy_stats()
        
        # For each tenant connection, publish to all sessions
        if tenant_id in manager.tenant_users:
            for user_id in manager.tenant_users[tenant_id]:
                self.publish_to_user_sessions(tenant_id, user_id, message.copy())

    # Legacy methods for backward compatibility
    def publish_to_user_device(self, tenant_id: str, user_id: str, message: dict, device_id: str = None, session_id: str = None):
        """Legacy method - updated to support session routing"""
        if session_id and device_id:
            # Send to specific session
            self.publish_to_session(tenant_id, user_id, device_id, session_id, message)
        elif device_id:
            # Send to all sessions on device
            self.publish_to_device_sessions(tenant_id, user_id, device_id, message)
        else:
            # Send to all user sessions
            self.publish_to_user_sessions(tenant_id, user_id, message)

    def get_task_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked tasks"""
        try:
            # Get all task keys
            session_task_keys = self.client.keys("tasks:*")
            device_task_keys = self.client.keys("device_tasks:*")
            
            session_task_count = 0
            device_task_count = 0
            
            for key in session_task_keys:
                session_task_count += self.client.scard(key)
            
            for key in device_task_keys:
                device_task_count += self.client.scard(key)
            
            return {
                "session_task_keys": len(session_task_keys),
                "device_task_keys": len(device_task_keys),
                "total_session_tasks": session_task_count,
                "total_device_tasks": device_task_count,
                "redis_healthy": self.is_healthy
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting task stats: {e}")
            return {"error": str(e)}

# Improved Redis listener with session-aware message processing
class RedisListener:
    """Improved Redis listener with session-aware message processing"""
    
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
        logger.info("🔄 Starting Redis WebSocket listener with session support")
        
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
            # Subscribe to all WebSocket channels (now session-aware)
            self.pubsub.psubscribe("ws:*")
            logger.info("🔄 Redis listener subscribed to ws:* channels (session-aware)")
            
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
        """Process a single Redis message with session awareness"""
        try:
            # Parse channel to extract tenant_id, user_id, device_id, and session_id
            channel = message['channel']
            parts = channel.split(':')
            
            if len(parts) >= 5 and parts[0] == 'ws':  # ws:tenant:user:device:session
                tenant_id = parts[1]
                user_id = parts[2]
                device_id = parts[3]
                session_id = parts[4]
                
                # Check if specific session is connected
                if manager.is_session_connected(tenant_id, user_id, device_id, session_id):
                    # Parse message data
                    data = json.loads(message['data'])
                    
                    # Forward to specific session
                    success = await manager.send_to_session(tenant_id, user_id, device_id, session_id, data)
                    
                    if success:
                        self.message_count += 1
                        self.last_message_time = time.time()
                        logger.debug(f"✅ Message delivered to session {tenant_id}|{user_id}|{device_id}|{session_id}")
                    else:
                        logger.warning(f"⚠️ Failed to deliver message to session {tenant_id}|{user_id}|{device_id}|{session_id}")
                else:
                    logger.debug(f"⚠️ Session not connected: {tenant_id}|{user_id}|{device_id}|{session_id}")
                    
            elif len(parts) >= 4:  # Backward compatibility for device-level messages
                tenant_id = parts[1]
                user_id = parts[2]
                device_id = parts[3] if len(parts) > 3 else None
                
                # Parse message data
                data = json.loads(message['data'])
                
                if device_id:
                    # Send to all sessions on device
                    results = await manager.send_to_device_sessions(tenant_id, user_id, device_id, data)
                else:
                    # Send to all user sessions
                    results = await manager.send_to_all_user_sessions(tenant_id, user_id, data)
                
                success = any(results.values()) if results else False
                if success:
                    self.message_count += 1
                    self.last_message_time = time.time()
            else:
                logger.warning(f"⚠️ Invalid channel format: {channel}")
                        
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