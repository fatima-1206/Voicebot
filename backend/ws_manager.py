import asyncio
import time
import json
import logging
from datetime import datetime
from typing import Dict, Set, Any, Optional, List
from collections import defaultdict
from fastapi import WebSocket, WebSocketDisconnect
import threading  # Add this import
from fastapi.websockets import WebSocketState
from uuid import uuid4

logger = logging.getLogger(__name__)

class ConnectionMetrics:
    """Track connection metrics and health"""
    def __init__(self):
        self.total_connections = 0
        self.failed_sends = 0
        self.successful_sends = 0
        self.connections_created = 0
        self.connections_closed = 0
        self.last_cleanup = time.time()
        self.message_rates = defaultdict(list)  # tenant_id:user_id -> list of timestamps
        
    def record_connection(self):
        self.connections_created += 1
        self.total_connections += 1
        
    def record_disconnection(self):
        self.connections_closed += 1
        self.total_connections = max(0, self.total_connections - 1)
        
    def record_send_success(self, key: str):
        self.successful_sends += 1
        self.message_rates[key].append(time.time())
        
    def record_send_failure(self, key: str):
        self.failed_sends += 1
        
    def get_message_rate(self, key: str, window_seconds: int = 60) -> int:
        """Get message rate for a connection in the last window_seconds"""
        now = time.time()
        cutoff = now - window_seconds
        self.message_rates[key] = [t for t in self.message_rates[key] if t > cutoff]
        return len(self.message_rates[key])
        
    def cleanup_old_rates(self):
        """Clean up old message rate data"""
        now = time.time()
        cutoff = now - 300  # Keep 5 minutes of data
        for key in list(self.message_rates.keys()):
            self.message_rates[key] = [t for t in self.message_rates[key] if t > cutoff]
            if not self.message_rates[key]:
                del self.message_rates[key]

class RateLimiter:
    """Simple rate limiter for WebSocket messages"""
    def __init__(self, max_messages: int = 100, window_seconds: int = 60):
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self.message_times: Dict[str, list] = defaultdict(list)
        
    def is_allowed(self, key: str) -> bool:
        """Check if a message is allowed for this key"""
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Clean up old messages
        self.message_times[key] = [t for t in self.message_times[key] if t > cutoff]
        
        # Check if under limit
        if len(self.message_times[key]) >= self.max_messages:
            return False
            
        # Record this message
        self.message_times[key].append(now)
        return True
        
    def cleanup(self):
        """Clean up old rate limit data"""
        now = time.time()
        cutoff = now - self.window_seconds * 2  # Keep some buffer
        for key in list(self.message_times.keys()):
            self.message_times[key] = [t for t in self.message_times[key] if t > cutoff]
            if not self.message_times[key]:
                del self.message_times[key]

class ConnectionManager:
    """Enhanced WebSocket connection manager with multi-device support"""
    
    def __init__(self, max_connections_per_user: int = 10, max_devices_per_user: int = 5, message_rate_limit: int = 100):
        # Change to support multiple devices per user
        self.active_connections: Dict[str, WebSocket] = {}  # maps tenant_id:user_id:device_id → WebSocket
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}  # Store connection metadata
        self.user_connections: Dict[str, Dict[str, str]] = defaultdict(dict)  # user_id → {device_id: connection_key}
        self.max_connections_per_user = max_connections_per_user
        self.max_devices_per_user = max_devices_per_user
        
        # Monitoring and metrics
        self.metrics = ConnectionMetrics()
        self.rate_limiter = RateLimiter(max_messages=message_rate_limit)
        
        # Health monitoring
        self._health_check_running = False
        self._cleanup_running = False
        
        # Use threading.Lock for synchronous methods and asyncio.Lock for async methods
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        
    async def connect(self, tenant_id: str, user_id: str, websocket: WebSocket, device_id: str = None) -> bool:
        """Connect a WebSocket with multi-device support"""
        if not device_id:
            device_id = str(uuid4())
            
        key = f"{tenant_id}:{user_id}:{device_id}"

        async with self._async_lock:
            # Check device limits per user
            if len(self.user_connections[user_id]) >= self.max_devices_per_user:
                logger.warning(f"❌ Device limit exceeded for user {user_id}")
                await websocket.close(code=1008, reason="Device limit exceeded")
                return False

            try:
                # Store connection and metadata
                self.active_connections[key] = websocket
                self.connection_metadata[key] = {
                    "connected_at": datetime.now(),
                    "last_activity": time.time(),
                    "message_count": 0,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "device_id": device_id
                }
                self.user_connections[user_id][device_id] = key

                # Update metrics
                self.metrics.record_connection()

                logger.info(f"✅ WebSocket connected: {key} (Total: {len(self.active_connections)})")
                return True

            except Exception as e:
                logger.error(f"❌ Error accepting WebSocket connection {key}: {e}")
                return False

    def disconnect(self, tenant_id: str, user_id: str, device_id: str = None) -> bool:
        """Disconnect with proper cleanup for specific device"""
        if device_id:
            key = f"{tenant_id}:{user_id}:{device_id}"
            return self._disconnect_single_device(key, user_id, device_id)
        else:
            # Disconnect all devices for user (backward compatibility)
            return self._disconnect_all_user_devices(tenant_id, user_id)
    
    def _disconnect_single_device(self, key: str, user_id: str, device_id: str) -> bool:
        """Disconnect a single device"""
        with self._sync_lock:
            if key in self.active_connections:
                try:
                    # Clean up all data structures
                    del self.active_connections[key]
                    if key in self.connection_metadata:
                        del self.connection_metadata[key]
                    
                    # Remove from user connections
                    if user_id in self.user_connections and device_id in self.user_connections[user_id]:
                        del self.user_connections[user_id][device_id]
                        if not self.user_connections[user_id]:
                            del self.user_connections[user_id]
                    
                    # Update metrics
                    self.metrics.record_disconnection()
                    
                    logger.info(f"🔌 WebSocket disconnected: {key} (Total: {len(self.active_connections)})")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Error during disconnect for {key}: {e}")
                    return False
            
            return False
    
    def _disconnect_all_user_devices(self, tenant_id: str, user_id: str) -> bool:
        """Disconnect all devices for a user"""
        success = True
        if user_id in self.user_connections:
            device_keys = list(self.user_connections[user_id].keys())
            for device_id in device_keys:
                key = f"{tenant_id}:{user_id}:{device_id}"
                if not self._disconnect_single_device(key, user_id, device_id):
                    success = False
        return success

    async def send_to_user_device(self, tenant_id: str, user_id: str, message: str, device_id: str = None) -> bool:
        """Send message to specific device or all user devices"""
        if device_id:
            key = f"{tenant_id}:{user_id}:{device_id}"
            return await self._send_to_connection(key, message)
        else:
            # Send to all user devices
            return await self.send_to_all_user_devices(tenant_id, user_id, message)
    
    async def send_to_all_user_devices(self, tenant_id: str, user_id: str, message: str) -> Dict[str, bool]:
        """Send message to all devices of a user"""
        results = {}
        if user_id in self.user_connections:
            for device_id, connection_key in self.user_connections[user_id].items():
                results[device_id] = await self._send_to_connection(connection_key, message)
        return results

    async def send_json_to_user_device(self, tenant_id: str, user_id: str, data: dict, device_id: str = None) -> bool:
        """Send JSON to specific device or all user devices"""
        if device_id:
            key = f"{tenant_id}:{user_id}:{device_id}"
            return await self._send_json_to_connection(key, data)
        else:
            # Send to all user devices
            return await self.send_json_to_all_user_devices(tenant_id, user_id, data)
    
    async def send_json_to_all_user_devices(self, tenant_id: str, user_id: str, data: dict) -> Dict[str, bool]:
        """Send JSON to all devices of a user"""
        results = {}
        if user_id in self.user_connections:
            for device_id, connection_key in self.user_connections[user_id].items():
                results[device_id] = await self._send_json_to_connection(connection_key, data)
        return results

    # Helper methods for sending to connections
    async def _send_to_connection(self, key: str, message: str) -> bool:
        """Send text message to a specific connection"""
        # Check rate limit
        if not self.rate_limiter.is_allowed(key):
            logger.warning(f"⚠️ Rate limit exceeded for {key}")
            return False
        
        ws = None
        with self._sync_lock:
            ws = self.active_connections.get(key)
            if ws:
                # Update activity
                if key in self.connection_metadata:
                    self.connection_metadata[key]["last_activity"] = time.time()
                    self.connection_metadata[key]["message_count"] += 1
        
        if ws and await self._is_websocket_alive(ws):
            try:
                await ws.send_text(message)
                self.metrics.record_send_success(key)
                return True
            except Exception as e:
                logger.warning(f"❌ Failed to send text to {key}: {e}")
                self.metrics.record_send_failure(key)
                self._cleanup_connection(key)
                return False
        else:
            if ws:
                self._cleanup_connection(key)
            return False

    async def _send_json_to_connection(self, key: str, data: dict) -> bool:
        """Send JSON data to a specific connection"""
        # Check rate limit
        if not self.rate_limiter.is_allowed(key):
            logger.warning(f"⚠️ Rate limit exceeded for {key}")
            return False
        
        # Validate payload size
        try:
            payload = json.dumps(data)
            if len(payload) > 1024 * 1024:  # 1MB limit
                logger.warning(f"⚠️ Payload too large for {key}: {len(payload)} bytes")
                return False
        except Exception as e:
            logger.error(f"❌ Invalid JSON payload for {key}: {e}")
            return False
        
        ws = None
        with self._sync_lock:
            ws = self.active_connections.get(key)
            if ws:
                # Update activity
                if key in self.connection_metadata:
                    self.connection_metadata[key]["last_activity"] = time.time()
                    self.connection_metadata[key]["message_count"] += 1
        
        if ws and await self._is_websocket_alive(ws):
            try:
                await ws.send_json(data)
                self.metrics.record_send_success(key)
                return True
            except Exception as e:
                logger.warning(f"❌ Failed to send JSON to {key}: {e}")
                self.metrics.record_send_failure(key)
                self._cleanup_connection(key)
                return False
        else:
            if ws:
                self._cleanup_connection(key)
            return False

    def get_user_devices(self, user_id: str) -> List[str]:
        """Get list of device IDs for a user"""
        return list(self.user_connections.get(user_id, {}).keys())
    
    def get_user_device_count(self, user_id: str) -> int:
        """Get number of connected devices for a user"""
        return len(self.user_connections.get(user_id, {}))

    def is_connected(self, tenant_id: str, user_id: str, device_id: str = None) -> bool:
        """Check if a user or specific device is connected"""
        if device_id:
            key = f"{tenant_id}:{user_id}:{device_id}"
            with self._sync_lock:
                return key in self.active_connections
        else:
            # Check if user has any connected devices
            return self.get_user_device_count(user_id) > 0

    async def _is_websocket_alive(self, websocket: WebSocket) -> bool:
        """Enhanced WebSocket health check"""
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                return False
            
            # Send a small ping to verify the connection
            await websocket.send_json({"type": "health_check"})
            return True
        except Exception:
            return False

    def _cleanup_connection(self, key: str) -> None:
        """Internal method to clean up a connection (synchronous)"""
        with self._sync_lock:
            if key in self.active_connections:
                del self.active_connections[key]
            if key in self.connection_metadata:
                user_id = self.connection_metadata[key].get("user_id")
                del self.connection_metadata[key]
                
                # Clean up user connections
                if user_id and user_id in self.user_connections:
                    self.user_connections[user_id].discard(key)
                    if not self.user_connections[user_id]:
                        del self.user_connections[user_id]

    async def cleanup_stale_connections(self, max_idle_seconds: int = 300) -> Dict[str, Any]:
        """Clean up stale connections with improved error handling"""
        if self._cleanup_running:
            return {"status": "cleanup_in_progress"}
            
        self._cleanup_running = True
        try:
            current_time = time.time()
            stale_keys = []
            
            with self._sync_lock:
                for key, metadata in self.connection_metadata.items():
                    last_activity = metadata.get("last_activity", 0)
                    if current_time - last_activity > max_idle_seconds:
                        stale_keys.append(key)
            
            # Clean up stale connections
            for key in stale_keys:
                ws = None
                with self._sync_lock:
                    ws = self.active_connections.get(key)
                
                if ws:
                    try:
                        await ws.close(code=1001, reason="Connection idle timeout")
                    except Exception:
                        pass  # Connection might already be closed
                    
                self._cleanup_connection(key)
                logger.info(f"🧹 Cleaned up stale connection: {key}")
            
            # Clean up rate limiter and metrics
            self.rate_limiter.cleanup()
            self.metrics.cleanup_old_rates()
            self.metrics.last_cleanup = current_time
            
            if stale_keys:
                logger.info(f"🧹 Cleaned up {len(stale_keys)} stale connections")
                
        except Exception as e:
            logger.error(f"❌ Error during connection cleanup: {e}")
        finally:
            self._cleanup_running = False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all connections"""
        if self._health_check_running:
            return {"status": "health_check_in_progress"}
            
        self._health_check_running = True
        try:
            dead_connections = []
            alive_count = 0
            
            # Check all connections
            connections_to_check = []
            with self._sync_lock:
                connections_to_check = list(self.active_connections.items())
            
            for key, ws in connections_to_check:
                if await self._is_websocket_alive(ws):
                    alive_count += 1
                else:
                    dead_connections.append(key)
            
            # Clean up dead connections
            for key in dead_connections:
                self._cleanup_connection(key)
            
            return {
                "status": "healthy",
                "total_connections": len(self.active_connections),
                "alive_connections": alive_count,
                "dead_connections_cleaned": len(dead_connections),
                "metrics": {
                    "total_created": self.metrics.connections_created,
                    "total_closed": self.metrics.connections_closed,
                    "successful_sends": self.metrics.successful_sends,
                    "failed_sends": self.metrics.failed_sends,
                    "last_cleanup": datetime.fromtimestamp(self.metrics.last_cleanup).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error during health check: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self._health_check_running = False

    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        with self._sync_lock:
            return len(self.active_connections)

    def get_connected_users(self) -> list:
        """Get list of connected users with metadata"""
        with self._sync_lock:
            return [
                {
                    "key": key,
                    "metadata": metadata
                }
                for key, metadata in self.connection_metadata.items()
            ]

    def get_user_connections(self, user_id: str) -> list:
        """Get all connections for a specific user"""
        with self._sync_lock:
            return list(self.user_connections.get(user_id, set()))

    def get_connection_info(self, tenant_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information for a specific user"""
        key = f"{tenant_id}:{user_id}"
        with self._sync_lock:
            if key in self.connection_metadata:
                return self.connection_metadata[key].copy()
            return None

    async def ping_all_connections(self) -> Dict[str, Any]:
        """Ping all connections to verify they're alive"""
        results = {"success": 0, "failed": 0, "errors": []}
        
        connections_to_ping = []
        with self._sync_lock:
            connections_to_ping = list(self.active_connections.items())
        
        for key, ws in connections_to_ping:
            try:
                if await self._is_websocket_alive(ws):
                    # Send a ping message
                    await ws.send_json({"type": "ping", "timestamp": time.time()})
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{key}: {str(e)}")
        
        return results

# Global instance
manager = ConnectionManager()

# Background tasks for monitoring and cleanup
async def connection_monitor():
    """Background task to monitor and clean up connections"""
    while True:
        try:
            await manager.cleanup_stale_connections()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logger.error(f"❌ Connection monitor error: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute on error

async def health_monitor():
    """Background task for periodic health checks"""
    while True:
        try:
            health_result = await manager.health_check()
            if health_result.get("status") != "healthy":
                logger.warning(f"⚠️ WebSocket health check: {health_result}")
            await asyncio.sleep(600)  # Run every 10 minutes
        except Exception as e:
            logger.error(f"❌ Health monitor error: {e}")
            await asyncio.sleep(120)  # Retry after 2 minutes on error
