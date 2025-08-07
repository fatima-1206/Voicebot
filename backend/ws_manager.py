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
        self.message_rates = defaultdict(list)  # connection_key -> list of timestamps
        
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
    """Enhanced WebSocket connection manager with tenant|user|device|session support"""
    
    def __init__(self, max_connections_per_user: int = 10, max_devices_per_user: int = 5, max_sessions_per_device: int = 10, message_rate_limit: int = 100):
        # Connection hierarchy: tenant|user|device|session
        self.active_connections: Dict[str, WebSocket] = {}  # maps tenant|user|device|session → WebSocket
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}  # Store connection metadata
        
        # Hierarchical tracking
        self.tenant_users: Dict[str, Set[str]] = defaultdict(set)  # tenant_id → {user_ids}
        self.user_devices: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))  # tenant|user → {device_id: {session_ids}}
        self.device_sessions: Dict[str, Set[str]] = defaultdict(set)  # tenant|user|device → {session_ids}
        
        # Limits
        self.max_connections_per_user = max_connections_per_user
        self.max_devices_per_user = max_devices_per_user
        self.max_sessions_per_device = max_sessions_per_device
        
        # Monitoring and metrics
        self.metrics = ConnectionMetrics()
        self.rate_limiter = RateLimiter(max_messages=message_rate_limit)
        
        # Health monitoring
        self._health_check_running = False
        self._cleanup_running = False
        
        # Use threading.Lock for synchronous methods and asyncio.Lock for async methods
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        
    def _create_connection_key(self, tenant_id: str, user_id: str, device_id: str, session_id: str) -> str:
        """Create standardized connection key"""
        return f"{tenant_id}|{user_id}|{device_id}|{session_id}"
    
    def _create_user_key(self, tenant_id: str, user_id: str) -> str:
        """Create user key for hierarchical tracking"""
        return f"{tenant_id}|{user_id}"
    
    def _create_device_key(self, tenant_id: str, user_id: str, device_id: str) -> str:
        """Create device key for hierarchical tracking"""
        return f"{tenant_id}|{user_id}|{device_id}"
    
    def _parse_connection_key(self, key: str) -> tuple:
        """Parse connection key into components"""
        try:
            parts = key.split('|')
            if len(parts) == 4:
                return parts[0], parts[1], parts[2], parts[3]  # tenant, user, device, session
            else:
                raise ValueError(f"Invalid connection key format: {key}")
        except Exception as e:
            logger.error(f"Error parsing connection key {key}: {e}")
            return None, None, None, None
        
    async def connect(self, tenant_id: str, user_id: str, websocket: WebSocket, session_id: str, device_id: str = None) -> bool:
        """Connect a WebSocket with tenant|user|device|session hierarchy"""
        if not device_id:
            device_id = str(uuid4())
        
        if not session_id:
            session_id = str(uuid4())
            
        connection_key = self._create_connection_key(tenant_id, user_id, device_id, session_id)
        user_key = self._create_user_key(tenant_id, user_id)
        device_key = self._create_device_key(tenant_id, user_id, device_id)

        async with self._async_lock:
            # Check limits
            if len(self.user_devices[user_key]) >= self.max_devices_per_user:
                logger.warning(f"❌ Device limit exceeded for user {user_key}")
                await websocket.close(code=1008, reason="Device limit exceeded")
                return False
                
            if len(self.device_sessions[device_key]) >= self.max_sessions_per_device:
                logger.warning(f"❌ Session limit exceeded for device {device_key}")
                await websocket.close(code=1008, reason="Session limit exceeded")
                return False

            try:
                # Store connection and metadata
                self.active_connections[connection_key] = websocket
                self.connection_metadata[connection_key] = {
                    "connected_at": datetime.now(),
                    "last_activity": time.time(),
                    "message_count": 0,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "device_id": device_id,
                    "session_id": session_id
                }
                
                # Update hierarchical tracking
                self.tenant_users[tenant_id].add(user_id)
                self.user_devices[user_key][device_id].add(session_id)
                self.device_sessions[device_key].add(session_id)

                # Update metrics
                self.metrics.record_connection()

                logger.info(f"✅ WebSocket connected: {connection_key} (Total: {len(self.active_connections)})")
                return True

            except Exception as e:
                logger.error(f"❌ Error accepting WebSocket connection {connection_key}: {e}")
                return False

    def disconnect(self, tenant_id: str, user_id: str, session_id: str, device_id: str = None) -> bool:
        """Disconnect specific session"""
        if not device_id:
            # If no device_id provided, find it from existing connections
            user_key = self._create_user_key(tenant_id, user_id)
            for dev_id, sessions in self.user_devices[user_key].items():
                if session_id in sessions:
                    device_id = dev_id
                    break
            
            if not device_id:
                logger.warning(f"Could not find device for session {session_id}")
                return False
        
        connection_key = self._create_connection_key(tenant_id, user_id, device_id, session_id)
        return self._disconnect_single_session(connection_key, tenant_id, user_id, device_id, session_id)
    
    def _disconnect_single_session(self, connection_key: str, tenant_id: str, user_id: str, device_id: str, session_id: str) -> bool:
        """Disconnect a single session"""
        with self._sync_lock:
            if connection_key in self.active_connections:
                try:
                    # Clean up connection
                    del self.active_connections[connection_key]
                    if connection_key in self.connection_metadata:
                        del self.connection_metadata[connection_key]
                    
                    # Clean up hierarchical tracking
                    user_key = self._create_user_key(tenant_id, user_id)
                    device_key = self._create_device_key(tenant_id, user_id, device_id)
                    
                    # Remove session from device
                    if device_key in self.device_sessions:
                        self.device_sessions[device_key].discard(session_id)
                        if not self.device_sessions[device_key]:
                            del self.device_sessions[device_key]
                    
                    # Remove session from user's device tracking
                    if user_key in self.user_devices and device_id in self.user_devices[user_key]:
                        self.user_devices[user_key][device_id].discard(session_id)
                        if not self.user_devices[user_key][device_id]:
                            del self.user_devices[user_key][device_id]
                        
                        # If user has no devices, remove from tenant
                        if not self.user_devices[user_key]:
                            del self.user_devices[user_key]
                            self.tenant_users[tenant_id].discard(user_id)
                            if not self.tenant_users[tenant_id]:
                                del self.tenant_users[tenant_id]
                    
                    # Update metrics
                    self.metrics.record_disconnection()
                    
                    logger.info(f"🔌 WebSocket disconnected: {connection_key} (Total: {len(self.active_connections)})")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Error during disconnect for {connection_key}: {e}")
                    return False
            
            return False

    async def send_to_session(self, tenant_id: str, user_id: str, device_id: str, session_id: str, data: dict) -> bool:
        """Send message to specific session"""
        connection_key = self._create_connection_key(tenant_id, user_id, device_id, session_id)
        return await self._send_json_to_connection(connection_key, data)

    async def send_to_all_user_sessions(self, tenant_id: str, user_id: str, data: dict) -> Dict[str, bool]:
        """Send message to all sessions across all devices for a user"""
        results = {}
        user_key = self._create_user_key(tenant_id, user_id)
        
        if user_key in self.user_devices:
            for device_id, sessions in self.user_devices[user_key].items():
                for session_id in sessions:
                    connection_key = self._create_connection_key(tenant_id, user_id, device_id, session_id)
                    success = await self._send_json_to_connection(connection_key, data)
                    results[f"{device_id}|{session_id}"] = success
        
        return results

    async def send_to_device_sessions(self, tenant_id: str, user_id: str, device_id: str, data: dict) -> Dict[str, bool]:
        """Send message to all sessions on a specific device"""
        results = {}
        device_key = self._create_device_key(tenant_id, user_id, device_id)
        
        if device_key in self.device_sessions:
            for session_id in self.device_sessions[device_key]:
                connection_key = self._create_connection_key(tenant_id, user_id, device_id, session_id)
                success = await self._send_json_to_connection(connection_key, data)
                results[session_id] = success
        
        return results

    async def send_to_tenant_sessions(self, tenant_id: str, data: dict) -> Dict[str, bool]:
        """Send message to all sessions in a tenant"""
        results = {}
        
        if tenant_id in self.tenant_users:
            for user_id in self.tenant_users[tenant_id]:
                user_results = await self.send_to_all_user_sessions(tenant_id, user_id, data)
                results.update(user_results)
        
        return results

    # Helper methods for sending to connections
    async def _send_json_to_connection(self, connection_key: str, data: dict) -> bool:
        """Send JSON data to a specific connection"""
        # Check rate limit
        if not self.rate_limiter.is_allowed(connection_key):
            logger.warning(f"⚠️ Rate limit exceeded for {connection_key}")
            return False
        
        # Validate payload size
        try:
            payload = json.dumps(data)
            if len(payload) > 1024 * 1024:  # 1MB limit
                logger.warning(f"⚠️ Payload too large for {connection_key}: {len(payload)} bytes")
                return False
        except Exception as e:
            logger.error(f"❌ Invalid JSON payload for {connection_key}: {e}")
            return False
        
        ws = None
        with self._sync_lock:
            ws = self.active_connections.get(connection_key)
            if ws:
                # Update activity
                if connection_key in self.connection_metadata:
                    self.connection_metadata[connection_key]["last_activity"] = time.time()
                    self.connection_metadata[connection_key]["message_count"] += 1
        
        if ws and await self._is_websocket_alive(ws):
            try:
                await ws.send_json(data)
                self.metrics.record_send_success(connection_key)
                return True
            except Exception as e:
                logger.warning(f"❌ Failed to send JSON to {connection_key}: {e}")
                self.metrics.record_send_failure(connection_key)
                await self._cleanup_connection_async(connection_key)
                return False
        else:
            if ws:
                await self._cleanup_connection_async(connection_key)
            return False

    def get_user_devices(self, tenant_id: str, user_id: str) -> Dict[str, List[str]]:
        """Get all devices and their sessions for a user"""
        user_key = self._create_user_key(tenant_id, user_id)
        result = {}
        
        if user_key in self.user_devices:
            for device_id, sessions in self.user_devices[user_key].items():
                result[device_id] = list(sessions)
        
        return result
    
    def get_device_sessions(self, tenant_id: str, user_id: str, device_id: str) -> List[str]:
        """Get all sessions for a specific device"""
        device_key = self._create_device_key(tenant_id, user_id, device_id)
        return list(self.device_sessions.get(device_key, set()))

    def is_session_connected(self, tenant_id: str, user_id: str, device_id: str, session_id: str) -> bool:
        """Check if a specific session is connected"""
        connection_key = self._create_connection_key(tenant_id, user_id, device_id, session_id)
        with self._sync_lock:
            return connection_key in self.active_connections

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

    async def _cleanup_connection_async(self, connection_key: str) -> None:
        """Async version of connection cleanup"""
        tenant_id, user_id, device_id, session_id = self._parse_connection_key(connection_key)
        if all([tenant_id, user_id, device_id, session_id]):
            self._disconnect_single_session(connection_key, tenant_id, user_id, device_id, session_id)

    async def cleanup_stale_connections(self, max_idle_seconds: int = 300) -> Dict[str, Any]:
        """Clean up stale connections with improved error handling"""
        if self._cleanup_running:
            return {"status": "cleanup_in_progress"}
            
        self._cleanup_running = True
        try:
            current_time = time.time()
            stale_keys = []
            
            with self._sync_lock:
                for connection_key, metadata in self.connection_metadata.items():
                    last_activity = metadata.get("last_activity", 0)
                    if current_time - last_activity > max_idle_seconds:
                        stale_keys.append(connection_key)
            
            # Clean up stale connections
            for connection_key in stale_keys:
                ws = None
                with self._sync_lock:
                    ws = self.active_connections.get(connection_key)
                
                if ws:
                    try:
                        await ws.close(code=1001, reason="Connection idle timeout")
                    except Exception:
                        pass  # Connection might already be closed
                
                await self._cleanup_connection_async(connection_key)
                logger.info(f"🧹 Cleaned up stale connection: {connection_key}")
            
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
            
            for connection_key, ws in connections_to_check:
                if await self._is_websocket_alive(ws):
                    alive_count += 1
                else:
                    dead_connections.append(connection_key)
            
            # Clean up dead connections
            for connection_key in dead_connections:
                await self._cleanup_connection_async(connection_key)
            
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

    def get_connection_hierarchy_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection hierarchy"""
        with self._sync_lock:
            return {
                "total_connections": len(self.active_connections),
                "tenants": len(self.tenant_users),
                "total_users": sum(len(users) for users in self.tenant_users.values()),
                "total_devices": sum(len(devices) for devices in self.user_devices.values()),
                "total_sessions": sum(len(sessions) for sessions in self.device_sessions.values()),
                "tenant_breakdown": {
                    tenant_id: {
                        "users": len(users),
                        "total_user_devices": sum(
                            len(self.user_devices.get(f"{tenant_id}|{user_id}", {})) 
                            for user_id in users
                        )
                    }
                    for tenant_id, users in self.tenant_users.items()
                }
            }

    async def ping_all_connections(self) -> Dict[str, Any]:
        """Ping all connections to verify they're alive"""
        results = {"success": 0, "failed": 0, "errors": []}
        
        connections_to_ping = []
        with self._sync_lock:
            connections_to_ping = list(self.active_connections.items())
        
        for connection_key, ws in connections_to_ping:
            try:
                if await self._is_websocket_alive(ws):
                    # Send a ping message
                    await ws.send_json({"type": "ping", "timestamp": time.time()})
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{connection_key}: {str(e)}")
        
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
