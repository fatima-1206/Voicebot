import json
import asyncio
from starlette.websockets import WebSocketState
import logging

logger = logging.getLogger(__name__)

# web sockets manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}  # maps tenant_user_id → WebSocket

    async def connect(self, tenant_id: str, user_id: str, websocket):
        await websocket.accept()
        key = f"{tenant_id}:{user_id}"
        self.active_connections[key] = websocket
        logger.info(f"✅ WebSocket connected: {key}")

    def disconnect(self, tenant_id: str, user_id: str):
        key = f"{tenant_id}:{user_id}"
        if key in self.active_connections:
            self.active_connections.pop(key)
            logger.info(f"🔌 WebSocket disconnected: {key}")

    async def send_to_user(self, tenant_id: str, user_id: str, message: str):
        key = f"{tenant_id}:{user_id}"
        ws = self.active_connections.get(key)
        if ws and await self._is_websocket_alive(ws):
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"❌ Failed to send text to {key}: {e}")
                self._cleanup_connection(key)

    async def send_json_to_user(self, tenant_id: str, user_id: str, data: dict):
        """Send JSON data to a specific user with connection state checking"""
        key = f"{tenant_id}:{user_id}"
        ws = self.active_connections.get(key)
        
        if ws and await self._is_websocket_alive(ws):
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.warning(f"❌ Failed to send JSON to {key}: {e}")
                self._cleanup_connection(key)
        else:
            # Connection is dead, clean it up
            if ws:
                self._cleanup_connection(key)

    def is_connected(self, tenant_id: str, user_id: str) -> bool:
        """Check if a user is connected"""
        key = f"{tenant_id}:{user_id}"
        return key in self.active_connections

    async def _is_websocket_alive(self, websocket) -> bool:
        """Check if WebSocket connection is still alive"""
        try:
            # Check the connection state
            if websocket.client_state != WebSocketState.CONNECTED:
                return False
            
            # Try a ping to verify the connection is responsive
            # Note: This is optional and might add latency
            # await websocket.ping()
            
            return True
        except Exception:
            return False

    def _cleanup_connection(self, key: str):
        """Clean up a dead connection"""
        if key in self.active_connections:
            self.active_connections.pop(key)
            logger.info(f"🧹 Cleaned up dead connection: {key}")

    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.active_connections)

    def get_connected_users(self) -> list:
        """Get list of connected users"""
        return list(self.active_connections.keys())
