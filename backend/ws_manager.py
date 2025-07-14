# web sockets manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}  # maps tenant_user_id → WebSocket

    async def connect(self, tenant_id: str, user_id: str, websocket):
        await websocket.accept()
        key = f"{tenant_id}:{user_id}"
        self.active_connections[key] = websocket

    def disconnect(self, tenant_id: str, user_id: str):
        key = f"{tenant_id}:{user_id}"
        self.active_connections.pop(key, None)

    async def send_to_user(self, tenant_id: str, user_id: str, message: str):
        key = f"{tenant_id}:{user_id}"
        ws = self.active_connections.get(key)
        if ws:
            await ws.send_text(message)
