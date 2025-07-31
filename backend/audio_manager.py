import asyncio
import time
import threading
from collections import defaultdict
from typing import Dict, Any
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio buffer management with size limits and proper cleanup
class AudioBufferManager:
    """Improved audio buffer management with size limits and cleanup"""
    
    def __init__(self, max_buffer_size: int = 1024 * 1024):  # Reduce to 1MB
        self.user_audio_buffers: Dict[str, bytearray] = defaultdict(bytearray)
        self.user_last_activity: Dict[str, float] = defaultdict(float)
        self.max_buffer_size = max_buffer_size  # 1MB default
        self.max_idle_time = 300  # 5 minutes default
        self._lock = threading.RLock()
        
        # Add automatic cleanup timer
        asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        while True:
            await asyncio.sleep(30)  # Clean up every 30 seconds
            self.cleanup_stale_buffers()
    
    def get_buffer(self, key: str) -> bytearray:
        """Get or create buffer for a user"""
        with self._lock:
            self.user_last_activity[key] = time.time()
            return self.user_audio_buffers[key]
    
    def append_audio(self, key: str, audio_bytes: bytes) -> bool:
        """Append audio data with size checking"""
        with self._lock:
            current_buffer = self.user_audio_buffers[key]
            
            # Check if adding this data would exceed the limit
            if len(current_buffer) + len(audio_bytes) > self.max_buffer_size:
                # Trim the buffer to half size to make room
                trim_size = self.max_buffer_size // 2
                current_buffer[:] = current_buffer[-trim_size:]
                logger.warning(f"⚠️ Buffer trimmed for {key}, was too large")
            
            current_buffer.extend(audio_bytes)
            self.user_last_activity[key] = time.time()
            return True
    
    def get_and_clear_buffer(self, key: str) -> bytes:
        """Get buffer contents and clear it"""
        with self._lock:
            buffer_data = bytes(self.user_audio_buffers[key])
            self.user_audio_buffers[key].clear()
            self.user_last_activity[key] = time.time()
            return buffer_data
    
    def clear_buffer(self, key: str):
        """Clear a specific buffer"""
        with self._lock:
            if key in self.user_audio_buffers:
                del self.user_audio_buffers[key]
            if key in self.user_last_activity:
                del self.user_last_activity[key]
    
    def cleanup_stale_buffers(self) -> int:
        """Clean up buffers that have been idle too long"""
        current_time = time.time()
        stale_keys = []
        
        with self._lock:
            for key, last_activity in self.user_last_activity.items():
                if current_time - last_activity > self.max_idle_time:
                    stale_keys.append(key)
        
        # Clean up stale buffers
        for key in stale_keys:
            self.clear_buffer(key)
        
        if stale_keys:
            logger.info(f"🧹 Cleaned up {len(stale_keys)} stale audio buffers")
        
        return len(stale_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            total_buffers = len(self.user_audio_buffers)
            total_size = sum(len(buffer) for buffer in self.user_audio_buffers.values())
            avg_size = total_size / total_buffers if total_buffers > 0 else 0
            
            return {
                "total_buffers": total_buffers,
                "total_size_bytes": total_size,
                "average_size_bytes": avg_size,
                "max_buffer_size": self.max_buffer_size,
                "max_idle_time": self.max_idle_time
            }
    def move_last_speech_to_front(self, key: str, speech_segment: bytes):
        """Move the last speech segment to the start of the buffer."""
        with self._lock:
            current_buffer = self.user_audio_buffers[key]
            # Remove the last speech segment from the buffer and add it at the front
            current_buffer.insert(0, speech_segment)  # Insert at the start

            # Ensure the buffer doesn't exceed max size by trimming it
            if len(current_buffer) > self.max_buffer_size:
                current_buffer[:] = current_buffer[:self.max_buffer_size]  # Trim to max size

            self.user_last_activity[key] = time.time()