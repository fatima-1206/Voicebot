import asyncio
import time
import threading
from collections import defaultdict
from typing import Dict, Any, List
import logging
import json
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioBufferManager:
    """Enhanced audio buffer management with server-side VAD and speech processing"""
    
    def __init__(self, max_buffer_size: int = 2 * 1024 * 1024):  # 2MB per buffer
        self.user_audio_buffers: Dict[str, bytearray] = defaultdict(bytearray)
        self.user_last_activity: Dict[str, float] = defaultdict(float)
        self.user_speech_states: Dict[str, Dict] = defaultdict(dict)  # Track speech state per user
        self.max_buffer_size = max_buffer_size
        self.max_idle_time = 300  # 5 minutes default
        self._lock = threading.RLock()
        
        # Speech detection parameters
        self.min_speech_duration = 0.5  # seconds
        self.max_silence_duration = 2.0  # seconds  
        self.speech_threshold = 0.3  # VAD confidence threshold
        self.buffer_process_interval = 0.1  # Process buffer every 100ms
        
        # Add automatic cleanup timer
        asyncio.create_task(self._periodic_cleanup())
        
    async def _periodic_cleanup(self):
        """Periodic cleanup and processing"""
        while True:
            await asyncio.sleep(30)  # Clean up every 30 seconds
            self.cleanup_stale_buffers()
            await self._process_pending_speech_segments()
    
    def get_buffer(self, key: str) -> bytearray:
        """Get or create buffer for a user"""
        with self._lock:
            self.user_last_activity[key] = time.time()
            # Initialize speech state if needed
            if key not in self.user_speech_states:
                self.user_speech_states[key] = {
                    'is_speaking': False,
                    'speech_start': None,
                    'silence_start': None,
                    'pending_segments': []
                }
            return self.user_audio_buffers[key]
    
    def append_audio(self, key: str, audio_bytes: bytes) -> Dict[str, Any]:
        """Append audio data and return processing status"""
        with self._lock:
            current_buffer = self.user_audio_buffers[key]
            
            # Check if adding this data would exceed the limit
            if len(current_buffer) + len(audio_bytes) > self.max_buffer_size:
                # Process what we have first, then trim
                self._process_buffer_for_speech(key)
                # Trim the buffer to half size to make room
                trim_size = self.max_buffer_size // 2
                current_buffer[:] = current_buffer[-trim_size:]
                logger.warning(f"⚠️ Buffer trimmed for {key}, was too large")
            
            current_buffer.extend(audio_bytes)
            self.user_last_activity[key] = time.time()
            
            # Process buffer for speech segments
            return self._process_buffer_for_speech(key)
    
    def _process_buffer_for_speech(self, key: str) -> Dict[str, Any]:
        """Process buffer for speech detection and segmentation"""
        try:
            current_buffer = self.user_audio_buffers[key]
            if len(current_buffer) < 3200:  # Less than 200ms at 16kHz
                return {"status": "insufficient_data"}
            
            # Convert buffer to numpy array for VAD
            audio_data = self._bytes_to_numpy_audio(bytes(current_buffer))
            if audio_data is None or len(audio_data) == 0:
                return {"status": "invalid_audio"}
            
            # Apply VAD to detect speech segments
            speech_segments = self._detect_speech_segments(audio_data)
            
            if speech_segments:
                # Update speech state and extract completed segments
                completed_segments = self._update_speech_state(key, speech_segments, audio_data)
                
                if completed_segments:
                    # Clear processed audio from buffer
                    # Keep last 1 second for continuity
                    keep_samples = 16000  # 1 second at 16kHz
                    keep_bytes = keep_samples * 2  # 2 bytes per sample (int16)
                    if len(current_buffer) > keep_bytes:
                        current_buffer[:] = current_buffer[-keep_bytes:]
                    
                    return {
                        "status": "speech_detected",
                        "segments": completed_segments,
                        "segment_count": len(completed_segments)
                    }
            
            return {"status": "processing"}
            
        except Exception as e:
            logger.error(f"❌ Error processing buffer for speech: {e}")
            return {"status": "error", "error": str(e)}
    
    def _bytes_to_numpy_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Assume 16-bit PCM audio
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32767.0
            return audio_data
        except Exception as e:
            logger.error(f"❌ Error converting bytes to numpy: {e}")
            return None
    
    def _detect_speech_segments(self, audio_data: np.ndarray) -> List[Dict]:
        """Detect speech segments using VAD"""
        try:
            # Import VAD model here to avoid circular imports
            from backend.workers.utilities import vad_model
            
            if vad_model is None:
                return []
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
            
            # Get speech timestamps
            from silero_vad import get_speech_timestamps
            speech_timestamps = get_speech_timestamps(
                audio_tensor, 
                vad_model, 
                sampling_rate=16000,
                threshold=self.speech_threshold,
                min_speech_duration_ms=int(self.min_speech_duration * 1000),
                min_silence_duration_ms=int(self.max_silence_duration * 1000),
                speech_pad_ms=100  # Add padding around speech
            )
            
            return speech_timestamps
            
        except Exception as e:
            logger.error(f"❌ Error in speech detection: {e}")
            return []
    
    def _update_speech_state(self, key: str, speech_segments: List[Dict], audio_data: np.ndarray) -> List[bytes]:
        """Update speech state and return completed speech segments"""
        speech_state = self.user_speech_states[key]
        completed_segments = []
        
        current_time = time.time()
        
        for segment in speech_segments:
            start_sample = segment['start']
            end_sample = segment['end']
            
            # Extract speech segment
            speech_audio = audio_data[start_sample:end_sample]
            
            # Convert back to bytes for processing
            speech_int16 = np.clip(speech_audio, -1.0, 1.0)
            speech_int16 = (speech_int16 * 32767).astype(np.int16)
            speech_bytes = speech_int16.tobytes()
            
            # Only include segments with sufficient duration
            duration = (end_sample - start_sample) / 16000.0  # Convert to seconds
            if duration >= self.min_speech_duration:
                completed_segments.append(speech_bytes)
                logger.info(f"🗣️ Speech segment detected: {duration:.2f}s for {key}")
        
        return completed_segments
    
    async def _process_pending_speech_segments(self):
        """Process any pending speech segments"""
        # This could be used for additional processing if needed
        pass
    
    def get_and_clear_buffer(self, key: str) -> bytes:
        """Get buffer contents and clear it"""
        with self._lock:
            buffer_data = bytes(self.user_audio_buffers[key])
            self.user_audio_buffers[key].clear()
            self.user_last_activity[key] = time.time()
            # Reset speech state
            if key in self.user_speech_states:
                self.user_speech_states[key] = {
                    'is_speaking': False,
                    'speech_start': None,
                    'silence_start': None,
                    'pending_segments': []
                }
            return buffer_data
    
    def clear_buffer(self, key: str):
        """Clear a specific buffer"""
        with self._lock:
            if key in self.user_audio_buffers:
                del self.user_audio_buffers[key]
            if key in self.user_last_activity:
                del self.user_last_activity[key]
            if key in self.user_speech_states:
                del self.user_speech_states[key]
    
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
            
            # Count active speech sessions
            active_speech_sessions = sum(
                1 for state in self.user_speech_states.values() 
                if state.get('is_speaking', False)
            )
            
            return {
                "total_buffers": total_buffers,
                "total_size_bytes": total_size,
                "average_size_bytes": avg_size,
                "max_buffer_size": self.max_buffer_size,
                "max_idle_time": self.max_idle_time,
                "active_speech_sessions": active_speech_sessions
            }
