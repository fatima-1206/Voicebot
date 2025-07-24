import asyncio
import websockets
import json
import base64
import sounddevice as sd
import soundfile as sf
import threading
import time
import numpy as np
from pathlib import Path
import tempfile
import io
import queue
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import datetime

class VoicebotSTTClient:
    def __init__(self, host="localhost", port=8000, tenant_id="d7acb015-1ce6-451d-b253-d7455070b4b4", user_id="f00ce248-5a04-41f5-97ca-828d24ea265a"):
        self.host = host
        self.port = port
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.websocket_url = f"ws://{host}:{port}/ws/{tenant_id}/{user_id}"
        self.websocket = None
        self.recording = False
        self.streaming = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        
        # Audio settings optimized for web-like streaming
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration_ms = 100
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # Buffer management
        self.audio_buffer = np.array([], dtype=np.int16)
        self.min_buffer_size = self.chunk_size * 2
        self.max_buffer_size = self.chunk_size * 20
        
        # CLI-style tracking
        self.chunk_counter = 0
        self.session_start_time = None
        
        # Thread pool for audio processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Create transcription log file like CLI
        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(f"=== Transcription Session Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
        # Create chat log file like CLI
        with open("chat.txt", "w", encoding="utf-8") as f:
            f.write(f"=== Chat Session Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    async def connect(self):
        """Connect to the WebSocket"""
        try:
            print(f"🔗 Connecting to {self.websocket_url}")
            self.websocket = await websockets.connect(self.websocket_url)
            print("✅ Connected to WebSocket")
            self.session_start_time = time.time()
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            print("🔌 Disconnected from WebSocket")
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback - mimics web MediaRecorder.ondataavailable"""
        if status:
            print(f"⚠️ Audio callback status: {status}")
        
        if self.recording and not self.stop_event.is_set():
            mono_audio = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()
            audio_int16 = (mono_audio * 32767).astype(np.int16)
            
            try:
                self.audio_queue.put_nowait(audio_int16)
            except queue.Full:
                print("⚠️ Audio queue full, dropping chunk")
    
    async def start_media_recorder_like_streaming(self, duration=None):
        """Start CLI-style continuous streaming"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        print("\n🎙️ Starting CLI-style Audio Processing Pipeline")
        print("=" * 60)
        print("🔄 Audio Recorder → STT → RAG → TTS Pipeline")
        print("📝 Output will match your CLI application format")
        if duration:
            print(f"⏰ Duration: {duration} seconds")
        else:
            print("⏰ Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            await self.start_recording()
            streaming_task = asyncio.create_task(
                self.process_audio_chunks_continuously(duration)
            )
            await streaming_task
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error in streaming: {e}")
        finally:
            await self.stop_recording()
    
    async def start_recording(self):
        """Start audio recording"""
        try:
            print("🎤 Starting audio capture...")
            print("🎧 Listening...")  # Match CLI output
            
            self.recording = True
            self.streaming = True
            self.stop_event.clear()
            self.chunk_counter = 0
            
            # Clear buffers
            self.audio_buffer = np.array([], dtype=np.int16)
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start audio stream
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
                dtype='float32',
                latency='low'
            )
            self.audio_stream.start()
            print("✅ Audio capture started")
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self.recording = False
            self.streaming = False
    
    async def stop_recording(self):
        """Stop audio recording"""
        print("🛑 Stopping audio capture...")
        self.recording = False
        self.streaming = False
        self.stop_event.set()
        
        if hasattr(self, 'audio_stream'):
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                print(f"⚠️ Warning stopping audio stream: {e}")
        
        # Send any remaining audio
        if len(self.audio_buffer) > 0:
            await self.send_audio_chunk(self.audio_buffer, is_final=True)
            self.audio_buffer = np.array([], dtype=np.int16)
        
        print("✅ Audio capture stopped")
    
    async def process_audio_chunks_continuously(self, duration=None):
        """Process audio chunks continuously"""
        start_time = time.time()
        last_send_time = start_time
        
        print("📡 Starting continuous audio processing...")
        
        while self.streaming and not self.stop_event.is_set():
            current_time = time.time()
            
            # Check duration limit
            if duration and (current_time - start_time) >= duration:
                print(f"\n⏰ Duration limit reached ({duration}s)")
                break
            
            try:
                audio_chunk = self.audio_queue.get(timeout=0.05)
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                
                # Check if we should send
                should_send = (
                    len(self.audio_buffer) >= self.min_buffer_size and
                    (current_time - last_send_time) >= 0.2
                ) or len(self.audio_buffer) >= self.max_buffer_size
                
                if should_send:
                    await self.send_audio_chunk(self.audio_buffer)
                    
                    # Reset buffer with overlap
                    overlap_size = self.chunk_size // 2
                    if len(self.audio_buffer) > overlap_size:
                        self.audio_buffer = self.audio_buffer[-overlap_size:]
                    else:
                        self.audio_buffer = np.array([], dtype=np.int16)
                    
                    last_send_time = current_time
                    self.chunk_counter += 1
            
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            except Exception as e:
                print(f"❌ Error processing audio: {e}")
                break
        
        print(f"✅ Audio processing complete")
    
    async def send_audio_chunk(self, audio_data, is_final=False):
        """Send audio chunk via WebSocket"""
        try:
            audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            
            message = {
                "type": "continuous_audio_stream",
                "audio": audio_b64,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "chunk_duration_ms": len(audio_data) * 1000 // self.sample_rate,
                "is_final": is_final,
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "tts_language": "en",
                "voice": "en-US-JennyNeural",
                "auto_respond": True,
                "timestamp": int(time.time() * 1000)
            }
            
            await self.websocket.send(json.dumps(message))
            
        except Exception as e:
            print(f"❌ Error sending audio chunk: {e}")
    
    async def start_conversation_session(self):
        """Start a conversation session"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            init_message = {
                "type": "start_conversation_session",
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "session_config": {
                    "language": "en",
                    "voice": "en-US-JennyNeural",
                    "auto_respond": True,
                    "use_rag": True,
                    "stream_audio": True
                },
                "client_info": {
                    "type": "cli_style_client",
                    "sample_rate": self.sample_rate,
                    "chunk_duration_ms": self.chunk_duration_ms
                }
            }
            
            await self.websocket.send(json.dumps(init_message))
            print("✅ Conversation session initialized")
            
        except Exception as e:
            print(f"❌ Error initializing session: {e}")
    
    async def end_conversation_session(self):
        """End conversation session"""
        if not self.websocket:
            return
        
        try:
            end_message = {
                "type": "end_conversation_session",
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "timestamp": int(time.time() * 1000)
            }
            
            await self.websocket.send(json.dumps(end_message))
            print("✅ Conversation session ended")
            
        except Exception as e:
            print(f"❌ Error ending session: {e}")
    
    async def listen_for_responses(self):
        """Listen for responses from WebSocket - CLI-style output"""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "unknown")
                    
                    # Handle different message types with CLI-style formatting
                    if message_type == "transcription_result":
                        await self.handle_transcription_result(data)
                        
                    elif message_type == "speech_detected":
                        await self.handle_speech_detected(data)
                        
                    elif message_type == "query_with_tts_result":
                        await self.handle_query_response(data)
                        
                    elif message_type == "audio_status":
                        await self.handle_audio_status(data)
                        
                    elif message_type == "audio_stream":
                        await self.handle_audio_stream(data)
                        
                    elif message_type == "session_status":
                        status = data.get('status', 'N/A')
                        print(f"📱 [SESSION] {status}")
                        
                    elif message_type == "error":
                        error_msg = data.get('message', 'Unknown error')
                        print(f"\n❌ [ERROR] {error_msg}")
                        
                    elif message_type == "pong":
                        print("🏓 Pong received")
                        
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON received: {message}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ Error listening for responses: {e}")
    
    async def handle_transcription_result(self, data):
        """Handle transcription results - CLI format"""
        transcript = data.get('transcript', '').strip()
        chunk_id = data.get('chunk_id', 'N/A')
        confidence = data.get('confidence', 0.0)
        processing_time = data.get('processing_time', 0.0)
        
        if transcript:
            # Format like CLI: [B001][0.00 - 3.45] transcript text
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            chunk_num = chunk_id.replace('chunk_', '') if 'chunk_' in chunk_id else chunk_id
            
            # Display like CLI
            print(f"📝 Transcribing at {timestamp}")
            
            # Log to file like CLI
            with open("transcription.txt", "a", encoding="utf-8") as f:
                f.write(f"\n[B{chunk_num}] {transcript}\n")
            
            print(f"⏱ Transcription took {processing_time:.2f} seconds")
            with open("transcription.txt", "a", encoding="utf-8") as f:
                f.write(f"Transcription took {processing_time:.2f} seconds\n")
            
            # If this triggers a query, show the generating message
            if len(transcript.split()) > 2:  # Only for meaningful transcripts
                print(f"🧑 Generating response for: {transcript}")
                with open("chat.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n[{timestamp}] 🧑: {transcript}\n")
    
    async def handle_speech_detected(self, data):
        """Handle speech detection"""
        transcript = data.get('transcript', 'N/A')
        duration = data.get('duration', 0.0)
        print(f"\n🗣️ [SPEECH] {transcript} ({duration:.1f}s)")
    
    async def handle_query_response(self, data):
        """Handle query response - CLI format"""
        response_text = data.get('response', 'N/A')
        query = data.get('query', 'N/A')
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        # Display like CLI
        print(f"🤖 Response: {response_text}")
        with open("chat.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] 🤖: {response_text}\n")
        
        # Show TTS processing like CLI
        print(f"🗣 Speaking: {response_text}")
    
    async def handle_audio_status(self, data):
        """Handle TTS audio status"""
        status = data.get('status', 'N/A')
        task_id = data.get('task_id', 'N/A')
        
        if status == "ready":
            filename = data.get('data', {}).get('filename', 'N/A')
            print(f"🔊 [TTS] Ready: {filename}")
        elif status == "generating":
            print(f"🔄 [TTS] Generating...")
        elif status == "streaming":
            print(f"🎵 [TTS] Streaming audio...")
    
    async def handle_audio_stream(self, data):
        """Handle audio stream"""
        task_id = data.get('task_id', 'N/A')
        print(f"🎵 [AUDIO] Stream received (Task: {task_id[:8]})")
    
    async def send_audio_file(self, file_path):
        """Send an audio file for transcription"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            data, sample_rate = sf.read(file_path)
            print(f"📂 Reading audio file: {file_path}")
            print(f"   Sample rate: {sample_rate}")
            print(f"   Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
            print(f"   Duration: {len(data) / sample_rate:.2f}s")
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                print(f"   Resampling from {sample_rate} to {self.sample_rate}")
                try:
                    import scipy.signal
                    data = scipy.signal.resample(data, int(len(data) * self.sample_rate / sample_rate))
                except ImportError:
                    data = np.interp(
                        np.linspace(0, len(data), int(len(data) * self.sample_rate / sample_rate)),
                        np.arange(len(data)),
                        data
                    )
            
            # Convert to 16-bit integers
            data = (data * 32767).astype(np.int16)
            
            # Send in chunks like CLI processing
            chunk_size = self.chunk_size * 5
            total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size > 0 else 0)
            
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                await self.send_audio_chunk(chunk, is_final=(i + chunk_size >= len(data)))
                
                chunk_num = i // chunk_size + 1
                if chunk_num % 5 == 0 or chunk_num == total_chunks:
                    print(f"📊 Sending chunk {chunk_num}/{total_chunks}")
                
                await asyncio.sleep(0.1)
            
            print("✅ Audio file sent for transcription")
            
        except Exception as e:
            print(f"❌ Error sending audio file: {e}")
    
    async def test_query_with_tts(self, query):
        """Test text query with TTS"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            # Show like CLI
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"🧑 Generating response for: {query}")
            with open("chat.txt", "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}] 🧑: {query}\n")
            
            message = {
                "type": "query_with_tts",
                "message": query,
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "language": "en",
                "voice": "en-US-JennyNeural",
                "max_tokens": 500,
                "temperature": 0.7,
                "stream_audio": True
            }
            
            await self.websocket.send(json.dumps(message))
            print(f"✅ Query sent: {query}")
            
        except Exception as e:
            print(f"❌ Error sending query: {e}")
    
    async def send_ping(self):
        """Send ping to test connection"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            message = {"type": "ping"}
            await self.websocket.send(json.dumps(message))
            print("🏓 Ping sent")
        except Exception as e:
            print(f"❌ Error sending ping: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_event.set()
        
        if hasattr(self, 'audio_stream'):
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                print(f"⚠️ Warning during cleanup: {e}")
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print("\n🛑 Received interrupt signal, shutting down...")
    sys.exit(0)

async def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    client = VoicebotSTTClient()
    
    try:
        if not await client.connect():
            return
        
        # Start listening for responses in background
        listen_task = asyncio.create_task(client.listen_for_responses())
        
        print("\n🎯 CLI-Style Voice Pipeline Client:")
        print("1. Test with audio file")
        print("2. CLI-style Real-time Pipeline (unlimited)")
        print("3. CLI-style Real-time Pipeline (30s)")
        print("4. CLI-style Real-time Pipeline (60s)")
        print("5. CLI-style Real-time Pipeline (custom duration)")
        print("6. Test text query with TTS")
        print("7. Send ping (connection test)")
        print("8. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-8): ").strip()
                
                if choice == "1":
                    file_path = input("Enter audio file path (or press Enter for 'sample.wav'): ").strip()
                    if not file_path:
                        file_path = "sample.wav"
                    
                    if Path(file_path).exists():
                        await client.start_conversation_session()
                        await client.send_audio_file(file_path)
                        await client.end_conversation_session()
                    else:
                        print(f"❌ File not found: {file_path}")
                
                elif choice == "2":
                    await client.start_conversation_session()
                    await client.start_media_recorder_like_streaming()
                    await client.end_conversation_session()
                
                elif choice == "3":
                    await client.start_conversation_session()
                    await client.start_media_recorder_like_streaming(30)
                    await client.end_conversation_session()
                    
                elif choice == "4":
                    await client.start_conversation_session()
                    await client.start_media_recorder_like_streaming(60)
                    await client.end_conversation_session()
                    
                elif choice == "5":
                    duration_str = input("Enter duration in seconds: ").strip()
                    try:
                        duration = int(duration_str)
                        await client.start_conversation_session()
                        await client.start_media_recorder_like_streaming(duration)
                        await client.end_conversation_session()
                    except ValueError:
                        print("❌ Invalid duration")
                
                elif choice == "6":
                    query = input("Enter your query: ").strip()
                    if query:
                        await client.test_query_with_tts(query)
                    else:
                        print("❌ Query cannot be empty")
                
                elif choice == "7":
                    await client.send_ping()
                    await asyncio.sleep(0.5)
                
                elif choice == "8":
                    print("👋 Exiting...")
                    break
                
                else:
                    print("❌ Invalid choice, please try again")
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except EOFError:
                print("\n🛑 Input stream ended")
                break
        
        listen_task.cancel()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()
        client.cleanup()

if __name__ == "__main__":
    print("🎙️ CLI-Style WebSocket Voice Client")
    print("=" * 60)
    print("Output will match your CLI application format:")
    print("📝 Transcribing at HH:MM:SS")
    print("⏱ Transcription took X.XX seconds")  
    print("🧑 Generating response for: [query]")
    print("🤖 Response: [response]")
    print("🗣 Speaking: [text]")
    print("📚 Source citations")
    print("=" * 60)
    
    asyncio.run(main())