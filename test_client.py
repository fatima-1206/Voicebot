import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
import numpy as np
import sounddevice as sd
import queue
import json
import base64
import uuid
import datetime
import websockets
import httpx
import sys
import signal
class VoicebotSTTClient:
    def __init__(self, host="localhost", port=8000, 
                 tenant_id="d7acb015-1ce6-451d-b253-d7455070b4b4",
                 user_id="f00ce248-5a04-41f5-97ca-828d24ea265a"):
        self.host = host
        self.port = port
        self.pending_queries: Dict[str, asyncio.Future] = {}

        self.token = None
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.websocket_url = f"ws://{host}:{port}/ws/{tenant_id}/{user_id}"
        self.websocket = None
        self.recording = False
        self.response_event = asyncio.Event()
        self.streaming = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        
        self.rag = True

        # Simplified audio settings - smaller chunks for backend buffering
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration_ms = 100  # Much smaller chunks - 100ms
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # Remove complex buffering from client
        self.chunk_counter = 0
        self.session_start_time = None
        
        # Thread pool for audio processing
        self.executor = ThreadPoolExecutor(max_workers=2)        
        self.input_device = None

    def list_input_devices(self):
        print("\nAvailable input devices:")
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        for idx, dev in enumerate(input_devices):
            print(f"{idx}: {dev['name']} (ID: {dev['index']})")
        return input_devices

    def select_input_device(self):
        input_devices = self.list_input_devices()
        if not input_devices:
            print("❌ No input devices found.")
            return
        try:
            choice = int(input("Select input device by number (or press Enter for default): ") or -1)
            if 0 <= choice < len(input_devices):
                self.input_device = input_devices[choice]['index']
                dev_info = sd.query_devices(self.input_device)
                default_samplerate = int(dev_info['default_samplerate'])
                print(f"✅ Selected device: {dev_info['name']}")
                print(f"   Default sample rate: {default_samplerate}")
                self.sample_rate = default_samplerate
            else:
                self.input_device = None
                print("ℹ️ Using default input device.")
        except Exception as e:
            self.input_device = None
            print(f"ℹ️ Using default input device. Error: {e}")

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
        """Simplified audio callback - just capture and queue"""
        if status:
            print(f"⚠️ Audio callback status: {status}")
        
        if self.recording and not self.stop_event.is_set():
            mono_audio = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()
            audio_int16 = (mono_audio * 32767).astype(np.int16)
            
            try:
                self.audio_queue.put_nowait(audio_int16)
            except queue.Full:
                print("⚠️ Audio queue full, dropping chunk")

    async def submit_text_query(self):
        query = input("Enter your text query: ").strip()
        self.api_url = f"http://{self.host}:{self.port}"

        if not self.token:
            self.token = input("Enter your JWT access token: ").strip()
        if not query:
            print("❌ Query cannot be empty")
            return

        url = f"{self.api_url}/query/submit"
        payload = {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "query": query,
            "use_rag": self.rag
        }
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    print(f"❌ Server error: {response.status_code}")
                    return

                result = response.json()
                query_id = result.get("query_id")
                if not query_id:
                    print("❌ No query_id returned from server.")
                    return

                # Create a future and wait for response
                loop = asyncio.get_event_loop()
                future = loop.create_future()
                self.pending_queries[query_id] = future

                print(f"📨 Query submitted. Waiting for response for query_id={query_id}...")
                response_data = await asyncio.wait_for(future, timeout=30.0)
                print(f"✅ Got response: {response_data}")

            except asyncio.TimeoutError:
                print("❌ Timed out waiting for response.")
                self.pending_queries.pop(query_id, None)
            except Exception as e:
                print(f"❌ Error sending or receiving query: {e}")

    async def start_cli_style_pipeline(self, duration=None):
        """Start simplified continuous streaming - backend does all the work"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        print("\n🎙️ Starting Simplified Voice Call Pipeline")
        print("=" * 60)
        print("🔄 Client: Small Audio Chunks → Backend: Buffer + VAD + STT → RAG → TTS")
        print("📝 Backend handles all buffering and speech detection")
        if duration:
            print(f"⏰ Duration: {duration} seconds")
        else:
            print("⏰ Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            # Start continuous conversation mode
            await self.start_continuous_conversation()
            
            # Start recording and streaming
            await self.start_recording()
            streaming_task = asyncio.create_task(
                self.stream_audio_continuously(duration)
            )
            await streaming_task
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error in streaming: {e}")
        finally:
            await self.stop_recording()
            await self.stop_continuous_conversation()
    
    async def start_continuous_conversation(self):
        """Start continuous conversation mode"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            message = {
                "type": "start_continuous_conversation"
            }
            
            await self.websocket.send(json.dumps(message))
            print("✅ Continuous conversation mode started")
            
        except Exception as e:
            print(f"❌ Error starting continuous conversation: {e}")
    
    async def stop_continuous_conversation(self):
        """Stop continuous conversation mode"""
        if not self.websocket:
            return
        
        try:
            message = {
                "type": "stop_continuous_conversation"
            }
            
            await self.websocket.send(json.dumps(message))
            print("✅ Continuous conversation mode stopped")
            
        except Exception as e:
            print(f"❌ Error stopping continuous conversation: {e}")
    
    async def start_recording(self):
        """Start audio recording"""
        try:
            print("🎤 Starting audio capture...")
            print("🎧 Listening...")
            
            self.recording = True
            self.streaming = True
            self.stop_event.clear()
            self.chunk_counter = 0
            
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start audio stream with smaller chunks
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.chunk_size,  # 100ms chunks
                dtype='float32',
                latency='low',
                device=self.input_device
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
        
        print("✅ Audio capture stopped")
    
    async def stream_audio_continuously(self, duration=None):
        """Simplified streaming - just send small chunks to backend"""
        start_time = time.time()
        
        print("📡 Starting continuous audio streaming (100ms chunks)...")

        while self.streaming and not self.stop_event.is_set():
            current_time = time.time()
            if duration and (current_time - start_time) >= duration:
                print(f"\n⏰ Duration limit reached ({duration}s)")
                break

            try:
                # Get audio chunk (100ms)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Send directly to backend - let backend handle buffering
                await self.send_small_audio_chunk(audio_chunk)
                self.chunk_counter += 1

            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            except Exception as e:
                print(f"❌ Error streaming audio: {e}")
                break
        
        print(f"✅ Audio streaming complete. Sent {self.chunk_counter} chunks")
    
    async def send_small_audio_chunk(self, audio_data):
        """Send small audio chunk to backend (100ms)"""
        try:
            audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            
            message = {
                "type": "continuous_audio_stream",
                "rag": self.rag,
                "audio": audio_b64,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "conversation_mode": True,
                "auto_respond": True,
                "tts_language": "en",
                "voice": "Fritz-PlayAI",
                "timestamp": int(time.time() * 1000)
            }
            await self.websocket.send(json.dumps(message))

        except Exception as e:
            print(f"❌ Error sending audio chunk: {e}")

    async def listen_for_responses(self):
        """Listen for responses from WebSocket - CLI-style output"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return

        try:
            async for message in self.websocket:
                data = json.loads(message)
                try:
                    message_type = data.get("type", "unknown") if isinstance(data, dict) else "unknown"

                    if message_type == "transcription_result":
                        await self.handle_transcription_result(data)
                    elif message_type in ("query_result", "query_with_tts_result", "text_response"):
                        query_id = data.get("query_id")
                        await self.handle_query_response(data)
                    elif message_type == "speech_detected":
                        await self.handle_speech_detected(data)
                    elif message_type == "audio_status":
                        await self.handle_audio_status(data)
                    elif message_type == "audio_stream":
                        await self.handle_audio_stream(data)
                    elif message_type == "continuous_conversation_started":
                        print("✅ Backend ready for continuous conversation")
                    elif message_type == "continuous_conversation_stopped":
                        print("✅ Backend stopped continuous conversation")
                    elif message_type == "error":
                        print(f"❌ Backend error: {data.get('message', 'Unknown error')}")
                    elif message_type == "ping":
                        await self.websocket.send(json.dumps({"type": "pong"}))
                    else:
                        print(f"📨 Unknown message: {message_type}")
                        
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
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"📝 [{timestamp}] Transcribed: {transcript}")
            print(f"⏱ Transcription took {processing_time:.2f} seconds")
            
    async def handle_speech_detected(self, data):
        """Handle speech detection"""
        transcript = data.get('transcript', 'N/A')
        duration = data.get('duration', 0.0)
        if transcript and len(transcript.strip()) > 2:
            print(f"\n🗣️ [SPEECH] {transcript} ({duration:.1f}s)")
    
    async def handle_query_response(self, data):
        """Handle query response - CLI format"""
        response_text = data.get('response', 'N/A')
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"🤖 [{timestamp}] Response: {response_text}")
       
    async def handle_audio_status(self, data):
        """Handle TTS audio status"""
        status = data.get('status', 'N/A')
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

async def cli_loop(client):
    print("\n🎯Voicebot Client:")
    print("1. Voicecall")
    print("2. Text Query")
    print("3. Record till keypress")
    print("4. turn rag off")       
    print("5. turn rag on") 
    print("6. Exit")
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice == "1":
                await client.start_cli_style_pipeline()
            elif choice == "2":
                await client.submit_text_query()
            elif choice == "3":
                await client.record_till_keypress()
            elif choice == "4":
                client.rag = False
                print("ℹ️ RAG processing disabled for voicecall")
            elif choice == "5":
                client.rag = True
                print("ℹ️ RAG processing enabled for voicecall")
            elif choice == "6":
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

async def main():
    signal.signal(signal.SIGINT, signal_handler)
    client = VoicebotSTTClient()
    try:
        if not await client.connect():
            return
        client.select_input_device()
        # Run both CLI and listener concurrently
        await asyncio.gather(
            client.listen_for_responses(),
            cli_loop(client)
        )
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()
        client.cleanup()

if __name__ == "__main__":
    print("🎙️ Style WebSocket Voice Client")
    print("=" * 60)

    print("📝 Transcribing at HH:MM:SS")
    print("⏱ Transcription took X.XX seconds")  
    print("🧑 Generating response for: [query]")
    print("🤖 Response: [response]")
    print("🗣 Speaking: [text]")
    print("📚 Source citations")
    print("=" * 60)
    
    asyncio.run(main())