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

class VoicebotSTTClient:
    def __init__(self, host="localhost", port=8000, tenant_id="d7acb015-1ce6-451d-b253-d7455070b4b4", user_id="f00ce248-5a04-41f5-97ca-828d24ea265a"):
        self.host = host
        self.port = port
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.websocket_url = f"ws://{host}:{port}/ws/{tenant_id}/{user_id}"
        self.websocket = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.stop_recording = threading.Event()
        
        # Audio settings
        self.chunk_size = 1024
        self.channels = 1
        self.rate = 16000
        
        # Thread pool for audio processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def connect(self):
        """Connect to the WebSocket"""
        try:
            print(f"🔗 Connecting to {self.websocket_url}")
            self.websocket = await websockets.connect(self.websocket_url)
            print("✅ Connected to WebSocket")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            print("🔌 Disconnected from WebSocket")
    
    async def send_audio_file(self, file_path):
        """Send an audio file for transcription"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            # Read audio file
            data, sample_rate = sf.read(file_path)
            print(f"📂 Reading audio file: {file_path}")
            print(f"   Sample rate: {sample_rate}")
            print(f"   Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
            print(f"   Duration: {len(data) / sample_rate:.2f}s")
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Resample if needed
            if sample_rate != self.rate:
                print(f"   Resampling from {sample_rate} to {self.rate}")
                try:
                    import scipy.signal
                    data = scipy.signal.resample(data, int(len(data) * self.rate / sample_rate))
                except ImportError:
                    # Fallback to simple interpolation
                    data = np.interp(
                        np.linspace(0, len(data), int(len(data) * self.rate / sample_rate)),
                        np.arange(len(data)),
                        data
                    )
            
            # Convert to 16-bit integers
            data = (data * 32767).astype(np.int16)
            
            # Send audio in chunks
            chunk_size = 4096
            total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size > 0 else 0)
            
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                
                # Encode audio chunk as base64
                audio_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                
                # Send audio chunk
                message = {
                    "type": "audio_stream",
                    "audio": audio_b64,
                    "buffer_size": 32000,
                    "overlap_size": 8000,
                    "conversation_mode": True,  # Add this line
                    "tts_language": "en",       # Add this line
                    "voice": "en-US-JennyNeural",  # Add this line
                    "auto_respond": True        # Add this line
                }

                
                await self.websocket.send(json.dumps(message))
                
                # Show progress
                chunk_num = i // chunk_size + 1
                if chunk_num % 10 == 0 or chunk_num == total_chunks:
                    print(f"📊 Sending chunk {chunk_num}/{total_chunks}")
                
                await asyncio.sleep(0.05)  # Small delay to simulate streaming
            
            # Send end of stream
            end_message = {
                "type": "audio_stream_end",
                "conversation_mode": True,  # Add this line
                "tts_language": "en",       # Add this line
                "voice": "en-US-JennyNeural",  # Add this line
                "auto_respond": True        # Add this line
            }
            await self.websocket.send(json.dumps(end_message))
            print("✅ Audio file sent for transcription")
            
        except Exception as e:
            print(f"❌ Error sending audio file: {e}")
    
    async def record_and_send_audio(self, duration=5):
        """Record audio from microphone in batch mode"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            print(f"🎤 Recording audio for {duration} seconds...")
            
            # Record audio
            recording = sd.rec(
                int(duration * self.rate),
                samplerate=self.rate,
                channels=self.channels,
                dtype='int16'
            )
            
            # Wait for recording to complete
            sd.wait()
            
            print("🔄 Recording complete, sending for transcription...")
            
            # Send audio in chunks
            chunk_size = 4096
            total_chunks = len(recording) // chunk_size + (1 if len(recording) % chunk_size > 0 else 0)
            
            for i in range(0, len(recording), chunk_size):
                chunk = recording[i:i + chunk_size]
                
                # Send audio chunk
                audio_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                message = {
                    "type": "audio_stream",
                    "audio": audio_b64,
                    "buffer_size": 32000,
                    "overlap_size": 8000,
                    "conversation_mode": True,  # Add this line
                    "tts_language": "en",       # Add this line
                    "voice": "en-US-JennyNeural",  # Add this line
                    "auto_respond": True        # Add this line
                }
                await self.websocket.send(json.dumps(message))
                
                # Show progress
                chunk_num = i // chunk_size + 1
                if chunk_num % 10 == 0 or chunk_num == total_chunks:
                    print(f"📊 Sending chunk {chunk_num}/{total_chunks}")
                
                await asyncio.sleep(0.01)  # Small delay
            
            # Send end of stream
            end_message = {
                "type": "audio_stream_end",
                "conversation_mode": True,  # Add this line
                "tts_language": "en",       # Add this line
                "voice": "en-US-JennyNeural",  # Add this line
                "auto_respond": True        # Add this line
            }
            await self.websocket.send(json.dumps(end_message))
            print("✅ Audio sent for transcription")
            
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
    
    async def record_and_send_audio_streaming(self, duration=5):
        """Record audio from microphone with real-time streaming"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            print(f"🎤 Starting real-time recording for {duration} seconds...")
            print("📡 Streaming audio...")
            
            # Clear the queue and reset stop event
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.stop_recording.clear()
            
            # Audio callback function
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"⚠️ Audio callback status: {status}")
                
                if not self.stop_recording.is_set():
                    # Convert to int16 and put in queue
                    chunk = (indata[:, 0] * 32767).astype(np.int16)
                    try:
                        self.audio_queue.put_nowait(chunk)
                    except queue.Full:
                        print("⚠️ Audio queue full, dropping chunk")
            
            # Start recording
            with sd.InputStream(
                callback=audio_callback,
                samplerate=self.rate,
                channels=self.channels,
                dtype='float32',
                blocksize=self.chunk_size
            ):
                start_time = time.time()
                chunk_count = 0
                
                while time.time() - start_time < duration:
                    try:
                        # Get audio chunk from queue (with timeout)
                        chunk = self.audio_queue.get(timeout=0.1)
                        
                        # Encode and send chunk
                        audio_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                        message = {
                            "type": "audio_stream",
                            "audio": audio_b64,
                            "buffer_size": 32000,
                            "overlap_size": 8000
                        }
                        await self.websocket.send(json.dumps(message))
                        
                        chunk_count += 1
                        
                    except queue.Empty:
                        # No audio chunk available, continue
                        pass
                    except Exception as e:
                        print(f"❌ Error sending audio chunk: {e}")
                        break
                    
                    # Show progress every second
                    elapsed = time.time() - start_time
                    if int(elapsed) != int(elapsed - 0.1):
                        print(f"📊 Streaming... {elapsed:.1f}s (sent {chunk_count} chunks)")
            
            # Stop recording
            self.stop_recording.set()
            
            # Send any remaining chunks
            remaining_chunks = 0
            while not self.audio_queue.empty():
                try:
                    chunk = self.audio_queue.get_nowait()
                    audio_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                    message = {
                        "type": "audio_stream",
                        "audio": audio_b64,
                        "buffer_size": 32000,
                        "overlap_size": 8000,
                        "conversation_mode": True,  # Add this line
                        "tts_language": "en",       # Add this line
                        "voice": "en-US-JennyNeural",  # Add this line
                        "auto_respond": True        # Add this line
                    }
                    await self.websocket.send(json.dumps(message))
                    remaining_chunks += 1
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"❌ Error sending remaining chunk: {e}")
            
            # Send end of stream
            end_message = {
                "type": "audio_stream_end",
                "conversation_mode": True,  # Add this line
                "tts_language": "en",       # Add this line
                "voice": "en-US-JennyNeural",  # Add this line
                "auto_respond": True        # Add this line
            }
            await self.websocket.send(json.dumps(end_message))
            
            print(f"✅ Real-time recording complete (sent {chunk_count + remaining_chunks} total chunks)")
            
        except Exception as e:
            print(f"❌ Error in streaming recording: {e}")
        finally:
            self.stop_recording.set()
    
    async def record_and_send_audio_conversation(self, duration=5):
        """Record audio from microphone with full conversation mode (STT + RAG + TTS)"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            print(f"🎤 Recording audio for conversation mode ({duration} seconds)...")
            print("🗣️ This will trigger: STT → RAG → TTS pipeline")
            
            # Record audio
            recording = sd.rec(
                int(duration * self.rate),
                samplerate=self.rate,
                channels=self.channels,
                dtype='int16'
            )
            
            # Wait for recording to complete
            sd.wait()
            
            print("🔄 Recording complete, sending for full conversation processing...")
            
            # Send audio in chunks with conversation mode enabled
            chunk_size = 4096
            total_chunks = len(recording) // chunk_size + (1 if len(recording) % chunk_size > 0 else 0)
            
            for i in range(0, len(recording), chunk_size):
                chunk = recording[i:i + chunk_size]
                
                # Send audio chunk with conversation mode
                audio_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                message = {
                    "type": "audio_stream",
                    "audio": audio_b64,
                    "buffer_size": 32000,
                    "overlap_size": 8000,
                    "conversation_mode": True,      # Enable full pipeline
                    "tts_language": "en",
                    "voice": "en-US-JennyNeural",
                    "auto_respond": True
                }
                await self.websocket.send(json.dumps(message))
                
                # Show progress
                chunk_num = i // chunk_size + 1
                if chunk_num % 10 == 0 or chunk_num == total_chunks:
                    print(f"📊 Sending chunk {chunk_num}/{total_chunks}")
                
                await asyncio.sleep(0.01)  # Small delay
        
            # Send end of stream with conversation mode
            end_message = {
                "type": "audio_stream_end",
                "conversation_mode": True,
                "tts_language": "en",
                "voice": "en-US-JennyNeural", 
                "auto_respond": True
            }
            await self.websocket.send(json.dumps(end_message))
            print("✅ Audio sent for full conversation processing (STT+RAG+TTS)")
            
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
    
    async def listen_for_responses(self):
        """Listen for responses from the WebSocket"""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "unknown")
                    
                    if message_type == "transcription_result":
                        transcript = data.get('transcript', 'N/A')
                        confidence = data.get('confidence', 0.0)
                        
                        print(f"\n📝 Transcription Result:")
                        print(f"   Chunk ID: {data.get('chunk_id', 'N/A')}")
                        print(f"   Transcript: '{transcript}'")
                        print(f"   Confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else f"   Confidence: {confidence}")
                        print(f"   Language: {data.get('detected_language', 'N/A')}")
                        
                        processing_time = data.get('processing_time', 'N/A')
                        if isinstance(processing_time, (int, float)):
                            print(f"   Processing Time: {processing_time:.2f}s")
                        else:
                            print(f"   Processing Time: {processing_time}")
                        
                        segments = data.get('segments', [])
                        print(f"   Segments: {len(segments)}")
                        
                        # Show segment details if available
                        if segments and len(segments) > 0:
                            print("   Segment Details:")
                            for i, segment in enumerate(segments[:3]):  # Show first 3 segments
                                start = segment.get('start', 'N/A')
                                end = segment.get('end', 'N/A')
                                text = segment.get('text', 'N/A')
                                print(f"     [{i+1}] {start}-{end}s: '{text}'")
                            if len(segments) > 3:
                                print(f"     ... and {len(segments) - 3} more segments")
                        
                        print()  # Empty line for readability
                        
                    elif message_type == "transcription_error":
                        print(f"\n❌ Transcription Error:")
                        print(f"   Chunk ID: {data.get('chunk_id', 'N/A')}")
                        print(f"   Error: {data.get('error', 'N/A')}")
                        print()
                        
                    elif message_type == "transcription_submitted":
                        transcription_id = data.get('transcription_id', 'N/A')
                        is_final = data.get('final', False)
                        final_text = " (final)" if is_final else ""
                        print(f"✅ Transcription Submitted: {transcription_id}{final_text}")
                        
                    elif message_type == "error":
                        print(f"\n❌ WebSocket Error: {data.get('message', 'N/A')}")
                        print()
                        
                    elif message_type == "pong":
                        print("🏓 Pong received")
                        
                    else:
                        print(f"\n📨 Received: {message_type}")
                        print(f"   Data: {json.dumps(data, indent=2)}")
                        print()
                        
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON received: {message}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ Error listening for responses: {e}")
    
    async def test_query_with_tts(self, query):
        """Test query with TTS functionality"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            message = {
                "type": "query_with_tts",
                "message": query,
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
            print("🏓 Ping sent (watch for pong response above)")
        except Exception as e:
            print(f"❌ Error sending ping: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_recording.set()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print("\n🛑 Received interrupt signal, shutting down...")
    sys.exit(0)

async def main():
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    client = VoicebotSTTClient()
    
    try:
        # Connect to WebSocket
        if not await client.connect():
            return
        
        # Start listening for responses in background
        listen_task = asyncio.create_task(client.listen_for_responses())
        
        print("\n🎯 STT Test Options:")
        print("1. Test with audio file")
        print("2. Record from microphone (transcription only)")
        print("3. Record from microphone (streaming transcription only)")
        print("4. Record from microphone (FULL CONVERSATION - STT+RAG+TTS)")  # Update this
        print("5. Test query with TTS")
        print("6. Send ping")
        print("7. Exit")  # Update numbering
        
        while True:
            try:
                choice = input("\nEnter your choice (1-7): ").strip()
                
                if choice == "1":
                    file_path = input("Enter audio file path (or press Enter for 'sample.wav'): ").strip()
                    if not file_path:
                        file_path = "sample.wav"
                    
                    if Path(file_path).exists():
                        await client.send_audio_file(file_path)
                    else:
                        print(f"❌ File not found: {file_path}")
                
                elif choice == "2":
                    duration = input("Recording duration in seconds (default 5): ").strip()
                    try:
                        duration = int(duration) if duration else 5
                        await client.record_and_send_audio(duration)
                    except ValueError:
                        print("❌ Invalid duration, using 5 seconds")
                        await client.record_and_send_audio(5)
                
                elif choice == "3":
                    duration = input("Streaming duration in seconds (default 10): ").strip()
                    try:
                        duration = int(duration) if duration else 10
                        await client.record_and_send_audio_streaming(duration)
                    except ValueError:
                        print("❌ Invalid duration, using 10 seconds")
                        await client.record_and_send_audio_streaming(10)
                
                elif choice == "4":
                    duration = input("Recording duration in seconds (default 5): ").strip()
                    duration = int(duration) if duration.isdigit() else 5
                    await client.record_and_send_audio_conversation(duration)
                
                elif choice == "5":
                    query = input("Enter your query: ").strip()
                    if query:
                        await client.test_query_with_tts(query)
                    else:
                        print("❌ Query cannot be empty")
                
                elif choice == "6":
                    await client.send_ping()
                    # Give a moment for the pong to be received and displayed
                    await asyncio.sleep(0.5)
                
                elif choice == "7":
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
        
        # Cancel listening task
        listen_task.cancel()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()
        client.cleanup()

if __name__ == "__main__":
    print("🤖 Voicebot STT Test Client")
    print("=" * 40)
    
    # Check if required packages are available
    try:
        import scipy.signal
    except ImportError:
        print("⚠️  Warning: scipy not installed. Audio resampling may not work properly.")
        print("   Install with: pip install scipy")
    
    asyncio.run(main())