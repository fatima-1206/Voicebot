import asyncio
import time
import queue
import json
import base64
import uuid
import datetime
import websockets
import httpx
import numpy as np
import sounddevice as sd

class SimpleVoicebotClient:
    """Simplified client - only handles audio capture and basic messaging"""
    
    def __init__(self, host="localhost", port=8000, 
                 tenant_id="d7acb015-1ce6-451d-b253-d7455070b4b4",
                 user_id="f00ce248-5a04-41f5-97ca-828d24ea265a"):
        self.host = host
        self.port = port
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.websocket_url = f"ws://{host}:{port}/ws/{tenant_id}/{user_id}"
        self.websocket = None
        self.recording = False
        
        # Simple audio queue - just for thread safety
        self.audio_queue = queue.Queue(maxsize=5)
        
        # Basic audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration_ms = 100  # 100ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        self.use_rag = True
        self.input_device = None

    def list_input_devices(self):
        """List available audio input devices"""
        print("\nAvailable input devices:")
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        for idx, dev in enumerate(input_devices):
            print(f"{idx}: {dev['name']}")
        return input_devices

    def select_input_device(self):
        """Let user select input device"""
        input_devices = self.list_input_devices()
        if not input_devices:
            print("❌ No input devices found.")
            return
        try:
            choice = int(input("Select device number (or Enter for default): ") or -1)
            if 0 <= choice < len(input_devices):
                self.input_device = input_devices[choice]['index']
                print(f"✅ Selected: {input_devices[choice]['name']}")
            else:
                print("ℹ️ Using default device.")
        except:
            print("ℹ️ Using default device.")

    async def connect(self):
        """Connect to WebSocket"""
        try:
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
            print("🔌 Disconnected")

    def audio_callback(self, indata, frames, time, status):
        """Simple audio callback - just capture and queue"""
        if status:
            print(f"⚠️ Audio status: {status}")
        
        if self.recording:
            try:
                # Convert to mono int16 - that's it!
                mono_audio = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()
                audio_int16 = (mono_audio * 32767).astype(np.int16)
                self.audio_queue.put_nowait(audio_int16)
            except queue.Full:
                pass  # Drop if queue full
            except Exception as e:
                print(f"❌ Audio callback error: {e}")

    async def start_voice_call(self):
        """Start a simple voice call - just send raw audio chunks"""
        if not self.websocket:
            print("❌ Not connected")
            return
        
        print("\n🎙️ Starting Simple Voice Call")
        print("📱 Client: Capture → Send → Done")
        print("🧠 Backend: Buffer → VAD → STT → Response")
        print("⏰ Press Ctrl+C to stop")
        
        try:
            # Tell backend we're starting
            await self.websocket.send(json.dumps({
                "type": "start_continuous_conversation"
            }))
            
            # Start recording
            self.recording = True
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
                dtype='float32',
                device=self.input_device
            )
            self.audio_stream.start()
            print("🎤 Recording started...")
            
            # Simple sender loop
            while self.recording:
                try:
                    # Get audio chunk (with timeout)
                    audio_chunk = await asyncio.get_event_loop().run_in_executor(
                        None, self.audio_queue.get, True, 0.1
                    )
                    
                    # Send immediately - no processing!
                    await self.send_raw_audio(audio_chunk)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    if self.recording:
                        print(f"❌ Send error: {e}")
                    break
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            await self.stop_recording()

    async def send_raw_audio(self, audio_data):
        """Send raw audio - backend handles everything else"""
        try:
            # Just encode and send - no fancy logic!
            audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            
            message = {
                "type": "audio_chunk",
                "audio": audio_b64,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "rag": self.use_rag,
                "auto_respond": True,
                "tts_language": "en",
                "voice": "Fritz-PlayAI"
            }
            
            await self.websocket.send(json.dumps(message))
            
        except Exception as e:
            print(f"❌ Send error: {e}")

    async def stop_recording(self):
        """Stop recording"""
        self.recording = False
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        if self.websocket:
            await self.websocket.send(json.dumps({
                "type": "stop_continuous_conversation"
            }))

    async def send_text_query(self):
        """Send a simple text query"""
        query = input("Enter your query: ").strip()
        if not query:
            return
        
        await self.websocket.send(json.dumps({
            "type": "query",
            "message": query,
            "rag": self.use_rag,
            "auto_respond": True
        }))

    async def listen_for_responses(self):
        """Listen for server responses"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                msg_type = data.get("type", "unknown")
                
                if msg_type == "transcription_result":
                    transcript = data.get('transcript', '').strip()
                    if transcript:
                        print(f"📝 You said: {transcript}")
                
                elif msg_type == "query_result":
                    response = data.get('response', '')
                    print(f"🤖 Response: {response}")
                
                elif msg_type == "audio_chunk_processed":
                    reason = data.get('reason', '')
                    segments = data.get('speech_segments', 0)
                    print(f"✅ Processed: {reason} ({segments} segments)")
                
                elif msg_type == "audio_chunk_accumulating":
                    duration = data.get('buffer_duration_ms', 0)
                    # print(f"🔄 Buffering: {duration:.0f}ms")
                
                elif msg_type == "error":
                    print(f"❌ Error: {data.get('message', 'Unknown')}")
                
                elif msg_type == "ping":
                    await self.websocket.send(json.dumps({"type": "pong"}))
                
        except Exception as e:
            print(f"❌ Listen error: {e}")

async def main():
    client = SimpleVoicebotClient()
    
    try:
        if not await client.connect():
            return
        
        client.select_input_device()
        
        print("\n🎯 Simple Voice Client:")
        print("1. Voice Call")
        print("2. Text Query") 
        print("3. Toggle RAG")
        print("4. Exit")
        
        # Start response listener
        listen_task = asyncio.create_task(client.listen_for_responses())
        
        while True:
            try:
                choice = input("\nChoice (1-4): ").strip()
                
                if choice == "1":
                    await client.start_voice_call()
                elif choice == "2":
                    await client.send_text_query()
                elif choice == "3":
                    client.use_rag = not client.use_rag
                    print(f"ℹ️ RAG: {'ON' if client.use_rag else 'OFF'}")
                elif choice == "4":
                    break
                else:
                    print("❌ Invalid choice")
                    
            except KeyboardInterrupt:
                break
        
        listen_task.cancel()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    print("📱 Simple Voice Client - Frontend Does Less!")
    print("=" * 50)
    print("✅ Client: Just capture and send audio")
    print("🧠 Backend: Handle all the smart stuff")
    print("=" * 50)
    
    asyncio.run(main())