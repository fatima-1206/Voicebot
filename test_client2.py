import asyncio
import websockets
import json
import base64
import sounddevice as sd
import numpy as np
import queue
import threading
import time

class ContinuousAudioClient:
    def __init__(self, host="localhost", port=8000, 
                 tenant_id="d7acb015-1ce6-451d-b253-d7455070b4b4", 
                 user_id="f00ce248-5a04-41f5-97ca-828d24ea265a"):
        self.host = host
        self.port = port
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.websocket_url = f"ws://{host}:{port}/ws/{tenant_id}/{user_id}"
        self.websocket = None
        
        # Audio settings
        self.chunk_size = 1024
        self.rate = 16000
        self.channels = 1
        
        # Streaming state
        self.audio_queue = queue.Queue()
        self.stop_streaming = threading.Event()
        self.is_streaming = False
        
    async def connect(self):
        """Connect to WebSocket"""
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
    
    async def listen_for_responses(self):
        """Listen for responses from server"""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "unknown")
                    
                    if message_type == "speech_detected":
                        transcript = data.get('transcript', '')
                        duration = data.get('duration', 0)
                        print(f"\n🗣️ Speech detected ({duration:.1f}s): '{transcript}'")
                        
                    elif message_type == "rag_result":
                        response = data.get('response', '')
                        print(f"🤖 Bot: {response}")
                        
                    elif message_type == "tts_audio_chunk":
                        print("🔊 Playing response audio...")
                        
                    elif message_type == "continuous_conversation_started":
                        print("✅ Continuous conversation mode started")
                        
                    elif message_type == "error":
                        error_msg = data.get('message', 'Unknown error')
                        print(f"❌ Error: {error_msg}")
                        
                    elif message_type in ["transcription_submitted", "query_with_tts_submitted"]:
                        # Suppress these for cleaner output
                        pass
                        
                    else:
                        print(f"📨 {message_type}: {data}")
                        
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON: {message}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 Connection closed")
        except Exception as e:
            print(f"❌ Listen error: {e}")
    
    async def start_continuous_conversation(self):
        """Start continuous audio streaming with server-side VAD"""
        print("\n🎤 Starting continuous conversation...")
        print("💡 Speak naturally - the server will detect your speech automatically")
        print("💡 Press Ctrl+C to stop")
        
        # Initialize conversation mode - FIX: use send() instead of send_json()
        start_message = {
            "type": "start_continuous_conversation",
            "tts_language": "en",
            "voice": "en-US-JennyNeural",
            "auto_respond": True
        }
        await self.websocket.send(json.dumps(start_message))
        
        # Wait for confirmation
        await asyncio.sleep(0.5)
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"⚠️ Audio status: {status}")
            
            if not self.stop_streaming.is_set():
                # Convert to int16 and queue
                chunk = (indata[:, 0] * 32767).astype(np.int16)
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    print("⚠️ Audio queue full")
        
        try:
            # Start audio capture
            with sd.InputStream(
                callback=audio_callback,
                samplerate=self.rate,
                channels=self.channels,
                dtype='float32',
                blocksize=self.chunk_size
            ):
                self.is_streaming = True
                chunk_count = 0
                
                # Stream audio continuously
                while not self.stop_streaming.is_set():
                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                        
                        # Send chunk to server for VAD processing
                        audio_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                        
                        message = {
                            "type": "continuous_audio_stream",
                            "audio": audio_b64,
                            "tts_language": "en",
                            "voice": "en-US-JennyNeural",
                            "auto_respond": True
                        }
                        
                        # FIX: use send() with json.dumps() instead of send_json()
                        await self.websocket.send(json.dumps(message))
                        
                        chunk_count += 1
                        if chunk_count % 100 == 0:  # Status every ~6 seconds
                            print(f"📊 Streaming... ({chunk_count} chunks sent)")
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"❌ Error sending audio: {e}")
                        break
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping conversation...")
        finally:
            self.stop_streaming.set()
            self.is_streaming = False
            
            # Stop conversation mode - FIX: use send() instead of send_json()
            try:
                stop_message = {"type": "stop_continuous_conversation"}
                await self.websocket.send(json.dumps(stop_message))
            except:
                pass

async def main():
    client = ContinuousAudioClient()
    
    if not await client.connect():
        return
    
    # Start response listener
    listen_task = asyncio.create_task(client.listen_for_responses())
    
    try:
        await client.start_continuous_conversation()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    finally:
        listen_task.cancel()
        await client.disconnect()

if __name__ == "__main__":
    print("🤖 Continuous Conversation Client (Server-side VAD)")
    print("=" * 55)
    asyncio.run(main())