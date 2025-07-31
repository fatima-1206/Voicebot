import asyncio
import websockets
import json
import base64
import sounddevice as sd
import threading
import time
import numpy as np
import uuid
from typing import Dict
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
        self.stop_event = threading.Event()

        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration_ms = 300
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)

        self.executor = ThreadPoolExecutor(max_workers=2)
        self.input_device = None

        self.audio_queue = asyncio.Queue()
        self.audio_stream = None

    def list_input_devices(self):
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        for idx, dev in enumerate(input_devices):
            print(f"{idx}: {dev['name']} (ID: {dev['index']})")
        return input_devices

    def select_input_device(self):
        input_devices = self.list_input_devices()
        if input_devices:
            try:
                choice = int(input("Select input device by number: ") or 0)
            except ValueError:
                choice = 0
            if 0 <= choice < len(input_devices):
                self.input_device = input_devices[choice]['index']
                dev_info = sd.query_devices(self.input_device)
                self.sample_rate = int(dev_info['default_samplerate'])
            else:
                print("Invalid choice, using default device.")
                self.input_device = None

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    def audio_callback(self, indata, frames, time_, status):
        if status:
            print(f"⚠️ Audio callback status: {status}")
        if self.recording and not self.stop_event.is_set():
            mono_audio = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()
            audio_int16 = (mono_audio * 32767).astype(np.int16)
            # Put audio data in queue for async sending
            try:
                self.audio_queue.put_nowait(audio_int16)
            except asyncio.QueueFull:
                print("⚠️ Audio queue is full, dropping audio chunk.")

    async def send_audio_chunks(self):
        while self.recording and not self.stop_event.is_set():
            try:
                audio_data = await self.audio_queue.get()
                audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                chunk_id = str(uuid.uuid4())
                message = {
                    "type": "continuous_audio_stream",
                    "audio": audio_b64,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "chunk_id": chunk_id
                }
                if self.websocket:
                    await self.websocket.send(json.dumps(message))
            except Exception as e:
                print(f"❌ Error sending audio chunk: {e}")

    async def start_recording(self):
        self.recording = True
        self.stop_event.clear()
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=self.chunk_size,
            dtype='float32',
            latency='low',
            device=self.input_device
        )
        self.audio_stream.start()

    async def stop_recording(self):
        self.recording = False
        self.stop_event.set()
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

    async def listen_for_responses(self):
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        try:
            async for message in self.websocket:
                data = json.loads(message)
                print(data)
        except Exception as e:
            print(f"❌ Error listening for responses: {e}")

    async def start_cli_style_pipeline(self):
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        try:
            await self.start_recording()
            send_task = asyncio.create_task(self.send_audio_chunks())
            listen_task = asyncio.create_task(self.listen_for_responses())
            await asyncio.wait([send_task, listen_task], return_when=asyncio.FIRST_COMPLETED)
        except Exception as e:
            print(f"❌ Error in streaming: {e}")
        finally:
            await self.stop_recording()

def signal_handler(signum, frame):
    print("\n🛑 Received interrupt signal, shutting down...")
    sys.exit(0)

async def main():
    client = VoicebotSTTClient()
    if not await client.connect():
        return
    client.select_input_device()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler, sig, None)
        except NotImplementedError:
            # add_signal_handler may not be implemented on some platforms
            pass
    try:
        await client.start_cli_style_pipeline()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
