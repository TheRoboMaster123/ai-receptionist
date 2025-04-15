import asyncio
import logging
import socketio
import uuid
import wave
import sounddevice as sd
import numpy as np
from asterisk.agi import *
from fastapi import FastAPI, WebSocket
import websockets
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app for WebSocket connections
app = FastAPI()

# Socket.IO for real-time communication
sio = socketio.AsyncServer(async_mode='asgi')
socket_app = socketio.ASGIApp(sio)

# Configuration
AUDIO_DIR = Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

LLM_WS_URL = "ws://localhost:8000/ws/llm"
TTS_WS_URL = "ws://localhost:8001/ws/tts"

class CallHandler:
    def __init__(self, agi, call_id):
        self.agi = agi
        self.call_id = call_id
        self.llm_ws = None
        self.tts_ws = None
        self.business_id = None
        self.conversation_id = None
        
    async def connect_services(self):
        """Connect to LLM and TTS WebSocket services"""
        try:
            self.llm_ws = await websockets.connect(f"{LLM_WS_URL}/{self.call_id}")
            self.tts_ws = await websockets.connect(f"{TTS_WS_URL}/{self.call_id}")
            logger.info(f"Connected to services for call {self.call_id}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to services: {str(e)}")
            return False
            
    async def handle_call(self):
        """Main call handling logic"""
        try:
            # Answer call
            self.agi.answer()
            
            # Get business ID from extension
            self.business_id = self.agi.get_variable('BUSINESS_ID')
            if not self.business_id:
                logger.error("No business ID provided")
                self.agi.hangup()
                return
                
            # Connect to services
            if not await self.connect_services():
                self.agi.hangup()
                return
                
            # Play welcome message
            welcome_text = "Hello! How can I help you today?"
            await self.play_tts(welcome_text)
            
            # Main conversation loop
            while True:
                # Record caller's speech
                audio_file = await self.record_audio()
                if not audio_file:
                    continue
                
                # Send audio to TTS service for transcription
                text = await self.get_transcription(audio_file)
                if not text:
                    continue
                
                # Send text to LLM service
                response = await self.get_llm_response(text)
                if not response:
                    continue
                
                # Convert response to speech and play it
                await self.play_tts(response)
                
        except AGIHangup:
            logger.info(f"Call {self.call_id} hung up")
        except Exception as e:
            logger.error(f"Error in call {self.call_id}: {str(e)}")
        finally:
            await self.cleanup()
            
    async def record_audio(self, timeout=5000, silence=2):
        """Record audio from the caller"""
        try:
            filename = AUDIO_DIR / f"{uuid.uuid4()}.wav"
            self.agi.record_file(
                str(filename),
                "wav",
                "#",
                timeout,
                0,
                True,
                silence
            )
            return filename
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            return None
            
    async def get_transcription(self, audio_file):
        """Get transcription from TTS service"""
        try:
            # Send audio file in chunks
            chunk_size = 32768
            with open(audio_file, 'rb') as f:
                while chunk := f.read(chunk_size):
                    await self.tts_ws.send_bytes(chunk)
            
            # Wait for transcription
            response = await self.tts_ws.receive_json()
            if response["type"] == "transcription":
                return response["text"]
            return None
        except Exception as e:
            logger.error(f"Error getting transcription: {str(e)}")
            return None
            
    async def get_llm_response(self, text):
        """Get response from LLM service"""
        try:
            await self.llm_ws.send_json({
                "type": "user_input",
                "text": text,
                "business_id": self.business_id,
                "conversation_id": self.conversation_id
            })
            
            response = await self.llm_ws.receive_json()
            if response["type"] == "llm_response":
                self.conversation_id = response["conversation_id"]
                return response["text"]
            return None
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return None
            
    async def play_tts(self, text):
        """Convert text to speech and play it"""
        try:
            # Request speech synthesis
            await self.tts_ws.send_json({
                "type": "synthesize",
                "text": text
            })
            
            # Receive audio chunks and save to file
            audio_file = AUDIO_DIR / f"{uuid.uuid4()}.wav"
            with open(audio_file, 'wb') as f:
                while True:
                    try:
                        chunk = await self.tts_ws.receive_bytes()
                        f.write(chunk)
                    except websockets.exceptions.WebSocketException:
                        break
            
            # Play the audio file
            self.agi.stream_file(str(audio_file))
            
            # Clean up
            os.remove(audio_file)
            
        except Exception as e:
            logger.error(f"Error in TTS playback: {str(e)}")
            
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.llm_ws:
                await self.llm_ws.close()
            if self.tts_ws:
                await self.tts_ws.close()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")

class AGIServer:
    def __init__(self, host='0.0.0.0', port=4573):
        self.host = host
        self.port = port
        
    async def handle_call(self, agi):
        """Handle incoming AGI connection"""
        call_id = str(uuid.uuid4())
        handler = CallHandler(agi, call_id)
        await handler.handle_call()
        
    def run(self):
        """Start the AGI server"""
        try:
            logger.info(f"Starting AGI server on {self.host}:{self.port}")
            server = FastAGIServer(self.handle_call, self.host, self.port)
            server.serve_forever()
        except Exception as e:
            logger.error(f"Error starting AGI server: {str(e)}")

if __name__ == "__main__":
    server = AGIServer()
    server.run() 