from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional, Any
import json
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Store active connections
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Store conversation states
        self.conversation_states: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str, service_type: str):
        """Connect a new client"""
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = {}
        self.active_connections[client_id][service_type] = websocket
        logger.info(f"Client {client_id} connected to {service_type} service")
        
    def disconnect(self, client_id: str, service_type: str):
        """Disconnect a client"""
        if client_id in self.active_connections:
            if service_type in self.active_connections[client_id]:
                del self.active_connections[client_id][service_type]
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected from {service_type} service")
        
    async def send_message(self, client_id: str, service_type: str, message: Dict[str, Any]):
        """Send a message to a specific client's service"""
        if client_id in self.active_connections and service_type in self.active_connections[client_id]:
            websocket = self.active_connections[client_id][service_type]
            try:
                await websocket.send_json(message)
                logger.debug(f"Message sent to client {client_id} on {service_type}")
            except Exception as e:
                logger.error(f"Error sending message to {client_id} on {service_type}: {str(e)}")
                
    async def broadcast(self, client_id: str, message: Dict[str, Any]):
        """Broadcast a message to all services for a client"""
        if client_id in self.active_connections:
            for service_type, websocket in self.active_connections[client_id].items():
                try:
                    await websocket.send_json(message)
                    logger.debug(f"Message broadcast to {service_type} for client {client_id}")
                except Exception as e:
                    logger.error(f"Error broadcasting to {service_type} for {client_id}: {str(e)}")
                    
    def update_conversation_state(self, client_id: str, updates: Dict[str, Any]):
        """Update the conversation state for a client"""
        if client_id not in self.conversation_states:
            self.conversation_states[client_id] = {}
        self.conversation_states[client_id].update(updates)
        self.conversation_states[client_id]["last_updated"] = datetime.now().isoformat()
        
    def get_conversation_state(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get the current conversation state for a client"""
        return self.conversation_states.get(client_id)
        
    async def handle_audio_stream(self, websocket: WebSocket, client_id: str):
        """Handle incoming audio stream"""
        try:
            while True:
                audio_chunk = await websocket.receive_bytes()
                # Process audio chunk and send to TTS service
                await self.send_message(client_id, "tts", {
                    "type": "audio_chunk",
                    "data": audio_chunk
                })
        except WebSocketDisconnect:
            self.disconnect(client_id, "audio")
            
    async def handle_text_stream(self, websocket: WebSocket, client_id: str):
        """Handle text stream between services"""
        try:
            while True:
                text_data = await websocket.receive_json()
                # Process text data and route between services
                if text_data.get("type") == "transcription":
                    await self.send_message(client_id, "llm", {
                        "type": "user_input",
                        "text": text_data["text"]
                    })
                elif text_data.get("type") == "llm_response":
                    await self.send_message(client_id, "tts", {
                        "type": "synthesize",
                        "text": text_data["text"]
                    })
        except WebSocketDisconnect:
            self.disconnect(client_id, "text")

# Create a global connection manager instance
manager = ConnectionManager() 