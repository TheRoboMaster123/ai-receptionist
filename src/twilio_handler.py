from fastapi import FastAPI, Request, WebSocket
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
import logging
import os
import uuid
from pathlib import Path
import websockets
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Receptionist Twilio Handler")

# Configuration
AUDIO_DIR = Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

LLM_WS_URL = "ws://localhost:8000/ws/llm"
TTS_WS_URL = "ws://localhost:8001/ws/tts"

# Twilio client setup
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
TWILIO_WEBHOOK_URL = os.getenv('TWILIO_WEBHOOK_URL')

# Validate webhook URL configuration
if not TWILIO_WEBHOOK_URL:
    logger.error("TWILIO_WEBHOOK_URL environment variable is not set")
    raise ValueError("TWILIO_WEBHOOK_URL environment variable is required")

# Ensure webhook URL is properly formatted
if not TWILIO_WEBHOOK_URL.startswith(('http://', 'https://')):
    logger.error(f"Invalid webhook URL format: {TWILIO_WEBHOOK_URL}")
    raise ValueError("Webhook URL must start with http:// or https://")

logger.info(f"Using webhook URL: {TWILIO_WEBHOOK_URL}")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

class CallHandler:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.llm_ws = None
        self.tts_ws = None
        self.business_id = None
        self.conversation_id = None
        
    async def connect_services(self):
        """Connect to LLM and TTS WebSocket services"""
        try:
            self.llm_ws = await websockets.connect(f"{LLM_WS_URL}/{self.call_sid}")
            self.tts_ws = await websockets.connect(f"{TTS_WS_URL}/{self.call_sid}")
            logger.info(f"Connected to services for call {self.call_sid}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to services: {str(e)}")
            return False
            
    async def get_llm_response(self, text: str) -> str:
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
            return "I'm sorry, I couldn't process that request."
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request."
            
    async def generate_speech(self, text: str) -> str:
        """Generate speech using TTS service"""
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
            
            return str(audio_file)
        except Exception as e:
            logger.error(f"Error in TTS generation: {str(e)}")
            return None
            
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.llm_ws:
                await self.llm_ws.close()
            if self.tts_ws:
                await self.tts_ws.close()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")

# Store active calls
active_calls: dict[str, CallHandler] = {}

@app.post("/incoming_call")
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio voice calls"""
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    business_id = form_data.get('To', 'default_business')  # Use the dialed number as business ID
    
    # Initialize call handler
    handler = CallHandler(call_sid)
    handler.business_id = business_id
    active_calls[call_sid] = handler
    
    # Connect to services
    await handler.connect_services()
    
    # Create TwiML response
    response = VoiceResponse()
    
    # Generate welcome message
    welcome_text = "Hello! How can I help you today?"
    audio_file = await handler.generate_speech(welcome_text)
    
    if audio_file:
        # Play welcome message and gather speech input
        gather = Gather(
            input='speech',
            action='/handle_speech',
            method='POST',
            speechTimeout='auto',
            enhanced='true'
        )
        gather.play(audio_file)
        response.append(gather)
    else:
        response.say("I apologize, but I'm having trouble initializing the conversation.")
    
    # Add the webhook for handling the call
    response.dial().client(
        "ai_receptionist",
        status_callback=TWILIO_WEBHOOK_URL,
        status_callback_event=['initiated', 'ringing', 'answered', 'completed']
    )
    
    return str(response)

@app.post("/handle_speech")
async def handle_speech(request: Request):
    """Handle speech input from Twilio"""
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    speech_result = form_data.get('SpeechResult')
    
    response = VoiceResponse()
    
    if call_sid in active_calls:
        handler = active_calls[call_sid]
        
        # Get LLM response
        llm_response = await handler.get_llm_response(speech_result)
        
        # Generate speech from response
        audio_file = await handler.generate_speech(llm_response)
        
        if audio_file:
            # Play response and gather next input
            gather = Gather(
                input='speech',
                action='/handle_speech',
                method='POST',
                speechTimeout='auto',
                enhanced='true'
            )
            gather.play(audio_file)
            response.append(gather)
        else:
            response.say("I apologize, but I'm having trouble generating the response.")
    else:
        response.say("I apologize, but I've lost track of our conversation. Please call back.")
        
    return str(response)

@app.post("/call_status")
async def handle_call_status(request: Request):
    """Handle call status updates"""
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    status = form_data.get('CallStatus')
    
    if status in ['completed', 'failed', 'busy', 'no-answer'] and call_sid in active_calls:
        handler = active_calls[call_sid]
        await handler.cleanup()
        del active_calls[call_sid]
        
    return {"status": "success"}

def update_webhook_url(new_url):
    """Update the webhook URL in Twilio"""
    # Update the webhook URL for your Twilio phone number
    incoming_phone_number = twilio_client.incoming_phone_numbers(TWILIO_PHONE_NUMBER).fetch()
    incoming_phone_number.update(
        voice_url=new_url,
        voice_method='POST'
    )
    
    # Also update the environment variable
    os.environ['TWILIO_WEBHOOK_URL'] = new_url
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 