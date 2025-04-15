from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig
import soundfile as sf
import os
from pathlib import Path
import uuid
import logging
import json
import asyncio
from ..websocket_manager import manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Receptionist TTS Service")

# Initialize TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = None

def init_tts():
    global tts
    try:
        logger.info(f"Initializing TTS model on device: {device}")
        if device == "cuda":
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA current device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(device)
        logger.info("TTS model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing TTS model: {str(e)}")
        return False

# Initialize model on startup
init_tts()

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speaker_wav: str = None  # Optional: path to speaker reference audio

class TTSResponse(BaseModel):
    audio_path: str
    duration: float

@app.websocket("/ws/tts/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id, "tts")
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "synthesize":
                try:
                    if tts is None:
                        if not init_tts():
                            await websocket.send_json({
                                "type": "error",
                                "message": "TTS model initialization failed"
                            })
                            continue
                    
                    # Create output directory if it doesn't exist
                    output_dir = Path("models/generated_audio")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate unique filename
                    output_file = output_dir / f"{uuid.uuid4()}.wav"
                    
                    # Generate speech
                    tts.tts_to_file(
                        text=data["text"],
                        file_path=str(output_file),
                        language=data.get("language", "en")
                    )
                    
                    # Get audio duration and read file
                    audio_data, sample_rate = sf.read(str(output_file))
                    duration = len(audio_data) / sample_rate
                    
                    # Send audio data in chunks
                    chunk_size = 32768  # 32KB chunks
                    with open(output_file, 'rb') as f:
                        while chunk := f.read(chunk_size):
                            await websocket.send_bytes(chunk)
                    
                    # Send completion message
                    await websocket.send_json({
                        "type": "synthesis_complete",
                        "duration": duration
                    })
                    
                    # Clean up the file
                    os.remove(output_file)
                    
                except Exception as e:
                    logger.error(f"Error in speech synthesis: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
    except WebSocketDisconnect:
        manager.disconnect(client_id, "tts")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        manager.disconnect(client_id, "tts")

@app.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    try:
        if tts is None:
            if not init_tts():
                raise HTTPException(
                    status_code=503,
                    detail="TTS model is not initialized and initialization failed"
                )
        
        # Create output directory if it doesn't exist
        output_dir = Path("models/generated_audio")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        output_file = output_dir / f"{uuid.uuid4()}.wav"
        
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        
        # Generate speech
        if request.speaker_wav:
            tts.tts_to_file(
                text=request.text,
                file_path=str(output_file),
                speaker_wav=request.speaker_wav,
                language=request.language
            )
        else:
            tts.tts_to_file(
                text=request.text,
                file_path=str(output_file),
                language=request.language
            )
        
        # Get audio duration
        audio_data, sample_rate = sf.read(str(output_file))
        duration = len(audio_data) / sample_rate
        
        logger.info(f"Speech generated successfully, duration: {duration:.2f}s")
        
        return TTSResponse(
            audio_path=str(output_file),
            duration=duration
        )
    
    except Exception as e:
        logger.error(f"Error in speech synthesis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if tts is not None else "initializing",
        "device": device,
        "cuda_available": torch.cuda.is_available() if device == "cuda" else False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 