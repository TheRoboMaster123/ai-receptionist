# AI Receptionist

An intelligent receptionist system powered by Mistral-7B and XTTS-v2, capable of handling multiple concurrent phone calls and providing natural conversations.

## Features

- Natural language conversation using Mistral-7B
- High-quality text-to-speech using XTTS-v2
- Multi-tenant support for different businesses
- Concurrent call handling (up to 8-10 simultaneous calls)
- Business hours management
- Conversation history tracking
- Twilio integration for phone calls

## System Requirements

- NVIDIA GPU with 24GB+ VRAM (optimized for RTX A5000)
- CUDA 12.1+
- Python 3.11+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/AI-Receptionist.git
cd AI-Receptionist
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. Initialize the database:
```bash
alembic upgrade head
```

## Services

The system consists of two main services:

1. LLM Service (Port 8000):
   - Handles natural language processing
   - Manages business logic and conversation flow
   - Provides REST API endpoints

2. TTS Service (Port 8001):
   - Handles text-to-speech conversion
   - Manages audio generation and streaming
   - Provides REST API endpoints

## Configuration

Key configuration files:
- `.env`: Environment variables and API keys
- `src/llm/service.py`: LLM service configuration
- `src/tts/tts_service.py`: TTS service configuration

## Deployment

The system is optimized for deployment on RunPod with an RTX A5000 GPU. See `deployment.md` for detailed deployment instructions.

## API Documentation

Once running, API documentation is available at:
- LLM Service: `http://localhost:8000/docs`
- TTS Service: `http://localhost:8001/docs`

## License

[Your License Here] 