#!/bin/bash

# Start the LLM service
python3 src/llm/service.py &
LLM_PID=$!

# Start the TTS service
python3 src/tts/tts_service.py &
TTS_PID=$!

# Start the dashboard
python3 src/dashboard/app.py &
DASHBOARD_PID=$!

# Function to handle shutdown
function shutdown {
    echo "Shutting down services..."
    kill $LLM_PID
    kill $TTS_PID
    kill $DASHBOARD_PID
    exit 0
}

# Trap SIGTERM and SIGINT
trap shutdown SIGTERM SIGINT

# Keep the container running
while true; do
    sleep 1
done 