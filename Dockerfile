# Use CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p models/mistral

# Expose ports
EXPOSE 8000 8001 8002

# Set environment variables for the services
ENV LLM_SERVICE_PORT=8000
ENV TTS_SERVICE_PORT=8001
ENV DASHBOARD_PORT=8002

# Start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"] 