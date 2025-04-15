import requests
import os
from pathlib import Path

def test_tts_service():
    # Test basic synthesis
    response = requests.post(
        "http://localhost:8000/synthesize",
        json={
            "text": "Hello, this is a test of the AI receptionist system.",
            "language": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "audio_path" in data
    assert "duration" in data
    
    # Verify audio file exists
    audio_path = Path(data["audio_path"])
    assert audio_path.exists()
    
    # Test health endpoint
    health_response = requests.get("http://localhost:8000/health")
    assert health_response.status_code == 200
    health_data = health_response.json()
    assert "status" in health_data
    assert health_data["status"] == "healthy"

if __name__ == "__main__":
    test_tts_service()
    print("All tests passed!") 