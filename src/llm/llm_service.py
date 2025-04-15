from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from pathlib import Path

app = FastAPI(title="AI Receptionist LLM Service")

class BusinessConfig(BaseModel):
    business_id: str
    name: str
    description: str
    business_hours: str
    custom_instructions: Optional[str] = None
    training_data: Optional[List[Dict[str, str]]] = None

class ConversationRequest(BaseModel):
    business_id: str
    user_input: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    context: Optional[Dict[str, str]] = None

class ConversationResponse(BaseModel):
    response: str
    confidence: float
    context_updates: Optional[Dict[str, str]] = None

# Initialize model and tokenizer
MODEL_PATH = "models/llm/meta-llama/Llama-2-7b-chat-hf"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_TEMPERATURE = 0.7

def load_model():
    """Load the LLM model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # Quantization for memory efficiency
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

model, tokenizer = load_model()
if not model or not tokenizer:
    raise RuntimeError("Failed to load model and tokenizer")

def generate_prompt(business_config: BusinessConfig, user_input: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Generate a prompt for the LLM based on business config and conversation history"""
    system_prompt = f"""You are an AI receptionist for {business_config.name}. 
Business Description: {business_config.description}
Business Hours: {business_config.business_hours}
{business_config.custom_instructions or ''}

Your role is to professionally handle customer inquiries, schedule appointments, and provide information about the business.
Always be polite, professional, and helpful while maintaining a natural conversation flow.
"""
    
    # Add conversation history
    conversation = ""
    if conversation_history:
        for msg in conversation_history[-5:]:  # Keep last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation += f"{role}: {content}\n"
    
    # Add current user input
    conversation += f"User: {user_input}\nAssistant:"
    
    return f"{system_prompt}\n\nConversation:\n{conversation}"

@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest) -> ConversationResponse:
    """Handle a conversation turn with the AI receptionist"""
    try:
        # TODO: Load business config from database
        # For now, use dummy config
        business_config = BusinessConfig(
            business_id=request.business_id,
            name="Example Business",
            description="A professional services company",
            business_hours="Monday-Friday 9AM-5PM",
        )
        
        # Generate prompt
        prompt = generate_prompt(
            business_config,
            request.user_input,
            request.conversation_history
        )
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=DEFAULT_MAX_LENGTH,
            temperature=DEFAULT_TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response from the full output
        response = response_text.split("Assistant:")[-1].strip()
        
        return ConversationResponse(
            response=response,
            confidence=0.95,  # TODO: Implement proper confidence scoring
            context_updates={}  # TODO: Implement context tracking
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/business/configure")
async def configure_business(config: BusinessConfig):
    """Configure or update a business profile"""
    try:
        # TODO: Implement business configuration storage
        # For now, just return success
        return {"status": "success", "message": "Business configured successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the service is healthy"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None,
        "device": str(model.device) if model else "unknown"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port than TTS service 