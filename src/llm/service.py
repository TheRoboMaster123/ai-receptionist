from fastapi import FastAPI, HTTPException, Depends, Query, status, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
from pathlib import Path
import os
from datetime import datetime, timedelta
from collections import Counter
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer
)

from .models import Business, Conversation, Message, ConversationStatus
from .db_ops import (
    get_db,
    get_business,
    create_or_update_business,
    create_conversation,
    get_conversation,
    add_message,
    get_conversation_messages,
    get_business_conversations
)
from .chains.base_chain import BusinessConversationChain
from .memory.memory_handler import ConversationMemoryHandler
from .agents.business_agent import BusinessAgent
from .analytics import AnalyticsManager
from .visualizations import VisualizationManager
from .business_hours import BusinessHoursManager
from .background_tasks import start_background_tasks
from .cleanup_monitoring import CleanupMetrics, get_cleanup_report
from .auth import (
    Token, User, authenticate_user, create_access_token,
    get_current_admin, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .rate_limiter import rate_limit_metrics, metrics_limiter
from .resource_manager import resource_manager
from .model_manager import ModelManager

app = FastAPI(title="AI Receptionist Multi-Tenant LLM Service")

# Model configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_CACHE_DIR = "models/mistral"
USE_4BIT = False  # With 24GB VRAM, we can use 8-bit for better quality
USE_8BIT = True   # Use 8-bit quantization for optimal quality/memory trade-off
DEVICE_MAP = "cuda:0"  # Use GPU explicitly
MAX_LENGTH = 2048
MAX_CONTEXT_WINDOW = 4096
TEMPERATURE = 0.7
MAX_MEMORY_MESSAGES = 15  # Increased from 10 due to more available memory

# Concurrent processing settings
MAX_CONCURRENT_CALLS = 8  # Maximum number of concurrent calls to handle
BATCH_SIZE = 4  # Batch size for concurrent processing

# Pydantic models for request/response
class BusinessProfile(BaseModel):
    business_id: str
    name: str
    description: Optional[str] = None
    business_hours: Dict[str, Any]
    business_metadata: Optional[Dict[str, Any]] = {}
    prompt_template: Optional[str] = None

class ConversationRequest(BaseModel):
    business_id: str
    user_input: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ConversationResponse(BaseModel):
    response: str
    conversation_id: str
    context_updates: Optional[Dict[str, Any]] = None

class BusinessHoursResponse(BaseModel):
    is_open: bool
    message: str
    current_day: str
    current_day_hours: Optional[Dict[str, str]] = None
    timezone: str
    next_opening_time: Optional[str] = None
    upcoming_special_dates: Optional[List[Dict[str, Any]]] = None
    holidays: Optional[List[Dict[str, Any]]] = None
    special_hours: Optional[List[Dict[str, Any]]] = None

# Check for Hugging Face token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    logger.warning("HUGGINGFACE_TOKEN not set. Some models may not be accessible.")

# Initialize model manager with token
model_manager = ModelManager(
    model_id=MODEL_ID,
    cache_dir=MODEL_CACHE_DIR,
    use_4bit=USE_4BIT,
    use_8bit=USE_8BIT,
    device_map=DEVICE_MAP,
    token=HF_TOKEN
)
memory_handler = ConversationMemoryHandler(max_messages=MAX_MEMORY_MESSAGES)
business_chains: Dict[str, BusinessConversationChain] = {}
business_agents: Dict[str, BusinessAgent] = {}

def load_model():
    """Initialize the Mistral model"""
    model_manager.initialize_model()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize model
        load_model()
        
        # Start background tasks
        await start_background_tasks()
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        # Cleanup model resources
        model_manager.cleanup()
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the model"""
    return model_manager.get_model_info()

@app.post("/model/reload")
async def reload_model():
    """Force reload the model"""
    try:
        model_manager.initialize_model(force_reload=True)
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading model: {str(e)}"
        )

@app.delete("/model/cache")
async def clear_model_cache():
    """Clear model cache"""
    try:
        model_manager.clear_cache()
        return {"status": "success", "message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )

def get_business_chain(
    db: Session,
    business_profile: BusinessProfile,
    conversation_id: Optional[str] = None
) -> BusinessConversationChain:
    """Create or retrieve a conversation chain for a business"""
    chain_key = f"{business_profile.business_id}:{conversation_id}" if conversation_id else business_profile.business_id
    
    if chain_key not in business_chains:
        business_chains[chain_key] = BusinessConversationChain(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            business_profile=business_profile.dict(),
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            max_memory_messages=MAX_MEMORY_MESSAGES
        )
    
    return business_chains[chain_key]

def get_business_agent(
    db: Session,
    business_profile: BusinessProfile
) -> BusinessAgent:
    """Create or retrieve a business agent"""
    if business_profile.business_id not in business_agents:
        pipe = pipeline(
            "text-generation",
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        business_agents[business_profile.business_id] = BusinessAgent(
            business_profile=business_profile.dict(),
            llm=pipe
        )
    
    return business_agents[business_profile.business_id]

@app.post("/chat", response_model=ConversationResponse)
async def chat(
    request: ConversationRequest,
    db: Session = Depends(get_db)
):
    """Handle a conversation turn with the AI receptionist"""
    start_time = time.time()
    initial_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    try:
        # Check resource availability
        resources_available, message = await resource_manager.check_resources()
        if not resources_available:
            raise HTTPException(
                status_code=503,
                detail=f"Service temporarily unavailable: {message}"
            )
        
        # Get business profile
        business = get_business(db, request.business_id)
        if not business:
            raise HTTPException(
                status_code=404,
                detail="Business not found"
            )
        
        business_profile = BusinessProfile(**business.__dict__)
        
        # Create new conversation if needed
        if not request.conversation_id:
            conversation = create_conversation(
                db,
                business_id=request.business_id,
                conversation_metadata=request.context if request.context else {}
            )
        else:
            conversation = get_conversation(db, request.conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Load conversation memory
        memory_handler.load_memory(conversation.id, db, request.business_id)
        
        # Get business chain and agent
        chain = get_business_chain(db, business_profile, conversation.id)
        agent = get_business_agent(db, business_profile)
        
        # Prepare context
        context_dict = request.context if request.context else {}
        
        # Generate response
        response = chain.generate_response(
            user_input=request.user_input,
            **context_dict
        )
        
        # Use agent's response as additional context for the chain
        additional_context = {}
        if request.context:
            additional_context.update(request.context)
        additional_context["agent_actions"] = agent_response["actions"]
        
        chain_response = await chain.process_message(
            request.user_input,
            additional_context
        )
        
        # Combine responses
        response = chain_response["response"]
        context_updates = {
            **chain_response.get("context_updates", {}),
            "agent_actions": agent_response["actions"]
        }
        
        # Add messages to memory and database
        memory_handler.add_message(conversation.id, "human", request.user_input)
        memory_handler.add_message(conversation.id, "ai", response)
        
        add_message(
            db,
            conversation.id,
            "human",
            request.user_input,
            request.context
        )
        add_message(
            db,
            conversation.id,
            "ai",
            response,
            context_updates
        )
        
        # Save memory to database
        memory_handler.save_memory(conversation.id, db, request.business_id)
        
        # Generate conversation summary
        summary = memory_handler.get_conversation_summary(conversation.id)
        
        return ConversationResponse(
            response=response,
            conversation_id=conversation.id,
            context_updates=context_updates,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
    finally:
        # Log resource usage
        end_time = time.time()
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_used = final_gpu_memory - initial_gpu_memory
            print(f"GPU Memory Used: {gpu_memory_used / 1024**2:.2f} MB")
        print(f"Processing Time: {end_time - start_time:.2f} seconds")

@app.post("/configure_business")
async def configure_business(
    business: BusinessProfile,
    db = Depends(get_db)
):
    """Configure or update a business profile"""
    try:
        # Validate business hours format
        hours_manager = BusinessHoursManager()
        if not hours_manager.validate_business_hours(business.business_hours):
            raise HTTPException(
                status_code=400,
                detail="Invalid business hours format. Must include regular_hours and optionally holidays and special_hours"
            )

        # Create or update business profile
        result = create_or_update_business(
            db,
            business.business_id,
            business.name,
            business.description,
            business.business_hours,
            business.business_metadata,
            business.prompt_template
        )
        
        return {"status": "success", "business": result}
    except Exception as e:
        logger.error(f"Error configuring business: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/analytics/business/{business_id}")
async def get_business_analytics(
    business_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db = Depends(get_db)
):
    """Get analytics for a specific business."""
    try:
        analytics_manager = AnalyticsManager(db)
        return await analytics_manager.get_business_analytics(
            business_id,
            start_date,
            end_date
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting business analytics: {str(e)}"
        )

@app.get("/analytics/conversation/{conversation_id}")
async def get_conversation_analytics(
    conversation_id: str,
    db = Depends(get_db)
):
    """Get detailed analytics for a specific conversation."""
    try:
        analytics_manager = AnalyticsManager(db)
        analytics = await analytics_manager.get_conversation_analytics(conversation_id)
        if not analytics:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return analytics
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting conversation analytics: {str(e)}"
        )

@app.get("/analytics/business/{business_id}/topics")
async def get_business_topics(
    business_id: str,
    db = Depends(get_db)
):
    """Get topic distribution and trends for a business."""
    try:
        # Get business
        business = get_business(db, business_id)
        if not business:
            raise HTTPException(status_code=404, detail="Business not found")
        
        # Get conversations from the last 30 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        conversations = self.db.query(Conversation).filter(
            and_(
                Conversation.business_id == business_id,
                Conversation.started_at >= start_date,
                Conversation.started_at <= end_date
            )
        ).all()
        
        # Aggregate topics
        all_topics = []
        topic_trends = {}
        
        for conv in conversations:
            if conv.primary_topic:
                all_topics.append(conv.primary_topic)
                
                # Track topic by date
                date_key = conv.started_at.date().isoformat()
                if date_key not in topic_trends:
                    topic_trends[date_key] = Counter()
                topic_trends[date_key][conv.primary_topic] += 1
        
        # Calculate overall distribution
        topic_distribution = Counter(all_topics)
        
        return {
            "overall_distribution": dict(topic_distribution),
            "daily_trends": {
                date: dict(topics)
                for date, topics in topic_trends.items()
            },
            "top_topics": dict(topic_distribution.most_common(5))
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting business topics: {str(e)}"
        )

@app.get("/visualizations/business/{business_id}/sentiment")
async def get_sentiment_chart(
    business_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    interval: str = Query('day', enum=['hour', 'day', 'week']),
    db = Depends(get_db)
):
    """Get sentiment timeline visualization data."""
    try:
        viz_manager = VisualizationManager(db)
        return await viz_manager.get_sentiment_timeline(
            business_id,
            start_date,
            end_date,
            interval
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating sentiment visualization: {str(e)}"
        )

@app.get("/visualizations/business/{business_id}/topics")
async def get_topic_chart(
    business_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db = Depends(get_db)
):
    """Get topic distribution visualization data."""
    try:
        viz_manager = VisualizationManager(db)
        return await viz_manager.get_topic_distribution_chart(
            business_id,
            start_date,
            end_date
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating topic visualization: {str(e)}"
        )

@app.get("/visualizations/business/{business_id}/volume")
async def get_volume_chart(
    business_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    interval: str = Query('day', enum=['hour', 'day', 'week']),
    db = Depends(get_db)
):
    """Get conversation volume visualization data."""
    try:
        viz_manager = VisualizationManager(db)
        return await viz_manager.get_conversation_volume_chart(
            business_id,
            start_date,
            end_date,
            interval
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating volume visualization: {str(e)}"
        )

@app.get("/visualizations/business/{business_id}/response-times")
async def get_response_times_chart(
    business_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db = Depends(get_db)
):
    """Get response time distribution visualization data."""
    try:
        viz_manager = VisualizationManager(db)
        return await viz_manager.get_response_times_chart(
            business_id,
            start_date,
            end_date
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response time visualization: {str(e)}"
        )

@app.get("/business_hours/{business_id}")
async def get_business_hours(
    business_id: str,
    db = Depends(get_db)
):
    """Get current business hours status and upcoming special dates"""
    try:
        business = get_business(db, business_id)
        if not business:
            raise HTTPException(status_code=404, detail="Business not found")

        hours_manager = BusinessHoursManager()
        status = hours_manager.get_business_status(
            business.business_hours,
            business.business_metadata.get("timezone") if business.business_metadata else None
        )

        next_opening = hours_manager.get_next_opening_time(
            business.business_hours,
            business.business_metadata.get("timezone") if business.business_metadata else None
        ) if not status['is_open'] else None

        return BusinessHoursResponse(
            is_open=status['is_open'],
            message=status['message'],
            current_day=status['current_day'],
            current_day_hours=status['current_day_hours'],
            timezone=business.business_metadata.get("timezone", "UTC"),
            next_opening_time=next_opening['message'] if next_opening else None,
            upcoming_special_dates=status.get('upcoming_special_dates'),
            holidays=business.business_hours.get('holidays'),
            special_hours=business.business_hours.get('special_hours')
        )
    except Exception as e:
        logger.error(f"Error getting business hours: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Authentication endpoints
@app.post("/token", response_model=Token, tags=["auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get access token for authentication."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "is_admin": user["is_admin"]},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Protected metrics endpoints
async def check_metrics_rate_limit(request: Request):
    """Dependency to check rate limits for metrics endpoints."""
    response = await rate_limit_metrics(request)
    if response:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.body
        )

@app.get("/metrics/cleanup", tags=["monitoring"])
async def get_cleanup_metrics(
    request: Request,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
    period_hours: Optional[int] = Query(24, description="Period in hours for performance metrics"),
    _: None = Depends(check_metrics_rate_limit)
) -> Dict[str, Any]:
    """
    Get comprehensive cleanup metrics including:
    - Conversation states and volumes
    - Cleanup performance over specified period
    - Memory efficiency metrics
    
    Requires admin authentication.
    Rate limited to 30 requests per minute with burst limit of 5.
    Responses are cached for 60 seconds.
    """
    # Check cache first
    cache_key = f"cleanup_metrics_{period_hours}"
    cached_response = metrics_limiter.get_cached_response(cache_key)
    if cached_response:
        return cached_response
        
    try:
        metrics = CleanupMetrics(db)
        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "conversation_metrics": metrics.get_conversation_metrics(),
            "cleanup_performance": metrics.get_cleanup_performance(period_hours),
            "memory_efficiency": metrics.get_memory_efficiency()
        }
        
        # Cache the response
        metrics_limiter.cache_response(cache_key, response)
        return response
        
    except Exception as e:
        logger.error(f"Error getting cleanup metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/cleanup/report", tags=["monitoring"])
async def get_full_cleanup_report(
    request: Request,
    current_user: User = Depends(get_current_admin),
    _: None = Depends(check_metrics_rate_limit)
) -> Dict[str, Any]:
    """
    Get a comprehensive cleanup report including metrics over multiple time periods
    and detailed performance analysis.
    
    Requires admin authentication.
    Rate limited to 30 requests per minute with burst limit of 5.
    Responses are cached for 60 seconds.
    """
    # Check cache first
    cache_key = "cleanup_report"
    cached_response = metrics_limiter.get_cached_response(cache_key)
    if cached_response:
        return cached_response
        
    try:
        response = get_cleanup_report()
        metrics_limiter.cache_response(cache_key, response)
        return response
    except Exception as e:
        logger.error(f"Error getting cleanup report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/cleanup/conversation_stats", tags=["monitoring"])
async def get_conversation_stats(
    request: Request,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
    _: None = Depends(check_metrics_rate_limit)
) -> Dict[str, Any]:
    """
    Get detailed statistics about conversations including:
    - Counts by status
    - Average messages per conversation
    - Summary statistics
    
    Requires admin authentication.
    Rate limited to 30 requests per minute with burst limit of 5.
    Responses are cached for 60 seconds.
    """
    # Check cache first
    cache_key = "conversation_stats"
    cached_response = metrics_limiter.get_cached_response(cache_key)
    if cached_response:
        return cached_response
        
    try:
        metrics = CleanupMetrics(db)
        response = metrics.get_conversation_metrics()
        metrics_limiter.cache_response(cache_key, response)
        return response
    except Exception as e:
        logger.error(f"Error getting conversation stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/cleanup/memory_efficiency", tags=["monitoring"])
async def get_memory_efficiency(
    request: Request,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
    _: None = Depends(check_metrics_rate_limit)
) -> Dict[str, Any]:
    """
    Get memory efficiency metrics including:
    - Total vs summarized messages
    - Compression ratios
    - Memory saved percentage
    
    Requires admin authentication.
    Rate limited to 30 requests per minute with burst limit of 5.
    Responses are cached for 60 seconds.
    """
    # Check cache first
    cache_key = "memory_efficiency"
    cached_response = metrics_limiter.get_cached_response(cache_key)
    if cached_response:
        return cached_response
        
    try:
        metrics = CleanupMetrics(db)
        response = metrics.get_memory_efficiency()
        metrics_limiter.cache_response(cache_key, response)
        return response
    except Exception as e:
        logger.error(f"Error getting memory efficiency metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/resources", tags=["monitoring"])
async def get_resource_metrics(
    request: Request,
    current_user: User = Depends(get_current_admin),
    minutes: int = Query(5, description="Period in minutes for metrics calculation"),
    _: None = Depends(check_metrics_rate_limit)
) -> Dict[str, Any]:
    """
    Get resource usage metrics including:
    - GPU memory usage
    - CPU usage
    - Memory usage
    - Request statistics
    
    Requires admin authentication.
    Rate limited to 30 requests per minute with burst limit of 5.
    Responses are cached for 60 seconds.
    """
    cache_key = f"resource_metrics_{minutes}"
    cached_response = metrics_limiter.get_cached_response(cache_key)
    if cached_response:
        return cached_response
        
    try:
        response = resource_manager.get_resource_metrics(minutes)
        metrics_limiter.cache_response(cache_key, response)
        return response
    except Exception as e:
        logger.error(f"Error getting resource metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/llm/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    db: Session = Depends(get_db)
):
    await manager.connect(websocket, client_id, "llm")
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "user_input":
                try:
                    # Get or create business chain
                    business_id = data.get("business_id")
                    if not business_id:
                        await websocket.send_json({
                            "type": "error",
                            "message": "business_id is required"
                        })
                        continue
                    
                    # Get business profile
                    business = get_business(db, business_id)
                    if not business:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Business {business_id} not found"
                        })
                        continue
                    
                    # Create business profile model
                    business_profile = BusinessProfile(
                        business_id=business.id,
                        name=business.name,
                        description=business.description,
                        business_hours=business.business_hours,
                        business_metadata=business.metadata,
                        prompt_template=business.prompt_template
                    )
                    
                    # Get conversation chain
                    conversation_id = data.get("conversation_id")
                    chain = get_business_chain(db, business_profile, conversation_id)
                    
                    # Process user input
                    response = await chain.agenerate_response(
                        user_input=data["text"],
                        conversation_id=conversation_id
                    )
                    
                    # Send response back through WebSocket
                    await websocket.send_json({
                        "type": "llm_response",
                        "text": response.response,
                        "conversation_id": response.conversation_id,
                        "context_updates": response.context_updates
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing user input: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
    except WebSocketDisconnect:
        manager.disconnect(client_id, "llm")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        manager.disconnect(client_id, "llm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 