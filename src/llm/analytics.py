from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from transformers import pipeline
import numpy as np
from collections import Counter

from .models import Conversation, Message, Business, ConversationStatus

class AnalyticsManager:
    def __init__(self, db: Session):
        self.db = db
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=1
        )
        # Initialize zero-shot classifier for topics
        self.topic_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Common business topics
        self.default_topics = [
            "Appointment Scheduling",
            "Business Hours",
            "Pricing Information",
            "Service Information",
            "Contact Information",
            "Complaints/Issues",
            "General Inquiry",
            "Technical Support",
            "Feedback",
            "Emergency"
        ]
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a message."""
        result = self.sentiment_analyzer(text)[0]
        return {
            "label": result["label"].lower(),
            "score": float(result["score"])
        }
    
    async def detect_topics(self, text: str, custom_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect topics in a message using zero-shot classification."""
        topics = custom_topics if custom_topics else self.default_topics
        result = self.topic_classifier(text, topics, multi_label=True)
        
        # Filter topics with confidence > 0.3
        detected_topics = [
            {"topic": label, "confidence": score}
            for label, score in zip(result["labels"], result["scores"])
            if score > 0.3
        ]
        
        return {
            "primary_topic": detected_topics[0]["topic"] if detected_topics else "General Inquiry",
            "subtopics": detected_topics[1:],
            "confidence_scores": {t["topic"]: t["confidence"] for t in detected_topics}
        }
    
    async def get_business_analytics(
        self,
        business_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive analytics for a business."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)  # Default to last 30 days
        if not end_date:
            end_date = datetime.utcnow()
        
        # Query base
        base_query = self.db.query(Conversation).filter(
            and_(
                Conversation.business_id == business_id,
                Conversation.started_at >= start_date,
                Conversation.started_at <= end_date
            )
        )
        
        # Get conversation metrics
        total_conversations = base_query.count()
        completed_conversations = base_query.filter(
            Conversation.status == ConversationStatus.COMPLETED
        ).count()
        
        # Get message metrics
        message_stats = self.db.query(
            func.count(Message.id).label('total_messages'),
            func.avg(func.json_extract(Message.metadata, '$.sentiment_score')).label('avg_sentiment'),
            func.count(func.distinct(Message.conversation_id)).label('conversations_with_messages')
        ).join(Conversation).filter(
            and_(
                Conversation.business_id == business_id,
                Message.timestamp >= start_date,
                Message.timestamp <= end_date
            )
        ).first()
        
        # Get topic distribution
        topics_query = self.db.query(
            Conversation.primary_topic,
            func.count(Conversation.id).label('count')
        ).filter(
            and_(
                Conversation.business_id == business_id,
                Conversation.started_at >= start_date,
                Conversation.started_at <= end_date
            )
        ).group_by(Conversation.primary_topic).all()
        
        topic_distribution = {topic: count for topic, count in topics_query}
        
        # Get sentiment distribution
        sentiment_distribution = self.db.query(
            Message.sentiment,
            func.count(Message.id).label('count')
        ).join(Conversation).filter(
            and_(
                Conversation.business_id == business_id,
                Message.timestamp >= start_date,
                Message.timestamp <= end_date
            )
        ).group_by(Message.sentiment).all()
        
        # Calculate response times
        response_times = self.db.query(
            func.avg(
                func.strftime('%s', Message.timestamp) -
                func.strftime('%s', func.lag(Message.timestamp).over(
                    partition_by=Message.conversation_id,
                    order_by=Message.timestamp
                ))
            )
        ).join(Conversation).filter(
            and_(
                Conversation.business_id == business_id,
                Message.role == 'assistant',
                Message.timestamp >= start_date,
                Message.timestamp <= end_date
            )
        ).scalar()
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "conversation_metrics": {
                "total": total_conversations,
                "completed": completed_conversations,
                "completion_rate": (completed_conversations / total_conversations) if total_conversations > 0 else 0
            },
            "message_metrics": {
                "total_messages": message_stats[0] if message_stats else 0,
                "average_sentiment_score": float(message_stats[1]) if message_stats and message_stats[1] else 0,
                "messages_per_conversation": (message_stats[0] / message_stats[2]) if message_stats and message_stats[2] > 0 else 0
            },
            "topic_distribution": topic_distribution,
            "sentiment_distribution": dict(sentiment_distribution),
            "average_response_time_seconds": float(response_times) if response_times else 0
        }
    
    async def get_conversation_analytics(self, conversation_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a specific conversation."""
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            return {}
        
        messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp).all()
        
        # Calculate message statistics
        message_count = len(messages)
        user_messages = sum(1 for m in messages if m.role == "user")
        assistant_messages = sum(1 for m in messages if m.role == "assistant")
        
        # Calculate sentiment progression
        sentiment_progression = [
            {
                "timestamp": m.timestamp.isoformat(),
                "sentiment": m.sentiment,
                "sentiment_score": m.metadata.get("sentiment_score", 0) if m.metadata else 0
            }
            for m in messages
        ]
        
        # Get topic changes
        topic_changes = [
            {
                "timestamp": m.timestamp.isoformat(),
                "detected_topics": m.metadata.get("detected_topics", []) if m.metadata else []
            }
            for m in messages
        ]
        
        return {
            "conversation_metrics": {
                "total_messages": message_count,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "duration_seconds": (messages[-1].timestamp - messages[0].timestamp).total_seconds() if messages else 0
            },
            "sentiment_analysis": {
                "overall_sentiment": conversation.metadata.get("overall_sentiment", "neutral"),
                "sentiment_progression": sentiment_progression
            },
            "topic_analysis": {
                "primary_topic": conversation.primary_topic,
                "subtopics": conversation.subtopics,
                "topic_changes": topic_changes
            },
            "status": conversation.status,
            "context": conversation.metadata
        } 