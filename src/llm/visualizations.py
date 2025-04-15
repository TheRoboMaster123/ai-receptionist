from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
import pandas as pd
import numpy as np

from .models import Conversation, Message, Business, ConversationStatus
from .analytics import AnalyticsManager

class VisualizationManager:
    def __init__(self, db: Session):
        self.db = db
        self.analytics = AnalyticsManager(db)
    
    async def get_sentiment_timeline(
        self,
        business_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = 'day'
    ) -> Dict[str, Any]:
        """Get sentiment data formatted for timeline visualization."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get all messages with their sentiment scores
        messages = self.db.query(
            Message.timestamp,
            Message.metadata['sentiment_score'].label('sentiment_score'),
            Message.role
        ).join(Conversation).filter(
            and_(
                Conversation.business_id == business_id,
                Message.timestamp >= start_date,
                Message.timestamp <= end_date
            )
        ).order_by(Message.timestamp).all()
        
        # Convert to pandas for easier time-based aggregation
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'sentiment_score': float(m.sentiment_score or 0),
                'role': m.role
            }
            for m in messages
        ])
        
        if df.empty:
            return {"labels": [], "datasets": []}
        
        # Resample based on interval
        if interval == 'hour':
            df['time_group'] = df['timestamp'].dt.floor('H')
        elif interval == 'day':
            df['time_group'] = df['timestamp'].dt.date
        elif interval == 'week':
            df['time_group'] = df['timestamp'].dt.isocalendar().week
        else:
            df['time_group'] = df['timestamp'].dt.date
        
        # Calculate average sentiment per time period
        sentiment_by_time = df.groupby('time_group').agg({
            'sentiment_score': 'mean'
        }).reset_index()
        
        return {
            "type": "line",
            "data": {
                "labels": sentiment_by_time['time_group'].astype(str).tolist(),
                "datasets": [{
                    "label": "Average Sentiment",
                    "data": sentiment_by_time['sentiment_score'].round(2).tolist(),
                    "borderColor": "rgb(75, 192, 192)",
                    "tension": 0.1
                }]
            }
        }
    
    async def get_topic_distribution_chart(
        self,
        business_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get topic distribution data formatted for pie/bar charts."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get topic counts
        topics = self.db.query(
            Conversation.primary_topic,
            func.count(Conversation.id).label('count')
        ).filter(
            and_(
                Conversation.business_id == business_id,
                Conversation.started_at >= start_date,
                Conversation.started_at <= end_date
            )
        ).group_by(Conversation.primary_topic).all()
        
        # Generate colors for each topic
        colors = [
            f"hsl({(i * 360) // len(topics)}, 70%, 50%)"
            for i in range(len(topics))
        ]
        
        return {
            "type": "pie",
            "data": {
                "labels": [t.primary_topic for t in topics],
                "datasets": [{
                    "data": [t.count for t in topics],
                    "backgroundColor": colors
                }]
            }
        }
    
    async def get_conversation_volume_chart(
        self,
        business_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = 'day'
    ) -> Dict[str, Any]:
        """Get conversation volume data formatted for bar chart."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get all conversations
        conversations = self.db.query(
            Conversation.started_at,
            Conversation.status
        ).filter(
            and_(
                Conversation.business_id == business_id,
                Conversation.started_at >= start_date,
                Conversation.started_at <= end_date
            )
        ).all()
        
        # Convert to pandas for time-based analysis
        df = pd.DataFrame([
            {
                'started_at': c.started_at,
                'status': c.status
            }
            for c in conversations
        ])
        
        if df.empty:
            return {"labels": [], "datasets": []}
        
        # Resample based on interval
        if interval == 'hour':
            df['time_group'] = df['started_at'].dt.floor('H')
        elif interval == 'day':
            df['time_group'] = df['started_at'].dt.date
        elif interval == 'week':
            df['time_group'] = df['started_at'].dt.isocalendar().week
        else:
            df['time_group'] = df['started_at'].dt.date
        
        # Count conversations by time period and status
        volume_by_time = df.groupby(['time_group', 'status']).size().unstack(fill_value=0)
        
        # Prepare datasets for each status
        datasets = []
        colors = {
            'active': 'rgb(54, 162, 235)',
            'completed': 'rgb(75, 192, 192)',
            'pending_followup': 'rgb(255, 206, 86)'
        }
        
        for status in volume_by_time.columns:
            datasets.append({
                "label": status.capitalize(),
                "data": volume_by_time[status].tolist(),
                "backgroundColor": colors.get(status, 'rgb(201, 203, 207)')
            })
        
        return {
            "type": "bar",
            "data": {
                "labels": volume_by_time.index.astype(str).tolist(),
                "datasets": datasets
            }
        }
    
    async def get_response_times_chart(
        self,
        business_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get response time distribution data formatted for histogram."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get message timestamps
        messages = self.db.query(
            Message.conversation_id,
            Message.timestamp,
            Message.role
        ).join(Conversation).filter(
            and_(
                Conversation.business_id == business_id,
                Message.timestamp >= start_date,
                Message.timestamp <= end_date
            )
        ).order_by(Message.conversation_id, Message.timestamp).all()
        
        # Calculate response times
        response_times = []
        prev_message = None
        
        for message in messages:
            if prev_message and \
               prev_message.conversation_id == message.conversation_id and \
               prev_message.role != message.role:
                response_time = (message.timestamp - prev_message.timestamp).total_seconds()
                response_times.append(response_time)
            prev_message = message
        
        if not response_times:
            return {"labels": [], "datasets": []}
        
        # Create histogram bins
        hist, bins = np.histogram(response_times, bins='auto')
        bin_labels = [f"{int(bins[i])}s-{int(bins[i+1])}s" for i in range(len(bins)-1)]
        
        return {
            "type": "bar",
            "data": {
                "labels": bin_labels,
                "datasets": [{
                    "label": "Response Time Distribution",
                    "data": hist.tolist(),
                    "backgroundColor": "rgb(75, 192, 192)"
                }]
            }
        } 