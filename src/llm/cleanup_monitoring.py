from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func
from .models import Conversation, Message, ConversationSummary, ConversationStatus

logger = logging.getLogger(__name__)

class CleanupMetrics:
    def __init__(self, db: Session):
        self.db = db
        
    def get_conversation_metrics(self) -> Dict[str, Any]:
        """Get metrics about conversation states and volumes."""
        try:
            # Count conversations by status
            status_counts = (
                self.db.query(
                    Conversation.status,
                    func.count(Conversation.id).label('count')
                )
                .group_by(Conversation.status)
                .all()
            )
            
            # Calculate average messages per conversation
            avg_messages = (
                self.db.query(func.avg(
                    self.db.query(func.count(Message.id))
                    .filter(Message.conversation_id == Conversation.id)
                    .correlate(Conversation)
                    .as_scalar()
                ))
                .scalar() or 0
            )
            
            # Get summary statistics
            summary_stats = (
                self.db.query(
                    func.count(ConversationSummary.id).label('total_summaries'),
                    func.avg(func.length(ConversationSummary.summary_text)).label('avg_summary_length')
                )
                .first()
            )
            
            return {
                "conversation_counts": {
                    status.value: count for status, count in status_counts
                },
                "avg_messages_per_conversation": round(float(avg_messages), 2),
                "summary_stats": {
                    "total_summaries": summary_stats[0] or 0,
                    "avg_summary_length": round(float(summary_stats[1] or 0), 2)
                }
            }
        except Exception as e:
            logger.error(f"Error getting conversation metrics: {str(e)}", exc_info=True)
            return {}

    def get_cleanup_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics about cleanup performance over the last N hours."""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            # Get conversations marked inactive
            inactive_count = (
                self.db.query(func.count(Conversation.id))
                .filter(
                    Conversation.status == ConversationStatus.INACTIVE,
                    Conversation.updated_at >= since
                )
                .scalar() or 0
            )
            
            # Get archived conversations
            archived_count = (
                self.db.query(func.count(Conversation.id))
                .filter(
                    Conversation.status == ConversationStatus.ARCHIVED,
                    Conversation.updated_at >= since
                )
                .scalar() or 0
            )
            
            # Get summarization stats
            summarized_messages = (
                self.db.query(func.count(Message.id))
                .filter(
                    Message.is_summarized == True,
                    Message.timestamp >= since
                )
                .scalar() or 0
            )
            
            return {
                "period_hours": hours,
                "conversations_processed": {
                    "marked_inactive": inactive_count,
                    "archived": archived_count
                },
                "messages_summarized": summarized_messages
            }
        except Exception as e:
            logger.error(f"Error getting cleanup performance metrics: {str(e)}", exc_info=True)
            return {}
    
    def get_memory_efficiency(self) -> Dict[str, Any]:
        """Calculate memory efficiency metrics."""
        try:
            # Get total vs summarized messages
            total_messages = (
                self.db.query(func.count(Message.id))
                .scalar() or 0
            )
            
            summarized_messages = (
                self.db.query(func.count(Message.id))
                .filter(Message.is_summarized == True)
                .scalar() or 0
            )
            
            # Calculate compression ratios
            if total_messages > 0:
                compression_ratio = summarized_messages / total_messages
            else:
                compression_ratio = 0
                
            return {
                "total_messages": total_messages,
                "summarized_messages": summarized_messages,
                "compression_ratio": round(compression_ratio, 2),
                "memory_saved_percentage": round(compression_ratio * 100, 1)
            }
        except Exception as e:
            logger.error(f"Error getting memory efficiency metrics: {str(e)}", exc_info=True)
            return {}

def get_cleanup_report() -> Dict[str, Any]:
    """Generate a comprehensive cleanup monitoring report."""
    from .db_ops import get_db
    
    with get_db() as db:
        metrics = CleanupMetrics(db)
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "conversation_metrics": metrics.get_conversation_metrics(),
            "cleanup_performance_24h": metrics.get_cleanup_performance(24),
            "cleanup_performance_7d": metrics.get_cleanup_performance(168),  # 7 days
            "memory_efficiency": metrics.get_memory_efficiency()
        }
        
        logger.info("Cleanup monitoring report generated")
        logger.debug(f"Report details: {json.dumps(report, indent=2)}")
        
        return report 