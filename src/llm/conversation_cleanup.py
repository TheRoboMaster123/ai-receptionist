from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging
from sqlalchemy import and_
from .models import Conversation, ConversationStatus, Message, ConversationSummary
from .db_ops import get_db
from .summarization import ConversationSummarizer
from .cleanup_monitoring import CleanupMetrics

logger = logging.getLogger(__name__)

class CleanupConfig:
    # Time after which a conversation is considered inactive
    INACTIVE_THRESHOLD = timedelta(hours=24)
    
    # Maximum number of messages to keep in active memory
    MAX_ACTIVE_MESSAGES = 100
    
    # Time after which to archive inactive conversations
    ARCHIVE_THRESHOLD = timedelta(days=7)
    
    # Batch size for cleanup operations
    CLEANUP_BATCH_SIZE = 50

class ConversationCleanup:
    def __init__(self, db: Session):
        self.db = db
        self.summarizer = ConversationSummarizer()
    
    def mark_inactive_conversations(self) -> List[str]:
        """Mark conversations as inactive if they haven't had activity within the threshold."""
        threshold_time = datetime.utcnow() - CleanupConfig.INACTIVE_THRESHOLD
        
        inactive_conversations = (
            self.db.query(Conversation)
            .filter(
                Conversation.status == ConversationStatus.ACTIVE,
                Conversation.last_message_at < threshold_time
            )
            .limit(CleanupConfig.CLEANUP_BATCH_SIZE)
            .all()
        )
        
        conversation_ids = []
        for conv in inactive_conversations:
            conv.status = ConversationStatus.INACTIVE
            conversation_ids.append(conv.id)
            logger.info(f"Marking conversation {conv.id} as inactive")
        
        self.db.commit()
        return conversation_ids
    
    def archive_old_conversations(self) -> List[str]:
        """Archive conversations that have been inactive for longer than the archive threshold."""
        archive_time = datetime.utcnow() - CleanupConfig.ARCHIVE_THRESHOLD
        
        to_archive = (
            self.db.query(Conversation)
            .filter(
                Conversation.status == ConversationStatus.INACTIVE,
                Conversation.last_message_at < archive_time
            )
            .limit(CleanupConfig.CLEANUP_BATCH_SIZE)
            .all()
        )
        
        conversation_ids = []
        for conv in to_archive:
            # Ensure we have a final summary before archiving
            self.cleanup_conversation_memory(conv.id, force_summarize=True)
            conv.status = ConversationStatus.ARCHIVED
            conversation_ids.append(conv.id)
            logger.info(f"Archiving conversation {conv.id}")
            
        self.db.commit()
        return conversation_ids
    
    def cleanup_conversation_memory(self, conversation_id: str, force_summarize: bool = False) -> Optional[int]:
        """Cleanup memory for a specific conversation by summarizing old messages."""
        conversation = (
            self.db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )
        
        if not conversation:
            return None
            
        # Get messages that haven't been summarized yet
        unsummarized_messages = (
            self.db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.is_summarized == False
            )
            .order_by(Message.timestamp)
            .all()
        )
        
        if len(unsummarized_messages) > CleanupConfig.MAX_ACTIVE_MESSAGES or force_summarize:
            messages_to_summarize = (
                unsummarized_messages[:-CleanupConfig.MAX_ACTIVE_MESSAGES]
                if not force_summarize else unsummarized_messages
            )
            
            # Get or create summary
            summary = conversation.summary
            if not summary:
                summary = ConversationSummary(
                    id=f"{conversation_id}_summary",
                    conversation_id=conversation_id
                )
                self.db.add(summary)
            
            try:
                # Update summary using the sophisticated summarizer
                self.summarizer.update_conversation_summary(summary, messages_to_summarize)
                
                # Mark messages as summarized
                for message in messages_to_summarize:
                    message.is_summarized = True
                
                self.db.commit()
                logger.info(
                    f"Summarized {len(messages_to_summarize)} messages "
                    f"for conversation {conversation_id}"
                )
                return len(messages_to_summarize)
                
            except Exception as e:
                logger.error(
                    f"Error summarizing messages for conversation {conversation_id}: {str(e)}",
                    exc_info=True
                )
                self.db.rollback()
                return None
        
        return 0

async def cleanup_inactive_conversations(
    db: Session,
    inactivity_threshold: int = 30,  # minutes
    max_conversations: int = 1000
) -> None:
    """Clean up inactive conversations and generate summaries."""
    # Get inactive conversations
    cutoff_time = datetime.utcnow() - timedelta(minutes=inactivity_threshold)
    inactive_conversations = db.query(Conversation).filter(
        and_(
            Conversation.status == ConversationStatus.ACTIVE,
            Conversation.last_message_at < cutoff_time
        )
    ).all()

    # Initialize summarizer
    summarizer = ConversationSummarizer()

    # Process each inactive conversation
    for conv in inactive_conversations:
        try:
            # Generate final summary
            summarizer.update_conversation_summary(
                db,
                conv.id,
                conv.messages
            )

            # Mark conversation as ended
            conv.status = ConversationStatus.ENDED
            conv.ended_at = datetime.utcnow()

        except Exception as e:
            print(f"Error processing conversation {conv.id}: {str(e)}")
            continue

    # Commit changes
    db.commit()

    # Archive old conversations if needed
    if max_conversations > 0:
        archive_old_conversations(db, max_conversations)

def archive_old_conversations(
    db: Session,
    max_active_conversations: int
) -> None:
    """Archive old conversations when the total exceeds the maximum limit."""
    # Count active and ended conversations
    total_conversations = db.query(Conversation).filter(
        Conversation.status != ConversationStatus.ARCHIVED
    ).count()

    if total_conversations > max_active_conversations:
        # Calculate how many conversations to archive
        to_archive = total_conversations - max_active_conversations

        # Get oldest ended conversations to archive
        old_conversations = db.query(Conversation).filter(
            Conversation.status == ConversationStatus.ENDED
        ).order_by(
            Conversation.ended_at.asc()
        ).limit(to_archive).all()

        # Archive conversations
        for conv in old_conversations:
            conv.status = ConversationStatus.ARCHIVED

        db.commit()

def cleanup_inactive_conversations() -> Dict[str, Any]:
    """Main cleanup function to be called periodically."""
    with get_db() as db:
        cleanup = ConversationCleanup(db)
        metrics = CleanupMetrics(db)
        
        try:
            # Get initial metrics
            initial_metrics = metrics.get_conversation_metrics()
            
            # Mark inactive conversations
            inactive_ids = cleanup.mark_inactive_conversations()
            
            # Archive old conversations
            archived_ids = cleanup.archive_old_conversations()
            
            # Cleanup memory for active conversations
            active_conversations = (
                db.query(Conversation)
                .filter(Conversation.status == ConversationStatus.ACTIVE)
                .all()
            )
            
            processed_count = 0
            for conv in active_conversations:
                if cleanup.cleanup_conversation_memory(conv.id) is not None:
                    processed_count += 1
            
            # Get final metrics
            final_metrics = metrics.get_conversation_metrics()
            memory_efficiency = metrics.get_memory_efficiency()
            
            return {
                "cleanup_results": {
                    "inactive_marked": len(inactive_ids),
                    "archived": len(archived_ids),
                    "active_processed": processed_count
                },
                "metrics_before": initial_metrics,
                "metrics_after": final_metrics,
                "memory_efficiency": memory_efficiency,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during conversation cleanup: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "cleanup_results": {
                    "inactive_marked": 0,
                    "archived": 0,
                    "active_processed": 0
                },
                "timestamp": datetime.utcnow().isoformat()
            } 