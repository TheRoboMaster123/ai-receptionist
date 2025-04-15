from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from langchain.schema import messages_from_dict, messages_to_dict
from .models import Conversation, Message, ConversationSummary, ConversationStatus

class MemoryManager:
    def __init__(self, db: Session):
        self.db = db
        self.summary_threshold = 10  # Number of messages before creating a new summary
    
    async def update_conversation_metadata(
        self,
        conversation_id: str,
        message_content: str,
        role: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update conversation metadata based on new message content."""
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            return
        
        # Update message count
        conversation.message_count += 1
        
        # Update metadata
        if metadata:
            current_metadata = conversation.metadata or {}
            current_metadata.update(metadata)
            conversation.metadata = current_metadata
        
        # Update last message timestamp
        conversation.last_message_at = datetime.utcnow()
        
        self.db.commit()
    
    async def should_create_summary(self, conversation_id: str) -> bool:
        """Determine if a new summary should be created."""
        # Get unsummarized messages
        unsummarized_count = self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_summarized == False
        ).count()
        
        return unsummarized_count >= self.summary_threshold
    
    async def create_conversation_summary(
        self,
        conversation_id: str,
        llm_chain: Any  # LangChain chain for generating summaries
    ) -> Optional[str]:
        """Create a new summary for unsummarized messages."""
        # Get unsummarized messages
        messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_summarized == False
        ).order_by(Message.timestamp).all()
        
        if not messages:
            return None
        
        # Prepare messages for summarization
        message_texts = []
        for msg in messages:
            prefix = "Customer" if msg.role == "user" else "AI Receptionist"
            message_texts.append(f"{prefix}: {msg.content}")
        
        # Generate summary using LLM
        summary_prompt = f"""Please provide a concise summary of the following conversation segment, 
        focusing on key points, decisions, and any action items:

        {'\n'.join(message_texts)}
        
        Summary:"""
        
        summary = await llm_chain.apredict(input=summary_prompt)
        
        # Create new summary record
        new_summary = ConversationSummary(
            conversation_id=conversation_id,
            summary_text=summary,
            message_start_id=messages[0].id,
            message_end_id=messages[-1].id
        )
        self.db.add(new_summary)
        
        # Update conversation's current summary
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            conversation.current_summary = summary
            conversation.summary_updated_at = datetime.utcnow()
        
        # Mark messages as summarized
        for message in messages:
            message.is_summarized = True
        
        self.db.commit()
        return summary
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: int = 10
    ) -> Dict[str, Any]:
        """Get the current conversation context including summary and recent messages."""
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            return {}
        
        # Get recent messages
        recent_messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp.desc()).limit(max_messages).all()
        
        # Format context
        context = {
            "summary": conversation.current_summary,
            "primary_topic": conversation.primary_topic,
            "subtopics": conversation.subtopics,
            "metadata": conversation.metadata,
            "recent_messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in reversed(recent_messages)
            ]
        }
        
        return context
    
    async def update_conversation_status(
        self,
        conversation_id: str,
        status: ConversationStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update conversation status and metadata."""
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            conversation.status = status
            if metadata:
                current_metadata = conversation.metadata or {}
                current_metadata.update(metadata)
                conversation.metadata = current_metadata
            self.db.commit() 