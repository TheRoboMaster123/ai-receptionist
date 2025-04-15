from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import messages_from_dict, messages_to_dict

class ConversationMemoryHandler:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.active_memories: Dict[str, ConversationBufferWindowMemory] = {}
        
    def get_memory(self, conversation_id: str) -> ConversationBufferWindowMemory:
        """Get or create memory for a conversation"""
        if conversation_id not in self.active_memories:
            self.active_memories[conversation_id] = ConversationBufferWindowMemory(
                k=self.max_messages,
                return_messages=True
            )
        return self.active_memories[conversation_id]
    
    def save_memory(
        self,
        conversation_id: str,
        db_session,
        business_id: str
    ) -> None:
        """Save conversation memory to database"""
        if conversation_id in self.active_memories:
            memory = self.active_memories[conversation_id]
            messages = messages_to_dict(memory.chat_memory.messages)
            
            # Update conversation in database
            conversation = db_session.query(Conversation).filter_by(id=conversation_id).first()
            if conversation:
                conversation.messages = messages
                conversation.last_updated = datetime.utcnow()
                db_session.commit()
    
    def load_memory(
        self,
        conversation_id: str,
        db_session,
        business_id: str
    ) -> Optional[ConversationBufferWindowMemory]:
        """Load conversation memory from database"""
        conversation = db_session.query(Conversation).filter_by(id=conversation_id).first()
        if conversation and conversation.messages:
            memory = ConversationBufferWindowMemory(
                k=self.max_messages,
                return_messages=True
            )
            memory.chat_memory.messages = messages_from_dict(conversation.messages)
            self.active_memories[conversation_id] = memory
            return memory
        return None
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to the conversation memory"""
        memory = self.get_memory(conversation_id)
        if role == "human":
            memory.chat_memory.add_user_message(content)
        else:
            memory.chat_memory.add_ai_message(content)
    
    def get_conversation_summary(
        self,
        conversation_id: str,
        max_length: int = 200
    ) -> str:
        """Generate a summary of the conversation"""
        memory = self.get_memory(conversation_id)
        messages = memory.chat_memory.messages
        
        # TODO: Implement better summarization logic
        # For now, just return the last few exchanges
        summary = []
        for msg in messages[-4:]:  # Last 2 exchanges
            summary.append(f"{msg.type}: {msg.content[:50]}...")
        
        return "\n".join(summary)
    
    def clear_memory(self, conversation_id: str) -> None:
        """Clear memory for a conversation"""
        if conversation_id in self.active_memories:
            del self.active_memories[conversation_id]
    
    def get_context_window(
        self,
        conversation_id: str,
        window_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Get the recent context window for a conversation"""
        memory = self.get_memory(conversation_id)
        messages = memory.chat_memory.messages[-window_size:]
        return messages_to_dict(messages)
    
    def update_context(
        self,
        conversation_id: str,
        context_updates: Dict[str, Any]
    ) -> None:
        """Update conversation context with new information"""
        memory = self.get_memory(conversation_id)
        context_msg = f"Context Update: {str(context_updates)}"
        memory.chat_memory.add_system_message(context_msg) 