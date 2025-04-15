from sqlalchemy import Column, String, JSON, DateTime, ForeignKey, Integer, Text, create_engine, Boolean, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

Base = declarative_base()

class ConversationStatus(str, Enum):
    ACTIVE = "active"
    ENDED = "ended"
    ARCHIVED = "archived"

class Business(Base):
    __tablename__ = "businesses"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    business_hours = Column(JSON)  # JSON structure for business hours
    business_metadata = Column(JSON, default={})  # For timezone, contact info, etc.
    custom_instructions = Column(Text)
    knowledge_base = Column(JSON)
    prompt_template = Column(Text)  # Custom prompt template for the business
    conversations = relationship("Conversation", back_populates="business")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True)
    business_id = Column(String, ForeignKey("businesses.id"), nullable=False)
    business = relationship("Business", back_populates="conversations")
    status = Column(SQLEnum(ConversationStatus), default=ConversationStatus.ACTIVE)
    conversation_metadata = Column(JSON, default={})  # For storing context, user info, etc.
    messages = relationship("Message", back_populates="conversation", order_by="Message.timestamp")
    summary = relationship("ConversationSummary", back_populates="conversation", uselist=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ended_at = Column(DateTime)
    last_message_at = Column(DateTime, default=datetime.utcnow)

class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    conversation = relationship("Conversation", back_populates="summary")
    summary_text = Column(Text)
    key_points = Column(JSON)  # List of key points from the conversation
    action_items = Column(JSON)  # List of action items identified
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    conversation = relationship("Conversation", back_populates="messages")
    role = Column(String, nullable=False)  # 'human' or 'ai'
    content = Column(Text, nullable=False)
    message_metadata = Column(JSON, default={})  # For storing additional message context
    is_summarized = Column(Boolean, default=False)  # Track if message is included in summary
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
def init_db(db_url: str):
    """Initialize the database with all tables."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    init_db() 