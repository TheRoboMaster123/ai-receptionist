from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid
from typing import Optional, Dict, List, Any
from sqlalchemy.orm.session import Session

from .models import Base, Business, Conversation, Message

DATABASE_URL = "sqlite:///./business_data.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_business(db, business_id: str) -> Optional[Business]:
    return db.query(Business).filter(Business.id == business_id).first()

def create_or_update_business(
    db: Session,
    business_id: str,
    name: str,
    description: str = None,
    business_hours: Dict[str, Any] = None,
    custom_instructions: str = None,
    knowledge_base: Dict = None,
    metadata: Dict[str, Any] = None,
    prompt_template: str = None
) -> Business:
    business = get_business(db, business_id)
    
    if business is None:
        business = Business(
            id=business_id,
            name=name,
            description=description,
            business_hours=business_hours or {},
            custom_instructions=custom_instructions,
            knowledge_base=knowledge_base or {},
            metadata=metadata or {"timezone": "UTC"},
            prompt_template=prompt_template,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(business)
    else:
        business.name = name
        if description is not None:
            business.description = description
        if business_hours is not None:
            business.business_hours = business_hours
        if custom_instructions is not None:
            business.custom_instructions = custom_instructions
        if knowledge_base is not None:
            business.knowledge_base = knowledge_base
        if metadata is not None:
            # Update metadata while preserving existing values
            current_metadata = business.metadata or {}
            current_metadata.update(metadata)
            business.metadata = current_metadata
        if prompt_template is not None:
            business.prompt_template = prompt_template
        business.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(business)
    return business

def create_conversation(db, business_id: str, metadata: Dict = None) -> Conversation:
    conversation = Conversation(
        id=str(uuid.uuid4()),
        business_id=business_id,
        metadata=metadata or {}
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

def get_conversation(db, conversation_id: str) -> Optional[Conversation]:
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()

def add_message(
    db,
    conversation_id: str,
    role: str,
    content: str
) -> Message:
    conversation = get_conversation(db, conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")
    
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content
    )
    
    # Update conversation last message timestamp
    conversation.last_message_at = datetime.utcnow()
    
    db.add(message)
    db.commit()
    db.refresh(message)
    return message

def get_conversation_messages(
    db,
    conversation_id: str,
    limit: int = None
) -> List[Message]:
    query = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp.asc())
    
    if limit:
        query = query.limit(limit)
    
    return query.all()

def get_business_conversations(
    db,
    business_id: str,
    limit: int = None
) -> List[Conversation]:
    query = db.query(Conversation).filter(
        Conversation.business_id == business_id
    ).order_by(Conversation.last_message_at.desc())
    
    if limit:
        query = query.limit(limit)
    
    return query.all()

def add_test_business():
    session = SessionLocal()
    try:
        # Create a test business
        test_business = Business(
            id=str(uuid.uuid4()),
            name="Tech Solutions Inc",
            description="A leading IT consulting firm specializing in software development and cloud solutions",
            business_hours="Monday-Friday: 9:00 AM - 5:00 PM EST",
            custom_instructions="""
            - Always greet callers professionally
            - Ask how you can assist them today
            - For technical support, collect basic information about their issue
            - For sales inquiries, note their contact information
            - Emergency support available 24/7 at premium rates
            """,
            knowledge_base={
                "services": [
                    "Software Development",
                    "Cloud Migration",
                    "IT Consulting",
                    "Technical Support"
                ],
                "support_tiers": {
                    "basic": "9-5 weekday support",
                    "premium": "24/7 support"
                },
                "common_issues": [
                    "Password reset",
                    "System access",
                    "Cloud services",
                    "Development consulting"
                ]
            }
        )
        
        session.add(test_business)
        session.commit()
        print(f"Test business added successfully! Business ID: {test_business.id}")
        return test_business.id
        
    except Exception as e:
        print(f"Error adding test business: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()

def update_business(
    db: Session,
    business_id: str,
    **kwargs
) -> Optional[Business]:
    """Update a business"""
    business = get_business(db, business_id)
    if business:
        for key, value in kwargs.items():
            if hasattr(business, key):
                setattr(business, key, value)
        db.commit()
        db.refresh(business)
    return business

def get_or_create_conversation(
    db: Session,
    business_id: str,
    conversation_id: Optional[str] = None
) -> Conversation:
    """Get an existing conversation or create a new one"""
    if conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.business_id == business_id
        ).first()
        if conversation:
            conversation.last_interaction = datetime.utcnow()
            db.commit()
            return conversation
    
    # Create new conversation
    conversation = Conversation(
        id=str(uuid.uuid4()),
        business_id=business_id,
        conversation_metadata={}
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

def update_conversation_metadata(
    db: Session,
    conversation_id: str,
    metadata_updates: Dict[str, Any]
) -> Optional[Conversation]:
    """Update conversation metadata"""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conversation:
        conversation.conversation_metadata = {
            **(conversation.conversation_metadata or {}),
            **metadata_updates
        }
        db.commit()
        db.refresh(conversation)
    return conversation

def update_business_hours(
    db: Session,
    business_id: str,
    business_hours: Dict[str, Any]
) -> Optional[Business]:
    """Update business hours specifically"""
    business = get_business(db, business_id)
    if business:
        business.business_hours = business_hours
        business.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(business)
    return business

def update_business_metadata(
    db: Session,
    business_id: str,
    metadata_updates: Dict[str, Any]
) -> Optional[Business]:
    """Update business metadata"""
    business = get_business(db, business_id)
    if business:
        current_metadata = business.metadata or {}
        current_metadata.update(metadata_updates)
        business.metadata = current_metadata
        business.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(business)
    return business

if __name__ == "__main__":
    add_test_business() 