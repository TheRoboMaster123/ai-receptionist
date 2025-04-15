from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
import json
from .models import Message, ConversationSummary
from .llm_utils import get_llm_pipeline

SUMMARY_PROMPT = """Analyze this conversation segment and provide a structured summary.
Focus on key information, action items, and maintaining context for future interactions.

Conversation:
{messages}

Provide a summary in the following JSON format:
{
    "main_topics": ["list of main topics discussed"],
    "key_points": ["important information or decisions"],
    "action_items": ["specific tasks or follow-ups needed"],
    "context": "brief summary of the conversation flow",
    "business_specific": {
        "preferences": ["any customer preferences mentioned"],
        "requirements": ["specific requirements or constraints"],
        "concerns": ["issues or concerns raised"]
    }
}

Summary:"""

class ConversationSummarizer:
    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_length: int = 512,
        temperature: float = 0.7
    ):
        self.pipeline = get_llm_pipeline(
            model_id=model_id,
            max_length=max_length,
            temperature=temperature
        )
    
    def format_messages(self, messages: List[Message]) -> str:
        """Format messages into a readable conversation format."""
        formatted = []
        for msg in messages:
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            formatted.append(f"[{timestamp}] {msg.role}: {msg.content}")
        return "\n".join(formatted)
    
    def summarize_messages(
        self,
        messages: List[Message],
        max_tokens: int = 200
    ) -> str:
        """Generate a summary of the conversation messages."""
        if not messages:
            return ""

        # Format messages for summarization
        formatted_messages = []
        for msg in messages:
            role = "User" if msg.role == "human" else "Assistant"
            formatted_messages.append(f"{role}: {msg.content}")

        # Create prompt for summarization
        prompt = (
            "Please provide a concise summary of the following conversation, "
            "focusing on the main topics discussed and key decisions made:\n\n"
            + "\n".join(formatted_messages)
        )

        # Generate summary
        response = self.pipeline(
            prompt,
            max_length=max_tokens,
            num_return_sequences=1
        )[0]["generated_text"]

        return response.strip()
    
    def merge_summaries(self, existing_summary: Dict[str, Any], new_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge a new summary with an existing one, removing duplicates."""
        if not existing_summary:
            return new_summary
            
        merged = {
            "main_topics": list(set(existing_summary.get("main_topics", []) + new_summary.get("main_topics", []))),
            "key_points": list(set(existing_summary.get("key_points", []) + new_summary.get("key_points", []))),
            "action_items": list(set(existing_summary.get("action_items", []) + new_summary.get("action_items", []))),
            "context": f"{existing_summary.get('context', '')} ... {new_summary.get('context', '')}",
            "business_specific": {
                "preferences": list(set(
                    existing_summary.get("business_specific", {}).get("preferences", []) +
                    new_summary.get("business_specific", {}).get("preferences", [])
                )),
                "requirements": list(set(
                    existing_summary.get("business_specific", {}).get("requirements", []) +
                    new_summary.get("business_specific", {}).get("requirements", [])
                )),
                "concerns": list(set(
                    existing_summary.get("business_specific", {}).get("concerns", []) +
                    new_summary.get("business_specific", {}).get("concerns", [])
                ))
            }
        }
        return merged
    
    def update_conversation_summary(
        self,
        db: Session,
        conversation_id: str,
        messages: List[Message]
    ) -> None:
        """Update the conversation summary in the database."""
        summary_text = self.summarize_messages(messages)
        
        # Update or create summary in database
        summary = db.query(ConversationSummary).filter_by(
            conversation_id=conversation_id
        ).first()

        if summary:
            summary.content = summary_text
            summary.updated_at = datetime.utcnow()
        else:
            summary = ConversationSummary(
                conversation_id=conversation_id,
                content=summary_text
            )
            db.add(summary)

        db.commit() 