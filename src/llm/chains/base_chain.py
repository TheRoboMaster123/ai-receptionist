from typing import Dict, Any, Optional
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch
import re
from datetime import datetime

class BusinessConversationChain:
    def __init__(
        self,
        model,
        tokenizer,
        business_profile: Dict[str, Any],
        max_length: int = 2048,
        temperature: float = 0.7,
        max_memory_messages: int = 10
    ):
        self.business_profile = business_profile
        self.max_memory_messages = max_memory_messages
        
        # Initialize the pipeline
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain LLM
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=max_memory_messages,
            return_messages=True
        )
        
        # Create the chain
        self.chain = self._create_chain()
    
    def _create_chain(self) -> ConversationChain:
        """Create a conversation chain with business-specific prompt"""
        # Create business-specific prompt template
        template = self._get_business_prompt_template()
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        return ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=True
        )
    
    def _get_business_prompt_template(self) -> str:
        """Generate a business-specific prompt template"""
        base_template = f"""You are an AI receptionist for {self.business_profile['name']}.

Business Details:
- Description: {self.business_profile['description']}
- Hours: {self.business_profile['business_hours']}

Instructions:
1. Always be professional and courteous
2. Stay within the business's operating hours
3. Handle scheduling and inquiries appropriately
4. Escalate emergency situations
5. Maintain context throughout the conversation

Current conversation:
{{history}}
Human: {{input}}
Assistant:"""

        # Add custom instructions if available
        if self.business_profile.get('prompt_template'):
            base_template = self.business_profile['prompt_template']
        
        return base_template
    
    async def process_message(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a message and return the response with context updates"""
        # Update context if provided
        if context:
            self.memory.chat_memory.add_user_message(f"Context: {str(context)}")
        
        # Get response from chain
        response = self.chain.predict(input=user_input)
        
        # Extract any context updates (e.g., appointments, tasks)
        context_updates = self._extract_context_updates(response)
        
        return {
            "response": response,
            "context_updates": context_updates
        }
    
    def _extract_context_updates(self, response: str) -> Dict[str, Any]:
        """Extract any context updates from the response"""
        context_updates = {
            "appointments": [],
            "tasks": [],
            "follow_ups": [],
            "contact_info": [],
            "preferences": {}
        }
        
        # Extract appointment details
        appointment_patterns = [
            r"scheduled.+?appointment.+?for\s+(.+?)(?:\.|$)",
            r"booked.+?for\s+(.+?)(?:\.|$)",
            r"appointment.*?:\s*(.+?)(?:\.|$)"
        ]
        for pattern in appointment_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                context_updates["appointments"].append({
                    "details": match.group(1).strip(),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Extract tasks
        task_patterns = [
            r"need to\s+(.+?)(?:\.|$)",
            r"will\s+(.+?)(?:\.|$)",
            r"task.*?:\s*(.+?)(?:\.|$)"
        ]
        for pattern in task_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                task = match.group(1).strip()
                if not any(task.lower() in t.lower() for t in context_updates["tasks"]):
                    context_updates["tasks"].append(task)
        
        # Extract follow-up items
        followup_patterns = [
            r"follow(?:-|\s)?up.*?:\s*(.+?)(?:\.|$)",
            r"(?:will|should)\s+follow(?:-|\s)?up\s+(.+?)(?:\.|$)",
            r"(?:will|should)\s+contact\s+(?:you|back)\s+(.+?)(?:\.|$)"
        ]
        for pattern in followup_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                followup = match.group(1).strip()
                if not any(followup.lower() in f.lower() for f in context_updates["follow_ups"]):
                    context_updates["follow_ups"].append(followup)
        
        # Extract contact information
        contact_patterns = [
            r"(?:phone|tel|contact)(?:\s+(?:number|info))?\s*:?\s*(\+?[\d\s-]+)",
            r"email\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
        ]
        for pattern in contact_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                contact = match.group(1).strip()
                if contact not in context_updates["contact_info"]:
                    context_updates["contact_info"].append(contact)
        
        # Extract customer preferences
        preference_patterns = {
            "preferred_time": r"prefer(?:red|s)?\s+time\s*:?\s*(.+?)(?:\.|$)",
            "preferred_contact": r"prefer(?:red|s)?\s+(?:to be )?contact(?:ed)?\s+(?:by|via)\s+(.+?)(?:\.|$)",
            "special_requests": r"special\s+request\s*:?\s*(.+?)(?:\.|$)"
        }
        for key, pattern in preference_patterns.items():
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                context_updates["preferences"][key] = match.group(1).strip()
        
        # Clean up empty lists and dicts
        return {k: v for k, v in context_updates.items() if v} 