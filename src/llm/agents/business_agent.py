from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, time, timedelta
import pytz
from zoneinfo import ZoneInfo
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
import re
import json

class BusinessAgent:
    def __init__(
        self,
        business_profile: Dict[str, Any],
        llm,
        tools: Optional[List[Tool]] = None
    ):
        self.business_profile = business_profile
        self.llm = llm
        self.tools = tools or self._get_default_tools()
        self.agent_executor = self._create_agent_executor()
    
    def _get_default_tools(self) -> List[Tool]:
        """Get default tools for the business agent"""
        return [
            Tool(
                name="check_business_hours",
                func=self._check_business_hours,
                description="Check if the business is currently open and get operating hours"
            ),
            Tool(
                name="schedule_appointment",
                func=self._schedule_appointment,
                description="Schedule an appointment with the business"
            ),
            Tool(
                name="handle_emergency",
                func=self._handle_emergency,
                description="Handle emergency situations and escalate if necessary"
            ),
            Tool(
                name="get_business_info",
                func=self._get_business_info,
                description="Get general information about the business"
            )
        ]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and prompt"""
        prompt = BusinessAgentPrompt(
            business_name=self.business_profile['name'],
            tools=self.tools
        )
        
        agent = LLMSingleActionAgent(
            llm_chain=self.llm,
            output_parser=self._parse_output,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
    
    async def handle_request(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle a user request using available tools"""
        try:
            # Add context to the input if available
            if context:
                user_input = f"Context: {str(context)}\nUser Request: {user_input}"
            
            # Run the agent
            response = await self.agent_executor.arun(user_input)
            
            # Extract any actions or updates
            actions = self._extract_actions(response)
            
            return {
                "response": response,
                "actions": actions
            }
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "actions": []
            }
    
    def _check_business_hours(self, query: str) -> str:
        """Check if the business is currently open"""
        hours = self.business_profile.get('business_hours', {})
        timezone_str = self.business_profile.get('metadata', {}).get('timezone', 'UTC')
        
        try:
            # Get current time in business timezone
            timezone = ZoneInfo(timezone_str)
            current_time = datetime.now(timezone)
            current_day = current_time.strftime('%A').lower()
            
            # Get regular hours for current day
            regular_hours = hours.get('regular_hours', {}).get(current_day, [])
            if not regular_hours:
                return f"The business is closed on {current_day.capitalize()}."
            
            # Check special dates and holidays
            date_str = current_time.strftime('%Y-%m-%d')
            special_hours = hours.get('special_hours', {}).get(date_str)
            holidays = hours.get('holidays', [])
            
            # Check if it's a holiday
            if date_str in holidays:
                return f"The business is closed today ({date_str}) as it's a holiday."
            
            # Use special hours if available, otherwise use regular hours
            today_hours = special_hours if special_hours else regular_hours
            
            # Convert current time to comparable format
            current_minutes = current_time.hour * 60 + current_time.minute
            
            # Check each time slot
            is_open = False
            time_slots = []
            for slot in today_hours:
                start_str, end_str = slot.split('-')
                start_hour, start_minute = map(int, start_str.split(':'))
                end_hour, end_minute = map(int, end_str.split(':'))
                
                start_minutes = start_hour * 60 + start_minute
                end_minutes = end_hour * 60 + end_minute
                
                time_slots.append(f"{start_str} to {end_str}")
                if start_minutes <= current_minutes <= end_minutes:
                    is_open = True
                    break
            
            if is_open:
                return f"The business is currently OPEN. Today's hours: {', '.join(time_slots)}"
            else:
                # Find next opening time
                next_slot = None
                for slot in today_hours:
                    start_str, _ = slot.split('-')
                    start_hour, start_minute = map(int, start_str.split(':'))
                    start_minutes = start_hour * 60 + start_minute
                    if start_minutes > current_minutes:
                        next_slot = slot
                        break
                
                if next_slot:
                    return f"The business is currently CLOSED. Next opening time today: {next_slot.split('-')[0]}. Today's hours: {', '.join(time_slots)}"
                else:
                    return f"The business is currently CLOSED for the day. Today's hours were: {', '.join(time_slots)}"
                
        except Exception as e:
            return f"Error checking business hours: {str(e)}. Please contact the business directly."
    
    def _schedule_appointment(self, details: str) -> str:
        """Schedule an appointment with the business"""
        try:
            # Parse appointment details
            parsed_details = self._parse_appointment_details(details)
            if not parsed_details:
                return "I couldn't understand the appointment details. Please provide a date and time for the appointment."

            date_str, time_str = parsed_details
            timezone_str = self.business_profile.get('metadata', {}).get('timezone', 'UTC')
            timezone = ZoneInfo(timezone_str)

            # Convert to datetime
            try:
                appointment_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                appointment_time = appointment_time.replace(tzinfo=timezone)
            except ValueError:
                return "Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time."

            # Validate appointment time
            validation_result = self._validate_appointment_time(appointment_time)
            if validation_result != "valid":
                return validation_result

            # Check if slot is available
            if not self._is_slot_available(appointment_time):
                return f"The requested time slot ({time_str} on {date_str}) is not available. Please choose another time."

            # Store appointment
            appointment = {
                "datetime": appointment_time.isoformat(),
                "status": "scheduled",
                "created_at": datetime.now(timezone).isoformat()
            }

            # Add to business profile's appointments
            if 'appointments' not in self.business_profile:
                self.business_profile['appointments'] = []
            self.business_profile['appointments'].append(appointment)

            return f"""APPOINTMENT: Successfully scheduled for {date_str} at {time_str}.
Please arrive 5-10 minutes early. If you need to reschedule or cancel, please let us know at least 24 hours in advance."""

        except Exception as e:
            return f"Error scheduling appointment: {str(e)}. Please try again with a valid date and time."

    def _parse_appointment_details(self, details: str) -> Optional[Tuple[str, str]]:
        """Parse date and time from appointment details"""
        # Look for date patterns (YYYY-MM-DD)
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        date_match = re.search(date_pattern, details)
        
        # Look for time patterns (HH:MM)
        time_pattern = r'(\d{1,2}:\d{2})'
        time_match = re.search(time_pattern, details)
        
        if date_match and time_match:
            return date_match.group(1), time_match.group(1)
        return None

    def _validate_appointment_time(self, appointment_time: datetime) -> str:
        """Validate the appointment time against business rules"""
        timezone = appointment_time.tzinfo
        current_time = datetime.now(timezone)
        
        # Check if appointment is in the past
        if appointment_time < current_time:
            return "Cannot schedule appointments in the past."
        
        # Check if appointment is too far in the future (e.g., > 3 months)
        max_future = current_time + timedelta(days=90)
        if appointment_time > max_future:
            return "Cannot schedule appointments more than 3 months in advance."
        
        # Check if appointment is during business hours
        weekday = appointment_time.strftime('%A').lower()
        hours = self.business_profile.get('business_hours', {})
        regular_hours = hours.get('regular_hours', {}).get(weekday, [])
        
        if not regular_hours:
            return f"We are not open on {weekday.capitalize()}."
        
        # Check special dates and holidays
        date_str = appointment_time.strftime('%Y-%m-%d')
        special_hours = hours.get('special_hours', {}).get(date_str)
        holidays = hours.get('holidays', [])
        
        if date_str in holidays:
            return f"Cannot schedule on {date_str} as it's a holiday."
        
        # Use special hours if available, otherwise use regular hours
        day_hours = special_hours if special_hours else regular_hours
        
        # Convert appointment time to minutes for comparison
        appt_minutes = appointment_time.hour * 60 + appointment_time.minute
        
        # Check if time falls within business hours
        is_valid_time = False
        for slot in day_hours:
            start_str, end_str = slot.split('-')
            start_hour, start_minute = map(int, start_str.split(':'))
            end_hour, end_minute = map(int, end_str.split(':'))
            
            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute
            
            if start_minutes <= appt_minutes <= end_minutes:
                is_valid_time = True
                break
        
        if not is_valid_time:
            hours_str = ', '.join(day_hours)
            return f"Appointment time must be during business hours: {hours_str}"
        
        return "valid"

    def _is_slot_available(self, appointment_time: datetime) -> bool:
        """Check if the requested time slot is available"""
        # Get existing appointments
        existing_appointments = self.business_profile.get('appointments', [])
        
        # Define appointment duration (e.g., 30 minutes)
        duration = timedelta(minutes=30)
        
        # Check for conflicts with existing appointments
        for appt in existing_appointments:
            existing_time = datetime.fromisoformat(appt['datetime'])
            
            # Check if appointment times overlap
            if (existing_time <= appointment_time < existing_time + duration or
                appointment_time <= existing_time < appointment_time + duration):
                return False
        
        return True

    def _get_available_slots(self, date_str: str) -> List[str]:
        """Get available time slots for a given date"""
        try:
            timezone_str = self.business_profile.get('metadata', {}).get('timezone', 'UTC')
            timezone = ZoneInfo(timezone_str)
            
            # Parse date
            date = datetime.strptime(date_str, "%Y-%m-%d")
            date = date.replace(tzinfo=timezone)
            
            # Get business hours for the day
            weekday = date.strftime('%A').lower()
            hours = self.business_profile.get('business_hours', {})
            regular_hours = hours.get('regular_hours', {}).get(weekday, [])
            special_hours = hours.get('special_hours', {}).get(date_str)
            
            if date_str in hours.get('holidays', []):
                return []
            
            day_hours = special_hours if special_hours else regular_hours
            if not day_hours:
                return []
            
            # Generate all possible slots
            available_slots = []
            for slot in day_hours:
                start_str, end_str = slot.split('-')
                start_hour, start_minute = map(int, start_str.split(':'))
                end_hour, end_minute = map(int, end_str.split(':'))
                
                current = datetime.combine(date.date(), time(start_hour, start_minute))
                current = current.replace(tzinfo=timezone)
                end = datetime.combine(date.date(), time(end_hour, end_minute))
                end = end.replace(tzinfo=timezone)
                
                # Generate 30-minute slots
                while current + timedelta(minutes=30) <= end:
                    if self._is_slot_available(current):
                        available_slots.append(current.strftime("%H:%M"))
                    current += timedelta(minutes=30)
            
            return available_slots
            
        except Exception as e:
            return []
    
    def _handle_emergency(self, situation: str) -> str:
        """Handle emergency situations"""
        # TODO: Implement emergency handling logic
        emergency_contacts = self.business_profile.get('emergency_contacts', [])
        return f"This is an emergency situation. Please contact: {emergency_contacts}"
    
    def _get_business_info(self, query: str) -> str:
        """Get business information"""
        return f"""
Business Name: {self.business_profile['name']}
Description: {self.business_profile['description']}
Hours: {self.business_profile['business_hours']}
"""
    
    def _parse_output(self, llm_output: str) -> AgentAction:
        """Parse LLM output into agent actions"""
        # Check if the output is a final answer
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Extract the action and input
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)
    
    def _extract_actions(self, response: str) -> List[Dict[str, Any]]:
        """Extract actions from the response"""
        actions = []
        
        # Look for specific patterns indicating actions
        if "APPOINTMENT:" in response:
            actions.append({
                "type": "appointment",
                "details": response.split("APPOINTMENT:")[-1].strip()
            })
        
        if "EMERGENCY:" in response:
            actions.append({
                "type": "emergency",
                "details": response.split("EMERGENCY:")[-1].strip()
            })
        
        return actions

class BusinessAgentPrompt(StringPromptTemplate):
    def __init__(self, business_name: str, tools: List[Tool]):
        super().__init__(template=self._get_template(), input_variables=["input", "agent_scratchpad"])
        self.business_name = business_name
        self.tools = tools
    
    def _get_template(self) -> str:
        """Get the prompt template"""
        return """You are an AI agent for {business_name}. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""
    
    def format(self, **kwargs) -> str:
        """Format the prompt template"""
        # Get the list of tool names and descriptions
        tool_strings = []
        tool_names = []
        for tool in self.tools:
            tool_strings.append(f"{tool.name}: {tool.description}")
            tool_names.append(tool.name)
        
        kwargs.update({
            "business_name": self.business_name,
            "tools": "\n".join(tool_strings),
            "tool_names": ", ".join(tool_names)
        })
        
        return self.template.format(**kwargs) 