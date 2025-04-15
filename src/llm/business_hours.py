from datetime import datetime, time, date, timedelta
from typing import Dict, Optional, List, Tuple, Any
import pytz
from dateutil.parser import parse
import json

class BusinessHoursManager:
    def __init__(self):
        self.default_timezone = "UTC"
    
    def parse_business_hours(self, hours_str: str) -> Dict[str, Any]:
        """
        Parse business hours string into structured format.
        Expected format: 
        {
            "regular_hours": {
                "monday": ["9:00-17:00", "18:00-22:00"],
                "tuesday": ["9:00-17:00"]
            },
            "holidays": [
                {"date": "2024-12-25", "name": "Christmas", "is_closed": true},
                {"date": "2024-12-31", "hours": ["10:00-15:00"], "name": "New Year's Eve"}
            ],
            "special_hours": [
                {"date": "2024-01-01", "hours": ["12:00-20:00"], "name": "New Year's Day"},
                {"date": "2024-02-14", "hours": ["9:00-23:00"], "name": "Valentine's Day"}
            ]
        }
        """
        try:
            hours_dict = json.loads(hours_str)
            parsed = {
                "regular_hours": {},
                "holidays": [],
                "special_hours": []
            }
            
            # Parse regular hours
            for day, time_ranges in hours_dict.get("regular_hours", {}).items():
                day = day.lower()
                parsed_ranges = []
                
                for time_range in time_ranges:
                    start_str, end_str = time_range.split('-')
                    start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
                    end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
                    parsed_ranges.append((start_time, end_time))
                
                parsed["regular_hours"][day] = parsed_ranges
            
            # Parse holidays
            for holiday in hours_dict.get("holidays", []):
                holiday_date = parse(holiday["date"]).date()
                parsed_holiday = {
                    "date": holiday_date,
                    "name": holiday["name"],
                    "is_closed": holiday.get("is_closed", True)
                }
                
                if "hours" in holiday and not holiday.get("is_closed"):
                    parsed_holiday["hours"] = [
                        tuple(
                            datetime.strptime(t.strip(), "%H:%M").time()
                            for t in time_range.split('-')
                        )
                        for time_range in holiday["hours"]
                    ]
                
                parsed["holidays"].append(parsed_holiday)
            
            # Parse special hours
            for special in hours_dict.get("special_hours", []):
                special_date = parse(special["date"]).date()
                parsed_special = {
                    "date": special_date,
                    "name": special["name"],
                    "hours": [
                        tuple(
                            datetime.strptime(t.strip(), "%H:%M").time()
                            for t in time_range.split('-')
                        )
                        for time_range in special["hours"]
                    ]
                }
                parsed["special_hours"].append(parsed_special)
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid business hours format: {e}")
    
    def is_business_hours(
        self,
        business_hours: str,
        timezone: Optional[str] = None,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if current time is within business hours."""
        try:
            # Parse business hours
            hours_dict = self.parse_business_hours(business_hours)
            
            # Get current time in business timezone
            tz = pytz.timezone(timezone or self.default_timezone)
            now = current_time or datetime.now(tz)
            current_date = now.date()
            current_day = now.strftime("%A").lower()
            current_time = now.time()
            
            # Check holidays
            for holiday in hours_dict["holidays"]:
                if holiday["date"] == current_date:
                    if holiday.get("is_closed", True):
                        return False, f"Closed for holiday: {holiday['name']}"
                    elif "hours" in holiday:
                        for start_time, end_time in holiday["hours"]:
                            if start_time <= current_time <= end_time:
                                return True, f"Open for holiday: {holiday['name']}"
                        return False, f"Outside holiday hours for {holiday['name']}"
            
            # Check special hours
            for special in hours_dict["special_hours"]:
                if special["date"] == current_date:
                    for start_time, end_time in special["hours"]:
                        if start_time <= current_time <= end_time:
                            return True, f"Open during special hours: {special['name']}"
                    return False, f"Outside special hours for {special['name']}"
            
            # Check regular hours
            if current_day not in hours_dict["regular_hours"]:
                return False, f"Business is closed on {current_day.capitalize()}"
            
            for start_time, end_time in hours_dict["regular_hours"][current_day]:
                if start_time <= current_time <= end_time:
                    return True, None
            
            # Find next opening time
            next_opening = self.get_next_opening_time(hours_dict, current_date, current_day, current_time)
            if next_opening:
                return False, f"Business is currently closed. Opens {next_opening}"
            
            return False, "Business is currently closed"
            
        except ValueError as e:
            return False, f"Error checking business hours: {e}"
    
    def get_next_opening_time(
        self,
        hours_dict: Dict[str, Any],
        current_date: date,
        current_day: str,
        current_time: time
    ) -> Optional[str]:
        """Find the next opening time considering holidays and special hours."""
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        current_day_idx = days.index(current_day)
        
        # Check remaining time slots for current day
        if current_day in hours_dict["regular_hours"]:
            for start_time, _ in hours_dict["regular_hours"][current_day]:
                if current_time < start_time:
                    return f"today at {start_time.strftime('%H:%M')}"
        
        # Check next 7 days
        for i in range(1, 8):
            next_date = current_date + timedelta(days=i)
            next_day_idx = (current_day_idx + i) % 7
            next_day = days[next_day_idx]
            
            # Check holidays
            holiday = next(
                (h for h in hours_dict["holidays"] if h["date"] == next_date),
                None
            )
            if holiday:
                if not holiday.get("is_closed") and "hours" in holiday:
                    day_name = "tomorrow" if i == 1 else next_date.strftime("%A")
                    return f"{day_name} at {holiday['hours'][0][0].strftime('%H:%M')} ({holiday['name']})"
                continue
            
            # Check special hours
            special = next(
                (s for s in hours_dict["special_hours"] if s["date"] == next_date),
                None
            )
            if special:
                day_name = "tomorrow" if i == 1 else next_date.strftime("%A")
                return f"{day_name} at {special['hours'][0][0].strftime('%H:%M')} ({special['name']})"
            
            # Check regular hours
            if next_day in hours_dict["regular_hours"] and hours_dict["regular_hours"][next_day]:
                day_name = "tomorrow" if i == 1 else next_day
                return f"{day_name} at {hours_dict['regular_hours'][next_day][0][0].strftime('%H:%M')}"
        
        return None
    
    def get_business_status(
        self,
        business_hours: str,
        timezone: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed business status including holidays and special hours."""
        is_open, message = self.is_business_hours(business_hours, timezone)
        hours_dict = self.parse_business_hours(business_hours)
        
        tz = pytz.timezone(timezone or self.default_timezone)
        now = datetime.now(tz)
        current_date = now.date()
        current_day = now.strftime("%A").lower()
        
        # Get current day's hours
        current_day_hours = []
        
        # Check holidays
        holiday = next(
            (h for h in hours_dict["holidays"] if h["date"] == current_date),
            None
        )
        if holiday:
            if holiday.get("is_closed", True):
                current_day_hours = ["Closed for " + holiday["name"]]
            else:
                current_day_hours = [
                    f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                    for start, end in holiday["hours"]
                ]
        
        # Check special hours
        special = next(
            (s for s in hours_dict["special_hours"] if s["date"] == current_date),
            None
        )
        if special:
            current_day_hours = [
                f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                for start, end in special["hours"]
            ]
        
        # Check regular hours
        if not current_day_hours and current_day in hours_dict["regular_hours"]:
            current_day_hours = [
                f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                for start, end in hours_dict["regular_hours"][current_day]
            ]
        
        # Get upcoming special dates
        upcoming_dates = []
        for item in sorted(
            hours_dict["holidays"] + hours_dict["special_hours"],
            key=lambda x: x["date"]
        ):
            if item["date"] > current_date:
                upcoming_dates.append({
                    "date": item["date"].strftime("%Y-%m-%d"),
                    "name": item["name"],
                    "type": "holiday" if "is_closed" in item else "special"
                })
                if len(upcoming_dates) >= 3:
                    break
        
        return {
            "is_open": is_open,
            "message": message,
            "current_day": current_day,
            "current_day_hours": current_day_hours,
            "timezone": timezone or self.default_timezone,
            "upcoming_dates": upcoming_dates
        } 