from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import time
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.tokens: Dict[str, Tuple[float, int]] = {}  # (last_request_time, remaining_tokens)
        
    def _get_tokens(self, key: str) -> Tuple[float, int]:
        """Get or initialize tokens for a key."""
        now = time.time()
        if key not in self.tokens:
            return now, self.burst_limit
        
        last_time, tokens = self.tokens[key]
        time_passed = now - last_time
        new_tokens = min(
            self.burst_limit,
            tokens + int(time_passed * (self.requests_per_minute / 60))
        )
        return now, new_tokens
    
    async def check_rate_limit(
        self, request: Request, key: Optional[str] = None
    ) -> Optional[JSONResponse]:
        """Check if request should be rate limited."""
        # Use IP address if no key provided
        if key is None:
            key = request.client.host if request.client else "default"
            
        now, tokens = self._get_tokens(key)
        
        if tokens <= 0:
            wait_time = (60 / self.requests_per_minute) * (1 - tokens)
            logger.warning(f"Rate limit exceeded for {key}. Wait time: {wait_time:.2f}s")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests",
                    "wait_seconds": round(wait_time, 2)
                }
            )
        
        self.tokens[key] = (now, tokens - 1)
        return None

class MetricsRateLimiter(RateLimiter):
    """Rate limiter specifically for metrics endpoints with caching."""
    def __init__(self):
        super().__init__(requests_per_minute=30, burst_limit=5)
        self.cache: Dict[str, Tuple[float, dict]] = {}
        self.cache_ttl = 60  # Cache TTL in seconds
    
    def get_cached_response(self, endpoint: str) -> Optional[dict]:
        """Get cached response if available and not expired."""
        if endpoint in self.cache:
            timestamp, data = self.cache[endpoint]
            if time.time() - timestamp < self.cache_ttl:
                return data
            del self.cache[endpoint]
        return None
    
    def cache_response(self, endpoint: str, data: dict):
        """Cache response data."""
        self.cache[endpoint] = (time.time(), data)
    
    def clear_cache(self):
        """Clear expired cache entries."""
        now = time.time()
        self.cache = {
            k: v for k, v in self.cache.items()
            if now - v[0] < self.cache_ttl
        }

# Create singleton instances
general_limiter = RateLimiter()
metrics_limiter = MetricsRateLimiter()

async def rate_limit_metrics(request: Request) -> Optional[JSONResponse]:
    """Rate limiting middleware for metrics endpoints."""
    return await metrics_limiter.check_rate_limit(request) 