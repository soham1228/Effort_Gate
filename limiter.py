import time
import asyncio
from collections import deque
from typing import Dict, Deque, Optional, Tuple
from config import settings

class RateLimitExceeded(Exception):
    def __init__(self, message: str, retry_after: int):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)

class BaseLimiter:
    async def check_and_record(self, key: str, estimated_tokens: int) -> None:
        raise NotImplementedError
    
    async def sync_headers(self, key: str, headers: Dict[str, str]) -> None:
        pass # Optional sync from provider headers

class InMemoryLimiter(BaseLimiter):
    def __init__(self):
        # Stores (timestamp_in_seconds, token_count)
        self.rpm_windows: Dict[str, Deque[Tuple[float, int]]] = {}
        self.rpd_windows: Dict[str, Deque[Tuple[float, int]]] = {}
        self.tpm_windows: Dict[str, Deque[Tuple[float, int]]] = {}
        self.lock = asyncio.Lock()
    
    def _clean_window(self, window: Deque[Tuple[float, int]], current_time: float, window_size: float) -> int:
        """Removes old entries and returns current sum of tokens/requests in the window."""
        while window and current_time - window[0][0] > window_size:
            window.popleft()
        return sum(item[1] for item in window)
    
    def _clean_rpm_window(self, window: Deque[Tuple[float, int]], current_time: float) -> int:
        """RPM window counts number of requests, not tokens. item[1] is 1 for RPM."""
        while window and current_time - window[0][0] > 60.0:
            window.popleft()
        return len(window)
        
    def _clean_rpd_window(self, window: Deque[Tuple[float, int]], current_time: float) -> int:
        while window and current_time - window[0][0] > 86400.0:
            window.popleft()
        return len(window)

    async def check_and_record(self, key: str, estimated_tokens: int = 0) -> None:
        now = time.time()
        
        async with self.lock:
            if key not in self.rpm_windows:
                self.rpm_windows[key] = deque()
                self.rpd_windows[key] = deque()
                self.tpm_windows[key] = deque()
                
            current_rpm = self._clean_rpm_window(self.rpm_windows[key], now)
            current_rpd = self._clean_rpd_window(self.rpd_windows[key], now)
            current_tpm = self._clean_window(self.tpm_windows[key], now, 60.0)
            
            if current_rpm + 1 > settings.MAX_RPM:
                oldest = self.rpm_windows[key][0][0]
                retry_after = max(1, int(60.0 - (now - oldest)))
                raise RateLimitExceeded(f"RPM limit exceeded. Max: {settings.MAX_RPM}", retry_after)
                
            if current_rpd + 1 > settings.MAX_RPD:
                oldest = self.rpd_windows[key][0][0]
                retry_after = max(1, int(86400.0 - (now - oldest)))
                raise RateLimitExceeded(f"RPD limit exceeded. Max: {settings.MAX_RPD}", retry_after)
                
            if estimated_tokens > 0 and current_tpm + estimated_tokens > settings.MAX_TPM:
                oldest = self.tpm_windows[key][0][0]
                retry_after = max(1, int(60.0 - (now - oldest)))
                raise RateLimitExceeded(f"TPM limit exceeded. Max: {settings.MAX_TPM}", retry_after)
            
            # Record the request
            self.rpm_windows[key].append((now, 1))
            self.rpd_windows[key].append((now, 1))
            if estimated_tokens > 0:
                self.tpm_windows[key].append((now, estimated_tokens))

class RedisLimiter(BaseLimiter):
    def __init__(self, redis_url: str):
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url, decode_responses=True)
        
    async def check_and_record(self, key: str, estimated_tokens: int = 0) -> None:
        now = time.time()
        pipeline = self.redis.pipeline()
        
        rpm_key = f"rate:rpm:{key}"
        rpd_key = f"rate:rpd:{key}"
        tpm_key = f"rate:tpm:{key}"
        
        # Clean old entries
        pipeline.zremrangebyscore(rpm_key, 0, now - 60)
        pipeline.zremrangebyscore(rpd_key, 0, now - 86400)
        pipeline.zremrangebyscore(tpm_key, 0, now - 60)
        
        # Count current
        pipeline.zcard(rpm_key)
        pipeline.zcard(rpd_key)
        # For TPM, we need the sum of scores or values, but zcard just counts elements. 
        # A better approach for Redis TPM is storing token counts in the member or scoring.
        # However, to keep it simple and accurate, we can store random UUID as member and token count as score? No, ZSET scores are used for time.
        # Standard approach for TPM in Redis sliding window:
        # Member: {timestamp}-{uuid}
        # Score: timestamp
        # Value doesn't easily sum without a LUA script.
        # Let's execute a Lua script for TPM, or simply fetch all elements and sum them up (ok for small windows).
        pipeline.zrange(tpm_key, 0, -1)
        
        results = await pipeline.execute()
        current_rpm = results[3]
        current_rpd = results[4]
        tpm_members = results[5]
        
        current_tpm = 0
        for member in tpm_members:
            try:
                # Format: timestamp:tokens:uuid
                parts = member.split(':')
                if len(parts) >= 2:
                    current_tpm += int(parts[1])
            except Exception:
                pass
                
        if current_rpm + 1 > settings.MAX_RPM:
            # Need to get oldest to calculate retry
            oldest_res = await self.redis.zrange(rpm_key, 0, 0, withscores=True)
            retry_after = 60
            if oldest_res:
                retry_after = max(1, int(60 - (now - oldest_res[0][1])))
            raise RateLimitExceeded(f"RPM limit exceeded. Max: {settings.MAX_RPM}", retry_after)
            
        if current_rpd + 1 > settings.MAX_RPD:
            oldest_res = await self.redis.zrange(rpd_key, 0, 0, withscores=True)
            retry_after = 86400
            if oldest_res:
                retry_after = max(1, int(86400 - (now - oldest_res[0][1])))
            raise RateLimitExceeded(f"RPD limit exceeded. Max: {settings.MAX_RPD}", retry_after)
            
        if estimated_tokens > 0 and current_tpm + estimated_tokens > settings.MAX_TPM:
            oldest_res = await self.redis.zrange(tpm_key, 0, 0, withscores=True)
            retry_after = 60
            if oldest_res:
                retry_after = max(1, int(60 - (now - oldest_res[0][1])))
            raise RateLimitExceeded(f"TPM limit exceeded. Max: {settings.MAX_TPM}", retry_after)
            
        import uuid
        req_id = str(uuid.uuid4())
        
        pipeline = self.redis.pipeline()
        pipeline.zadd(rpm_key, {f"{now}:1:{req_id}": now})
        pipeline.expire(rpm_key, 60)
        
        pipeline.zadd(rpd_key, {f"{now}:1:{req_id}": now})
        pipeline.expire(rpd_key, 86400)
        
        if estimated_tokens > 0:
            pipeline.zadd(tpm_key, {f"{now}:{estimated_tokens}:{req_id}": now})
            pipeline.expire(tpm_key, 60)
            
        await pipeline.execute()

def get_limiter() -> BaseLimiter:
    if settings.REDIS_URL:
        return RedisLimiter(settings.REDIS_URL)
    return InMemoryLimiter()

limiter = get_limiter()
