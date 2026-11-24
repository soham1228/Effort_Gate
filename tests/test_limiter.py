import pytest
import time
import asyncio
from effortgate.limiter import InMemoryLimiter, RateLimitExceeded

@pytest.mark.asyncio
async def test_in_memory_limiter_rpm():
    limiter = InMemoryLimiter()
    # Mock settings max RPM to 2 for testing
    from effortgate.config import settings
    original_rpm = settings.MAX_RPM
    settings.MAX_RPM = 2
    
    try:
        await limiter.check_and_record("test_key", 100)
        await limiter.check_and_record("test_key", 100)
        
        with pytest.raises(RateLimitExceeded):
            await limiter.check_and_record("test_key", 100)
            
    finally:
        settings.MAX_RPM = original_rpm

@pytest.mark.asyncio
async def test_in_memory_limiter_window_cleanup():
    limiter = InMemoryLimiter()
    # We will inject old records
    key = "cleanup_key"
    now = time.time()
    
    await limiter.check_and_record(key, 10)
    
    # Manually push an old entry 61 seconds ago
    async with limiter.lock:
        limiter.rpm_windows[key].appendleft((now - 61.0, 1))
        
    # The next check should clean it up
    await limiter.check_and_record(key, 10)
    
    async with limiter.lock:
        assert len(limiter.rpm_windows[key]) == 2 # Only the two recent ones should be there

@pytest.mark.asyncio
async def test_in_memory_limiter_isolated_keys():
    limiter = InMemoryLimiter()
    from effortgate.config import settings
    original_rpm = settings.MAX_RPM
    settings.MAX_RPM = 1
    
    try:
        # User A makes 1 request -> OK
        await limiter.check_and_record("user_a", 50)
        
        # User A makes 2nd request -> Exceeds RPM
        with pytest.raises(RateLimitExceeded):
            await limiter.check_and_record("user_a", 50)
            
        # User B makes a request -> Must SUCCEED because quotas are isolated per key!
        await limiter.check_and_record("user_b", 50)
    finally:
        settings.MAX_RPM = original_rpm

def test_resolve_rate_limit_key():
    from effortgate.router import resolve_rate_limit_key
    # Explicit key
    assert resolve_rate_limit_key({}, rate_limit_key="custom_key") == "custom_key"
    # From body
    assert resolve_rate_limit_key({"rate_limit_key": "body_key"}) == "body_key"
    # Hashed API key
    key = resolve_rate_limit_key({"api_key": "sk-secret-12345"})
    assert key.startswith("key:") and len(key) == 20
    # Fallback to global
    assert resolve_rate_limit_key({}) == "global"

