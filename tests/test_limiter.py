import pytest
import time
import asyncio
from limiter import InMemoryLimiter, RateLimitExceeded

@pytest.mark.asyncio
async def test_in_memory_limiter_rpm():
    limiter = InMemoryLimiter()
    # Mock settings max RPM to 2 for testing
    from config import settings
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
