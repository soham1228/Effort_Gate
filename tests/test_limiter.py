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

@pytest.mark.asyncio
async def test_sync_headers_retry_after():
    limiter = InMemoryLimiter()
    key = "upstream_test_key"
    
    # Simulate receiving a 429 or high load Retry-After header from upstream
    await limiter.sync_headers(key, {"retry-after": "15"})
    
    with pytest.raises(RateLimitExceeded) as exc:
        await limiter.check_and_record(key, 10)
    
    assert exc.value.retry_after >= 14
    assert "cooldown active" in exc.value.message

@pytest.mark.asyncio
async def test_sync_headers_remaining_requests_zero():
    limiter = InMemoryLimiter()
    key = "upstream_depleted_key"
    
    # Simulate upstream indicating 0 remaining requests with reset in 25 seconds
    await limiter.sync_headers(key, {
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "25"
    })
    
    with pytest.raises(RateLimitExceeded) as exc:
        await limiter.check_and_record(key, 10)
        
    assert exc.value.retry_after >= 24
    assert "cooldown active" in exc.value.message

@pytest.mark.asyncio
async def test_in_memory_limiter_boundary_zero_rpm():
    limiter = InMemoryLimiter()
    from effortgate.config import settings
    original_rpm = settings.MAX_RPM
    settings.MAX_RPM = 0  # Zero limit edge case
    
    try:
        # First request on empty deque must raise RateLimitExceeded without IndexError
        with pytest.raises(RateLimitExceeded) as exc:
            await limiter.check_and_record("zero_rpm_key", 10)
        assert exc.value.retry_after >= 1
    finally:
        settings.MAX_RPM = original_rpm

@pytest.mark.asyncio
async def test_in_memory_limiter_boundary_tpm_exceeded_on_empty_window():
    limiter = InMemoryLimiter()
    from effortgate.config import settings
    original_tpm = settings.MAX_TPM
    settings.MAX_TPM = 50
    
    try:
        # First request exceeding TPM on empty window must raise RateLimitExceeded without IndexError
        with pytest.raises(RateLimitExceeded) as exc:
            await limiter.check_and_record("tpm_key", 100)
        assert exc.value.retry_after >= 1
    finally:
        settings.MAX_TPM = original_tpm

@pytest.mark.asyncio
async def test_redis_limiter_fallback_on_connection_error():
    from unittest.mock import AsyncMock
    from effortgate.limiter import RedisLimiter
    
    limiter = RedisLimiter("redis://localhost:6379/0")
    # Simulate Redis connection failure
    limiter.redis.ttl = AsyncMock(side_effect=ConnectionError("Redis connection refused"))
    
    # Should not raise ConnectionError; should fallback to InMemoryLimiter
    await limiter.check_and_record("fallback_client", 100)
    
    # Verify fallback has recorded the window
    assert len(limiter.fallback.rpm_windows["fallback_client"]) == 1

@pytest.mark.asyncio
async def test_redis_limiter_fallback_sync_headers():
    from unittest.mock import AsyncMock
    from effortgate.limiter import RedisLimiter
    
    limiter = RedisLimiter("redis://localhost:6379/0")
    # Simulate Redis failure on set
    limiter.redis.set = AsyncMock(side_effect=ConnectionError("Redis set failed"))
    
    # Should fallback to InMemoryLimiter sync_headers
    await limiter.sync_headers("fallback_key", {"retry-after": "20"})
    
    assert "fallback_key" in limiter.fallback.cooldowns



