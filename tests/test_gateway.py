import pytest
import json
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from effortgate.server import app

class MockChunk:
    def __init__(self, content: str = "Hello from EffortGate!"):
        self.content = content
        
    def model_dump(self, **kwargs):
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {"content": self.content},
                "index": 0,
                "finish_reason": None
            }]
        }

async def mock_streaming_response(*args, **kwargs):
    yield MockChunk("Hello from ")
    yield MockChunk("EffortGate!")

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_list_models_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1

@pytest.mark.asyncio
async def test_chat_completions_endpoint_mocked():
    transport = ASGITransport(app=app)
    with patch("effortgate.router.litellm.acompletion", new=AsyncMock(side_effect=mock_streaming_response)):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/v1/chat/completions", json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hello"}],
                "stream": True
            })
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        content = response.text
        assert "data: " in content
        assert "Hello from EffortGate!" in content or "Hello from " in content
        assert "data: [DONE]" in content

@pytest.mark.asyncio
async def test_chat_completions_non_streaming_mocked():
    transport = ASGITransport(app=app)
    mock_response = {
        "id": "chatcmpl-mock-nonstream",
        "object": "chat.completion",
        "choices": [{
            "message": {"role": "assistant", "content": "Hello non-streaming!"},
            "finish_reason": "stop",
            "index": 0
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    }
    with patch("effortgate.router.litellm.acompletion", new=AsyncMock(return_value=mock_response)):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/v1/chat/completions", json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hello"}],
                "stream": False
            })
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        data = response.json()
        assert data["id"] == "chatcmpl-mock-nonstream"
        assert data["choices"][0]["message"]["content"] == "Hello non-streaming!"

@pytest.mark.asyncio
async def test_chat_completions_missing_messages_returns_400():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json={
            "model": "openai/gpt-4o-mini"
            # messages omitted
        })
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"

@pytest.mark.asyncio
async def test_chat_completions_rate_limit_returns_429():
    from effortgate.limiter import RateLimitExceeded
    transport = ASGITransport(app=app)
    with patch("effortgate.router.limiter.check_and_record", side_effect=RateLimitExceeded("RPM limit exceeded", retry_after=45)):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/v1/chat/completions", json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True  # Even for streaming requests, 429 is returned BEFORE stream starts
            })
    assert response.status_code == 429
    assert response.headers["retry-after"] == "45"
    assert response.json()["error"]["type"] == "rate_limit_exceeded"

@pytest.mark.asyncio
async def test_chat_completions_bearer_token_passthrough():
    transport = ASGITransport(app=app)
    mock_response = {
        "id": "chatcmpl-mock-auth",
        "object": "chat.completion",
        "choices": [{
            "message": {"role": "assistant", "content": "Auth test"},
            "finish_reason": "stop",
            "index": 0
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    }
    with patch("effortgate.router.litellm.acompletion", new=AsyncMock(return_value=mock_response)) as mock_call:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "fix button color"}],
                    "stream": False
                },
                headers={"Authorization": "Bearer sk-test-key-12345"}
            )
        assert response.status_code == 200
        mock_call.assert_awaited_once()
        _, kwargs = mock_call.call_args
        assert kwargs.get("api_key") == "sk-test-key-12345"

@pytest.mark.asyncio
async def test_chat_completions_auto_routing_fast():
    transport = ASGITransport(app=app)
    mock_response = {
        "id": "chatcmpl-mock-fast",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Fast edit"}, "index": 0}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    }
    with patch("effortgate.router.litellm.acompletion", new=AsyncMock(return_value=mock_response)) as mock_call:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "expensive-unnecessary-model",
                    "messages": [{"role": "user", "content": "fix button color in modal"}],
                    "stream": False
                }
            )
        assert response.status_code == 200
        assert response.headers["x-effortgate-tier"] == "FAST"
        assert response.headers["x-effortgate-routed-model"] == "openai/gpt-4o-mini"
        assert response.headers["x-effortgate-model-locked"] == "false"
        _, kwargs = mock_call.call_args
        assert kwargs.get("model") == "openai/gpt-4o-mini"

@pytest.mark.asyncio
async def test_chat_completions_auto_routing_agent():
    transport = ASGITransport(app=app)
    mock_response = {
        "id": "chatcmpl-mock-agent",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Agent plan"}, "index": 0}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 15, "total_tokens": 30}
    }
    with patch("effortgate.router.litellm.acompletion", new=AsyncMock(return_value=mock_response)) as mock_call:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "refactor database migration across entire pipeline"}],
                    "stream": False
                }
            )
        assert response.status_code == 200
        assert response.headers["x-effortgate-tier"] == "AGENT"
        assert response.headers["x-effortgate-routed-model"] == "anthropic/claude-3-5-sonnet"
        assert response.headers["x-effortgate-model-locked"] == "false"
        _, kwargs = mock_call.call_args
        assert kwargs.get("model") == "anthropic/claude-3-5-sonnet"

@pytest.mark.asyncio
async def test_chat_completions_model_lock_override():
    transport = ASGITransport(app=app)
    mock_response = {
        "id": "chatcmpl-mock-locked",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Locked model response"}, "index": 0}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    }
    with patch("effortgate.router.litellm.acompletion", new=AsyncMock(return_value=mock_response)) as mock_call:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o",
                    "messages": [{"role": "user", "content": "fix button color in modal"}],
                    "stream": False
                },
                headers={"X-EffortGate-Lock-Model": "true"}
            )
        assert response.status_code == 200
        assert response.headers["x-effortgate-tier"] == "FAST"
        # User locked the model, so it must stay gpt-4o instead of being downscaled to gpt-4o-mini!
        assert response.headers["x-effortgate-routed-model"] == "openai/gpt-4o"
        assert response.headers["x-effortgate-model-locked"] == "true"
        _, kwargs = mock_call.call_args
        assert kwargs.get("model") == "openai/gpt-4o"




