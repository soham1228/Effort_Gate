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
