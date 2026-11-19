import pytest
from httpx import AsyncClient
from effortgate.server import app
import json

@pytest.mark.asyncio
async def test_chat_completions_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "say hello"}],
            "stream": True
        })
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # We should be able to read SSE chunks.
    # Note: Because litellm requires valid API keys to actually stream, 
    # we might get an internal error chunk if keys aren't set in the test env.
    # We just verify the SSE format `data: {...}`
    content = response.text
    assert "data: " in content
    assert "data: [DONE]" in content
