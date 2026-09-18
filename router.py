import json
import logging
from typing import AsyncGenerator, Dict, Any, List
import litellm
from config import settings
from limiter import limiter
from classifier import classify_effort
from budget import trim_messages
from tokenizer import estimate_prompt_tokens
from agent_loop import run_agent_loop

logger = logging.getLogger(__name__)

async def route_request(request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    Main routing logic.
    1. Estimate tokens.
    2. Check quota.
    3. Classify effort.
    4. Trim budget.
    5. Route to direct stream or agent loop.
    """
    model = request_data.get("model", settings.DEFAULT_MODEL)
    messages = request_data.get("messages", [])
    
    # 1. Estimate tokens
    estimated_tokens = estimate_prompt_tokens(messages)
    
    # 2. Check quota
    # Use API key or model as the rate limit key for simplicity. 
    # In a real setup, this might be a user ID.
    key = "global" 
    await limiter.check_and_record(key, estimated_tokens)
    
    # 3. Classify effort
    tier = await classify_effort(messages)
    
    # 4. Trim budget
    trimmed_messages = trim_messages(messages, tier)
    
    # Ensure model configuration is passed for litellm
    request_data["model"] = model
    request_data["messages"] = trimmed_messages
    
    # Remove unsupported top-level params if any (litellm handles most, but just in case)
    # We enforce stream=True for our output
    request_data["stream"] = True
    
    # Emit a system message indicating the tier (optional, for visibility)
    # yield f"data: {json.dumps({'choices': [{'delta': {'content': f'[EffortGate: {tier} tier assigned]\\n'}}]})}\n\n"

    if tier in ["FAST", "NORMAL"]:
        # Direct SSE streaming
        async for chunk in _stream_litellm(request_data):
            yield chunk
    else:
        # AGENT tier: structured Plan-and-Execute loop
        async for chunk in run_agent_loop(request_data):
            yield chunk

async def _stream_litellm(request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    try:
        response = await litellm.acompletion(**request_data)
        async for chunk in response:
            # chunk is a ModelResponse object, we can dump its dict
            # or directly format as OpenAI SSE
            data = chunk.model_dump(exclude_unset=True)
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in litellm stream: {str(e)}")
        error_msg = {"error": {"message": str(e), "type": "internal_error", "param": None, "code": "500"}}
        yield f"data: {json.dumps(error_msg)}\n\n"
        yield "data: [DONE]\n\n"
