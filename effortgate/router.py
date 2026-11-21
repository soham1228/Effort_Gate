import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
import litellm
from effortgate.config import settings
from effortgate.limiter import limiter
from effortgate.classifier import classify_effort
from effortgate.budget import trim_messages
from effortgate.tokenizer import estimate_prompt_tokens
from effortgate.agent_loop import run_agent_loop

logger = logging.getLogger(__name__)

async def prepare_request(request_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Common preparation:
    1. Estimate tokens.
    2. Check rate limit quota (raises RateLimitExceeded if quota exceeded).
    3. Classify effort tier.
    4. Resolve routed model based on tier.
    5. Trim message history context budget.
    """
    requested_model = request_data.get("model", settings.DEFAULT_MODEL)
    messages = request_data.get("messages", [])
    
    # 1. Estimate tokens
    estimated_tokens = estimate_prompt_tokens(messages)
    
    # 2. Check quota
    key = "global" 
    await limiter.check_and_record(key, estimated_tokens)
    
    # 3. Classify effort
    tier = await classify_effort(messages)

    # 4. Resolve routed model based on tier
    routed_model = settings.get_model_for_tier(tier, requested_model)
    
    # 5. Trim budget
    trimmed_messages = trim_messages(messages, tier)
    
    # Ensure model configuration is passed for litellm
    request_data["model"] = routed_model
    request_data["messages"] = trimmed_messages
    return tier, routed_model

async def route_streaming_request(request_data: Dict[str, Any], tier: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    Handles streaming chat completion requests (stream=True).
    """
    if tier is None:
        tier, _ = await prepare_request(request_data)
    request_data["stream"] = True

    if tier in ["FAST", "NORMAL"]:
        async for chunk in _stream_litellm(request_data):
            yield chunk
    else:
        async for chunk in run_agent_loop(request_data):
            yield chunk

# Backward compatibility alias
route_request = route_streaming_request

async def route_non_streaming_request(request_data: Dict[str, Any], tier: Optional[str] = None) -> Dict[str, Any]:
    """
    Handles standard non-streaming chat completion requests (stream=False).
    Returns standard OpenAI-compatible JSON dict.
    """
    if tier is None:
        tier, _ = await prepare_request(request_data)
    request_data["stream"] = False

    if tier in ["FAST", "NORMAL"]:
        response = await litellm.acompletion(**request_data)
        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_unset=True)
        return dict(response)
    else:
        # AGENT mode non-streaming: inject planning instruction and call acompletion
        messages = request_data.get("messages", [])
        planning_instruction = (
            "You are in AGENT mode. Follow this structured process:\n"
            "1. Create a brief 3-step plan.\n"
            "2. Execute the steps logically.\n"
            "3. Provide a unified summary with code diffs.\n"
        )
        inserted = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = str(msg["content"]) + "\n\n" + planning_instruction
                inserted = True
                break
        if not inserted:
            messages.insert(0, {"role": "system", "content": planning_instruction})
        request_data["messages"] = messages
        
        response = await litellm.acompletion(**request_data)
        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_unset=True)
        return dict(response)

async def _stream_litellm(request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    try:
        response = await litellm.acompletion(**request_data)
        async for chunk in response:
            data = chunk.model_dump(exclude_unset=True)
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in litellm stream: {str(e)}")
        error_msg = {"error": {"message": str(e), "type": "internal_error", "param": None, "code": "500"}}
        yield f"data: {json.dumps(error_msg)}\n\n"
        yield "data: [DONE]\n\n"
