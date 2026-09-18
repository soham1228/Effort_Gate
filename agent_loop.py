import json
import logging
from typing import AsyncGenerator, Dict, Any, List
import litellm

logger = logging.getLogger(__name__)

async def run_agent_loop(request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    Executes the AGENT tier structured Plan-and-Execute loop.
    1. Emit planning progress chunks via SSE.
    2. Prompt planner to generate a structured 3-step action DAG.
    3. Execute steps and emit unified summary.
    """
    def emit_chunk(text: str) -> str:
        return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"
        
    yield emit_chunk("[AGENT] Starting planning phase...\n")
    
    # In a fully realized system, we would:
    # - Run a specific prompt to generate the plan
    # - Parse the plan DAG
    # - Execute each step using litellm
    # - Handle retries for failures
    
    # For now, we simulate the structured loop by making a single larger call 
    # but injecting a planning instruction, and streaming it.
    
    messages = request_data.get("messages", [])
    
    # Inject planning instructions
    planning_instruction = (
        "You are in AGENT mode. Follow this structured process:\n"
        "1. Create a brief 3-step plan.\n"
        "2. Execute the steps logically.\n"
        "3. Provide a unified summary with code diffs.\n"
    )
    
    # Try to insert into system prompt
    inserted = False
    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = str(msg["content"]) + "\n\n" + planning_instruction
            inserted = True
            break
            
    if not inserted:
        messages.insert(0, {"role": "system", "content": planning_instruction})
        
    request_data["messages"] = messages
    
    yield emit_chunk("[AGENT] Executing steps...\n\n")
    
    try:
        response = await litellm.acompletion(**request_data)
        async for chunk in response:
            data = chunk.model_dump(exclude_unset=True)
            yield f"data: {json.dumps(data)}\n\n"
            
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in agent execution: {str(e)}")
        error_msg = {"error": {"message": str(e), "type": "agent_internal_error", "param": None, "code": "500"}}
        yield f"data: {json.dumps(error_msg)}\n\n"
        yield "data: [DONE]\n\n"
