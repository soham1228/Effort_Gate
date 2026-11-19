import copy
from typing import List, Dict, Any

def trim_messages(messages: List[Dict[str, Any]], tier: str) -> List[Dict[str, Any]]:
    """
    Trims the message payload based on the tier.
    FAST: 8,000 chars (~2k tokens)
    NORMAL: 32,000 chars (~8k tokens)
    AGENT: 64,000 chars (~16k tokens)
    
    Keeps system prompts and latest user turns intact; truncates middle history.
    """
    if tier == "FAST":
        max_chars = 8_000
    elif tier == "NORMAL":
        max_chars = 32_000
    else:
        max_chars = 64_000

    # Calculate total current chars
    def get_content_len(msg: Dict[str, Any]) -> int:
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        elif isinstance(content, list):
            return sum(len(p.get("text", "")) for p in content if isinstance(p, dict) and "text" in p)
        return 0

    total_chars = sum(get_content_len(m) for m in messages)
    if total_chars <= max_chars:
        # For FAST, we also inject a system directive
        if tier == "FAST":
            return inject_fast_directive(messages)
        return copy.deepcopy(messages)

    # Need to trim middle history
    trimmed = []
    
    # Always keep first message (often system prompt)
    if messages:
        trimmed.append(copy.deepcopy(messages[0]))
        
    # Always keep the last 2 messages (usually user + assistant or just user)
    tail_msgs = []
    if len(messages) > 1:
        tail_count = min(2, len(messages) - 1)
        tail_msgs = copy.deepcopy(messages[-tail_count:])
        
    current_len = sum(get_content_len(m) for m in trimmed) + sum(get_content_len(m) for m in tail_msgs)
    
    # Fill in from the middle starting from the most recent
    middle_msgs = messages[1:-len(tail_msgs)] if tail_msgs else []
    
    # Add from end of middle to beginning
    added_middle = []
    for msg in reversed(middle_msgs):
        msg_len = get_content_len(msg)
        if current_len + msg_len <= max_chars:
            added_middle.insert(0, copy.deepcopy(msg))
            current_len += msg_len
        else:
            # If a single message is too big but we still have budget, we could truncate it, 
            # but for simplicity, we just skip it or stop adding middle messages.
            break
            
    final_messages = trimmed + added_middle + tail_msgs
    
    if tier == "FAST":
        return inject_fast_directive(final_messages)
    return final_messages

def inject_fast_directive(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Injects a minimalist context directive for FAST tasks."""
    final = copy.deepcopy(messages)
    directive = "PERFORM PRECISE, LOCALIZED EDITS WITH ZERO UNSOLICITED REFACTORING."
    
    # Try to find a system prompt to append to
    for msg in final:
        if msg.get("role") == "system":
            if isinstance(msg["content"], str):
                msg["content"] += "\n" + directive
            return final
            
    # If no system prompt, inject one
    final.insert(0, {"role": "system", "content": directive})
    return final
