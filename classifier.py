import re
import litellm
from typing import List, Dict, Any, Tuple
from config import settings

# Keywords sets
NARROW_KEYWORDS = {
    "button", "label", "typo", "copy", "text", "style", "css", "color", "colour", 
    "tooltip", "rename", "padding", "margin", "wording", "spacing", "icon", "modal", 
    "header", "footer", "badge", "avatar", "border", "hide", "show", "disable", 
    "enable", "center", "align", "truncate", "dropdown", "navbar"
}

BROAD_KEYWORDS = {
    "system", "refactor", "architecture", "migrate", "migration", "redesign", 
    "overhaul", "entire", "across", "pipeline", "integration", "rewrite", "rearchitect"
}

PATH_REGEX = re.compile(r'(?:[\w\-\.]+\/)+[\w\-\.]+\.\w+')

async def fallback_micro_model(prompt: str) -> str:
    """Uses a cheap micro-model to classify ambiguous tasks."""
    try:
        response = await litellm.acompletion(
            model=settings.EFFORT_MICRO_MODEL,
            messages=[
                {"role": "system", "content": "You are a task effort classifier. Return exactly one word: FAST, NORMAL, or AGENT based on task complexity. Answer FAST for trivial UI tweaks/typos, NORMAL for standard logic changes, AGENT for broad architectural refactors. Only return the word."},
                {"role": "user", "content": prompt[:1000]}
            ],
            temperature=0.1,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip().upper()
        if content in ["FAST", "NORMAL", "AGENT"]:
            return content
        return "NORMAL"
    except Exception:
        return "NORMAL"

def get_word_count_score(prompt: str) -> float:
    words = len(prompt.split())
    if words <= 8: return 0.10
    if words <= 20: return 0.35
    if words <= 40: return 0.60
    if words <= 80: return 0.80
    return 1.00

def get_keyword_score(prompt: str) -> float:
    prompt_lower = prompt.lower()
    words = set(re.findall(r'\b\w+\b', prompt_lower))
    
    # CRITICAL BROAD PRECEDENCE
    if any(bk in words for bk in BROAD_KEYWORDS):
        return 0.95
        
    if any(nk in words for nk in NARROW_KEYWORDS):
        return 0.10
        
    return 0.50

def get_path_score(prompt: str) -> float:
    paths = PATH_REGEX.findall(prompt)
    num_paths = len(paths)
    if num_paths == 0: return 0.50
    if num_paths == 1: return 0.15
    if 2 <= num_paths <= 3: return 0.40
    return 0.75

async def classify_effort(messages: List[Dict[str, Any]]) -> str:
    """
    Returns tier: "FAST", "NORMAL", or "AGENT"
    """
    if not messages:
        return "NORMAL"
        
    # Get the latest user prompt
    last_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_prompt = content
            elif isinstance(content, list):
                last_prompt = " ".join([p.get("text", "") for p in content if isinstance(p, dict) and "text" in p])
            break

    # Manual Overrides Check (Priority 0)
    prompt_lower = last_prompt.lower().strip()
    if prompt_lower.startswith("!fast") or "!fast" in prompt_lower:
        return "FAST"
    if prompt_lower.startswith("!normal") or "!normal" in prompt_lower:
        return "NORMAL"
    if prompt_lower.startswith("!agent") or "!agent" in prompt_lower:
        return "AGENT"

    # Conversational Multi-Turn Protection
    if len(messages) > 3:
        # Check if the prompt is brief or deictic
        if len(prompt_lower.split()) <= 15:
            return "NORMAL" # Force minimum NORMAL

    # Signals
    s1 = get_word_count_score(last_prompt)
    s2 = get_keyword_score(last_prompt)
    s3 = get_path_score(last_prompt)
    s4 = settings.DEFAULT_HISTORICAL_RATE # Could be dynamic later
    
    score = (
        s1 * settings.WEIGHT_WORD_COUNT +
        s2 * settings.WEIGHT_KEYWORDS +
        s3 * settings.WEIGHT_FILE_PATHS +
        s4 * settings.WEIGHT_HISTORY
    )
    
    # Classification Thresholds
    tier = "NORMAL"
    if score <= 0.33:
        tier = "FAST"
    elif score > 0.66:
        tier = "AGENT"
        
    # Micro-Model Fallback for ambiguity or missing keywords
    if 0.30 <= score <= 0.40 and s2 == 0.50:
        micro_tier = await fallback_micro_model(last_prompt)
        # Conservative Guardrail
        if micro_tier == "FAST":
            tier = "NORMAL" # downgrade safely to NORMAL for safety
        else:
            tier = micro_tier

    return tier
