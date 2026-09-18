import tiktoken
from typing import List, Dict, Any

# Fallback ratio: 3 characters per token for source code
CHARS_PER_TOKEN = 3

def _get_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def estimate_text_tokens(text: str) -> int:
    """Estimate the number of tokens in a string."""
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:
            pass
    return len(text) // CHARS_PER_TOKEN

def estimate_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate the number of tokens in a list of messages."""
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += estimate_text_tokens(value)
            elif isinstance(value, list):
                # Handle multi-part messages (e.g., image_url, text parts)
                for part in value:
                    if isinstance(part, dict) and "text" in part:
                        num_tokens += estimate_text_tokens(part["text"])
            if key == "name":  # If there's a name, the role is omitted
                num_tokens -= 1  # Role is usually omitted when name is present
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return max(0, num_tokens)
