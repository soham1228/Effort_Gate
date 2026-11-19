import pytest
from effortgate.budget import trim_messages, inject_fast_directive

def test_trim_messages_within_budget():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a python function to add two numbers."}
    ]
    trimmed = trim_messages(messages, "NORMAL")
    assert len(trimmed) == 2
    assert trimmed[0]["content"] == "You are a helpful assistant."
    assert trimmed[1]["content"] == "Write a python function to add two numbers."

def test_fast_directive_appended_to_system():
    messages = [
        {"role": "system", "content": "Base system prompt."},
        {"role": "user", "content": "Fix typo"}
    ]
    trimmed = trim_messages(messages, "FAST")
    assert "PERFORM PRECISE, LOCALIZED EDITS WITH ZERO UNSOLICITED REFACTORING." in trimmed[0]["content"]
    assert "Base system prompt." in trimmed[0]["content"]

def test_fast_directive_injected_when_no_system_prompt():
    messages = [
        {"role": "user", "content": "Fix typo"}
    ]
    trimmed = trim_messages(messages, "FAST")
    assert len(trimmed) == 2
    assert trimmed[0]["role"] == "system"
    assert "PERFORM PRECISE, LOCALIZED EDITS WITH ZERO UNSOLICITED REFACTORING." in trimmed[0]["content"]
    assert trimmed[1]["content"] == "Fix typo"

def test_trim_middle_messages_when_over_budget():
    # FAST tier budget is 8,000 chars.
    system_msg = {"role": "system", "content": "System prompt " + "s" * 100}
    # Create several large middle messages of 3,000 chars each (total 9,000 chars)
    middle_1 = {"role": "user", "content": "Middle 1 " + "m" * 3000}
    middle_2 = {"role": "assistant", "content": "Middle 2 " + "m" * 3000}
    middle_3 = {"role": "user", "content": "Middle 3 " + "m" * 3000}
    # Tail message (last user message)
    tail_msg = {"role": "user", "content": "Recent prompt " + "r" * 500}
    
    messages = [system_msg, middle_1, middle_2, middle_3, tail_msg]
    trimmed = trim_messages(messages, "FAST")
    
    # System message must be preserved
    assert trimmed[0]["role"] == "system"
    # Tail message must be preserved
    assert trimmed[-1]["content"] == tail_msg["content"]
    # Total characters (excluding injected directive) should fit within max_chars
    # The middle messages cannot all fit (3000*3 = 9000 > 8000), so some middle turns are dropped
    assert len(trimmed) < len(messages)
    contents = [m["content"] for m in trimmed]
    # middle_1 is the oldest middle turn, so it should be the first to get dropped
    assert middle_1["content"] not in contents

def test_multipart_content_length_calculation():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [
            {"type": "text", "text": "part 1 "},
            {"type": "text", "text": "part 2"}
        ]}
    ]
    trimmed = trim_messages(messages, "NORMAL")
    assert len(trimmed) == 2
    assert isinstance(trimmed[1]["content"], list)
