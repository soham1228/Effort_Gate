import pytest
from effortgate.classifier import classify_effort, get_keyword_score, get_path_score

@pytest.mark.asyncio
async def test_fast_classification():
    messages = [{"role": "user", "content": "fix the button color in the modal"}]
    # 'button', 'color', 'modal' are narrow keywords.
    # very short message (7 words) -> low word count score
    tier = await classify_effort(messages)
    assert tier == "FAST"

@pytest.mark.asyncio
async def test_agent_broad_keyword_override():
    # 'button' is narrow, but 'refactor', 'across', 'pipeline' are broad
    messages = [{"role": "user", "content": "refactor modal button across the pipeline"}]
    tier = await classify_effort(messages)
    assert tier == "AGENT"

@pytest.mark.asyncio
async def test_manual_override_agent():
    messages = [{"role": "user", "content": "!agent fix typo"}]
    tier = await classify_effort(messages)
    assert tier == "AGENT"
    
@pytest.mark.asyncio
async def test_manual_override_fast():
    messages = [{"role": "user", "content": "!fast refactor entire system"}]
    tier = await classify_effort(messages)
    assert tier == "FAST"

@pytest.mark.asyncio
async def test_multi_turn_protection():
    # Even if it's brief, a 4-turn conversation shouldn't be FAST.
    messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "response 1"},
        {"role": "user", "content": "second turn"},
        {"role": "assistant", "content": "response 2"},
        {"role": "user", "content": "why did it fail?"} # Brief, but multi-turn
    ]
    tier = await classify_effort(messages)
    # The rule says: return "NORMAL" (force minimum NORMAL)
    # Let's ensure it's not FAST
    assert tier != "FAST"
