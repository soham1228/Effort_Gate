# EffortGate

OpenAI-compatible AI proxy that sits between your coding tools (Cursor, Continue, Aider) and LLM providers to cut token costs and prevent `429 Too Many Requests`.

## How It Works

1. **Quota Limiter:** Tracks RPM, RPD, and TPM with rolling sliding windows (In-Memory or Redis).
2. **Effort Classifier:** Sub-millisecond heuristics categorize requests into `FAST`, `NORMAL`, or `AGENT`.
3. **Context Budgeter:** Trims bloated conversation history based on the task tier (e.g. `FAST` tasks are capped at ~2k tokens).
4. **Universal Router:** Forwards requests to any provider via `litellm`.

---

## Developer Quickstart

### 1. Install & Setup
```bash
git clone https://github.com/soham1228/Effort_Gate.git
cd Effort_Gate
pip install -e .[dev]
cp .env.example .env
```

Set your API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) in `.env`.

### 2. Run the Gateway
```bash
# Via CLI
effortgate --port 8000 --reload

# Or via Python module
python -m effortgate.server --port 8000
```

### 3. Point Your Tools to EffortGate
Set your tool's OpenAI Base URL to:
```text
http://localhost:8000/v1
```

### 4. Run Tests
```bash
pytest
```

---

## Prompt Tags (Overrides)
You can force a tier inside any prompt:
- `!fast <prompt>`: Forces minimal context (~2k tokens) and localized edits.
- `!normal <prompt>`: Default single-turn flow.
- `!agent <prompt>`: Full context and planning mode.
