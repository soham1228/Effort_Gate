# EffortGate

Universal Multi-Provider AI Gateway with Adaptive Effort Sizing & Token Protection.

EffortGate sits between your coding interfaces (Cursor, Continue.dev, Claude Code, Aider, Neovim) and upstream AI providers (OpenAI, Anthropic Claude, Google Gemini, etc.).
## Features
- **Quota Protection:** Actively enforce rolling sliding-window limits for RPM, RPD, and TPM per provider/key to prevent `429 Too Many Requests`.
- **Adaptive Effort Sizing:** Zero-cost heuristics classify tasks into `FAST`, `NORMAL`, and `AGENT` tiers.
- **Dynamic Context Budgeting:** Trim massive histories down to focused context blocks based on task effort tier.
- **Routing:** Unifies calling across multiple providers via `litellm`.

## Setup

1. Copy `.env.example` to `.env` and fill in your API keys.
2. Run with Docker Compose:

```bash
docker-compose up -d
```

3. Point your AI tooling to `http://localhost:8000/v1` for standard OpenAI drop-in support.

## Testing

Install dependencies and run pytest:

```bash
pip install -e .[dev]
pytest
```
