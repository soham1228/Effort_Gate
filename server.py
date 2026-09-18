import time
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from router import route_request
from limiter import RateLimitExceeded

app = FastAPI(
    title="EffortGate",
    description="Universal Multi-Provider AI Gateway with Adaptive Effort Sizing & Token Protection",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For web-based IDEs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": {"message": exc.message, "type": "rate_limit_exceeded"}},
        headers={"Retry-After": str(exc.retry_after)}
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    # Example metrics endpoint
    return {
        "status": "ok",
        "description": "Metrics not fully implemented in mock",
    }

@app.get("/v1/models")
async def list_models():
    # Mocking standard OpenAI models response
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o-mini", "object": "model", "created": int(time.time()), "owned_by": "openai"},
            {"id": "gpt-4", "object": "model", "created": int(time.time()), "owned_by": "openai"},
            {"id": "claude-3-5-sonnet-20240620", "object": "model", "created": int(time.time()), "owned_by": "anthropic"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON"}})
        
    return StreamingResponse(
        route_request(body),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
