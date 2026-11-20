import time
import argparse
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from effortgate.router import route_streaming_request, route_non_streaming_request, route_request
from effortgate.limiter import RateLimitExceeded

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
    return {
        "status": "ok",
        "description": "Metrics not fully implemented in mock",
    }

@app.get("/v1/models")
async def list_models():
    # Standard OpenAI models response
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o-mini", "object": "model", "created": int(time.time()), "owned_by": "openai"},
            {"id": "gpt-4o", "object": "model", "created": int(time.time()), "owned_by": "openai"},
            {"id": "claude-3-5-sonnet-20240620", "object": "model", "created": int(time.time()), "owned_by": "anthropic"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})
        
    is_stream = body.get("stream", False)
    
    if is_stream:
        return StreamingResponse(
            route_streaming_request(body),
            media_type="text/event-stream"
        )
    else:
        try:
            result = await route_non_streaming_request(body)
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "internal_error"}}
            )

def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="EffortGate AI Gateway Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    
    uvicorn.run("effortgate.server:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
