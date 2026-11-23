import time
import argparse
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from effortgate.router import (
    prepare_request,
    route_streaming_request,
    route_non_streaming_request,
    route_request
)
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
    # 1. Body parsing validation
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}}
        )
        
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Request body must be a JSON object", "type": "invalid_request_error"}}
        )

    # 2. Schema validation
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "'messages' field is required and must be a non-empty list", "type": "invalid_request_error"}}
        )

    # 3. Extract Authorization / API Key header if provided
    auth_header = request.headers.get("authorization")
    api_key = None
    if auth_header:
        if auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()
        else:
            api_key = auth_header.strip()
    elif "api-key" in request.headers:
        api_key = request.headers.get("api-key").strip()

    if api_key and "api_key" not in body:
        body["api_key"] = api_key

    # 4. Check for client model lock override flags
    # Supported: Header 'X-EffortGate-Lock-Model: true' or 'X-EffortGate-Auto-Route: false'
    # Or in JSON body: 'lock_model: true' or 'auto_route: false'
    lock_model = False
    lock_header = request.headers.get("x-effortgate-lock-model", "").lower()
    auto_route_header = request.headers.get("x-effortgate-auto-route", "").lower()
    if lock_header in ("true", "1") or auto_route_header in ("false", "0"):
        lock_model = True
    elif body.get("lock_model") is True or body.get("auto_route") is False:
        lock_model = True

    # 5. Pre-flight check & preparation (quota tracking, effort classification, model routing, budgeting)
    # Any RateLimitExceeded raised here is caught by FastAPI exception handler returning HTTP 429
    tier, routed_model = await prepare_request(body, lock_model=lock_model)
    response_headers = {
        "X-EffortGate-Tier": tier,
        "X-EffortGate-Routed-Model": routed_model,
        "X-EffortGate-Model-Locked": str(lock_model).lower()
    }

    is_stream = body.get("stream", False)
    
    if is_stream:
        return StreamingResponse(
            route_streaming_request(body, tier=tier, lock_model=lock_model),
            media_type="text/event-stream",
            headers=response_headers
        )
    else:
        try:
            result = await route_non_streaming_request(body, tier=tier, lock_model=lock_model)
            return JSONResponse(content=result, headers=response_headers)
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            return JSONResponse(
                status_code=status_code,
                content={"error": {"message": str(e), "type": "internal_error"}},
                headers=response_headers
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
