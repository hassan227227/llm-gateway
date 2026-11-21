"""FastAPI front end for the gateway.

Run:  uvicorn api:app --reload
Auth: send `x-api-key: <any string>`; each key gets its own rate-limit bucket and usage ledger.
Providers: primary Claude model, with a cheaper model as automatic fallback.
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from gateway import (
    AllProvidersFailed,
    AnthropicProvider,
    ChatRequest,
    Gateway,
    RateLimited,
    RateLimiter,
)

app = FastAPI(title="LLM Gateway")
_gateway: Gateway | None = None


def get_gateway() -> Gateway:
    global _gateway
    if _gateway is None:
        _gateway = Gateway(
            providers=[
                AnthropicProvider(name="primary"),
                AnthropicProvider(name="fallback", model_override=os.getenv("FALLBACK_MODEL", "claude-haiku-5-5")),
            ],
            limiter=RateLimiter(
                capacity=float(os.getenv("RATE_BURST", "10")), rate=float(os.getenv("RATE_PER_SEC", "1"))
            ),
        )
    return _gateway


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class ChatBody(BaseModel):
    model: str = os.getenv("DEFAULT_MODEL", "claude-opus-5-5")
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat")
def chat(body: ChatBody, x_api_key: str = Header(...), gw: Gateway = Depends(get_gateway)):
    request = ChatRequest.of(body.model, [m.model_dump() for m in body.messages], body.max_tokens, body.temperature)
    try:
        response = gw.chat(x_api_key, request)
    except RateLimited as exc:
        raise HTTPException(429, detail=str(exc), headers={"Retry-After": str(int(exc.retry_after) + 1)})
    except AllProvidersFailed as exc:
        raise HTTPException(502, detail=str(exc))
    return {
        "text": response.text,
        "model": response.model,
        "provider": response.provider,
        "cached": response.cached,
        "usage": {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
    }


@app.get("/v1/usage")
def usage(x_api_key: str = Header(...), gw: Gateway = Depends(get_gateway)):
    return gw.costs.by_key.get(x_api_key, {"requests": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
