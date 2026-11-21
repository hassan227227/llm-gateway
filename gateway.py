"""LLM gateway core: rate limiting, response caching, retries with fallback, cost tracking.

Everything here is plain Python with injectable clocks and providers, so it is fully
unit-testable without network access or an API key.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Protocol


# ---------------------------------------------------------------- data types

@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple  # tuple of (role, content) pairs, hashable
    max_tokens: int = 1024
    temperature: float = 0.0

    @staticmethod
    def of(model: str, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> "ChatRequest":
        return ChatRequest(
            model=model,
            messages=tuple((m["role"], m["content"]) for m in messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def cache_key(self) -> str:
        payload = json.dumps(
            [self.model, self.messages, self.max_tokens, self.temperature], sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class ChatResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool = False
    provider: str = ""


class Provider(Protocol):
    name: str

    def complete(self, request: ChatRequest) -> ChatResponse: ...


class RateLimited(Exception):
    def __init__(self, retry_after: float):
        super().__init__(f"rate limited, retry in {retry_after:.2f}s")
        self.retry_after = retry_after


class AllProvidersFailed(Exception):
    pass


# --------------------------------------------------------------- rate limiter

class TokenBucket:
    """Classic token bucket: `capacity` burst, refilled at `rate` tokens per second."""

    def __init__(self, capacity: float, rate: float, clock: Callable[[], float] = time.monotonic):
        self.capacity = capacity
        self.rate = rate
        self._clock = clock
        self._tokens = capacity
        self._last = clock()

    def try_acquire(self, cost: float = 1.0) -> float:
        """Return 0.0 if allowed, otherwise seconds until `cost` tokens are available."""
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now
        if self._tokens >= cost:
            self._tokens -= cost
            return 0.0
        return (cost - self._tokens) / self.rate


class RateLimiter:
    """One token bucket per API key."""

    def __init__(self, capacity: float = 10, rate: float = 1.0, clock: Callable[[], float] = time.monotonic):
        self._capacity, self._rate, self._clock = capacity, rate, clock
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def check(self, api_key: str) -> None:
        with self._lock:
            bucket = self._buckets.setdefault(
                api_key, TokenBucket(self._capacity, self._rate, self._clock)
            )
            wait = bucket.try_acquire()
        if wait > 0:
            raise RateLimited(wait)


# ---------------------------------------------------------------------- cache

class ResponseCache:
    """LRU cache with per-entry TTL. Only deterministic requests (temperature 0) are cached."""

    def __init__(self, max_entries: int = 256, ttl: float = 300.0, clock: Callable[[], float] = time.monotonic):
        self.max_entries, self.ttl, self._clock = max_entries, ttl, clock
        self._data: OrderedDict[str, tuple[float, ChatResponse]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> ChatResponse | None:
        entry = self._data.get(key)
        if entry is None or self._clock() - entry[0] > self.ttl:
            self._data.pop(key, None)
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return entry[1]

    def put(self, key: str, response: ChatResponse) -> None:
        self._data[key] = (self._clock(), response)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)


# ----------------------------------------------------------------------- cost

# USD per million tokens (input, output). Example values: set your own via CostTracker(prices=...).
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (5.0, 25.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-haiku-5-5": (1.0, 5.0),
}


@dataclass
class CostTracker:
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    by_key: dict[str, dict] = field(default_factory=dict)

    def record(self, api_key: str, response: ChatResponse) -> float:
        in_price, out_price = self.prices.get(response.model, (0.0, 0.0))
        cost = (response.input_tokens * in_price + response.output_tokens * out_price) / 1_000_000
        row = self.by_key.setdefault(
            api_key, {"requests": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        )
        row["requests"] += 1
        if response.cached:
            row["cache_hits"] += 1
            return 0.0  # a cache hit costs nothing
        row["input_tokens"] += response.input_tokens
        row["output_tokens"] += response.output_tokens
        row["cost_usd"] = round(row["cost_usd"] + cost, 8)
        return cost


# -------------------------------------------------------------------- gateway

class Gateway:
    """Request path: rate limit -> cache -> providers (retry, then fall back) -> cost accounting."""

    def __init__(
        self,
        providers: list[Provider],
        limiter: RateLimiter | None = None,
        cache: ResponseCache | None = None,
        costs: CostTracker | None = None,
        retries: int = 2,
        backoff: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.limiter = limiter or RateLimiter()
        self.cache = cache or ResponseCache()
        self.costs = costs or CostTracker()
        self.retries, self.backoff, self._sleep = retries, backoff, sleep

    def chat(self, api_key: str, request: ChatRequest) -> ChatResponse:
        self.limiter.check(api_key)

        cacheable = request.temperature == 0
        key = request.cache_key()
        if cacheable:
            hit = self.cache.get(key)
            if hit is not None:
                response = ChatResponse(**{**hit.__dict__, "cached": True})
                self.costs.record(api_key, response)
                return response

        response = self._call_providers(request)
        if cacheable:
            self.cache.put(key, response)
        self.costs.record(api_key, response)
        return response

    def _call_providers(self, request: ChatRequest) -> ChatResponse:
        errors: list[str] = []
        for provider in self.providers:
            for attempt in range(self.retries + 1):
                try:
                    response = provider.complete(request)
                    response.provider = provider.name
                    return response
                except Exception as exc:  # noqa: BLE001 - any provider failure triggers retry/fallback
                    errors.append(f"{provider.name}#{attempt}: {exc}")
                    if attempt < self.retries:
                        self._sleep(self.backoff * (2**attempt))
        raise AllProvidersFailed("; ".join(errors))


# ---------------------------------------------------------- real provider

class AnthropicProvider:
    """Thin adapter over the Anthropic SDK. Imported lazily so tests need no SDK or key."""

    def __init__(self, name: str = "anthropic", model_override: str | None = None):
        import anthropic  # local import on purpose

        self.name = name
        self._client = anthropic.Anthropic()
        self._model_override = model_override

    def complete(self, request: ChatRequest) -> ChatResponse:
        model = self._model_override or request.model
        kwargs = {}
        if request.temperature > 0:  # some newer models reject explicit sampling params
            kwargs["temperature"] = request.temperature
        msg = self._client.messages.create(
            model=model,
            max_tokens=request.max_tokens,
            messages=[{"role": r, "content": c} for r, c in request.messages],
            **kwargs,
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return ChatResponse(
            text=text, model=model, input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens
        )
