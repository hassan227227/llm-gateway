# LLM Gateway

![tests](https://github.com/hassan227227/llm-gateway/actions/workflows/ci.yml/badge.svg)

A small, production-minded gateway that sits between your apps and an LLM provider. It adds the things every team ends up building after the first prototype: per-key rate limiting, response caching, retries with fallback to a second model, and per-key cost tracking. Python, FastAPI, Claude.

```
client ──► POST /v1/chat ──► rate limit (token bucket per key)
                                  │
                                  ▼
                            cache lookup ── hit ──► response (cost: $0)
                                  │ miss
                                  ▼
                   primary model ──fail──► retry w/ exponential backoff
                                  │            │ still failing
                                  │            ▼
                                  │        fallback model
                                  ▼
                       cost ledger per API key ──► GET /v1/usage
```

## Design choices

- **Core is plain Python.** `gateway.py` has no web or SDK dependency; clocks, sleep and providers are injected, so every behaviour is unit-tested deterministically with no network or API key.
- **Token bucket per key.** Allows short bursts, refills at a steady rate, and tells the caller how long to wait (`429` with `Retry-After`).
- **Cache only deterministic calls.** Requests with `temperature > 0` are never cached. Entries are keyed by a SHA-256 of model, messages and parameters, evicted LRU and by TTL.
- **Retry, then fall back.** Each provider gets `retries` attempts with exponential backoff before the next provider is tried; if all fail the API returns `502` with the error trail.
- **Cost accounting.** Token usage is priced per model (example prices in `DEFAULT_PRICES`, override them for your contract). Cache hits are counted but cost nothing.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # Windows: set ANTHROPIC_API_KEY=...
uvicorn api:app --reload

curl -X POST localhost:8000/v1/chat -H "x-api-key: team-a" -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Summarise token buckets in one sentence."}]}'
curl localhost:8000/v1/usage -H "x-api-key: team-a"

pytest -q                               # 14 tests, offline
```

Environment: `DEFAULT_MODEL`, `FALLBACK_MODEL`, `RATE_BURST`, `RATE_PER_SEC`.

## Files

| File | What it does |
|---|---|
| `gateway.py` | Token bucket, LRU+TTL cache, cost tracker, retry/fallback gateway, Anthropic adapter |
| `api.py` | FastAPI app: `/health`, `/v1/chat`, `/v1/usage` |
| `test_gateway.py` | Unit tests with fake providers and a fake clock |
| `Dockerfile` | Non-root image with a health check |

## Next steps

- Persist the usage ledger (PostgreSQL) and export Prometheus metrics
- Semantic cache using embeddings alongside the exact-match cache
- Streaming responses and per-key model allowlists

## Author

Hassan Raza · [Portfolio](https://hassan227227.github.io) · [LinkedIn](https://www.linkedin.com/in/iamhassan890)
