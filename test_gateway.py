import pytest

from gateway import (
    AllProvidersFailed,
    ChatRequest,
    ChatResponse,
    CostTracker,
    Gateway,
    RateLimited,
    RateLimiter,
    ResponseCache,
    TokenBucket,
)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeProvider:
    def __init__(self, name="fake", fail_times=0, model="claude-opus-5-5"):
        self.name, self.fail_times, self.model, self.calls = name, fail_times, model, 0

    def complete(self, request):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("boom")
        return ChatResponse(text="hi", model=self.model, input_tokens=1000, output_tokens=500)


def req(text="hello", temperature=0.0):
    return ChatRequest.of("claude-opus-5-5", [{"role": "user", "content": text}], temperature=temperature)


def gw(*providers, **kw):
    kw.setdefault("sleep", lambda s: None)
    return Gateway(list(providers), **kw)


# --- rate limiting -------------------------------------------------------

def test_token_bucket_allows_burst_then_blocks_then_refills():
    clock = Clock()
    b = TokenBucket(capacity=2, rate=1.0, clock=clock)
    assert b.try_acquire() == 0 and b.try_acquire() == 0
    assert b.try_acquire() == pytest.approx(1.0)
    clock.t = 1.0
    assert b.try_acquire() == 0


def test_rate_limit_is_per_key():
    clock = Clock()
    limiter = RateLimiter(capacity=1, rate=0.1, clock=clock)
    limiter.check("a")
    limiter.check("b")  # separate bucket
    with pytest.raises(RateLimited):
        limiter.check("a")


def test_gateway_rejects_when_limited():
    g = gw(FakeProvider(), limiter=RateLimiter(capacity=1, rate=0.001, clock=Clock()))
    g.chat("k", req("one"))
    with pytest.raises(RateLimited):
        g.chat("k", req("two"))


# --- cache ---------------------------------------------------------------

def test_second_identical_request_is_served_from_cache():
    p = FakeProvider()
    g = gw(p)
    first = g.chat("k", req())
    second = g.chat("k", req())
    assert p.calls == 1 and not first.cached and second.cached


def test_nonzero_temperature_is_never_cached():
    p = FakeProvider()
    g = gw(p)
    g.chat("k", req(temperature=0.7))
    g.chat("k", req(temperature=0.7))
    assert p.calls == 2


def test_cache_entries_expire():
    clock = Clock()
    cache = ResponseCache(ttl=10, clock=clock)
    r = ChatResponse("x", "m", 1, 1)
    cache.put("k", r)
    clock.t = 5
    assert cache.get("k") is r
    clock.t = 20
    assert cache.get("k") is None


def test_cache_evicts_least_recently_used():
    cache = ResponseCache(max_entries=2)
    r = ChatResponse("x", "m", 1, 1)
    cache.put("a", r)
    cache.put("b", r)
    cache.get("a")  # a becomes most recent
    cache.put("c", r)
    assert cache.get("b") is None and cache.get("a") is r


# --- retries and fallback ------------------------------------------------

def test_retries_same_provider_before_giving_up():
    p = FakeProvider(fail_times=2)
    g = gw(p, retries=2)
    assert g.chat("k", req()).text == "hi"
    assert p.calls == 3


def test_falls_back_to_next_provider():
    bad, good = FakeProvider("bad", fail_times=99), FakeProvider("good")
    g = gw(bad, good, retries=1)
    response = g.chat("k", req())
    assert response.provider == "good" and bad.calls == 2


def test_all_providers_failing_raises():
    g = gw(FakeProvider("a", fail_times=99), FakeProvider("b", fail_times=99), retries=0)
    with pytest.raises(AllProvidersFailed):
        g.chat("k", req())


def test_backoff_is_exponential():
    waits = []
    g = Gateway([FakeProvider(fail_times=3)], retries=3, backoff=0.5, sleep=waits.append)
    g.chat("k", req())
    assert waits == [0.5, 1.0, 2.0]


# --- cost ----------------------------------------------------------------

def test_cost_is_computed_per_model_price():
    costs = CostTracker(prices={"claude-opus-5-5": (5.0, 25.0)})
    g = gw(FakeProvider(), costs=costs)
    g.chat("k", req())
    # 1000 in * $5/M + 500 out * $25/M = 0.005 + 0.0125
    assert costs.by_key["k"]["cost_usd"] == pytest.approx(0.0175)


def test_cache_hits_cost_nothing_but_are_counted():
    costs = CostTracker(prices={"claude-opus-5-5": (5.0, 25.0)})
    g = gw(FakeProvider(), costs=costs)
    g.chat("k", req())
    g.chat("k", req())
    row = costs.by_key["k"]
    assert row["requests"] == 2 and row["cache_hits"] == 1
    assert row["cost_usd"] == pytest.approx(0.0175)


def test_empty_provider_list_rejected():
    with pytest.raises(ValueError):
        Gateway([])
