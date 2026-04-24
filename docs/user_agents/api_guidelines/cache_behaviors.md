# Cache Behaviours

Source of truth: `src/proxai/types.py` (the `CacheOptions`,
`ConnectionOptions`, `CacheLookFailReason`, `ResultSource`, and
`ConnectionMetadata` dataclasses), `src/proxai/caching/query_cache.py`
(query cache), `src/proxai/caching/model_cache.py` (model cache), and
`src/proxai/connectors/provider_connector.py` (the per-call cache
hooks). If this document disagrees with those files, the files win —
update this document.

This is the definitive user-facing reference for ProxAI's two caches —
what each one stores, what controls turn them on or off, what shows up
on a returned `CallRecord`, and the surprises every caller eventually
hits. Read this before relying on cache replay in tests, sharing a
cache across processes, or wiring `skip_cache` / `override_cache_value`
into production code paths.

See also: `developer_agents/cache_internals.md` (storage layout,
sharding, freshness invariant — read this if you are editing the cache
code), `px_client_api.md` (where `CacheOptions` lives in the client
constructor), `px_generate_api.md` (where `ConnectionOptions` plugs
into a single call), and `call_record.md` (the cache fields on a
returned `CallRecord`).

---

## 1. Cache surface (current)

```
CacheOptions(...)                                    # client-level, on px.connect / px.Client
│
│   # Query cache (response replay)
├── cache_path: str | None = None                   # enables both caches
├── unique_response_limit: int | None = 1           # responses per query before replay
├── clear_query_cache_on_connect: bool = False      # wipe on construction
│
│   # User-level model cache (probe results stored at cache_path)
├── disable_model_cache: bool = False               # turn off the user-level layer
├── clear_model_cache_on_connect: bool = False      # wipe on construction
└── model_cache_duration: int | None = None         # seconds; None = no expiry

ConnectionOptions(...)                               # per-call, on px.generate(...)
│
├── skip_cache: bool | None = None                  # bypass entirely (no read, no write)
└── override_cache_value: bool | None = None        # bypass on read; overwrite on success

# Always-on default model cache (no user control beyond px.reset_state):
#   path: platformdirs user_cache_dir, or tempfile.TemporaryDirectory() fallback
#   TTL : 4 hours when in the platform dir; per-process when in temp dir
```

The query cache is **off by default**. The model cache is **on by
default** via the always-on built-in layer — `cache_options` only
controls whether a second user-level model cache stacks on top of it.

---

## 2. What ProxAI caches

### 2.1 Query cache — `(query) → response`

When `cache_options.cache_path` is set, every successful
`px.generate(...)` call writes its `CallRecord.result` to disk keyed by
a hash of the full `QueryRecord` (prompt, messages, system_prompt,
provider/model, parameters, tools, output_format). On a subsequent call
with an identical query, the stored response is replayed instead of
hitting the provider.

Off by default. Failed provider calls are not written — only successes
are cached.

### 2.2 Model cache — `(provider, model, output_format) → ModelStatus`

Stores which models passed the most recent health probe per
`output_format`, plus the per-model probe `CallRecord` in
`ModelStatus.provider_queries`. This is what makes
`px.models.list_working_models(...)` near-instant on the second call.

Two layers, both file-backed:

- **Built-in default model cache** — always exists. Lives in the
  platform user cache dir (4-hour TTL); falls back to a per-process
  `tempfile.TemporaryDirectory()` if the platform dir isn't writable.
  You cannot point this at a custom location — only `px.reset_state()`
  clears it.
- **User-level model cache** — added on top when
  `cache_options.cache_path` is set, *unless*
  `disable_model_cache=True`. Honours `model_cache_duration` (seconds;
  default `None` = no expiry).

Setting `cache_options.cache_path` does **not** replace the default; it
adds a second layer.

---

## 3. When the cache helps

- **Iterating on prompt engineering.** Repeat-runs of the same prompt
  cost zero provider calls.
- **Re-running an eval harness.** Identical inputs replay deterministic
  outputs; you only pay once per query.
- **Shared dev caches across machines.** The on-disk layout is
  designed to be file-system-shared and is restart-safe (see
  `cache_internals.md`).
- **Avoiding the model-discovery round-trip.** The default model cache
  makes `px.models.list_working_models()` cheap after the first call,
  even with no `cache_options`.

```python
import proxai as px

# Enable the query cache for a dev session.
px.connect(cache_options=px.CacheOptions(cache_path="/tmp/proxai_cache"))
px.set_model(provider_model=("openai", "gpt-4o"))

a = px.generate_text(prompt="Summarise the Iliad in three sentences.")
b = px.generate_text(prompt="Summarise the Iliad in three sentences.")
# Second call is served from cache — same string, no provider call.
```

---

## 4. When the cache hurts

- **Production traffic.** End users expect their request to hit the
  model. Don't ship a query cache to prod — leave `cache_options` unset
  or pass `connection_options=ConnectionOptions(skip_cache=True)` on
  the request path.
- **Non-deterministic evaluations.** If your eval depends on response
  variance (sampling, jitter, multi-trial scoring), cache replay
  collapses every retry to the same output. Use `skip_cache=True` or
  `override_cache_value=True` per call, or use
  `unique_response_limit > 1` (§5.1).
- **Stale model-availability data.** A regressed model can stay in the
  working set until the model cache expires. Use `clear_model_cache=True`
  on `px.models.list_working_models()` to force a retest, or
  `px.reset_state()` to clear the default model cache.

```python
# Production-safe pattern: no query cache configured at all.
px.connect()
rec = px.generate(prompt=user_input)

# Or: cache configured for tests, but bypassed for live traffic.
rec = px.generate(
    prompt=user_input,
    connection_options=px.ConnectionOptions(skip_cache=True),
)
```

---

## 5. Cache options reference

### 5.1 `CacheOptions` — client-level

Pass to `px.connect(cache_options=...)` or
`px.Client(cache_options=...)`. Validated at construction time; see
`px_client_api.md` §2.2 for the construction-time errors.

| Field | Default | Effect |
|---|---|---|
| `cache_path` | `None` | Directory on disk where query and user-level model caches are stored. Required when `cache_options` is set, **unless** `disable_model_cache=True` (which lets you run with neither cache). |
| `unique_response_limit` | `1` | Number of distinct responses to collect per query before replay starts. With `1`, the first response is cached and every subsequent identical call replays it. With `N`, the cache asks the provider until it has `N` distinct responses, then round-robins across them. |
| `clear_query_cache_on_connect` | `False` | Wipe the query cache directory at construction. Destructive; every previously cached response is gone. |
| `disable_model_cache` | `False` | Skip the user-level model cache entirely. The built-in default model cache (§2.2) still applies. The one escape that lets you set `cache_options` without `cache_path`. |
| `clear_model_cache_on_connect` | `False` | Wipe the user-level model cache at `cache_path` at construction. Does **not** clear the built-in default model cache — use `px.reset_state()` for that. |
| `model_cache_duration` | `None` | Seconds before user-level model cache entries are considered stale (re-probed on read). `None` = no expiry. The default model cache uses 4 hours independently. |

### 5.2 `ConnectionOptions` — per-call

Pass to `px.generate(..., connection_options=...)`. Each field is
scoped to a single call. Both default to `None` (use the cache if
configured).

| Field | Effect on read | Effect on write |
|---|---|---|
| `skip_cache=True` | Skip the cache lookup | Skip the cache write — the call is invisible to the cache |
| `override_cache_value=True` | Skip the lookup; always call the provider | On success, wipe any existing entry for this query hash and store the fresh result as a single-response bucket (`call_count` reset to 0) |

`skip_cache` wins if both are set: the cache stays untouched.
`override_cache_value=True` requires a working query cache — passing
it without one raises `ValueError` synchronously (§10).

### 5.3 Behaviour matrix for a query whose bucket has 3 cached responses

For a query already cached with `results=[R1, R2, R3]`, `call_count=5`,
`unique_response_limit=3`:

| Connection flags | Returned to caller | Bucket after the call |
|---|---|---|
| _(none)_ | `R3` (call_count → 6) | `[R1, R2, R3]`, cc=6 |
| `skip_cache=True` | provider response, cache untouched | `[R1, R2, R3]`, cc=5 |
| `override_cache_value=True` | provider response `R_new` | `[R_new]`, cc=0 |

After `override_cache_value=True`, `unique_response_limit > 1` will
refill the bucket from the provider on the next few calls.

---

## 6. Reading cache info from a `CallRecord`

The fields you'll inspect after a call:

```python
from proxai.types import ResultSource, CacheLookFailReason

rec = px.generate(prompt="Hello")

rec.connection.result_source       # → ResultSource.CACHE | ResultSource.PROVIDER
rec.connection.cache_look_fail_reason
#   → CacheLookFailReason | None — set when result_source == PROVIDER
#     AND the cache subsystem ran a lookup that didn't return a result.
#     Stays None when the cache was bypassed (skip_cache /
#     override_cache_value) or when no cache is configured.

rec.result.timestamp.response_time         # provider latency, always set
rec.result.timestamp.cache_response_time   # cache lookup latency, only on hits
```

The `CacheLookFailReason` values you can see on a returned record:

```
CACHE_NOT_FOUND                    no entry for this query hash
CACHE_NOT_MATCHED                  entry exists but query fingerprint differed
UNIQUE_RESPONSE_LIMIT_NOT_REACHED  bucket still collecting responses
CACHE_UNAVAILABLE                  cache configured but not in WORKING state (path unwritable, etc.)
```

See `call_record.md` §3.9 (cache hit) and §3.10 (cache miss reasons)
for full example shapes.

---

## 7. Examples

### 7.1 Dev cache for prompt iteration

Re-runs of the same prompt cost zero provider calls — the canonical
"why we built this" pattern.

```python
import proxai as px

px.connect(cache_options=px.CacheOptions(cache_path="/tmp/proxai_dev_cache"))
px.set_model(provider_model=("openai", "gpt-4o"))

for temperature in (0.2, 0.4, 0.6):
    rec = px.generate(
        prompt="Summarise the Iliad in three sentences.",
        parameters=px.ParameterType(temperature=temperature),
    )
    # Each (prompt, temperature) pair is a different bucket. Re-running
    # this loop replays from cache instead of re-billing each time.
```

### 7.2 Production: no cache at all

Every call goes to the provider. The model cache still works (the
built-in default is always on), but no query replay.

```python
import proxai as px

px.connect()  # no cache_options → query cache disabled
rec = px.generate(prompt=user_request)
```

### 7.3 Shared dev cache, but live traffic bypasses it

CI and a developer share a checkpoint cache so eval re-runs are free.
Live request handlers pass `skip_cache=True` so end users never get a
replay.

```python
px.connect(cache_options=px.CacheOptions(cache_path="/shared/cache"))

# Eval harness (free re-runs)
for q in golden_questions:
    rec = px.generate(prompt=q)

# Live request handler (always hits the provider)
def handle_user(prompt: str) -> str:
    rec = px.generate(
        prompt=prompt,
        connection_options=px.ConnectionOptions(skip_cache=True),
    )
    return rec.result.output_text
```

### 7.4 Force-refresh a single cached answer

You changed the upstream data; you want this one query re-fetched and
the cache updated, without wiping anything else.

```python
rec = px.generate(
    prompt="Summarise the latest earnings report.",
    connection_options=px.ConnectionOptions(override_cache_value=True),
)
assert rec.connection.result_source == px.ResultSource.PROVIDER
# The bucket for this query is now [rec.result], call_count=0.
```

### 7.5 Collect diverse responses with `unique_response_limit`

The first three calls go to the provider and contribute distinct
responses; the fourth onward round-robins from the cached three.

```python
client = px.Client(
    cache_options=px.CacheOptions(
        cache_path="/tmp/diverse_cache",
        unique_response_limit=3,
    ),
)
client.set_model(provider_model=("openai", "gpt-4o"))

for i in range(6):
    rec = client.generate(
        prompt="Give me a creative cat name.",
        parameters=px.ParameterType(temperature=0.9),
    )
    print(i, rec.connection.result_source, rec.result.output_text)
# i=0..2 → PROVIDER (collecting); i=3..5 → CACHE (round-robin replay).
```

### 7.6 Force a model retest after a probe went stale

The default model cache may still report a regressed model as working.
Three escape hatches, ordered from narrowest to broadest.

```python
# (a) Force a retest for the working-models query you're about to run.
working = px.models.list_working_models(clear_model_cache=True)

# (b) Full diagnostic re-probe of every registered model.
status = px.models.check_health()
for m, q in status.provider_queries.items():
    if q.result.status == px.ResultStatusType.FAILED:
        print(f"  {m} regressed: {q.result.error}")

# (c) Hardest reset — drops the default client and clears its caches.
px.reset_state()
```

### 7.7 Cache-hit telemetry off `CallRecord`

Compute hit ratio and lookup latency for monitoring.

```python
from proxai.types import ResultSource

rec = px.generate(prompt=user_request)

is_hit = rec.connection.result_source == ResultSource.CACHE
miss_reason = rec.connection.cache_look_fail_reason  # None on hits / bypasses
lookup_ms = (
    rec.result.timestamp.cache_response_time.total_seconds() * 1000
    if rec.result.timestamp.cache_response_time
    else None
)
emit_metric("cache.hit", 1 if is_hit else 0, tags={"reason": str(miss_reason)})
if lookup_ms is not None:
    emit_metric("cache.lookup_ms", lookup_ms)
```

### 7.8 Two isolated clients with separate caches

Compare two model configurations side-by-side without their caches
contaminating each other. See `px_client_api.md` §5.1 for the
isolation rules.

```python
exp_a = px.Client(cache_options=px.CacheOptions(cache_path="/data/exp_a"))
exp_b = px.Client(cache_options=px.CacheOptions(cache_path="/data/exp_b"))

exp_a.set_model(provider_model=("openai", "gpt-4o"))
exp_b.set_model(provider_model=("anthropic", "claude-3-5-sonnet"))

# Two caches; the same prompt is fetched independently for each model.
ans_a = exp_a.generate_text(prompt="Explain entropy.")
ans_b = exp_b.generate_text(prompt="Explain entropy.")
```

---

## 8. Implications

- **Cache key is content-based.** The hash covers the full
  `QueryRecord` — prompt text, messages, system_prompt, provider,
  model, parameters, tools, output_format. Changing any of them is a
  miss. Whitespace and casing matter.
- **Auto-uploaded files contribute their content hash, not their local
  path.** Renaming or moving a file you reference inside `messages`
  doesn't change the cache key as long as the bytes are the same.
- **Fallback chains cache only the model that succeeded.** With
  `connection_options.fallback_models=[...]`, the chain may try several
  models; only the response from the model that actually returned is
  written to the cache, keyed by *that* model's `QueryRecord`. The
  primary's failed attempts are not cached.
- **Failed calls don't update the query cache.** A `result.status =
  FAILED` CallRecord (when `suppress_provider_errors=True`) is returned
  to the caller but nothing is written. Errors are never replayed from
  the cache, so a transient provider failure won't poison subsequent
  calls.
- **Sharing a cache across processes or machines is supported.** The
  on-disk layout is restart-safe and has no fsck step. See
  `cache_internals.md` §4 / §10 for the freshness invariant and
  recovery path.
- **The default model cache always exists.** Even without
  `cache_options`, model discovery (`px.models.*`) is cached for 4
  hours in the platform user dir, or per-process in a tempdir. A
  regressed model can keep showing as "working" until that TTL elapses
  — use `clear_model_cache=True` on the working-models calls or
  `px.reset_state()` to force a refresh.

---

## 9. Common surprises and how to handle them

**"Same prompt, same response, every time."** That is the cache doing
its job. Either accept it, pass `skip_cache=True` per call, or add a
distinguishing field (e.g., a timestamp in the prompt, a random
parameter) so the hash changes.

**"`unique_response_limit=3` cost me 3× on the first calls."** Each
fresh query pays one provider call per response collected, until the
bucket fills. After that, calls round-robin from cache. Document this
to callers before turning it on.

**"`override_cache_value=True` only stored one response."** Override
wipes the bucket and writes a single fresh entry. With
`unique_response_limit > 1`, the next few calls will refill the bucket
from the provider. Use `override_cache_value` for "force-refresh this
one query," not as a way to bulk-rebuild a multi-response bucket.

**"`px.models.list_working_models()` reports a model that doesn't
work."** The default model cache (or the user-level one) is serving a
stale entry. Force a retest:

```python
px.models.list_working_models(clear_model_cache=True)
# or
px.models.check_health()       # always re-probes
# or, hardest reset:
px.reset_state()               # drops the default client and its caches
```

**"`override_cache_value=True` raised on me."** Synchronous
`ValueError("override_cache_value is True but query cache is not
configured")`. The override flag requires a working query cache.
Either set `cache_options.cache_path`, or use `skip_cache=True` if
your goal is just to bypass.

**"`skip_cache` and `override_cache_value` look interchangeable."**
They are not. `skip_cache=True` makes the call invisible to the cache:
no read, no write. `override_cache_value=True` still writes — it just
ignores the cached value on read and overwrites it on success. Choose
based on whether you want this call's result to affect future lookups.

---

## 10. Errors

Validation errors raised synchronously by the cache surface. None of
these are routed through `suppress_provider_errors`.

| Trigger | Error |
|---|---|
| `cache_options` set but `cache_path is None` and `disable_model_cache is False` | `ValueError: cache_path is required while setting cache_options` |
| `connection_options.override_cache_value=True` while no query cache is configured (or it isn't in `WORKING` state) | `ValueError: override_cache_value is True but query cache is not configured.` |
| `cache_options.cache_path` points at a non-writable directory | Logged as a warning at construction; the manager moves to `CACHE_PATH_NOT_WRITABLE` and every subsequent `look()` returns `CacheLookFailReason.CACHE_UNAVAILABLE` (no raise — the cache degrades to "always miss"). |

A cache that is configured but cannot reach `WORKING` status never
crashes a `generate()` call; it surfaces as `CACHE_UNAVAILABLE` on
`connection.cache_look_fail_reason` so callers can detect the
degradation without a try/except.

---

## 11. See also

- `developer_agents/cache_internals.md` — sharding, on-disk layout,
  freshness invariant, restart recovery. Read this when editing cache
  code.
- `px_client_api.md` §2.2 — `CacheOptions` field reference at
  construction time, plus `disable_model_cache` semantics.
- `px_generate_api.md` §5.9 — calling-side patterns for `skip_cache` and
  `override_cache_value`.
- `call_record.md` §3.9, §3.10, §2.14 — concrete `ConnectionMetadata`
  shapes for cache hits and the five `CacheLookFailReason` values.
- `px_models_api.md` §5 — health-check caching rules and the
  `clear_model_cache=True` escape hatch on working-models methods.
