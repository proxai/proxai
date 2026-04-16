# `px.models` — Model Discovery API

User-facing reference for ProxAI's model discovery system. Covers how to
list, filter, and verify models via `px.models.*` (module-level) or
`client.models.*` (instance-level). Source of truth:
`src/proxai/client.py` (`ModelConnector` class) and `src/proxai/types.py`
(`ProviderModelType`, `ModelSizeType`, `FeatureTagType`, `ModelStatus`).

---

## 0. `px.models` at a glance

```
px.models
│
│   # Configured models (no network call)
├── .list_models(...)              → list[ProviderModelType]
├── .list_providers(...)           → list[str]
├── .list_provider_models(...)     → list[ProviderModelType]
├── .get_model(...)                → ProviderModelType
│
│   # Working models (health-checked, may call providers)
├── .list_working_models(...)      → list[ProviderModelType] | ModelStatus
├── .list_working_providers(...)   → list[str]
├── .list_working_provider_models(...) → list[ProviderModelType] | ModelStatus
└── .get_working_model(...)        → ProviderModelType
```

**Configured** = registered in ProxAI's model registry and your API keys
are present. No network call needed.

**Working** = configured AND verified to respond successfully. Triggers
health checks on first call; results are cached.

### Common filter parameters

```
Filter parameters (shared across most methods)
│
├── model_size: str | ModelSizeType | None      # "small" | "medium" | "large" | "largest"
├── features: list[str] | list[FeatureTagType] | None
│   │                                           # filter by required capabilities
│   │   Feature tags:
│   │     "prompt", "messages", "system_prompt",
│   │     "temperature", "max_tokens", "stop", "n", "thinking",
│   │     "web_search",
│   │     "response_text", "response_image", "response_audio",
│   │     "response_video", "response_json", "response_pydantic",
│   │     "response_multi_modal"
│   │
├── call_type: str | CallType = "text"          # "text" | "image" | "audio" | "video"
└── recommended_only: bool = True               # True = curated subset only
```

### Return type: `ProviderModelType`

```
ProviderModelType (frozen dataclass)
├── provider: str                               # "openai", "anthropic", ...
├── model: str                                  # "gpt-4o", "claude-3-5-sonnet", ...
└── provider_model_identifier: str              # "gpt-4o-2024-08-06", ...
```

Printed as `(openai, gpt-4o)`. Usable directly as the `provider_model`
argument to `px.generate()`.

### Return type: `ModelStatus` (when `return_all=True`)

```
ModelStatus
├── unprocessed_models: set[ProviderModelType]  # not yet tested
├── working_models: set[ProviderModelType]      # passed health check
├── failed_models: set[ProviderModelType]       # failed health check
├── filtered_models: set[ProviderModelType]     # excluded by filters
└── provider_queries: dict[ProviderModelType, CallRecord]
                                                # per-model test CallRecord
```

---

## 1. Configured model methods

These methods read from the model registry. They never contact a
provider — they only check whether your API keys are present.

### 1.1 `list_models()`

Returns all models matching the filters.

```python
px.models.list_models(
    model_size: str | ModelSizeType | None = None,
    features: list[str] | list[FeatureTagType] | None = None,
    call_type: str | CallType = "text",
    recommended_only: bool = True,
) → list[ProviderModelType]
```

```python
# All recommended text models
models = px.models.list_models()

# Large models only
models = px.models.list_models(model_size="large")

# Models that support JSON output and system prompts
models = px.models.list_models(
    features=["response_json", "system_prompt"]
)

# Include non-recommended models
models = px.models.list_models(recommended_only=False)

# Image-generation models
models = px.models.list_models(call_type="image")
```

### 1.2 `list_providers()`

Returns provider names that have API keys configured.

```python
px.models.list_providers(
    call_type: str | CallType = "text",
    recommended_only: bool = True,
) → list[str]
```

```python
providers = px.models.list_providers()
# → ["anthropic", "google", "openai"]
```

### 1.3 `list_provider_models()`

Returns models from a specific provider.

```python
px.models.list_provider_models(
    provider: str,                              # required
    model_size: str | ModelSizeType | None = None,
    features: list[str] | list[FeatureTagType] | None = None,
    call_type: str | CallType = "text",
    recommended_only: bool = True,
) → list[ProviderModelType]
```

```python
openai_models = px.models.list_provider_models("openai")
# → [(openai, gpt-4o), (openai, gpt-4), ...]

# Only small OpenAI models
small = px.models.list_provider_models("openai", model_size="small")
```

Raises `ValueError` if the provider's API key is not found.

### 1.4 `get_model()`

Returns a single model by provider and name.

```python
px.models.get_model(
    provider: str,                              # required
    model: str,                                 # required
    call_type: str | CallType = "text",
) → ProviderModelType
```

```python
model = px.models.get_model("openai", "gpt-4o")
# → ProviderModelType(provider="openai", model="gpt-4o",
#     provider_model_identifier="gpt-4o-2024-08-06")

# Use it directly in generate()
rec = px.generate(prompt="Hello", provider_model=model)
```

Raises `ValueError` if the API key is missing, the model does not
exist, or the model does not support the specified `call_type`.

---

## 2. Working model methods

These methods perform health checks — they send a test request to each
model and return only models that respond successfully. Results are
cached (see §4).

### 2.1 `list_working_models()`

The most commonly used working-model method. Returns models that pass
health checks.

```python
px.models.list_working_models(
    model_size: str | ModelSizeType | None = None,
    features: list[str] | list[FeatureTagType] | None = None,
    verbose: bool = True,
    return_all: bool = False,
    clear_model_cache: bool = False,
    call_type: str | CallType = "text",
    recommended_only: bool = True,
) → list[ProviderModelType] | ModelStatus
```

```python
# Basic usage (prints progress to stdout)
working = px.models.list_working_models()

# Silent
working = px.models.list_working_models(verbose=False)

# Full diagnostics
status = px.models.list_working_models(return_all=True)
print(f"{len(status.working_models)} working")
print(f"{len(status.failed_models)} failed")
for model, record in status.provider_queries.items():
    print(f"  {model}: {record.result.timestamp.response_time}")

# Force retest (ignore cached results)
working = px.models.list_working_models(clear_model_cache=True)
```

### 2.2 `list_working_providers()`

Returns providers with at least one working model.

```python
px.models.list_working_providers(
    verbose: bool = True,
    clear_model_cache: bool = False,
    call_type: str | CallType = "text",
    recommended_only: bool = True,
) → list[str]
```

```python
providers = px.models.list_working_providers(verbose=False)
# → ["anthropic", "openai"]
```

### 2.3 `list_working_provider_models()`

Returns working models from a specific provider.

```python
px.models.list_working_provider_models(
    provider: str,                              # required
    model_size: str | ModelSizeType | None = None,
    features: list[str] | list[FeatureTagType] | None = None,
    verbose: bool = True,
    return_all: bool = False,
    clear_model_cache: bool = False,
    call_type: str | CallType = "text",
    recommended_only: bool = True,
) → list[ProviderModelType] | ModelStatus
```

```python
openai_working = px.models.list_working_provider_models(
    "openai", verbose=False
)
# → [(openai, gpt-4o), (openai, gpt-4)]
```

Raises `ValueError` if the provider's API key is not found.

### 2.4 `get_working_model()`

Verifies a single model is working and returns it.

```python
px.models.get_working_model(
    provider: str,                              # required
    model: str,                                 # required
    verbose: bool = False,
    clear_model_cache: bool = False,
    call_type: str | CallType = "text",
) → ProviderModelType
```

```python
model = px.models.get_working_model("openai", "gpt-4o")
rec = px.generate(prompt="Hello", provider_model=model)
```

Raises `ValueError` if the API key is missing, the model does not
exist, or the model fails the health check.

---

## 3. Common patterns

### 3.1 Discovery before generation

```python
import proxai as px

# Find what's available, then generate
models = px.models.list_models()
print(f"{len(models)} models configured")

rec = px.generate(
    prompt="Hello",
    provider_model=models[0],
)
```

### 3.2 Filter by capability

```python
# Models that support structured output
pydantic_models = px.models.list_models(
    features=["response_pydantic"]
)

# Models that support thinking/reasoning
thinking_models = px.models.list_models(features=["thinking"])

# Models that support web search
search_models = px.models.list_models(features=["web_search"])
```

### 3.3 Filter by size

```python
# Cost-effective small models
small = px.models.list_models(model_size="small")

# Flagship models
flagship = px.models.list_models(model_size="largest")
```

### 3.4 Verify before use

```python
# Ensure a specific model is reachable before committing to it
try:
    model = px.models.get_working_model("openai", "gpt-4o")
    rec = px.generate(prompt="Hello", provider_model=model)
except ValueError as e:
    print(f"Model not available: {e}")
```

### 3.5 Full health-check report

```python
status = px.models.list_working_models(
    return_all=True, verbose=False
)

print("Working:")
for m in sorted(status.working_models, key=str):
    latency = status.provider_queries[m].result.timestamp.response_time
    print(f"  {m} — {latency.total_seconds():.2f}s")

print("Failed:")
for m in sorted(status.failed_models, key=str):
    error = status.provider_queries[m].result.error
    print(f"  {m} — {error}")
```

### 3.6 Iterate over providers

```python
for provider in px.models.list_providers():
    models = px.models.list_provider_models(provider)
    print(f"{provider}: {len(models)} models")
    for m in models:
        print(f"  {m.model} ({m.provider_model_identifier})")
```

### 3.7 Non-text models

```python
# Image generation models
image_models = px.models.list_models(call_type="image")

# All model types (not just recommended)
all_models = px.models.list_models(
    call_type="image", recommended_only=False
)
```

### 3.8 Client-instance model discovery

```python
client = px.Client(
    cache_options=px.CacheOptions(cache_path="/tmp/cache"),
)

# Same API, scoped to this client
models = client.models.list_models()
working = client.models.list_working_models(verbose=False)
```

Module-level `px.models` and `client.models` are fully isolated — they
use separate clients with separate model caches.

### 3.9 Force-refresh cached health-check results

```python
# Cached results may be stale. Force a retest:
working = px.models.list_working_models(clear_model_cache=True)

# Or reset everything (including the default client):
px.reset_state()
```

---

## 4. Health-check caching

Working-model methods cache their results to avoid repeated network
calls. Key behaviours:

- A built-in default model cache (4-hour TTL) always exists, even
  without `cache_options`. It lives in a per-user platform directory.
- If `cache_options.cache_path` is set, a user-level model cache is
  added on top of the default.
- `clear_model_cache=True` on any working-model method forces a retest.
- `px.reset_state()` clears the default model cache entirely.
- Health checks may use multiprocessing (controlled by
  `allow_multiprocessing` on the client). Disable in Jupyter, Lambda,
  or environments that cannot fork cleanly.
- `model_test_timeout` (default 25s) controls how long to wait per
  model before marking it failed.

See `px_client_analysis.md` §2.7, §2.8, §5.3 for full details.

---

## 5. Errors

| Method | Trigger | Error |
|--------|---------|-------|
| `list_provider_models()` | Provider API key not in env | `ValueError` |
| `list_working_provider_models()` | Provider API key not in env | `ValueError` |
| `get_model()` | API key missing, model not found, or wrong `call_type` | `ValueError` |
| `get_working_model()` | API key missing, model not found, or health check failed | `ValueError` |

These are all synchronous `ValueError`s — they are not affected by
`suppress_provider_errors`.

---

## 6. Module-level vs client instance

```python
# Module-level (uses hidden default client)
import proxai as px
models = px.models.list_models()

# Client instance (independent)
client = px.Client(...)
models = client.models.list_models()
```

Same API, same methods, same parameters. Fully isolated — separate
model caches, separate health-check state. See `px_client_analysis.md`
§5.1.
