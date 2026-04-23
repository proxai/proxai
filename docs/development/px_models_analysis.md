# `px.models` API Comprehensive Use Case Analysis

Source of truth: `src/proxai/client.py` (the `ModelConnector` class —
both module-level `px.models` and instance-level `client.models` use
it) and `src/proxai/types.py` (`ProviderModelType`, `ModelSizeType`,
`FeatureTag`, `ToolTag`, `InputFormatType`, `OutputFormatType`,
`ModelStatus`, `ProviderModelConfig`). If this document disagrees with
those files, the files win — update this document.

This is the definitive reference for the model discovery surface —
every method on `px.models`, what each filter parameter accepts, when
the network is hit vs. when a cached answer is served, and the health
check caching rules. Read this before adding a new discovery method,
a new filter axis, or changing how health checks propagate.

See also: `px_client_analysis.md` (client construction — where
`model_probe_options` lives), `call_record_analysis.md` (CallRecord
shape — what `provider_queries` holds per model), and
`px_generate_analysis.md` (how returned `ProviderModelType` values
plug into `generate()`).

---

## 1. `px.models` structure (current)

```
px.models                                            # same on client.models
│
│   # Configured models (read from registry, never hit the network)
├── .list_models(...)                → list[ProviderModelType]
├── .list_providers(...)             → list[str]
├── .list_provider_models(...)       → list[ProviderModelType]
├── .get_model(...)                  → ProviderModelType
├── .get_model_config(...)           → ProviderModelConfig
├── .get_default_model_list()        → list[ProviderModelType]
│
│   # Working models (run health probes; cached)
├── .list_working_models(...)        → list[ProviderModelType] | ModelStatus
├── .list_working_providers(...)     → list[str]
├── .list_working_provider_models(...) → list[ProviderModelType] | ModelStatus
├── .get_working_model(...)          → ProviderModelType
└── .check_health(...)               → ModelStatus
```

**Configured** = registered in ProxAI's model registry and your API
keys are present. No network call needed.

**Working** = configured AND verified to respond successfully. Triggers
health checks on first call; results are cached (§5).

### 1.1 Common filter parameters

Configured and working methods share the same declared-capability
filter surface. The working surface just adds operational flags for
the probing harness (`verbose`, `return_all`, `clear_model_cache`)
and restricts `output_format` to the subset the harness can probe
cheaply — see §1.2 below.

```
Filter parameters (configured and working, list_models /
list_provider_models / list_working_models / list_working_provider_models)
│
├── model_size: ModelSizeIdentifierType | None     # ModelSizeType | str | None
│                                                  # "small" | "medium" | "large" | "largest"
├── input_format: InputFormatTypeParam | None      # single value, list, or str
│   │                                              # InputFormatType values:
│   │                                              #   TEXT, IMAGE, DOCUMENT, AUDIO,
│   │                                              #   VIDEO, JSON, PYDANTIC
│   │
├── output_format: OutputFormatTypeParam           # default: OutputFormatType.TEXT
│   │                                              # OutputFormatType values:
│   │                                              #   TEXT, IMAGE, AUDIO, VIDEO,
│   │                                              #   JSON, PYDANTIC, MULTI_MODAL
│   │                                              # Working methods: refuses
│   │                                              # IMAGE/AUDIO/VIDEO (§1.2).
│   │
├── feature_tags: FeatureTagParam | None           # single value, list, or str
│   │                                              # FeatureTag values:
│   │                                              #   prompt, messages, system_prompt,
│   │                                              #   temperature, max_tokens, stop, n,
│   │                                              #   thinking
│   │
├── tool_tags: ToolTagParam | None                 # single value, list, or str
│   │                                              # ToolTag values:
│   │                                              #   web_search
│   │
└── recommended_only: bool = True                  # True = curated subset only

Working methods — additional operational flags
│
├── verbose: bool = True                           # progress on stdout
├── return_all: bool = False                       # True → ModelStatus instead of list
└── clear_model_cache: bool = False                # force retest
```

Filter values are **conjunctive** — a model must satisfy every filter
you pass. Passing `output_format="json"` and `feature_tags=["thinking"]`
returns only models that support JSON output *and* thinking.

Declared-capability filtering happens **before** probing on working
methods, so only models that claim to support the requested shape are
ever tested — no wasted network calls for unreachable combinations.

### 1.2 Probe-safe output formats

Working methods send a real provider call per model. For text-shaped
output (TEXT / JSON / PYDANTIC) and MULTI_MODAL the probe is cheap and
allowed. For IMAGE / AUDIO / VIDEO each probe would generate real media,
which is prohibitive for bulk probing and still costly for a single
model — so the working methods refuse those formats up front:

```
output_format                     Working methods
─────────────                     ────────────────
TEXT                              ✓ allowed (default)
JSON                              ✓ allowed
PYDANTIC                          ✓ allowed
MULTI_MODAL                       ✓ allowed
IMAGE / AUDIO / VIDEO             ✗ ValueError
```

The refusal error message points callers at the cheaper alternative:
`list_models(output_format='image')` for declared capability, or
`generate_image(provider_model=...)` to verify a specific model.

Under `provider_call_options.feature_mapping_strategy=STRICT`, a model
whose support for the requested `output_format` is `BEST_EFFORT` (not
`SUPPORTED`) is rejected during the declared-capability filter pass
— so the probe never runs and the model is placed in
`ModelStatus.filtered_models`. Under `BEST_EFFORT` strategy, the same
model is probed and, on success, classified as working.

### 1.3 Return types

```
ProviderModelType (frozen dataclass)
├── provider: str                                # "openai", "anthropic", ...
├── model: str                                    # "gpt-4o", "claude-3-5-sonnet", ...
└── provider_model_identifier: str               # "gpt-4o-2024-08-06", ...
```

Printed as `(openai, gpt-4o)`. Usable directly as the `provider_model`
argument to `px.generate()`.

```
ProviderModelConfig (from get_model_config)
├── provider_model: ProviderModelType
├── pricing: ProviderModelPricingType
│   ├── input_token_cost_nano_usd_per_token: int | None
│   └── output_token_cost_nano_usd_per_token: int | None
├── features: FeatureConfigType
│   ├── prompt / messages / system_prompt: FeatureSupportType | None
│   ├── parameters: ParameterConfigType | None
│   ├── tools: ToolConfigType | None
│   ├── input_format: InputFormatConfigType | None
│   └── output_format: OutputFormatConfigType | None
└── metadata: ProviderModelMetadataType
    ├── is_recommended: bool | None
    └── model_size_tags: list[ModelSizeType] | None
```

```
ModelStatus (from working methods with return_all=True, and check_health)
├── unprocessed_models: set[ProviderModelType]    # not yet tested
├── working_models: set[ProviderModelType]        # passed health check
├── failed_models: set[ProviderModelType]         # failed health check
├── filtered_models: set[ProviderModelType]       # excluded by filters
└── provider_queries: dict[ProviderModelType, CallRecord]
                                                  # per-model test CallRecord
```

---

## 2. Configured model methods

These methods read from the in-process model registry. They never
contact a provider — they only check whether your API keys are present
and whether the model declares support for the requested capabilities.

### 2.1 `list_models()`

Returns all models matching the filters.

```python
px.models.list_models(
    model_size: ModelSizeIdentifierType | None = None,
    input_format: InputFormatTypeParam | None = None,
    output_format: OutputFormatTypeParam = OutputFormatType.TEXT,
    feature_tags: FeatureTagParam | None = None,
    tool_tags: ToolTagParam | None = None,
    recommended_only: bool = True,
) → list[ProviderModelType]
```

```python
# All recommended text models
models = px.models.list_models()

# Large text models only
models = px.models.list_models(model_size="large")

# Models that support JSON output AND system prompts
models = px.models.list_models(
    output_format="json",
    feature_tags=["system_prompt"],
)

# Multi-modal models that accept image input
models = px.models.list_models(
    output_format="multi_modal",
    input_format="image",
)

# Models with web-search tool support
models = px.models.list_models(tool_tags=["web_search"])

# Include non-recommended models
models = px.models.list_models(recommended_only=False)

# Image-generation models
models = px.models.list_models(output_format="image")
```

### 2.2 `list_providers()`

Returns provider names that have at least one matching model with an
API key configured.

```python
px.models.list_providers(
    output_format: OutputFormatTypeParam = OutputFormatType.TEXT,
    recommended_only: bool = True,
) → list[str]
```

```python
providers = px.models.list_providers()
# → ["anthropic", "google", "openai"]
```

### 2.3 `list_provider_models()`

Returns models from a specific provider.

```python
px.models.list_provider_models(
    provider: str,                                # required
    model_size: ModelSizeIdentifierType | None = None,
    input_format: InputFormatTypeParam | None = None,
    output_format: OutputFormatTypeParam = OutputFormatType.TEXT,
    feature_tags: FeatureTagParam | None = None,
    tool_tags: ToolTagParam | None = None,
    recommended_only: bool = True,
) → list[ProviderModelType]
```

```python
openai_models = px.models.list_provider_models("openai")
# → [(openai, gpt-4o), (openai, gpt-4), ...]

# Only small OpenAI models with JSON support
small = px.models.list_provider_models(
    "openai", model_size="small", output_format="json",
)
```

Raises `ValueError` if the provider's API key is not found.

### 2.4 `get_model()`

Returns a single model by provider and name. No output-format filter —
resolution is purely by provider/model identity.

```python
px.models.get_model(
    provider: str,                                # required
    model: str,                                   # required
) → ProviderModelType
```

```python
model = px.models.get_model("openai", "gpt-4o")
# → ProviderModelType(provider="openai", model="gpt-4o",
#     provider_model_identifier="gpt-4o-2024-08-06")

# Use it directly in generate()
rec = px.generate(prompt="Hello", provider_model=model)
```

Raises `ValueError` if the API key is missing or the model does not
exist.

### 2.5 `get_model_config()`

Returns the full `ProviderModelConfig` for a model — including pricing,
feature support flags, and metadata. Use this to introspect model
capabilities without running a health check.

```python
px.models.get_model_config(
    provider: str,                                # required
    model: str,                                   # required
) → ProviderModelConfig
```

```python
cfg = px.models.get_model_config("gemini", "gemini-2.5-flash")
print(cfg.features.input_format.image)   # → SUPPORTED
print(cfg.pricing.input_token_cost_nano_usd_per_token)   # → 300
```

Raises `KeyError` if the model doesn't exist.

### 2.6 `get_default_model_list()`

Returns the ordered default model priority list used by the lazy
model-selection fallback when you never call `set_model` (see
`px_client_analysis.md` §5.5).

```python
px.models.get_default_model_list() → list[ProviderModelType]
```

```python
models = px.models.get_default_model_list()
# → [(gemini, gemini-3-pro), (openai, gpt-4o), ...]
print("Default fallback head:", models[0])
```

---

## 3. Working model methods

These methods run the health-check harness — they send a test request
to each model and return only models that respond successfully.
Declared-capability filters run **before** probing, so unreachable
combinations never hit the network. Results are cached (§5). The first
call may take several seconds; subsequent calls are near-instant.

The two `list_working_models` / `list_working_provider_models` methods
accept the full filter set (see §1.1). `list_working_providers` and
`get_working_model` keep the narrower surface their configured
counterparts already had. All four methods refuse
`output_format in {IMAGE, AUDIO, VIDEO}` per §1.2.

### 3.1 `list_working_models()`

The most commonly used working-model method.

```python
px.models.list_working_models(
    model_size: ModelSizeIdentifierType | None = None,
    input_format: InputFormatTypeParam | None = None,
    output_format: OutputFormatTypeParam = OutputFormatType.TEXT,
    feature_tags: FeatureTagParam | None = None,
    tool_tags: ToolTagParam | None = None,
    verbose: bool = True,
    return_all: bool = False,
    clear_model_cache: bool = False,
    recommended_only: bool = True,
) → list[ProviderModelType] | ModelStatus
```

```python
# Basic usage (prints progress to stdout)
working = px.models.list_working_models()

# Silent
working = px.models.list_working_models(verbose=False)

# Full diagnostics via ModelStatus
status = px.models.list_working_models(return_all=True)
print(f"{len(status.working_models)} working")
print(f"{len(status.failed_models)} failed")
for model, record in status.provider_queries.items():
    print(f"  {model}: {record.result.timestamp.response_time}")

# Force retest (ignore cached results)
working = px.models.list_working_models(clear_model_cache=True)

# Narrow by declared capability before probing — only models that
# declare thinking support get health-checked.
thinking = px.models.list_working_models(
    feature_tags=["thinking"], verbose=False,
)
```

### 3.2 `list_working_providers()`

Returns providers with at least one working model for the given
`output_format`.

```python
px.models.list_working_providers(
    verbose: bool = True,
    clear_model_cache: bool = False,
    output_format: OutputFormatTypeParam = OutputFormatType.TEXT,
    recommended_only: bool = True,
) → list[str]
```

```python
providers = px.models.list_working_providers(verbose=False)
# → ["anthropic", "openai"]
```

### 3.3 `list_working_provider_models()`

Returns working models from a specific provider.

```python
px.models.list_working_provider_models(
    provider: str,                                # required
    model_size: ModelSizeIdentifierType | None = None,
    input_format: InputFormatTypeParam | None = None,
    output_format: OutputFormatTypeParam = OutputFormatType.TEXT,
    feature_tags: FeatureTagParam | None = None,
    tool_tags: ToolTagParam | None = None,
    verbose: bool = True,
    return_all: bool = False,
    clear_model_cache: bool = False,
    recommended_only: bool = True,
) → list[ProviderModelType] | ModelStatus
```

```python
openai_working = px.models.list_working_provider_models(
    "openai", verbose=False,
)
# → [(openai, gpt-4o), (openai, gpt-4)]

# Narrow to web-search-capable OpenAI models before probing.
openai_search = px.models.list_working_provider_models(
    "openai", tool_tags=["web_search"], verbose=False,
)
```

Raises `ValueError` if the provider's API key is not found.

### 3.4 `get_working_model()`

Verifies a single model is working and returns it.

```python
px.models.get_working_model(
    provider: str,                                # required
    model: str,                                   # required
    verbose: bool = False,
    clear_model_cache: bool = False,
    output_format: OutputFormatTypeParam = OutputFormatType.TEXT,
) → ProviderModelType
```

```python
model = px.models.get_working_model("openai", "gpt-4o")
rec = px.generate(prompt="Hello", provider_model=model)
```

Raises `ValueError` if the API key is missing, the model does not
exist, or the model fails the health check.

### 3.5 `check_health()`

Full diagnostic sweep. Always clears the model cache and retests every
model (unlike the `list_working_*` methods, which consult the cache by
default). Uses the client's `model_probe_options` for timeout and
parallelism.

```python
px.models.check_health(
    verbose: bool = True,
) → ModelStatus
```

```python
status = px.models.check_health(verbose=False)
print(f"Working: {len(status.working_models)}")
print(f"Failed: {len(status.failed_models)}")
for m, rec in status.provider_queries.items():
    if rec.result.status == px.ResultStatusType.FAILED:
        print(f"  {m} ← {rec.result.error}")
```

Configure `timeout` / `allow_multiprocessing` via
`ModelProbeOptions` at client construction (see `px_client_analysis.md`
§2.6); `check_health()` itself only takes `verbose`.

---

## 4. Common patterns

### 4.1 Discovery before generation

```python
import proxai as px

# Find what's available, then generate
models = px.models.list_models()
print(f"{len(models)} models configured")

rec = px.generate(prompt="Hello", provider_model=models[0])
```

### 4.2 Filter by capability

```python
# Models that support structured Pydantic output
pydantic_models = px.models.list_models(output_format="pydantic")

# Models that support thinking / reasoning
thinking_models = px.models.list_models(feature_tags=["thinking"])

# Models that support web search
search_models = px.models.list_models(tool_tags=["web_search"])

# Models that accept image input AND produce JSON output
vision_json = px.models.list_models(
    input_format="image",
    output_format="json",
)
```

### 4.3 Filter by size

```python
# Cost-effective small models
small = px.models.list_models(model_size="small")

# Flagship models
flagship = px.models.list_models(model_size="largest")
```

### 4.4 Verify before use

```python
# Ensure a specific model is reachable before committing to it
try:
    model = px.models.get_working_model("openai", "gpt-4o")
    rec = px.generate(prompt="Hello", provider_model=model)
except ValueError as e:
    print(f"Model not available: {e}")
```

### 4.5 Full health-check report

```python
status = px.models.list_working_models(
    return_all=True, verbose=False,
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

### 4.6 Iterate over providers

```python
for provider in px.models.list_providers():
    models = px.models.list_provider_models(provider)
    print(f"{provider}: {len(models)} models")
    for m in models:
        print(f"  {m.model} ({m.provider_model_identifier})")
```

### 4.7 Non-text models

```python
# Image generation models
image_models = px.models.list_models(output_format="image")

# All image models (not just recommended)
all_image = px.models.list_models(
    output_format="image", recommended_only=False,
)

# Audio generation models
audio_models = px.models.list_models(output_format="audio")

# Video generation models
video_models = px.models.list_models(output_format="video")
```

### 4.8 Inspecting a model's declared capabilities

```python
cfg = px.models.get_model_config("openai", "gpt-4o")

# Pricing (nano-USD per token — see call_record_analysis.md §2.12)
print(cfg.pricing.input_token_cost_nano_usd_per_token)

# Feature flags
print(cfg.features.input_format.image)    # SUPPORTED | BEST_EFFORT | NOT_SUPPORTED
print(cfg.features.output_format.json)
print(cfg.features.parameters.thinking)
print(cfg.features.tools.web_search)

# Metadata
print(cfg.metadata.is_recommended)
print(cfg.metadata.model_size_tags)
```

### 4.9 Client-instance model discovery

```python
client = px.Client(
    cache_options=px.CacheOptions(cache_path="/tmp/cache"),
)

# Same API, scoped to this client
models = client.models.list_models()
working = client.models.list_working_models(verbose=False)
```

Module-level `px.models` and `client.models` are fully isolated — they
use separate clients with separate model caches (see §7).

### 4.10 Force-refresh cached health-check results

```python
# Cached results may be stale. Force a retest:
working = px.models.list_working_models(clear_model_cache=True)

# Or run a full fresh sweep:
status = px.models.check_health(verbose=False)

# Or reset everything (including the default client):
px.reset_state()
```

---

## 5. Health-check caching

Working-model methods cache their results to avoid repeated network
calls. Key behaviours:

- A built-in default model cache (4-hour TTL) always exists, even
  without `cache_options`. It lives in a per-user platform directory.
- If `cache_options.cache_path` is set, a user-level model cache is
  added on top of the default.
- `clear_model_cache=True` on any working-model method forces a retest
  for the models that method touches.
- `check_health()` *always* clears before probing — it exists
  specifically for "I want a fresh full sweep."
- `px.reset_state()` clears the default model cache entirely.
- Health checks may use multiprocessing (controlled by
  `model_probe_options.allow_multiprocessing` on the client). Disable
  in Jupyter, AWS Lambda, or environments that cannot fork cleanly.
- `model_probe_options.timeout` (default 25s) controls how long to
  wait per model before marking it failed.

See `px_client_analysis.md` §2.6 for the full `ModelProbeOptions`
contract and §5.3 for the default model cache semantics.

---

## 6. Errors

| Method | Trigger | Error |
|--------|---------|-------|
| `list_provider_models()` | Provider API key not in env | `ValueError` |
| `list_working_provider_models()` | Provider API key not in env | `ValueError` |
| `get_model()` | API key missing or model not found | `ValueError` |
| `get_model_config()` | Model not found | `KeyError` |
| `get_working_model()` | API key missing, model not found, or health check failed | `ValueError` |
| `list_working_models()` / `list_working_providers()` / `list_working_provider_models()` / `get_working_model()` | `output_format` is IMAGE / AUDIO / VIDEO | `ValueError` (see §1.2) |

These are all synchronous exceptions — they are not affected by
`suppress_provider_errors`. A health-check *failure* inside
`list_working_*` / `check_health` surfaces as a `FAILED` CallRecord
inside `ModelStatus.provider_queries`, not as a raise. A model whose
declared support for the requested format / feature / tool resolves to
`BEST_EFFORT` under `feature_mapping_strategy=STRICT` is placed in
`ModelStatus.filtered_models` during the pre-probe pass — it never
reaches the probe, so the caller sees neither a raise nor a CallRecord
for it.

---

## 7. Module-level vs client instance

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
