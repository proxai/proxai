# `px.Client` Control Parameters — Redesign Proposal

Partner document to `px_client_analysis.md`, `px_generate_analysis.md`, and
`call_record_analysis.md`. Those docs describe the current state; this doc
proposes a cleaner grouping for the four "loose" client-level parameters
(`feature_mapping_strategy`, `suppress_provider_errors`,
`allow_multiprocessing`, `model_test_timeout`) and two related concerns
(`keep_raw_provider_response`, per-call `ConnectionOptions`).

---

## 0. Proposed `px.Client(...)` at a glance

```
px.Client(                                          # or px.connect(...)
│
├── experiment_path: str | None = None              # group related calls
│
│   # Infrastructure
├── cache_options: CacheOptions | None = None
│   ├── cache_path: str | None
│   ├── unique_response_limit: int | None = 1
│   ├── retry_if_error_cached: bool = False
│   ├── clear_query_cache_on_connect: bool = False
│   ├── disable_model_cache: bool = False
│   ├── clear_model_cache_on_connect: bool = False
│   └── model_cache_duration: int | None = None
│
├── logging_options: LoggingOptions | None = None
│   ├── logging_path: str | None
│   ├── stdout: bool = False
│   └── hide_sensitive_content: bool = False
│
├── proxdash_options: ProxDashOptions | None = None
│   ├── stdout: bool = False
│   ├── hide_sensitive_content: bool = False
│   ├── disable_proxdash: bool = False
│   ├── api_key: str | None
│   └── base_url: str | None
│
│   # Provider call defaults (overridable per-call)
├── provider_call_options: ProviderCallOptions | None = None
│   ├── feature_mapping_strategy: FeatureMappingStrategy = BEST_EFFORT
│   └── suppress_provider_errors: bool = False
│
│   # Model probing (which models are available / working)
├── model_probe_options: ModelProbeOptions | None = None
│   ├── allow_multiprocessing: bool = True
│   └── timeout: int = 25                          # seconds, per-model
│
│   # Developer diagnostics
└── debug_options: DebugOptions | None = None
    └── keep_raw_provider_response: bool = False
)
```

### Current (flat) vs proposed (grouped)

```
CURRENT                                  PROPOSED
─────────────────────────────────        ─────────────────────────────────
feature_mapping_strategy  ─┐             provider_call_options:
suppress_provider_errors  ─┘ mixed         ├── feature_mapping_strategy
                             at            └── suppress_provider_errors
allow_multiprocessing  ────┐ same
model_test_timeout  ───────┘ level       model_probe_options:
                                           ├── allow_multiprocessing
keep_raw_provider_response ── orphan       └── timeout

                                         debug_options:
                                           └── keep_raw_provider_response
```

---

## 1. The problem

After the `px.generate()` and `CallRecord` refactorings, those APIs have
clear semantic grouping: **what** to generate (`prompt`, `messages`,
`system_prompt`), **which** model (`provider_model`), **how** to generate
(`parameters`, `tools`, `response_format`), and **per-call behaviour**
(`connection_options`). The client constructor, by contrast, has four
parameters dangling at the top level with no obvious relationship:

```
px.Client(
    ...
    feature_mapping_strategy: FeatureMappingStrategy = BEST_EFFORT   # generation behaviour
    suppress_provider_errors: bool = False                           # error handling
    allow_multiprocessing: bool = True                               # health-check infra
    model_test_timeout: int = 25                                     # health-check infra
)
```

Problems:

1. **Mixed concerns at the same level.** `feature_mapping_strategy` shapes
   every `generate()` call. `allow_multiprocessing` and `model_test_timeout`
   only affect health checks / model discovery. `suppress_provider_errors`
   is a default that can be overridden per-call. These are three different
   categories presented as peers.

2. **`suppress_provider_errors` appears in two places.** It lives on
   `Client(...)` as a default *and* on `ConnectionOptions` as a per-call
   override. The name is identical, but the semantics differ (`bool` vs
   `bool | None` where `None` means "inherit"). This is correct but
   confusing — users have to learn the inheritance rule.

3. **`allow_multiprocessing` and `model_test_timeout` are health-check
   knobs exposed at the top level of every client.** Most users never
   call `check_health()` or `list_working_models()`. These parameters
   add cognitive load to the constructor for a niche use case.

4. **No room for future generation-level defaults.** If we later want a
   client-wide default for `skip_cache`, `endpoint`, or new generation
   behaviours, we have no structured place to put them — they'd become
   more loose top-level parameters.

5. **`keep_raw_provider_response` is a debug flag mixed in with production
   knobs.** It has a fundamentally different audience (library developers
   debugging provider SDK responses) from the other parameters.

---

## 2. Design goals

- **Semantic grouping**: parameters that serve the same purpose live
  together in a named object, not as unrelated keyword arguments.
- **Progressive disclosure**: the minimal `px.Client()` or `px.connect()`
  call should need zero of these. Advanced knobs are discoverable when
  you look at the options object, not when you read the constructor
  signature.
- **Consistency with `px.generate()`**: the generate call uses structured
  option objects (`ParameterType`, `ConnectionOptions`, `ResponseFormat`).
  The client constructor should follow the same pattern.
- **Backward compatibility**: existing code that passes these as keyword
  args should keep working with minimal change. Deprecation, not breakage.

---

## 3. Proposed structure

```
px.Client(
    experiment_path: str | None = None,
    cache_options: CacheOptions | None = None,
    logging_options: LoggingOptions | None = None,
    proxdash_options: ProxDashOptions | None = None,
    provider_call_options: ProviderCallOptions | None = None,       # NEW
    model_probe_options: ModelProbeOptions | None = None,   # NEW
    debug_options: DebugOptions | None = None,                # NEW
)
```

### 3.1 `ProviderCallOptions` — client-wide defaults for provider calls

```python
@dataclasses.dataclass
class ProviderCallOptions:
  feature_mapping_strategy: FeatureMappingStrategy = BEST_EFFORT
  suppress_provider_errors: bool = False
```

**Why this grouping:** Both parameters directly affect what happens inside
every `generate()` call. `feature_mapping_strategy` controls how the
feature adapter handles unsupported features. `suppress_provider_errors`
controls whether provider failures raise or return. Both can conceptually
be thought of as "how should generation calls behave by default?"

**Why not put these in `ConnectionOptions`?** `ConnectionOptions` is
per-call and travels with the `QueryRecord` into the cache key and
ProxDash logs. `provider_call_options` is client-level policy that is
*not* part of the query identity. Mixing them would mean the same type
serves two different scopes.

**Why the name `ProviderCallOptions`?** It communicates:
- These govern how the client *calls providers* (feature compatibility +
  error handling).
- The `_options` suffix is consistent with every other group
  (`cache_options`, `logging_options`, `proxdash_options`).
- It pairs naturally with `ConnectionOptions` — `ProviderCallOptions`
  sets client-wide defaults, `ConnectionOptions` overrides per call.

Alternative names considered and rejected:
- `GenerateDefaults` — requires knowing that `generate` refers to the
  `px.generate()` function. Not obvious at first encounter.
- `GenerationPolicy` — too abstract, sounds like rate limiting.
- `ProviderOptions` — could be confused with provider-specific config
  like API keys or base URLs.
- `BehaviorOptions` — too vague, could mean anything.
- `CallOptions` — might blur with `ConnectionOptions`.

**Per-call override story:**

`suppress_provider_errors` is already overridable via
`ConnectionOptions.suppress_provider_errors`. No change needed.

`feature_mapping_strategy` currently has no per-call override. This is
a separate design decision — if we want to add one, it belongs in
`ConnectionOptions` as `feature_mapping_strategy: FeatureMappingStrategy | None = None`.
This proposal does not require that change but is compatible with it.

### 3.2 `ModelProbeOptions` — configuration for model probing

```python
@dataclasses.dataclass
class ModelProbeOptions:
  allow_multiprocessing: bool = True
  timeout: int = 25  # seconds, per-model
```

**Why this grouping:** Both parameters *exclusively* affect
`check_health()` and `list_working_models()`. They have zero impact on
`generate()`. Grouping them makes this scope immediately obvious and
removes two parameters from the main constructor that most users never
need to touch.

**Why `timeout` instead of `model_test_timeout`?** Inside a
`ModelProbeOptions` object, the context is already "model probing." The
`model_test_` prefix is redundant. Just `timeout` is clear and
consistent with how other Python libraries name timeouts inside a
scoped options object (e.g., `requests.Session.timeout`,
`httpx.Timeout.connect`).

**Should `check_health(...)` still accept these as direct kwargs?**
Yes — `px.check_health(allow_multiprocessing=False, timeout=25)` is
cleaner for one-off calls than constructing an options object. The
standalone `check_health()` function creates an ephemeral client anyway,
so its kwargs are not "client defaults" — they are direct config. It
can also accept `model_probe_options=ModelProbeOptions(...)` for
consistency.

### 3.3 `DebugOptions` — developer-only escape hatches

```python
@dataclasses.dataclass
class DebugOptions:
  keep_raw_provider_response: bool = False
```

**Why separate?** `keep_raw_provider_response` is not a production knob.
It attaches the raw SDK response object to every `CallRecord`, which
increases memory usage and breaks serialization. Isolating it in
`DebugOptions` signals "you probably don't need this" and keeps the
main constructor clean.

**Future residents:** If we later add `trace_adapter_transforms`,
`log_cache_decisions`, or other diagnostic flags, they go here.

---

## 4. `ConnectionOptions` refinement

The current `ConnectionOptions` is well-designed for per-call overrides.
One small naming improvement:

```python
@dataclasses.dataclass
class ConnectionOptions:
  fallback_models: list[ProviderModelType] | None = None
  suppress_provider_errors: bool | None = None
  endpoint: str | None = None
  skip_cache: bool | None = None
  override_cache_value: bool | None = None
```

**No changes proposed.** `ConnectionOptions` is already clean. The
`suppress_provider_errors` field here correctly uses `bool | None` to
mean "override the client default or inherit." This works well with the
proposed `provider_call_options.suppress_provider_errors` as the default
source.

One thing worth noting: the `ConnectionOptions` naming pattern
(suppression, caching, endpoint) is internally consistent — all fields
are "how should this single call connect to the provider?" That's the
right boundary.

---

## 5. Migration path

### 5.1 Phase 1 — Add the new option types, keep old kwargs

```python
# Both work:
px.connect(suppress_provider_errors=True)                      # old
px.connect(provider_call_options=ProviderCallOptions(              # new
    suppress_provider_errors=True))
```

Internally, old-style kwargs construct the corresponding options object.
If both are provided, raise `ValueError`.

### 5.2 Phase 2 — Deprecation warnings on old-style kwargs

```python
# This still works but emits DeprecationWarning:
px.connect(allow_multiprocessing=False)
# Preferred:
px.connect(model_probe_options=ModelProbeOptions(
    allow_multiprocessing=False))
```

### 5.3 Phase 3 — Remove old-style kwargs

After a major version bump, the constructor accepts only the
structured forms.

---

## 6. Before and after

### Before

```python
client = px.Client(
    cache_options=px.CacheOptions(cache_path="/tmp/cache"),
    logging_options=px.LoggingOptions(stdout=True),
    feature_mapping_strategy=px.FeatureMappingStrategy.STRICT,
    suppress_provider_errors=True,
    allow_multiprocessing=False,
    model_test_timeout=30,
    keep_raw_provider_response=True,
)
```

Seven keyword arguments. No visual grouping. A reader must already know
which ones affect generation, which affect health checks, and which are
debug-only.

### After

```python
client = px.Client(
    cache_options=px.CacheOptions(cache_path="/tmp/cache"),
    logging_options=px.LoggingOptions(stdout=True),
    provider_call_options=px.ProviderCallOptions(
        feature_mapping_strategy=px.FeatureMappingStrategy.STRICT,
        suppress_provider_errors=True,
    ),
    model_probe_options=px.ModelProbeOptions(
        allow_multiprocessing=False,
        timeout=30,
    ),
    debug_options=px.DebugOptions(
        keep_raw_provider_response=True,
    ),
)
```

Five top-level keyword arguments, each a named group. A reader unfamiliar
with ProxAI can see at a glance: this client has caching, logging, strict
provider call settings, custom health-check settings, and debug mode on.

### Minimal usage (unchanged)

```python
# Still works — zero options objects needed
px.connect()
text = px.generate_text(prompt="Hello")
```

### Common usage

```python
# The 80% case: cache + logging, everything else defaults
px.connect(
    cache_options=px.CacheOptions(cache_path="/tmp/cache"),
    logging_options=px.LoggingOptions(stdout=True),
)
```

No `provider_call_options`, `model_probe_options`, or `debug_options`
needed in the common case. Progressive disclosure works.

---

## 7. Full constructor tree (proposed)

```
px.Client(
│
├── experiment_path: str | None = None
│
├── cache_options: CacheOptions | None = None
│   ├── cache_path: str | None
│   ├── unique_response_limit: int | None = 1
│   ├── retry_if_error_cached: bool = False
│   ├── clear_query_cache_on_connect: bool = False
│   ├── disable_model_cache: bool = False
│   ├── clear_model_cache_on_connect: bool = False
│   └── model_cache_duration: int | None = None
│
├── logging_options: LoggingOptions | None = None
│   ├── logging_path: str | None
│   ├── stdout: bool = False
│   └── hide_sensitive_content: bool = False
│
├── proxdash_options: ProxDashOptions | None = None
│   ├── stdout: bool = False
│   ├── hide_sensitive_content: bool = False
│   ├── disable_proxdash: bool = False
│   ├── api_key: str | None
│   └── base_url: str | None
│
├── provider_call_options: ProviderCallOptions | None = None
│   ├── feature_mapping_strategy: FeatureMappingStrategy = BEST_EFFORT
│   └── suppress_provider_errors: bool = False
│
├── model_probe_options: ModelProbeOptions | None = None
│   ├── allow_multiprocessing: bool = True
│   └── timeout: int = 25
│
└── debug_options: DebugOptions | None = None
    └── keep_raw_provider_response: bool = False
)
```

### Naming pattern across all option groups

| Group | What it configures | When it matters |
|-------|-------------------|-----------------|
| `cache_options` | Query and model caching | Every call (if configured) |
| `logging_options` | File and stdout logging | Every call (if configured) |
| `proxdash_options` | ProxDash monitoring | Every call (if configured) |
| `provider_call_options` | Provider call defaults | Every `generate()` call |
| `model_probe_options` | Model probing / availability | `check_health()`, `list_working_models()` |
| `debug_options` | Developer diagnostics | Never in production |

The pattern: `{domain}_options` consistently across all groups.

---

## 8. Alternative designs considered

### 8.1 Flat kwargs with better names (no new types)

```python
px.Client(
    strict_feature_mapping=True,            # was feature_mapping_strategy
    suppress_errors=True,                   # was suppress_provider_errors
    parallel_model_probes=True,             # was allow_multiprocessing
    model_probe_timeout=25,                 # was model_test_timeout
)
```

**Rejected.** Renaming helps readability but doesn't solve the mixed-
concerns problem. Four unrelated parameters at the same level are still
four unrelated parameters, regardless of naming.

### 8.2 Single `behavior_options` bag

```python
class BehaviorOptions:
    feature_mapping_strategy: ...
    suppress_provider_errors: ...
    allow_multiprocessing: ...
    model_test_timeout: ...
    keep_raw_provider_response: ...
```

**Rejected.** This is just moving the problem one level down. The five
fields still have three different scopes (generation, health-check,
debug). A single bag doesn't communicate those boundaries.

### 8.3 Merge into existing option groups

Put `suppress_provider_errors` into `ConnectionOptions` as a default.
Put `allow_multiprocessing` into `CacheOptions` (since model cache is
related).

**Rejected.** This would overload existing types with unrelated
concerns. `CacheOptions` is about disk caching, not process management.
`ConnectionOptions` is per-call, not per-client.

### 8.4 Two groups instead of three (merge debug into health-check)

```python
class ProviderCallOptions:
    feature_mapping_strategy: ...
    suppress_provider_errors: ...

class InfraOptions:
    allow_multiprocessing: ...
    timeout: ...
    keep_raw_provider_response: ...
```

**Viable but weaker.** `keep_raw_provider_response` has nothing to do
with health checks or multiprocessing. Putting it in `InfraOptions`
would be misleading. Three groups is cleaner even though `DebugOptions`
currently has one field — it's the right conceptual bucket for future
debug flags.

---

## 9. Open questions

1. **Should `feature_mapping_strategy` get a per-call override?**
   Adding `feature_mapping_strategy: FeatureMappingStrategy | None`
   to `ConnectionOptions` would be consistent with how
   `suppress_provider_errors` works. But it adds complexity for a
   feature nobody has requested. Recommendation: don't add it now,
   but the design supports it.

2. **Should `provider_call_options` eventually hold default `parameters`?**
   E.g., `ProviderCallOptions(default_temperature=0.7)`. This is tempting
   but dangerous — it creates a merge problem (what happens when both
   the default and the per-call `parameters` set `temperature`?). The
   current `set_model()` approach avoids this. Recommendation: leave
   default parameters out of `ProviderCallOptions`.

3. **Should `ModelProbeOptions` be passable to `check_health()` directly?**
   E.g., `px.check_health(model_probe_options=ModelProbeOptions(...))`.
   This would give users one type for both client-level and standalone
   probing. Recommendation: yes — accept both the options object and
   the flat kwargs for convenience.

4. **`ProviderCallOptions` vs `ConnectionOptions` — is the boundary clear?**
   `ProviderCallOptions` is client-wide defaults; `ConnectionOptions` is
   per-call overrides. The word "provider call" appears in both concepts.
   In practice the scoping rule is simple: `ProviderCallOptions` lives on
   the client, `ConnectionOptions` lives on the `generate()` call. If
   user testing reveals confusion, consider renaming `ConnectionOptions`
   to `CallOverrides` to sharpen the contrast.
