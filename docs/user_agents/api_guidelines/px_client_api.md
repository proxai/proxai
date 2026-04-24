# `px.Client` Comprehensive Use Case Analysis

Source of truth: `src/proxai/types.py` (the option dataclasses),
`src/proxai/client.py` (the `ProxAIClient` constructor, `set_model()`,
and `generate()` method), and `src/proxai/proxai.py` (the module-level
functions). If this document disagrees with those files, the files win
— update this document.

This is the definitive reference for every knob a caller of
`px.Client(...)` or `px.connect(...)` can turn, every implicit
behaviour they observe when they don't, and every validation error the
client can raise up front. Read this before adding a new client
option, a new `ConnectionOptions` field, or changing how default
lookups cascade.

See also: `call_record_analysis.md` (the `CallRecord` shape and
request pipeline) and `px_generate_analysis.md` (calling conventions
for `px.generate()` and its wrappers).

---

## 1. Option structure (current)

### 1.1 Construction-time options

```
ProxAIClient(                                  # same kwargs on px.connect(...)
│
├── experiment_path: str | None = None
│
├── cache_options: CacheOptions | None = None
│   ├── cache_path: str | None = None
│   ├── unique_response_limit: int | None = 1
│   ├── clear_query_cache_on_connect: bool = False
│   ├── disable_model_cache: bool = False
│   ├── clear_model_cache_on_connect: bool = False
│   └── model_cache_duration: int | None = None
│
├── logging_options: LoggingOptions | None = None
│   ├── logging_path: str | None = None
│   ├── stdout: bool = False
│   └── hide_sensitive_content: bool = False
│
├── proxdash_options: ProxDashOptions | None = None
│   ├── stdout: bool = False
│   ├── hide_sensitive_content: bool = False
│   ├── disable_proxdash: bool = False
│   ├── api_key: str | None = None
│   └── base_url: str | None = "https://proxainest-production.up.railway.app"
│
├── provider_call_options: ProviderCallOptions | None = None
│   ├── feature_mapping_strategy: FeatureMappingStrategy = BEST_EFFORT
│   ├── suppress_provider_errors: bool = False
│   └── allow_parallel_file_operations: bool = True
│
├── model_probe_options: ModelProbeOptions | None = None
│   ├── allow_multiprocessing: bool = True
│   └── timeout: int = 25                     # seconds
│
└── debug_options: DebugOptions | None = None
    └── keep_raw_provider_response: bool = False
)
```

The exact same kwargs are accepted by the module-level façade
`px.connect(...)` and by `px.Client(...)`. The module-level form
writes to a hidden global client used by `px.generate(...)`,
`px.set_model(...)`, `px.models.*`. See §5.1 for the two calling
styles and their isolation rules.

### 1.2 Per-call options

```
# Passed via client.generate(..., connection_options=ConnectionOptions(...))
ConnectionOptions(
│
├── fallback_models: list[ProviderModelIdentifierType] | None = None
│                                              # each entry is either a
│                                              # ProviderModelType or a
│                                              # (provider, model) tuple
├── suppress_provider_errors: bool | None = None   # None = inherit client default
├── endpoint: str | None = None
├── skip_cache: bool | None = None
└── override_cache_value: bool | None = None
)
```

---

## 2. Client-level options

### 2.1 `experiment_path: str | None`

An identifier that groups all calls made through this client under a
common name. Default `None`.

**What it affects:**

- **File logs.** If `logging_options.logging_path` is also set, the
  effective log directory becomes
  `{logging_options.logging_path}/{experiment_path}`. The directory
  is created at construction time if it does not exist.
- **ProxDash uploads.** Every call this client makes is reported to
  ProxDash under this `experiment_path`, so uploads are grouped
  server-side.

**Errors.** Invalid path format → `ValueError` from
`experiment.validate_experiment_path`.

### 2.2 `cache_options: CacheOptions | None`

Configures the query cache (replays identical calls) and the user-level
model cache (stores which models are reachable). Default `None`, which
means: no query cache, and health checks fall back to a built-in
default model cache (see §5.3). You do **not** need `cache_options` to
get working model discovery — you only need it to get query replay.

See `cache_behaviors.md` for the user-facing cache contract — what each
field does at runtime, how to read cache metadata off a `CallRecord`,
and when to bypass via `ConnectionOptions`.

#### 2.2.1 `cache_path: str | None`

Directory on disk where both caches are stored. Default `None`.

- **Required** if you pass a non-`None` `cache_options`, **unless** you
  also set `disable_model_cache=True`. Otherwise the constructor raises
  `ValueError: cache_path is required while setting cache_options`.
- The directory is created on demand.

#### 2.2.2 `unique_response_limit: int | None`

Number of distinct responses to collect per unique query before the
cache starts serving replays. Default `1`.

- With `1`, the **first** response for a query is cached immediately
  and every subsequent call replays it.
- With `3`, the cache keeps asking the provider until it has seen 3
  distinct responses for the same query; after that, calls are served
  from cache, rotating across the collected responses.
- Useful for generating diversity on deterministic prompts without
  losing cacheability. The CallRecord for a call that is still
  "collecting" sets `connection.cache_look_fail_reason =
  UNIQUE_RESPONSE_LIMIT_NOT_REACHED`.

#### 2.2.3 `clear_query_cache_on_connect: bool`

If `True`, the query cache directory is **wiped** during client
construction. Default `False`. Destructive — every previously cached
response is gone. Use in tests or when you intentionally want to
invalidate prior results.

#### 2.2.4 `disable_model_cache: bool`

If `True`, the user-level model cache at `cache_path` is not used.
Default `False`. Health check results still flow through the built-in
default model cache (§5.3), so model discovery still works.

This is the one escape that lets you pass `cache_options` without
setting `cache_path` — useful if you want the query cache somewhere
and do not want a separate model cache at all.

#### 2.2.5 `clear_model_cache_on_connect: bool`

If `True`, the user-level model cache at `cache_path` is wiped during
client construction. Default `False`. Same destructive semantics as
`clear_query_cache_on_connect`. Does **not** wipe the built-in default
model cache — use `px.reset_state()` for that.

#### 2.2.6 `model_cache_duration: int | None`

Time-to-live for model cache entries, in seconds. Default `None` (no
expiry on user-level cache entries). Stale entries are discarded on
read.

### 2.3 `logging_options: LoggingOptions | None`

Controls local file logging and stdout mirroring. Default `None`,
which means no file logging and no stdout output. Omitting this option
still leaves `client.logging_options` readable — fields just report
their defaults — so downstream code can check `stdout` safely without
a `None` check.

#### 2.3.1 `logging_path: str | None`

Directory where log files are written. Default `None` (no file logging).

- If set together with `experiment_path`, the effective directory
  becomes `{logging_path}/{experiment_path}`.
- The directory is `makedirs(..., exist_ok=True)`'d at construction,
  so pointing at a non-existent path is fine as long as the parent
  chain is writable.

#### 2.3.2 `stdout: bool`

Mirror log entries to stdout. Default `False`. Independent of file
logging — you can enable stdout logging without a `logging_path`, and
vice versa.

#### 2.3.3 `hide_sensitive_content: bool`

Redact prompt and response bodies in written logs. Default `False`.
Only affects what is written; the in-memory `CallRecord` you receive
is not modified.

### 2.4 `proxdash_options: ProxDashOptions | None`

ProxDash is ProxAI's hosted monitoring backend. These options control
whether and how calls are reported. Default `None`, which means
ProxDash is still enabled if a `PROXDASH_API_KEY` environment variable
is present — omitting the option is **not** the same as opting out.
See §5.4.

#### 2.4.1 `stdout: bool`

Print ProxDash connection status messages (connect, disconnect, upload
failures) to stdout. Default `False`.

#### 2.4.2 `hide_sensitive_content: bool`

Redact prompt and response bodies in uploads. Default `False`. Same
local-only guarantee as `logging_options.hide_sensitive_content`.

#### 2.4.3 `disable_proxdash: bool`

Hard-off switch. Default `False`. If `True`, no ProxDash connection is
established even if an API key is present. This is the only way to
fully opt out without deleting the env var.

#### 2.4.4 `api_key: str | None`

ProxDash API key. Default `None`. When `None`, the client falls back
to the `PROXDASH_API_KEY` environment variable. If neither is set,
ProxDash is effectively disabled but you get a status of
`API_KEY_NOT_FOUND` on the connection (not an error).

#### 2.4.5 `base_url: str | None`

ProxDash server URL. Default
`"https://proxainest-production.up.railway.app"`. Override only if you
are pointing at a staging server or a self-hosted deployment.

### 2.5 `provider_call_options: ProviderCallOptions | None`

Client-wide defaults for what happens when a provider call is
actually executed. Default `None`, which resolves to the `ProviderCallOptions`
defaults listed below.

#### 2.5.1 `feature_mapping_strategy: FeatureMappingStrategy`

How strict the client is when matching your request against a model's
endpoint capabilities. Default `FeatureMappingStrategy.BEST_EFFORT`.

- `BEST_EFFORT` — if the endpoint only partially supports your
  request (e.g., the model has no native system prompt but the
  adapter can fold it into the first user message), the client
  accepts it and adapts. The returned CallRecord notes the strategy
  in `connection.feature_mapping_strategy`.
- `STRICT` — anything below full support is rejected. The client
  raises `ValueError` before contacting the provider if no endpoint
  fully supports your request.

Currently there is **no per-call override**. Strategy is chosen once
at client construction and applies to every call.

#### 2.5.2 `suppress_provider_errors: bool`

What to do when a provider call fails (rate limit, HTTP 500, JSON
parse error, etc.). Default `False`.

- `False` — the original exception is raised out of `.generate()`.
  No CallRecord is returned.
- `True` — a CallRecord with `result.status = FAILED` and a
  stringified `result.error` is returned instead. You need to check
  `result.status` in your caller.

Can be overridden on a single call via
`ConnectionOptions.suppress_provider_errors` (§3.2). See also §5.6
for how fallback chains force this on internally.

#### 2.5.3 `allow_parallel_file_operations: bool`

If `True`, the files manager may upload / fetch media attachments in
parallel when a call carries multiple `MessageContent` media blocks.
Default `True`. Disable in environments that cannot run concurrent
I/O reliably (some sandboxed or single-threaded runtimes).

Does **not** affect text-only calls.

### 2.6 `model_probe_options: ModelProbeOptions | None`

Controls the health-check harness used by `client.models.check_health()`,
`client.models.list_working_models()`, and the lazy model-selection
fallback (§5.5). Default `None`, which resolves to the
`ModelProbeOptions` defaults below.

#### 2.6.1 `allow_multiprocessing: bool`

Whether model probing may spin up a process pool to probe models in
parallel. Default `True`.

Disable in environments that cannot fork cleanly — Jupyter notebooks on
macOS, AWS Lambda, some Windows configurations. With `False`, model
probing is sequential and slower but never crashes on spawn.

Does **not** affect normal `.generate()` calls — only the health-check
harness.

#### 2.6.2 `timeout: int`

Seconds to wait for any one model during a health check before marking
it failed. Default `25`. Must be `>= 1` (setter raises `ValueError:
ModelProbeOptions.timeout must be >= 1.` otherwise).

Does **not** apply to normal `.generate()` calls; those use the
provider SDK's own timeouts.

### 2.7 `debug_options: DebugOptions | None`

Developer-only diagnostics. Default `None`, which resolves to the
`DebugOptions` defaults below.

#### 2.7.1 `keep_raw_provider_response: bool`

If `True`, the raw response object from the provider SDK is attached
to `CallRecord.debug.raw_provider_response` (see
`call_record_analysis.md` §1). Default `False`. None of `DebugInfo`
is serialised to the cache or ProxDash — the raw response can hold a
live SDK object that is not portable. Intended purely for local
debugging.

---

## 3. Per-call options: `ConnectionOptions`

Pass to `client.generate(..., connection_options=...)`. Each field is
scoped to a single call. `None` on any field means "use the client
default" (where applicable) or "off".

### 3.1 `fallback_models: list[ProviderModelIdentifierType] | None`

Ordered list of models to try if the primary model fails. Default
`None` (no fallback). Each entry may be a `ProviderModelType` instance
or a `(provider, model)` tuple — tuples are resolved by the client
before dispatch.

**Behaviour:**

- On the primary's success, the fallbacks are never tried.
- On the primary's failure, the client tries each fallback in order
  and returns on the first success.
- If every model fails, the client returns the **last** failed
  CallRecord — not the first, and not a list.
- The returned CallRecord's `query.provider_model` tells you which
  model actually answered. `connection.failed_fallback_models` lists
  the models that failed before the one that returned.
- Setting `fallback_models` triggers a shallow-copy of the caller's
  `ConnectionOptions` before internal mutations (see §5.6); the
  caller's instance stays untouched.

**Conflicts (all raise `ValueError` up front):**

- With `suppress_provider_errors=True` on the same `ConnectionOptions`
  — the fallback loop owns error suppression and cannot coexist with
  an external override.
- With `endpoint` on the same `ConnectionOptions` — different models
  have different endpoints, one override cannot apply to all of them.

### 3.2 `suppress_provider_errors: bool | None`

Overrides the client-level `provider_call_options.suppress_provider_errors`
for this single call. Default `None` (inherit). Same semantics as
§2.5.2 once resolved.

Conflicts with `fallback_models` on the same `ConnectionOptions`
(see §3.1).

### 3.3 `endpoint: str | None`

Force a specific endpoint within the selected provider. Default `None`
(let the client pick via the feature adapter). The string is the
provider-specific endpoint key, e.g. `"chat.completions.create"`,
`"beta.chat.completions.parse"`, `"responses.create"`.

- Must still pass the support-level check under the current
  `feature_mapping_strategy`. If you force an endpoint that cannot
  handle your request, the client raises `ValueError` before the call.
- Conflicts with `fallback_models` on the same `ConnectionOptions`.

### 3.4 `skip_cache: bool | None`

If `True`, the query cache is bypassed completely for this call — no
lookup, no write. Default `None` (use the cache if it is configured).

Wins over `override_cache_value`: if both are set, `skip_cache` applies
and the cache is untouched.

### 3.5 `override_cache_value: bool | None`

If `True`, the query cache is ignored on read (always calls the
provider) but the result is **still written back** on success.
Default `None`. The typical use is "force-refresh the cached answer for
this question."

---

## 4. Errors & parameter conflicts

Every validation error a caller can hit. All of these raise
`ValueError` synchronously — they are programmer errors and are never
routed through `suppress_provider_errors`.

| Where | Trigger | Message (abbreviated) |
|-------|---------|------------------------|
| `Client()` / `connect()` | `cache_options` set, no `cache_path`, no `disable_model_cache=True` | `cache_path is required while setting cache_options` |
| `Client()` / `connect()` | `model_probe_options.timeout < 1` | `ModelProbeOptions.timeout must be >= 1.` |
| `Client()` / `connect()` | `root_logging_path` (derived from `logging_options.logging_path`) points at a path whose parent cannot be created | `Root logging path does not exist: ...` |
| `client.generate()` | `prompt` and `messages` both set | `prompt and messages cannot be used together` |
| `client.generate()` | `system_prompt` and `messages` both set | `system_prompt and messages cannot be used together. Please use "system" message in messages...` |
| `client.generate()` | `connection_options.fallback_models` and `connection_options.suppress_provider_errors=True` | `suppress_provider_errors and fallback_models cannot be used together` |
| `client.generate()` | `connection_options.fallback_models` and `connection_options.endpoint` | `endpoint and fallback_models cannot be used together` |
| `client.generate()` | forced `endpoint` not supported by the resolved model | `endpoint <name> is not supported` (or `... is not supported in STRICT mode`) |
| `client.generate()` | no endpoint at all compatible with the request | `No compatible endpoint found for the query record...` |
| `client.set_model()` | both `provider_model` and `generate_text` given | `provider_model and generate_text cannot be set at the same time. Please set one of them.` |
| `client.set_model()` | no kwargs given at all | `At least one model must be specified. Use provider_model, generate_text, generate_json, generate_pydantic, generate_image, generate_audio, or generate_video.` |
| First `generate()` without `set_model` | no model in the default priority list is reachable with your env keys | `No working models found in current environment. Please check your environment variables...` |

Any provider-side exception (rate limit, auth, HTTP 5xx, JSON parse
failure) follows a **separate** path — it is raised out of
`.generate()` unless `suppress_provider_errors` is `True`, in which
case you get a `CallRecord` with `result.status = FAILED`.

---

## 5. Implicit behaviours

Things that happen without you configuring them — what your code will
observe "for free."

### 5.1 Two calling styles, completely isolated

`px.connect(...)` writes to a **hidden global default client** used by
`px.generate`, `px.set_model`, `px.models`, `px.get_current_options`,
`px.reset_state`. Calling `px.Client(...)` constructs a **new, separate
client instance** that lives alongside the default — it never replaces
it.

Consequence: code that mixes `px.generate(...)` with a manually
constructed `client = px.Client(...); client.generate(...)` is talking
to **two independent systems**. Two caches, two log files, two ProxDash
connections. Tests rely on this.

### 5.2 Default client lifecycle

- First access to any `px.generate` / `px.set_model` / `px.models`
  call lazily creates a default client with all defaults.
- Calling `px.connect(...)` replaces the default client wholesale
  (not in place). Previously configured options are dropped; any
  query cache state from the old default client is forgotten.
- Calling `px.connect()` twice in a program → the second call wins.
- `px.reset_state()` drops the default client to `None` and, if the
  default model cache was stored in a platform directory, clears it.
  The next API call will construct a fresh default client.

### 5.3 There is always a working model cache

Even if you never set `cache_options`, the client keeps a built-in
default model cache in a per-user platform directory (4-hour TTL), so
health checks and `client.models.list_working_models()` always have
somewhere to store results. On systems where the platform directory
cannot be created, the client falls back to a temporary directory.

You cannot point this default model cache at a location of your
choosing — that is what `cache_options.cache_path` is for. Setting
`cache_options.cache_path` does not *replace* the default; it adds a
user-level cache on top of it.

`px.reset_state()` clears the default model cache; nothing else in the
public API does.

### 5.4 ProxDash is on if you have an API key in your environment

If `PROXDASH_API_KEY` is set, ProxDash is active even if you never pass
`proxdash_options`. To fully opt out without deleting the env var, pass
`proxdash_options=ProxDashOptions(disable_proxdash=True)`.

ProxDash failures never break a call. If the upload fails, the
CallRecord still returns successfully and the ProxDash layer logs the
error through the normal logging options.

### 5.5 Lazy model selection when you never call `set_model`

If your code calls `client.generate()` without ever calling
`set_model` and without passing `provider_model=`, the client walks a
built-in default priority list, pings models in order, and picks the
first one that responds. The answer is cached for subsequent calls.

If your environment has **no** reachable models in the default
priority list, the client falls back to probing any working model it
can find. Only if that also fails does it raise the "no working
models" error (§4). Translation: misconfiguring your env keys is
detected on your first `generate()` call, not at construction time.

### 5.6 Fallback chains force-enable error suppression

When `connection_options.fallback_models` is set, the fallback loop
internally enables error suppression for every attempt so a single
provider failure doesn't abort the chain. Your `ConnectionOptions`
instance is not mutated — the client operates on its own copy.

That is why §3.1 forbids setting `suppress_provider_errors=True` and
`fallback_models` together — the loop already owns suppression and a
second source of truth would be ambiguous.

You always get exactly one `CallRecord` back from a fallback call. If
every model failed, `status=FAILED`; if any succeeded, `status=SUCCESS`
and `connection.failed_fallback_models` lists the models that failed
before the one that returned.

### 5.7 Health checks do not touch the default client's state

`client.models.check_health()` retests every model fresh using the
client's `model_probe_options`. It does not disturb the **default
model cache**'s in-memory selection that `px.generate()` has been
using for the current session.

`px.models.check_health()` runs against the **default** client — so
it reuses that default client's `model_probe_options` / cache, not an
ephemeral throwaway.

### 5.8 Option objects you omit still report defaults

If you pass `logging_options=None` (or just leave it out),
`client.logging_options` is still readable — its fields return the
blank defaults (`stdout=False`, `hide_sensitive_content=False`,
`logging_path=None`). Same for `proxdash_options`,
`provider_call_options`, `model_probe_options`, `debug_options`. You
never have to guard `client.logging_options.stdout` or
`client.provider_call_options.feature_mapping_strategy` with a `None`
check.

`cache_options` is the exception: leaving it out leaves
`client.cache_options` as `None`. That is intentional — "no query
cache" needs to be distinguishable from "query cache with defaults."

### 5.9 `get_current_options()` gives you the effective configuration

`client.get_current_options()` returns a `RunOptions` dataclass
snapshot of what the client is currently using, including the values
ProxAI filled in for options you omitted. Pass `json=True` to get a
JSON-serialisable dict suitable for reproducibility manifests.

---

## 6. Examples

### 6.1 Minimal setup (module API, default client)

```python
import proxai as px

px.connect(
    cache_options=px.CacheOptions(cache_path="/tmp/proxai_cache"),
    logging_options=px.LoggingOptions(stdout=True),
)
px.set_model(provider_model=("openai", "gpt-4o"))
rec = px.generate(prompt="What is the capital of France?")
print(rec.result.output_text)
```

### 6.2 Isolated client per experiment

Two independent clients with their own caches and logs. Useful in
notebooks where you want to compare configurations without them
stepping on each other. Note that behavioural knobs like
`feature_mapping_strategy` now live inside `ProviderCallOptions`.

```python
exp_a = px.Client(
    experiment_path="exp_a",
    cache_options=px.CacheOptions(cache_path="/data/exp_a/cache"),
    logging_options=px.LoggingOptions(logging_path="/data/exp_a/logs"),
    provider_call_options=px.ProviderCallOptions(
        feature_mapping_strategy=px.FeatureMappingStrategy.STRICT,
    ),
)
exp_b = px.Client(
    experiment_path="exp_b",
    cache_options=px.CacheOptions(cache_path="/data/exp_b/cache"),
    logging_options=px.LoggingOptions(logging_path="/data/exp_b/logs"),
)

exp_a.set_model(provider_model=("openai", "gpt-4o"))
exp_b.set_model(provider_model=("anthropic", "claude-3-5-sonnet"))
```

### 6.3 Per-call fallback chain

```python
rec = px.generate(
    prompt="Explain quantum entanglement.",
    provider_model=("openai", "gpt-4"),
    connection_options=px.ConnectionOptions(
        # Tuples and ProviderModelType entries are both accepted.
        fallback_models=[
            ("anthropic", "claude-3-5-sonnet"),
            px.ProviderModelType(
                provider="google",
                model="gemini-2.5-pro",
                provider_model_identifier="gemini-2.5-pro",
            ),
        ],
    ),
)
if rec.connection.failed_fallback_models:
    print("Primary failed, recovered via:", rec.query.provider_model)
```

### 6.4 Refreshing a cached answer

```python
# Do the provider call, ignore the cached value on read, but update the
# cache so future calls get the refreshed answer.
rec = client.generate(
    prompt="Summarise the latest earnings report.",
    connection_options=px.ConnectionOptions(override_cache_value=True),
)
assert rec.connection.result_source == px.ResultSource.PROVIDER
```

### 6.5 Suppressing errors for a single call

```python
# The client default raises on failure; override for one diagnostic call.
rec = client.generate(
    prompt="Ping",
    connection_options=px.ConnectionOptions(suppress_provider_errors=True),
)
if rec.result.status == px.ResultStatusType.FAILED:
    print("Provider said:", rec.result.error)
```

### 6.6 Diversity via `unique_response_limit`

```python
# Collect 3 different responses to the same prompt before caching
# kicks in. The first 3 calls go to the provider and are all stored;
# subsequent calls serve one of the stored responses.
client = px.Client(
    cache_options=px.CacheOptions(
        cache_path="/tmp/diversity_cache",
        unique_response_limit=3,
    ),
)
```

### 6.7 Health check in a constrained environment

Probing options are set at client construction — `check_health()`
itself only takes `verbose`.

```python
# Jupyter on macOS: disable multiprocessing to avoid spawn crashes.
client = px.Client(
    model_probe_options=px.ModelProbeOptions(
        allow_multiprocessing=False,
        timeout=15,
    ),
)
status = client.models.check_health(verbose=False)
print(len(status.working_models), "working;",
      len(status.failed_models), "failed")
```

### 6.8 Opting out of ProxDash when the env var is set

```python
# Your CI sets PROXDASH_API_KEY globally. For this run, you do not
# want to report anything.
client = px.Client(
    proxdash_options=px.ProxDashOptions(disable_proxdash=True),
)
```

### 6.9 Snapshotting what the client is actually using

```python
# Reproducibility manifest for an experiment run.
import json
with open("run_config.json", "w") as f:
    json.dump(client.get_current_options(json=True), f, indent=2)
```

### 6.10 Attaching raw provider responses for local debugging

```python
# Keeps the provider SDK's raw response on CallRecord.debug.
# Not serialised to cache or ProxDash — local-only.
client = px.Client(
    debug_options=px.DebugOptions(keep_raw_provider_response=True),
)
rec = client.generate(prompt="Hello")
print(type(rec.debug.raw_provider_response))
```
