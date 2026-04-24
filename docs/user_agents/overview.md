# User Agents — Overview

Source of truth: `src/proxai/proxai.py` (the `px.*` module façade),
`src/proxai/client.py` (the `ProxAIClient`, `ModelConnector`, and
`FileConnector` implementations that every call delegates to), and
`src/proxai/types.py` (the option, parameter, and record dataclasses).
If this document disagrees with those files, the files win — update
this document.

This is the landing page for agents (and humans) using the ProxAI
library in their own code. It shows the complete `import proxai as
px` surface, maps common tasks to the doc that answers them, and
states the library-wide conventions that every other doc assumes.
Read this before touching any other `user_agents/` doc — the
decision tree in §2 will save you a wrong turn, and the global
rules in §3 are assumed throughout.

ProxAI is a lightweight abstraction over a dozen foundational LLM
providers (OpenAI, Anthropic, Google, Mistral, Cohere, DeepSeek,
Grok/xAI, HuggingFace, Databricks, …). One Python API,
`import proxai as px`, calls any of them — with uniform parameters,
uniform return records, optional caching, model discovery, fallback
chains, and multi-modal file handling. The library is on PyPI as
`proxai`.

See also: [`api_guidelines/px_generate_api.md`](./api_guidelines/px_generate_api.md)
(the call surface that does the work),
[`api_guidelines/px_client_api.md`](./api_guidelines/px_client_api.md)
(client construction and options), and
[`troubleshooting.md`](./troubleshooting.md) (symptom-first error
lookup). For contributors editing ProxAI itself, cross over to
[`../developer_agents/overview.md`](../developer_agents/overview.md).

---

## 1. `px.*` surface (current)

```
proxai                                              # import proxai as px
│
│   # Configuration — one hidden default client per process
├── px.connect(...)                                 → None
├── px.Client(...)                                  → ProxAIClient   (separate instance)
├── px.set_model(...)                               → None
├── px.get_current_options(json=False)              → RunOptions | dict
├── px.reset_state()                                → None
│
│   # Generation — seven functions, one CallRecord shape
├── px.generate(...)                                → CallRecord
├── px.generate_text(...)                           → str
├── px.generate_json(...)                           → dict
├── px.generate_pydantic(..., output_format=Model)  → BaseModel
├── px.generate_image(...)                          → MessageContent
├── px.generate_audio(...)                          → MessageContent
├── px.generate_video(...)                          → MessageContent
│
│   # Discovery — declared vs. working (health-probed)
├── px.models.list_models(...)                      → list[ProviderModelType]
├── px.models.list_providers(...)                   → list[str]
├── px.models.list_provider_models(...)             → list[ProviderModelType]
├── px.models.get_model(...)                        → ProviderModelType
├── px.models.get_model_config(...)                 → ProviderModelConfig
├── px.models.get_default_model_list()              → list[ProviderModelType]
├── px.models.list_working_models(...)              → list[ProviderModelType] | ModelStatus
├── px.models.list_working_providers(...)           → list[str]
├── px.models.list_working_provider_models(...)     → list[ProviderModelType] | ModelStatus
├── px.models.get_working_model(...)                → ProviderModelType
├── px.models.check_health(verbose=True)            → ModelStatus
│
│   # Registry mutation — register / swap / export models
├── px.models.model_config.register_provider_model_config(cfg)  → None
├── px.models.model_config.unregister_model(...)               → None
├── px.models.model_config.unregister_all_models()             → None
├── px.models.model_config.override_default_model_priority_list(...) → None
├── px.models.model_config.load_model_registry_from_json_string(...) → None
├── px.models.model_config.export_to_json(file_path)           → None
├── px.models.model_config.get_default_model_priority_list()   → list[ProviderModelType]
│
│   # Files — per-provider upload / list / remove / download
├── px.files.upload(...)                            → MessageContent  (mutated)
├── px.files.download(...)                          → MessageContent  (mutated)
├── px.files.list(...)                              → list[MessageContent]
├── px.files.remove(...)                            → MessageContent  (mutated)
├── px.files.is_upload_supported(...)               → bool
├── px.files.is_download_supported(...)             → bool
│
│   # Conversations — build-then-generate
├── px.Chat(system_prompt=None, messages=None)      → Chat
├── px.Message(role, content)                       → Message
│
│   # Re-exported types (for annotations and construction)
├── px.CacheOptions / LoggingOptions / ProxDashOptions
├── px.ProviderCallOptions / ModelProbeOptions / DebugOptions
├── px.FeatureMappingStrategy                       # BEST_EFFORT | STRICT
├── px.ConnectionOptions                            # per-call overrides
├── px.ParameterType / ThinkingType / Tools
├── px.OutputFormat / OutputFormatType
├── px.ProviderModelType
└── px.MessageRoleType / ContentType
```

All `px.*` functions delegate to a **hidden default client** created
lazily on first access. A separately-constructed `client = px.Client(...)`
is a fully isolated instance — see §3.4 and
[`px_client_api.md`](./api_guidelines/px_client_api.md) §5.1.

---

## 2. Decision tree

Find your task on the left, read the doc on the right.

```
I want to …                                         → Read
│
│   # Making calls
├── generate text / JSON / pydantic / image / ...   →  px_generate_api.md
├── build a multi-turn conversation                 →  px_chat_api.md
├── attach an image / PDF / audio to a request      →  px_files_api.md §3
├── understand what came back from a call           →  call_record.md
│
│   # Picking a model
├── list / filter models by capability              →  px_models_api.md §2
├── verify a model actually works (health probe)    →  px_models_api.md §3
├── register / unregister / swap models in the      →  px_models_model_config_api.md
│   registry, override the fallback priority list
├── see which providers support which features      →  provider_feature_support_summary.md
│
│   # Configuring the client
├── initialize or tune the client                   →  px_client_api.md
├── set cache paths / behaviors                     →  cache_behaviors.md
├── connect to ProxDash for observability           →  px_client_api.md §2.4
│
│   # Managing files
├── upload once, reuse across calls                 →  px_files_api.md §2–§3
├── list or delete previously-uploaded files        →  px_files_api.md §2
│
│   # Control flow / resilience
├── try multiple models with automatic fallback     →  px_generate_api.md §5.8
├── bypass or force-refresh the cache               →  cache_behaviors.md §5
├── handle provider errors without exceptions       →  px_client_api.md §2.5.2
│
│   # Debugging
├── read the raw provider SDK response              →  raw_provider_response.md
├── something broke and I want a symptom table      →  troubleshooting.md
└── see why a call was / wasn't cached              →  cache_behaviors.md §6
```

A task that isn't on this tree is probably covered by
[`call_record.md`](./api_guidelines/call_record.md) (every field on
the returned record) or
[`px_client_api.md`](./api_guidelines/px_client_api.md) (every knob
on the client). If it still isn't clear, open an issue at
<https://github.com/proxai/proxai/issues>.

---

## 3. Global rules

Conventions every other `user_agents/` doc assumes.

### 3.1 Import as `px`

Every code sample in these docs uses `import proxai as px`. The
module-level names (`px.generate`, `px.connect`, `px.models`,
`px.files`, `px.Chat`, `px.CacheOptions`, …) all come from this
alias. No other import path is documented; binding the module under
a different name may work but will not match the docs.

### 3.2 `px.models.*` is the runtime source of truth for capabilities

Provider capabilities — which models exist, what they support, which
endpoints they expose, how each feature is classified — live in the
`px.models` API at runtime and in the per-provider JSON under
`src/proxai/connectors/model_configs_data/`. Static tables like
[`provider_feature_support_summary.md`](./api_guidelines/provider_feature_support_summary.md)
are snapshots that age with the code. When they disagree, trust the
API:

```python
import proxai as px

# Declared capability — no network call
cfg = px.models.get_model_config("openai", "gpt-4o")
print(cfg.features.output_format.json)      # SUPPORTED | BEST_EFFORT | NOT_SUPPORTED

# Verified — runs a probe, result is cached
working = px.models.list_working_models(feature_tags=["thinking"])
```

Never hard-code "provider X supports feature Y" in long-lived code.
Ask `px.models` at runtime or accept that the assumption will rot.

### 3.3 Costs are integer nano-USD

`CallRecord.result.usage.estimated_cost` is an integer in nano-USD
(1 USD = 1,000,000,000 nano-USD). Pricing in
`ProviderModelPricingType` (`input_token_cost`, `output_token_cost`)
is the same unit — nano-USD per token. To display USD, divide by
`1_000_000_000`:

```python
rec = px.generate(prompt="Hello")
cost_usd = rec.result.usage.estimated_cost / 1_000_000_000
```

This convention exists so arithmetic on costs is exact — no floating
point drift in aggregates. See
[`call_record.md`](./api_guidelines/call_record.md) §2.11 for the
full contract.

### 3.4 Two calling styles, fully isolated

`px.connect(...)` configures a **hidden global default client** that
`px.generate`, `px.set_model`, `px.models`, `px.files`, and
`px.get_current_options` all read. `client = px.Client(...)`
constructs a **new, separate instance** that lives alongside the
default — it never replaces it.

```python
import proxai as px

# Style 1: module-level (default client)
px.connect(cache_options=px.CacheOptions(cache_path="/tmp/cache"))
px.generate_text(prompt="Hello")           # uses default client

# Style 2: explicit instance (isolated)
client = px.Client(cache_options=px.CacheOptions(cache_path="/tmp/exp_a"))
client.generate_text(prompt="Hello")       # uses `client`, NOT default
```

Two styles → two caches, two log files, two ProxDash connections.
Never assume state set through one is visible to the other. See
[`px_client_api.md`](./api_guidelines/px_client_api.md) §5.1 for the
isolation rules and §5.2 for default-client lifecycle.

### 3.5 Errors split into two classes

ProxAI distinguishes **programmer errors** from **provider errors**:

- **Programmer errors** — bad arguments, conflicting options,
  missing API keys, unsupported endpoints. Raised synchronously as
  `ValueError` (sometimes `KeyError` / `TypeError`) *before* any
  network call. Never routed through `suppress_provider_errors`.
  Examples: passing both `prompt` and `messages`, setting
  `override_cache_value=True` without a configured cache, asking
  for a model whose API key is not in your environment.
- **Provider errors** — rate limits, HTTP 5xx, timeouts, JSON
  parsing failures, validation errors raised by the provider SDK.
  By default the exception propagates out of `.generate()`. Setting
  `provider_call_options.suppress_provider_errors=True` (or the
  per-call override on `ConnectionOptions`) turns these into a
  `CallRecord` with `result.status = FAILED` and a stringified
  `result.error`.

When in doubt, the per-API **Errors** tables list every exception a
caller can hit at that surface:
[`px_client_api.md`](./api_guidelines/px_client_api.md) §4,
[`px_generate_api.md`](./api_guidelines/px_generate_api.md) §6,
[`px_models_api.md`](./api_guidelines/px_models_api.md) §6,
[`px_files_api.md`](./api_guidelines/px_files_api.md) §6,
[`px_chat_api.md`](./api_guidelines/px_chat_api.md) §7,
[`cache_behaviors.md`](./api_guidelines/cache_behaviors.md) §10,
[`raw_provider_response.md`](./api_guidelines/raw_provider_response.md) §5.
For a symptom-first lookup when something has already gone wrong,
jump to [`troubleshooting.md`](./troubleshooting.md).

---

## 4. Minimal example

The shortest working program exercising the library — no config, no
model selection. ProxAI auto-selects a reachable model from the
default priority list (see
[`px_client_api.md`](./api_guidelines/px_client_api.md) §5.5) and
caches the choice for the rest of the process.

```python
import proxai as px

text = px.generate_text(prompt="What is the capital of France?")
print(text)
```

A slightly more realistic setup — explicit client config, pinned
model, cache enabled, per-call fallback:

```python
import proxai as px

px.connect(
    cache_options=px.CacheOptions(cache_path="/tmp/proxai_cache"),
    logging_options=px.LoggingOptions(stdout=True),
)
px.set_model(provider_model=("openai", "gpt-4o"))

rec = px.generate(
    prompt="Summarize this article in one sentence.",
    connection_options=px.ConnectionOptions(
        fallback_models=[("anthropic", "claude-3-5-sonnet")],
    ),
)
print(rec.result.output_text)
print("model that answered:", rec.query.provider_model)
print("cost (USD):", rec.result.usage.estimated_cost / 1_000_000_000)
```

---

## 5. Further reading

The per-topic deep dives in [`api_guidelines/`](./api_guidelines/)
cover every parameter, every field, and every edge case:

| Topic | Doc |
|---|---|
| Client construction, options, lifecycle | [`px_client_api.md`](./api_guidelines/px_client_api.md) |
| Generation — seven functions, full parameter surface | [`px_generate_api.md`](./api_guidelines/px_generate_api.md) |
| Model discovery and health probes | [`px_models_api.md`](./api_guidelines/px_models_api.md) |
| Model registry mutation — register / swap / export | [`px_models_model_config_api.md`](./api_guidelines/px_models_model_config_api.md) |
| File upload / list / remove / download | [`px_files_api.md`](./api_guidelines/px_files_api.md) |
| Conversations — `Chat`, `Message`, `MessageContent` | [`px_chat_api.md`](./api_guidelines/px_chat_api.md) |
| `CallRecord` shape — every field returned from a call | [`call_record.md`](./api_guidelines/call_record.md) |
| Cache behavior — options, bypass, replay, metadata | [`cache_behaviors.md`](./api_guidelines/cache_behaviors.md) |
| Provider capability cheat-sheet | [`provider_feature_support_summary.md`](./api_guidelines/provider_feature_support_summary.md) |
| Local-debug escape hatch for raw SDK responses | [`raw_provider_response.md`](./api_guidelines/raw_provider_response.md) |
| Symptom-first error lookup | [`troubleshooting.md`](./troubleshooting.md) |

Task-oriented workflows (setup, migrating an existing OpenAI /
Anthropic / Gemini codebase, production resilience, debugging) are
planned as **bundled Claude skills** — see
[`documentation_outline.md`](../documentation_outline.md) §3 for the
Layer B plan. Until those ship, the equivalent content is being
staged under [`recipes/`](./recipes/).

If you are editing ProxAI itself rather than calling it, cross over
to [`../developer_agents/overview.md`](../developer_agents/overview.md).
