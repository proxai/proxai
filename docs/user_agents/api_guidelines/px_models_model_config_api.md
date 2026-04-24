# `px.models.model_config` API

Source of truth: `ModelConfigConnector` in `src/proxai/client.py`,
`ModelConfigs` in `src/proxai/connectors/model_configs.py`, and the
`ModelRegistry` / `ProviderModelConfig` / `ProviderModelType`
dataclasses in `src/proxai/types.py`. If this document disagrees with
those files, the files win — update this document.

This is the definitive reference for the registry-mutation surface —
every method on `px.models.model_config`, what it writes into the
client's in-process model registry, when it raises, and how the two
handles (`px.models.model_config` and `client.models.model_config`)
stay isolated. Read this when you need to register a custom model,
override the default fallback priority list, swap the registry in
bulk from JSON, or export the current registry for inspection.

See also: `px_models_api.md` (the read-only discovery surface —
`list_models`, `get_model`, health probes), `px_client_api.md` (where
`ProxAIClient` itself is constructed and hosts the underlying
`model_configs_instance`), and `call_record.md` (the
`ProviderModelType` shape callers pass into these methods and get
back from the discovery surface).

---

## 1. `px.models.model_config` structure (current)

```
px.models.model_config                               # same on client.models.model_config
│
│   # Registry mutation — write to the client's in-process model registry
├── .register_provider_model_config(cfg)             → None
├── .unregister_model(provider_model)                → None
├── .unregister_all_models()                         → None
├── .override_default_model_priority_list(models)    → None
│
│   # Bulk replace / export
├── .load_model_registry_from_json_string(json_str)  → None
├── .export_to_json(file_path)                       → None
│
│   # Inspection
└── .get_default_model_priority_list()               → list[ProviderModelType]
```

Every method is a thin forwarder onto the client's
`model_configs_instance`. The module-level `px.models.model_config`
writes into the hidden default client; `client.models.model_config`
writes into that specific instance. The two are fully isolated — see
§5.

### 1.1 Where `model_config` lives on the API

```
px.models                                            # the discovery surface
├── .list_models(...) / .get_model(...) / ...        # read-only queries
├── .get_default_model_list()                        # alias (see §4)
└── .model_config                                    # registry mutation (this doc)
    ├── .register_provider_model_config(...)
    ├── .unregister_model(...)
    ├── ...
```

`px.models` and `px.models.model_config` share the same underlying
registry — a mutation through `model_config` is immediately visible
through `list_models` / `get_model` / `list_working_models` on the
same client.

### 1.2 Building a `ProviderModelConfig`

Three of the seven methods take or return dataclasses from
`proxai.types`. The construction types are NOT re-exported at the
`px.*` level — import them directly:

```python
import proxai as px
import proxai.types as types
```

The minimum shape of a `ProviderModelConfig`:

```
ProviderModelConfig
├── provider_model: ProviderModelType
│   ├── provider: str                                 # "openai", "anthropic", ...
│   ├── model: str                                    # "gpt-4o", ...
│   └── provider_model_identifier: str                # SDK-facing model id
├── pricing: ProviderModelPricingType
│   ├── input_token_cost: int | None                  # nano-USD per token
│   └── output_token_cost: int | None
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

A fully-constructed example lives in §3.1 below. For the detailed
semantics of each `features.*` cell (SUPPORTED / BEST_EFFORT /
NOT_SUPPORTED), see `provider_feature_support_summary.md`.

---

## 2. Registry mutation methods

### 2.1 `register_provider_model_config()`

Adds a new `(provider, model)` entry to the client's registry. The
provider slot is created on demand — registering
`("my_provider", "my_model")` for the first time creates the
`my_provider` bucket.

```python
px.models.model_config.register_provider_model_config(
    provider_model_config: types.ProviderModelConfig,
) → None
```

```python
import proxai as px
import proxai.types as types

S = types.FeatureSupportType.SUPPORTED
NS = types.FeatureSupportType.NOT_SUPPORTED

config = types.ProviderModelConfig(
    provider_model=types.ProviderModelType(
        provider="openai",
        model="gpt-4o-custom",
        provider_model_identifier="gpt-4o-2024-08-06"),
    pricing=types.ProviderModelPricingType(
        input_token_cost=2_500,     # nano-USD per token
        output_token_cost=10_000),
    features=types.FeatureConfigType(
        prompt=S, messages=S, system_prompt=S,
        parameters=types.ParameterConfigType(
            temperature=S, max_tokens=S, stop=S, n=NS, thinking=NS),
        tools=types.ToolConfigType(web_search=NS),
        input_format=types.InputFormatConfigType(
            text=S, image=S, document=NS, audio=NS, video=NS,
            json=NS, pydantic=NS),
        output_format=types.OutputFormatConfigType(
            text=S, json=S, pydantic=S, image=NS, audio=NS,
            video=NS, multi_modal=NS),
    ),
    metadata=types.ProviderModelMetadataType(
        is_recommended=False,
        model_size_tags=[types.ModelSizeType.LARGE]),
)
px.models.model_config.register_provider_model_config(config)

# Immediately visible through the discovery surface on the same client.
print(px.models.get_model("openai", "gpt-4o-custom"))
```

Raises `ValueError` if a model with the same `(provider, model)` key
is already registered — re-registering an existing model is refused,
not overwritten. Call `unregister_model` first if you intend to
replace an entry.

### 2.2 `unregister_model()`

Removes one `(provider, model)` entry from the registry.

```python
px.models.model_config.unregister_model(
    provider_model: types.ProviderModelType,
) → None
```

```python
pm = px.models.get_model("openai", "gpt-4o-custom")
px.models.model_config.unregister_model(pm)
```

The passed `ProviderModelType` must match the registered config
exactly — including `provider_model_identifier`. Passing a
`ProviderModelType` whose identifier differs from the registered one
raises `ValueError`, even if the `(provider, model)` pair matches.

Raises `ValueError` if the provider is not registered, the model is
not registered under that provider, or the `provider_model_identifier`
on the passed value disagrees with the stored config.

Note: `unregister_model` does NOT remove the unregistered model from
the default priority list. If the model you just removed was part of
the default priority list, the subsequent priority-list validation
will still accept it (since the priority list is re-validated only on
`load_model_registry_from_json_string` and on
`override_default_model_priority_list`), but `get_default_provider_model()`
will fail over to the next entry when it cannot resolve the model
through the registry. Call `override_default_model_priority_list([...])`
after to keep the two in sync.

### 2.3 `unregister_all_models()`

Clears every registered model AND the default priority list in one
call. The registry's `metadata` block is preserved.

```python
px.models.model_config.unregister_all_models() → None
```

```python
px.models.model_config.unregister_all_models()
assert px.models.list_models(recommended_only=False) == []
```

Use this as the first step when you want to replace the bundled
registry with a custom set — see the pattern in §4.3.

### 2.4 `override_default_model_priority_list()`

Replaces the ordered fallback list that `px.generate()` walks when no
explicit model is set via `px.set_model()`. Every entry must already
be registered.

```python
px.models.model_config.override_default_model_priority_list(
    default_model_priority_list: list[types.ProviderModelType],
) → None
```

```python
# Order matters — first entry is tried first.
px.models.model_config.override_default_model_priority_list([
    px.models.get_model("openai", "gpt-4o"),
    px.models.get_model("anthropic", "claude-3-5-sonnet"),
    px.models.get_model("google", "gemini-2.5-pro"),
])

# Now px.generate() with no explicit model will try in this order.
rec = px.generate_text("Hello")
```

Raises `ValueError` on the first entry whose provider or model is not
in the registry. The entire call is rejected — the priority list is
either replaced whole or not at all. See
`px_client_api.md` §5.5 for how the priority list feeds the lazy
default-model selection.

---

## 3. Bulk replace and export

### 3.1 `load_model_registry_from_json_string()`

Replaces the ENTIRE registry with one deserialized from a JSON
string. Use this to install a registry published elsewhere (your
company's curated registry, a ProxDash snapshot, a pinned file
checked into a repo).

```python
px.models.model_config.load_model_registry_from_json_string(
    json_string: str,
) → None
```

```python
with open("/path/to/registry.json") as f:
    px.models.model_config.load_model_registry_from_json_string(f.read())
```

The payload is validated before the swap is applied. If validation
fails, the method raises `ValueError` with the message prefix
`Failed to load model registry:` and the original registry is left
untouched. Validation covers:

- **Schema shape** — JSON must decode into a `ModelRegistry`.
- **Per-model invariants** — `provider_model.provider` must match the
  outer provider key; `provider_model.model` must match the outer
  model key; `pricing.input_token_cost` and `output_token_cost` must
  be non-negative when set.
- **Priority-list consistency** — every entry in
  `default_model_priority_list` must resolve in `provider_model_configs`.
- **`min_proxai_version`** — if the registry metadata declares a
  minimum proxai version, the running version must satisfy it.
  This gate is skipped for registries marked as `BUILT_IN`; all other
  origins (including `PROXDASH` and a freshly loaded JSON) go through
  the check.

Note: this replaces the registry, it does not merge. Any custom
models you registered via `register_provider_model_config` before the
load are dropped. To extend a loaded registry, call
`load_model_registry_from_json_string` first, then
`register_provider_model_config` for each addition.

### 3.2 `export_to_json()`

Writes the current registry to a file as pretty-printed JSON (keys
recursively sorted) so the output is stable across runs and friendly
to version control.

```python
px.models.model_config.export_to_json(
    file_path: str,
) → None
```

```python
px.models.model_config.export_to_json("/tmp/my_registry.json")

# Round-trip on the same or a fresh client.
with open("/tmp/my_registry.json") as f:
    px.models.model_config.load_model_registry_from_json_string(f.read())
```

Any existing file at `file_path` is overwritten without warning.
`metadata`, `default_model_priority_list`, and `provider_model_configs`
are emitted in that order; `provider_model_configs` is sorted
alphabetically by provider name and then by model name.

---

## 4. Inspection

### 4.1 `get_default_model_priority_list()`

Returns the ordered list of models used by the lazy default-model
fallback when no explicit `px.set_model()` call has been made.

```python
px.models.model_config.get_default_model_priority_list()
    → list[types.ProviderModelType]
```

```python
priority = px.models.model_config.get_default_model_priority_list()
print("First fallback:", priority[0])

# Inspect without mutating — make a copy if you want to rearrange.
reordered = [priority[1], priority[0], *priority[2:]]
px.models.model_config.override_default_model_priority_list(reordered)
```

The returned list is the live registry list (not a copy). Do not
mutate it in place — call `override_default_model_priority_list()` to
publish a changed list. If you do mutate the returned list directly
you'll bypass the validation pass and may leave the registry in a
state that `px.generate()` cannot resolve.

This is the same list exposed by `px.models.get_default_model_list()`
in `px_models_api.md` §2.6 — both handles are aliases for the same
underlying method. The older `get_default_model_list()` stays for
backward compatibility; new code should prefer the
`model_config.get_default_model_priority_list()` name which matches
the surface that writes the list.

---

## 5. Common patterns

### 5.1 Register one custom model on top of the bundled registry

```python
import proxai as px
import proxai.types as types

# Keep the bundled registry. Add one private model.
S = types.FeatureSupportType.SUPPORTED
NS = types.FeatureSupportType.NOT_SUPPORTED

px.models.model_config.register_provider_model_config(
    types.ProviderModelConfig(
        provider_model=types.ProviderModelType(
            provider="openai", model="gpt-4o-internal",
            provider_model_identifier="gpt-4o-finetune-2026-04"),
        pricing=types.ProviderModelPricingType(
            input_token_cost=2_500, output_token_cost=10_000),
        features=types.FeatureConfigType(
            prompt=S, messages=S, system_prompt=S,
            parameters=types.ParameterConfigType(
                temperature=S, max_tokens=S, stop=S, n=NS, thinking=NS),
            tools=types.ToolConfigType(web_search=NS),
            input_format=types.InputFormatConfigType(
                text=S, image=NS, document=NS, audio=NS, video=NS,
                json=NS, pydantic=NS),
            output_format=types.OutputFormatConfigType(
                text=S, json=S, pydantic=S, image=NS, audio=NS,
                video=NS, multi_modal=NS)),
        metadata=types.ProviderModelMetadataType(
            is_recommended=False,
            model_size_tags=[types.ModelSizeType.LARGE])))

rec = px.generate_text(
    "Hello", provider_model=("openai", "gpt-4o-internal"))
```

### 5.2 Custom fallback priority list

```python
# Prefer a cheap model first, fall back to a flagship on failure.
px.models.model_config.override_default_model_priority_list([
    px.models.get_model("openai", "gpt-4o-mini"),
    px.models.get_model("anthropic", "claude-3-5-haiku"),
    px.models.get_model("openai", "gpt-4o"),
])

# With no explicit set_model, the first working model in this list
# is chosen lazily on first generate() call.
rec = px.generate_text("Hello")
```

See `px_client_api.md` §5.5 for how the priority list interacts with
`px.set_model()` (set explicitly → priority list is ignored; not set
→ the first working entry becomes the registered default).

### 5.3 Replace the entire registry with a private JSON

```python
# Step 1: clear the bundled registry.
px.models.model_config.unregister_all_models()

# Step 2: load your own.
with open("/opt/proxai/company_registry.json") as f:
    px.models.model_config.load_model_registry_from_json_string(f.read())

# Equivalent shorter form — load() replaces atomically, so the
# explicit unregister_all_models() is only necessary if you want to
# observe an empty registry between the two calls.
```

### 5.4 Capture the current registry for auditing

```python
# Bundled + any programmatic changes made so far.
px.models.model_config.export_to_json("/tmp/proxai_snapshot.json")

# Reload into a separate client later.
reader = px.Client()
with open("/tmp/proxai_snapshot.json") as f:
    reader.models.model_config.load_model_registry_from_json_string(f.read())
```

### 5.5 Per-client isolation

```python
client_a = px.Client()
client_b = px.Client()

client_a.models.model_config.unregister_all_models()
client_a.models.model_config.register_provider_model_config(...)

# client_b still has the bundled registry.
assert len(client_b.models.list_models(recommended_only=False)) > 0
```

Each `ProxAIClient` owns its own `model_configs_instance`. Mutating
through one client's `model_config` connector never touches another
client's registry, and never touches the module-level
`px.models.model_config`. See `px_client_api.md` §5.1.

### 5.6 Scripted registry rebuild (test-harness pattern)

```python
# A fresh registry with exactly three mock models — common in tests.
px.models.model_config.unregister_all_models()
for provider, model in [("mock_a", "m1"), ("mock_b", "m2"), ("mock_c", "m3")]:
    px.models.model_config.register_provider_model_config(
        _make_mock_config(provider, model))

px.models.model_config.override_default_model_priority_list([
    px.models.get_model("mock_a", "m1"),
    px.models.get_model("mock_b", "m2"),
])
```

The equivalent helpers in the repo's own tests live in
`tests/conftest.py` and `tests/test_client.py::_register_mock_providers`.

---

## 6. Errors

| Method | Trigger | Error |
|---|---|---|
| `register_provider_model_config()` | `(provider, model)` already registered | `ValueError` — `"Model X already registered for provider Y. Please use a different model or delete the existing model."` |
| `unregister_model()` | Provider not registered | `ValueError` — `"Provider X not registered."` |
| `unregister_model()` | Model not registered under that provider | `ValueError` — `"Model X not registered for provider Y."` |
| `unregister_model()` | `provider_model_identifier` on the passed value differs from the stored config | `ValueError` — `"Provider model identifier mismatch: …"` |
| `override_default_model_priority_list()` | Any entry references an unregistered provider | `ValueError` — `"Provider X not registered."` |
| `override_default_model_priority_list()` | Any entry references an unregistered model | `ValueError` — `"Model X not registered for provider Y."` |
| `load_model_registry_from_json_string()` | Any validation failure (schema, per-model invariants, priority-list resolution, or min-proxai-version gate) | `ValueError` prefixed with `"Failed to load model registry: …"` |
| `export_to_json()` | Filesystem-level failure (e.g. parent directory missing, permission denied) | `OSError` from Python's `open(..., 'w')` |

These are all synchronous exceptions raised directly by the connector
method; they are NOT routed through `suppress_provider_errors`, which
only affects calls into provider SDKs.

`load_model_registry_from_json_string` is atomic — if any of the
checks above fail, the original registry is preserved. Partial
loads never happen.

---

## 7. Module-level vs client instance

```python
import proxai as px

# Module-level — writes to the hidden default client.
px.models.model_config.register_provider_model_config(cfg)

# Per-instance — fully isolated.
client = px.Client()
client.models.model_config.register_provider_model_config(other_cfg)
```

The two handles share nothing: separate `ProxAIClient` instances have
separate `model_configs_instance` objects, and each `model_config`
connector holds a getter back to its own client. A mutation through
one is never visible through the other. Use
`client.models.model_config.<func>` whenever you want registry
changes scoped to a single client lifetime — for example, inside a
test that should not leak state into the next test, or a job that
wants a private curated registry while other jobs in the same
process see the defaults. See `px_client_api.md` §5.1 for the
isolation guarantees between clients.
