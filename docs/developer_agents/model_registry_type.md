# `ModelRegistry` Type

Source of truth: `ModelRegistry` and its supporting types in
`src/proxai/types.py` (the dataclass), `ModelConfigs` in
`src/proxai/connectors/model_configs.py` (the StateControlled owner
that mutates and validates it), `encode_model_registry` /
`decode_model_registry` in `src/proxai/serializers/type_serializer.py`
(the JSON round-trip), and `ProxDashConnection.get_model_registry` in
`src/proxai/connections/proxdash.py` (the optional remote source). If
this document disagrees with those files, the files win — update this
document.

This is the definitive reference for the `ModelRegistry` dataclass —
its three fields, the supporting nested types, the lifecycle that
loads it (bundled JSON → optional ProxDash override → optional
user-supplied JSON), the validation invariants enforced at the load
boundary, the JSON serializer, and every consumer that reads off it.
Read this before changing any of those files, before adding a new
field to `ModelRegistry` or `ProviderModelConfig`, before bumping the
bundled `v1.x.x.json` schema, or before adding a new validation gate.

`ModelRegistry` is the **internal in-process schema** that the rest
of the library consumes. The user-facing mutation methods are
documented in `../user_agents/api_guidelines/px_models_model_config_api.md`;
that doc is the right reference for "how do I register a custom
model from my code." This doc is the right reference for "how does
the registry actually live inside the process, and what invariants
must hold."

See also: `../user_agents/api_guidelines/px_models_model_config_api.md`
(user-facing mutation surface — `register_provider_model_config`,
`override_default_model_priority_list`, `load_model_registry_from_json_string`,
etc.), `../user_agents/api_guidelines/call_record.md`
(`ProviderModelType` / `ProviderModelConfig` field reference for the
caller-facing types this doc references), `state_controller.md`
(the `StateControlled` machinery that wraps `ModelConfigs` and
propagates `ModelRegistry` changes through `ModelConfigsState`),
`adding_a_new_provider.md` (where to add a new provider's row to
the bundled `v1.3.0.json` registry, and the
`FeatureSupportType` taxonomy that fills `provider_model_configs`),
and `dependency_graph.md` (the layering — `types` → `model_configs`
→ everything else).

---

## 1. `ModelRegistry` structure (current)

```
ModelRegistry                                          (src/proxai/types.py)
│
├── metadata: ModelConfigsSchemaMetadataType | None
│   ├── version: str | None                            # e.g. "1.3.0"
│   ├── released_at: datetime | None
│   ├── min_proxai_version: str | None                 # PEP 440 specifier set
│   ├── config_origin: ConfigOriginType | None        # BUILT_IN | PROXDASH
│   └── release_notes: str | None
│
├── default_model_priority_list: list[ProviderModelType] | None
│   │   # Ordered fallback chain consumed by the lazy default-model
│   │   # selection in client.get_default_provider_model().
│   └── ProviderModelType
│       ├── provider: str                              # "openai", "claude", ...
│       ├── model: str                                 # "gpt-4o", ...
│       └── provider_model_identifier: str             # SDK-facing model id
│
└── provider_model_configs: dict[str, dict[str, ProviderModelConfig]] | None
    │   # Two-level mapping: provider name → model name → config.
    │   # Keys MUST equal config.provider_model.{provider, model};
    │   # _validate_provider_model_configs enforces this.
    └── ProviderModelConfig
        ├── provider_model: ProviderModelType          # see above
        ├── pricing: ProviderModelPricingType
        │   ├── input_token_cost: int | None           # nano-USD per token
        │   └── output_token_cost: int | None          # nano-USD per token
        ├── features: FeatureConfigType
        │   ├── prompt: FeatureSupportType | None      # SUPPORTED | BEST_EFFORT | NOT_SUPPORTED
        │   ├── messages: FeatureSupportType | None
        │   ├── system_prompt: FeatureSupportType | None
        │   ├── add_system_to_messages: bool | None
        │   ├── parameters: ParameterConfigType | None
        │   │   ├── temperature, max_tokens, stop, n, thinking
        │   ├── tools: ToolConfigType | None
        │   │   └── web_search
        │   ├── input_format: InputFormatConfigType | None
        │   │   ├── text, image, document, audio, video, json, pydantic
        │   └── output_format: OutputFormatConfigType | None
        │       ├── text, image, audio, video, json, pydantic
        └── metadata: ProviderModelMetadataType
            ├── is_recommended: bool | None            # filtered by `recommended_only=True`
            ├── model_size_tags: list[ModelSizeType] | None  # SMALL | MEDIUM | LARGE | LARGEST
            └── tags: list[str] | None                 # unstructured; not filtered on
```

Three fields. `metadata` describes *which* registry this is and
when it was published. `default_model_priority_list` is the ordered
fallback chain. `provider_model_configs` is the actual model
catalog — for each `(provider, model)` pair, ProxAI knows the price,
the per-feature support cells, and the discovery metadata.

Every nested type is a plain `@dataclass` in `types.py`. None of
them carry behavior — they are pure data shells. All behavior lives
on `ModelConfigs` (the StateControlled owner — see §4).

### 1.1 The three fields are all `Optional`

The dataclass declaration in `types.py` does not mark them
`Optional`, but in practice every reader treats `None` as a valid
state. The serializer's decoder leaves any field absent in the JSON
payload as `None`; `_validate_provider_model_configs` and
`_validate_default_model_priority_list` short-circuit on `None`;
`ProxDashConnection.get_model_registry` rejects a payload whose
`metadata is None or provider_model_configs is None`. The result:
the only registry instances the library actually operates on always
have `metadata` and `provider_model_configs` populated, and
`default_model_priority_list` either populated or empty. Treat
`None` as the "fresh decoder output before validation" state.

---

## 2. The three fields in detail

### 2.1 `metadata: ModelConfigsSchemaMetadataType`

Provenance and gating info. Five sub-fields, all optional:

- `version: str | None` — schema version. The bundled file is named
  `v{version}.json` (currently `v1.3.0.json`).
  `ModelConfigs.LOCAL_CONFIG_VERSION = "v1.3.0"` is the version
  loaded by default from
  `src/proxai/connectors/model_configs_data/`.
- `released_at: datetime | None` — ISO timestamp of publication.
  Encoded with `.isoformat()`, decoded with
  `datetime.fromisoformat()`. Informational only — no consumer
  branches on it.
- `min_proxai_version: str | None` — PEP 440 specifier set
  (e.g. `">=0.2.20"`). Compared against the running proxai version
  in `ModelConfigs._validate_min_proxai_version`. Skipped for
  `config_origin == BUILT_IN`; enforced for every other origin
  (see §5.3).
- `config_origin: ConfigOriginType | None` — the enum has two
  values, `BUILT_IN` and `PROXDASH`. The bundled
  `v1.3.0.json` declares `BUILT_IN`; ProxDash's `/models/configs`
  endpoint stamps `PROXDASH`. A user-supplied JSON loaded via
  `load_model_registry_from_json_string` carries whatever
  `config_origin` was written into the file — if it is `BUILT_IN`,
  the version gate is skipped, so honor the convention: only
  bundled files should carry `BUILT_IN`.
- `release_notes: str | None` — human-readable changelog snippet.
  Surfaces in cache files for debugging but no code branches on it.

### 2.2 `default_model_priority_list: list[ProviderModelType]`

Ordered fallback chain. Consumed in two places:

- `ModelConfigs.get_default_model_priority_list()` returns it as-is
  to `ProxAIClient.get_default_provider_model()`, which walks it on
  the first `generate_text` call when no explicit
  `px.set_model()` has been issued.
- `px.models.model_config.get_default_model_priority_list()` (the
  user-facing connector method) is a thin forwarder onto the same
  field.

Order matters — first entry is tried first. Every entry must
resolve in `provider_model_configs`; `_validate_default_model_priority_list`
enforces this on every load through `reload_from_registry`.

The list is the **live registry list, not a copy**. Mutating it in
place (e.g.,
`registry.default_model_priority_list.append(...)`) bypasses
validation. Always go through
`ModelConfigs.override_default_model_priority_list(...)`, which
re-runs the resolution check before swapping the list in.

### 2.3 `provider_model_configs: dict[str, dict[str, ProviderModelConfig]]`

The model catalog. Two-level mapping, keyed first by provider
(`"openai"`, `"claude"`, `"gemini"`, …) and then by model name
(`"gpt-4o"`, `"haiku-4.5"`, …). The leaf `ProviderModelConfig`
carries:

- `provider_model: ProviderModelType` — the same `(provider, model,
  provider_model_identifier)` triple stored in the leaf must match
  the outer dict keys. `_validate_provider_model_configs` enforces
  the key-match invariant; a mismatch raises with the specific path
  (`provider_model_configs['openai']['gpt-4o']: provider_model.provider
  is 'claude'`).
- `pricing: ProviderModelPricingType` — `input_token_cost` and
  `output_token_cost`, both nano-USD per token. Both must be
  non-negative when set; the validator rejects a registry that
  carries a negative cost. The integer nano-USD convention is
  authoritative — see `types.py:ProviderModelPricingType`.
- `features: FeatureConfigType` — the SUPPORTED / BEST_EFFORT /
  NOT_SUPPORTED matrix. `FeatureAdapter` reads this to pick the
  endpoint; `ResultAdapter` reads `output_format` to drive the
  TEXT → JSON / PYDANTIC conversion. See `feature_adapters_logic.md`
  for the full pipeline.
- `metadata: ProviderModelMetadataType` — `is_recommended` (filtered
  by the default `recommended_only=True` of
  `list_models`), `model_size_tags`, and `tags` (unstructured —
  never used for filtering, sometimes used for display strings such
  as `display::model_name=...`).

The mapping is a regular Python `dict` of regular Python `dict`s.
Inserts and deletes go through `register_provider_model_config` /
`unregister_model` on `ModelConfigs`; **direct mutation of these
nested dicts is unsupported**. The mutation methods enforce the
key-match invariant on insert; bypassing them allows the registry
to drift into a state that fails the next `reload_from_registry`.

---

## 3. Lifecycle — how a registry is loaded

There are three load paths. Every load goes through
`ModelConfigs.reload_from_registry` (or its `__init__`-time
equivalent). The three differ only in *where the `ModelRegistry`
came from*.

```
Process startup
  │
  ├── px.connect() → ProxAIClient.__init__
  │     └── self.model_configs_instance = ModelConfigs()
  │           ├── (no init_from_params) → BUILT_IN load (§3.1)
  │           └── self.model_registry = ... (from bundled JSON)
  │
  ├── ProxAIClient.__init__ → _maybe_fetch_model_registry_from_proxdash
  │     ├── skipped if RunType.TEST or proxdash DISABLED or already fetched
  │     └── proxdash_connection.get_model_registry()
  │           ├── HTTP GET /models/configs?proxaiVersion=...
  │           ├── decode_model_registry(response.data) → ModelRegistry
  │           └── reject if metadata None / provider_model_configs None
  │                       / len(provider_model_configs) < 2
  │     └── if not None: model_configs_instance.reload_from_registry(...)
  │           └── PROXDASH override replaces the BUILT_IN registry (§3.2)
  │
  └── (any time after connect) user code:
        ├── px.models.model_config.load_model_registry_from_json_string(s)
        │     └── decode_model_registry → reload_from_registry (§3.3)
        ├── px.models.model_config.register_provider_model_config(cfg)
        │     └── direct mutation of the live registry (§4.2)
        └── px.models.model_config.override_default_model_priority_list(...)
              └── live mutation of the priority field (§4.2)
```

### 3.1 `BUILT_IN` — bundled JSON at construction

`ModelConfigs.__init__` with no `init_from_params.model_registry`
calls `_load_model_registry_from_local_files`, which reads
`src/proxai/connectors/model_configs_data/{LOCAL_CONFIG_VERSION}.json`
via `importlib.resources.files(...)`. The current bundled version is
`v1.3.0.json`; bump
`LOCAL_CONFIG_VERSION` and ship a new JSON file alongside (and
update the `pyproject.toml` `include` list) when a new schema
version is released.

The `__init__` path is slightly different from
`reload_from_registry`: it copies `metadata` and
`default_model_priority_list` straight through but rebuilds
`provider_model_configs` model-by-model via
`register_provider_model_config`. This rebuild path runs the
"already registered" check, so a bundled file that names the same
`(provider, model)` pair twice at the JSON level fails fast at
load time.

The `BUILT_IN` registry is **trusted by construction** — the
`min_proxai_version` gate is intentionally skipped (the bundled
file ships in the same wheel as the running code, so the
specifier set is always satisfied tautologically). All other
validators still run.

### 3.2 `PROXDASH` — optional override on `px.connect()`

The constructor of `ProxAIClient` calls
`_maybe_fetch_model_registry_from_proxdash` after the proxdash
connection is up. The fetch happens at most once per client
lifetime, controlled by `model_configs_requested_from_proxdash`.

Skipped when:

- `run_type == RunType.TEST` — tests must not hit the network.
- `proxdash_connection is None`, or its status is `DISABLED` — the
  user opted out.
- The flag is already set (idempotent).

Otherwise, **the fetch attempts the request regardless of
proxdash auth status**: `/models/configs` is a public endpoint that
serves fresh model metadata to open-source users without an API
key, and to authenticated users with a key. `CONNECTED` sends the
`X-API-Key` header; everything else (including
`API_KEY_NOT_FOUND`) sends an unauthenticated request. Failures —
HTTP non-200, `success: false`, decode exception, or a payload that
looks degenerate (missing `metadata`, missing `provider_model_configs`,
or fewer than 2 providers) — are logged and the bundled `BUILT_IN`
registry is left in place. The fetch is best-effort; a network
outage never breaks `px.connect()`.

When the fetch succeeds, `reload_from_registry` is called and the
PROXDASH-origin registry replaces the BUILT_IN one. The
`min_proxai_version` gate runs (this is the typical motivation for
shipping a registry from ProxDash — to roll out new providers /
features that older proxai versions cannot support).

### 3.3 User-supplied JSON via `load_model_registry_from_json_string`

`px.models.model_config.load_model_registry_from_json_string(s)`
forwards onto `ModelConfigs.load_model_registry_from_json_string`,
which decodes via `decode_model_registry` and then runs
`reload_from_registry`. Whatever `config_origin` is encoded in the
payload is honored — if the payload declares `BUILT_IN`, the
version gate is skipped (so don't ship third-party registries with
that origin). Per `px_models_model_config_api.md` §3.1, the user-facing
behavior is atomic: validation failures preserve the original
registry; partial loads never happen.

The same path is used to install a snapshot exported via
`export_to_json` — the round-trip is lossless across the three
fields plus all nested types (see §6.2).

---

## 4. State integration

### 4.1 `ModelConfigsState` and `StateControlled`

The `ModelRegistry` instance lives on
`ModelConfigsState.model_registry`, the dataclass that
`ModelConfigs` exposes through the `StateControlled` machinery:

```python
# types.py
@dataclasses.dataclass
class ModelConfigsState(StateContainer):
  model_registry: ModelRegistry | None = None

# model_configs.py
class ModelConfigs(state_controller.StateControlled):
  _model_configs_state: types.ModelConfigsState | None
  LOCAL_CONFIG_VERSION = "v1.3.0"

  @property
  def model_registry(self) -> types.ModelRegistry:
    return self.get_property_value('model_registry')

  @model_registry.setter
  def model_registry(self, value: types.ModelRegistry):
    self.set_property_value('model_registry', value)
```

Reads and writes go through `get_property_value` /
`set_property_value`, which route into the `StateContainer`. That
means a `ModelRegistry` swap is observable to anyone holding a
reference to the `ModelConfigsState` and is properly captured in
the StateControlled snapshot. See `state_controller.md` for the
propagation contract.

`get_internal_state_property_name` returns
`'_model_configs_state'`; `get_internal_state_type` returns
`types.ModelConfigsState`. Both are required by the StateControlled
base.

### 4.2 The mutation surface

`ModelConfigs` exposes five mutation methods. All five operate on
the live `model_registry` field; none of them deep-copy the input.
Whether a method fully replaces the registry or just edits one
field of it matters for invariant tracking:

| Method | Effect on `ModelRegistry` |
|---|---|
| `register_provider_model_config(cfg)` | Mutates `provider_model_configs[provider][model]` in place. Raises `ValueError` if the `(provider, model)` pair is already registered. Does NOT re-validate the default priority list. |
| `unregister_model(provider_model)` | Deletes one leaf from `provider_model_configs`. Raises if the provider/model is not registered, or the `provider_model_identifier` mismatches. Does NOT prune the entry from `default_model_priority_list`. |
| `unregister_all_models()` | Replaces the registry with a new `ModelRegistry` whose `metadata` is preserved, `default_model_priority_list = []`, and `provider_model_configs = {}`. |
| `override_default_model_priority_list(list)` | Replaces the field in place after validating that every entry resolves. Atomic — entire list is rejected on the first unresolvable entry. |
| `reload_from_registry(registry)` | Full validation (per-model invariants + priority-list resolution + `min_proxai_version` for non-BUILT_IN), then full replace. The error wrapper "Failed to load model registry: …" is added at this boundary. |

The first three operate on **live nested dicts** without revalidating
the priority list. The intentional consequence is that
`unregister_model("openai", "gpt-4o")` may leave an orphan in
`default_model_priority_list` pointing at the now-missing model. The
priority list is checked again only on the next
`reload_from_registry` or `override_default_model_priority_list` call,
not eagerly. Caller-side guidance is in
`px_models_model_config_api.md` §2.2 — "Call
`override_default_model_priority_list([...])` after to keep the
two in sync."

### 4.3 Per-client isolation

Each `ProxAIClient` owns its own `model_configs_instance`. Two
clients in the same Python process hold two independent
`ModelRegistry` instances; mutating one is invisible to the
other. The module-level `px.models.model_config` writes into the
hidden default client only; `client.models.model_config` writes
into that specific instance. See
`px_models_model_config_api.md` §5 / §7 for the user-facing view of
this guarantee, and `state_controller.md` for the StateControlled
mechanism behind it.

---

## 5. Validation invariants

Every load through `reload_from_registry` runs three validators in
sequence, all defined as `@staticmethod` on `ModelConfigs`. Any
`ValueError` raised by any of them is re-raised wrapped with the
prefix `Failed to load model registry: `, so the caller sees one
clear failure at the load boundary instead of a deep deserialization
trace.

### 5.1 `_validate_provider_model_configs`

Per-leaf invariants, applied to every `ProviderModelConfig` in
`provider_model_configs`:

- `config.provider_model is not None` — a leaf without an embedded
  `ProviderModelType` is invalid.
- `config.provider_model.provider == outer_provider_key` — the
  outer dict key must match the inner provider name.
- `config.provider_model.model == outer_model_key` — same for the
  model key.
- `config.pricing.input_token_cost >= 0` and
  `config.pricing.output_token_cost >= 0` (when each is set).
  Negative pricing is rejected — the cost-estimation path in
  `provider_connector.py` assumes non-negative integers.

The error messages name the exact path
(`provider_model_configs['openai']['gpt-4o']: ...`) so a registry
with a mistyped provider key surfaces the offending row directly.

### 5.2 `_validate_default_model_priority_list`

Resolution check, applied to every entry in
`default_model_priority_list`:

- `entry.provider in provider_model_configs`
- `entry.model in provider_model_configs[entry.provider]`

Empty list short-circuits as valid. The validator does NOT compare
`provider_model_identifier` between the priority entry and the
registered config — that field is only checked in
`unregister_model` and `check_provider_model_identifier_type`.

### 5.3 `_validate_min_proxai_version`

PEP 440 specifier-set check using the `packaging` library. The
running proxai version (`importlib.metadata.version("proxai")`) is
parsed as a `Version` and tested against the registry's specifier
set. Two specific errors are differentiated:

- `InvalidSpecifier` — the registry shipped a malformed specifier
  string (bug in the registry author's release process).
- `InvalidVersion` — the running proxai version cannot be parsed
  as PEP 440 (bug in the proxai release process).

Both are wrapped by the outer "Failed to load model registry:"
prefix.

This validator is **skipped for `config_origin == BUILT_IN`**:

```python
# model_configs.py
if model_registry.metadata is not None:
  config_origin = model_registry.metadata.config_origin
  if config_origin != types.ConfigOriginType.BUILT_IN:
    self._validate_min_proxai_version(...)
```

The bundled JSON ships in the same wheel as the running code, so
the specifier set is satisfied tautologically. PROXDASH and
user-supplied JSONs go through the gate, which is the typical
motivation for setting it ("v1.4.0 needs proxai>=0.3.0 because it
references the new multi-modal output format").

### 5.4 The error wrapper

`reload_from_registry` re-raises any `ValueError` from the three
validators with the prefix `Failed to load model registry: `:

```python
try:
  ...
  self._validate_provider_model_configs(model_registry)
  self._validate_default_model_priority_list(model_registry)
except ValueError as e:
  raise ValueError(f'Failed to load model registry: {e}') from e
self.model_registry = model_registry
```

The `from e` chain preserves the original traceback. Note the
ordering: **the swap happens after every validator passes**. If
any validator raises, `self.model_registry` is left untouched —
the registry is atomic with respect to `reload_from_registry`. The
mutation methods in §4.2 do not benefit from this atomicity; they
edit live nested dicts.

---

## 6. JSON serialization

### 6.1 `encode_model_registry` / `decode_model_registry`

Defined in `src/proxai/serializers/type_serializer.py`. The encoder
emits a dict with up to three keys; each is omitted when the
corresponding field is `None`:

```python
def encode_model_registry(model_registry):
  record = {}
  if model_registry.metadata is not None:
    record['metadata'] = encode_model_configs_schema_metadata_type(...)
  if model_registry.default_model_priority_list is not None:
    record['default_model_priority_list'] = encode_default_model_priority_list(...)
  if model_registry.provider_model_configs is not None:
    record['provider_model_configs'] = encode_provider_model_configs_mapping_type(...)
  return record
```

The decoder is symmetric and lenient — missing keys decode to
`None`, never raise. This is intentional: a partial payload is
allowed at the deserializer level so the validator can produce
the actual error with a meaningful "Failed to load model
registry:" wrapper.

The nested encoders / decoders follow the same conventions:

- `ModelConfigsSchemaMetadataType.released_at` round-trips through
  `datetime.isoformat()` ↔ `datetime.fromisoformat()`.
- `ConfigOriginType` round-trips through its `.value` (string).
- `ProviderModelType.provider_model_identifier` is serialized as a
  raw string. Per the freeform comment in `types.py:60`, the dict
  variant of this field is supported; the encoder accepts dict
  values verbatim. Confirm against `encode_provider_model_type` if
  you are adding new fields to `ProviderModelType`.

### 6.2 Round-trip via `export_to_json` and
`load_model_registry_from_json_string`

`ModelConfigs.export_to_json(file_path)` calls
`encode_model_registry`, recursively sorts the
`provider_model_configs` mapping, enforces the top-level key order
`metadata` → `default_model_priority_list` → `provider_model_configs`,
and writes pretty-printed JSON with `indent=2` and a trailing
newline. The recursive sort ensures byte-stable output across
runs — the same in-memory registry produces the same file content,
so a registry checked into git gets clean diffs.

`ModelConfigs.load_model_registry_from_json_string(s)` is the
inverse: `json.loads(s)` → `decode_model_registry` →
`reload_from_registry`. The full validation pipeline runs.

Round-trip is lossless across the three top-level fields and all
nested dataclasses. There is no "raw" or "untyped" escape hatch —
every field that is decoded back is one declared on the dataclass.

### 6.3 Wire format with ProxDash

`/models/configs` returns a JSON envelope of the form:

```json
{
  "success": true,
  "data": { /* ModelRegistry payload — same shape as encode_model_registry */ }
}
```

`ProxDashConnection.get_model_registry` unwraps `response.data` and
calls `decode_model_registry` directly. The envelope-level
`success: false` short-circuits to an `INFO` log and `None`; an
exception out of `decode_model_registry` is caught, logged with the
full traceback, and also returns `None` — the bundled BUILT_IN
registry stays in place for the rest of the client's lifetime.

A "looks degenerate" payload guard runs after a successful decode:
`metadata is None or provider_model_configs is None or
len(provider_model_configs) < 2`. This catches a misbehaving
endpoint that returned a one-provider registry that would render
the client effectively single-provider.

---

## 7. Consumers

### 7.1 `ModelConfigs` — the owner

Every other consumer reads the registry through the
`model_configs_instance` on a `ProxAIClient`, not directly. The
read methods on `ModelConfigs` form a thin facade:

| Method | Reads from |
|---|---|
| `get_provider_model(identifier)` | `provider_model_configs[provider][model].provider_model` |
| `get_provider_model_config(identifier)` | `provider_model_configs[provider][model]` (full leaf) |
| `get_all_models(provider, model_size, recommended_only)` | iterates `provider_model_configs`, filters by metadata |
| `get_default_model_priority_list()` | returns `default_model_priority_list` directly |
| `check_provider_model_identifier_type(...)` | walks `provider_model_configs` to verify a `(provider, model)` exists and the `provider_model_identifier` matches |

These are the workhorse read paths used everywhere in the codebase.

### 7.2 `ProxDashConnection.get_model_registry`

The only place where a `ModelRegistry` instance is constructed
*outside* `ModelConfigs.__init__` and `decode_model_registry` calls
under `ModelConfigs`. Returns `types.ModelRegistry | None`; the
caller in `client.py:_maybe_fetch_model_registry_from_proxdash`
treats `None` as "stay on the bundled BUILT_IN registry."

The function is **read-only with respect to local state** — it
never writes back into `ModelConfigs`. The replacement happens in
the caller via `model_configs_instance.reload_from_registry(...)`.

### 7.3 `AvailableModels` — indirect consumer

`AvailableModels` (in `connections/available_models.py`) is the
discovery surface behind `px.models.list_working_models` etc. It
holds a `model_configs_instance` and reads through it
(`self.model_configs_instance.get_provider_model(...)`); it never
touches the `model_registry` field directly.

The name collision is worth noting: `available_models.py` imports
`proxai.connectors.model_registry as model_registry` — that module
is the **helper module** in `connectors/model_registry.py` that
exposes `_MODEL_CONNECTOR_MAP` and `get_model_connector(provider)`,
not the `ModelRegistry` *type*. The connector-factory map and the
`ModelRegistry` dataclass are unrelated artifacts that happen to
share a stem name. Cross-reference: see §8 ("Common pitfalls"
below) and `dependency_graph.md` for the layering.

### 7.4 `ProviderConnector.generate` — pricing and feature lookup

The request pipeline reads `ProviderModelConfig.features` (through
`FeatureAdapter` and `ResultAdapter` — see
`feature_adapters_logic.md`) and `ProviderModelConfig.pricing`
(for `ResultRecord.usage.estimated_cost` computation). Both reads
go through `model_configs_instance.get_provider_model_config(...)`;
the connector executor never touches the registry directly.

This is why a `register_provider_model_config(...)` call is
visible to `px.generate_text` immediately on the same client — the
adapters and the cost estimator both refresh from the live
`ModelConfigs` on every call.

### 7.5 `px.models.model_config.*` — the user-facing connector

`ModelConfigConnector` in `client.py` is a thin wrapper around
`ModelConfigs` that exposes the mutation surface to user code. Each
method forwards onto the same-named method on the client's
`model_configs_instance`. Documented in
`../user_agents/api_guidelines/px_models_model_config_api.md`.

---

## 8. Common pitfalls

### 8.1 The `model_registry` name collision

There are two unrelated artifacts both called `model_registry`:

- The `ModelRegistry` **dataclass** in `types.py` — this document.
- The `model_registry` **module** in
  `src/proxai/connectors/model_registry.py` — a tiny helper that
  exposes `_MODEL_CONNECTOR_MAP` (a hardcoded dict of provider name
  → connector class) and `get_model_connector(provider)` (a
  factory). Has nothing to do with the registry data.

`available_models.py` uses the helper module (for connector
factories); `proxdash.py` uses the type. Don't conflate them when
reading import statements.

### 8.2 Mutation methods do not deep-copy

`register_provider_model_config(cfg)` stores the caller's
`ProviderModelConfig` instance directly into the registry. If the
caller subsequently mutates the same instance (e.g., flips a
feature cell), the change is visible through the registry. The
intended pattern is build-once-then-register; treat the registered
instance as owned by the registry from the call onward.

The same applies to
`override_default_model_priority_list([...])` — the list is
stored by reference; appending to the list later bypasses
validation.

### 8.3 `unregister_model` does not prune the priority list

Already noted in §4.2 / `px_models_model_config_api.md` §2.2.
Calling `unregister_model("openai", "gpt-4o")` while
`default_model_priority_list` still references that entry leaves
the registry internally inconsistent until the next
`override_default_model_priority_list` or `reload_from_registry`.
The lazy default-model selection in
`ProxAIClient.get_default_provider_model` will fail over to the
next entry rather than crash, but explicit listings via
`px.models.model_config.get_default_model_priority_list()` will
return the stale list.

### 8.4 `BUILT_IN` skips the version gate

The `min_proxai_version` validator is skipped for any registry
declaring `config_origin == BUILT_IN`. If you ship a third-party
registry to a customer and write `BUILT_IN` into the metadata, the
gate that protects them from running an outdated proxai version
will be silently disabled. Reserve `BUILT_IN` for the bundled
`v{version}.json` files in
`src/proxai/connectors/model_configs_data/`.

### 8.5 Bumping the bundled schema version

Adding a new field to `ModelRegistry`, `ProviderModelConfig`, or
their nested types requires four coordinated changes:

1. The dataclass in `types.py`.
2. The encode / decode pair in
   `serializers/type_serializer.py` — and matching
   tests in `tests/serializers/`.
3. The bundled `v{x.y.z}.json` under
   `src/proxai/connectors/model_configs_data/`, plus a bump of
   `ModelConfigs.LOCAL_CONFIG_VERSION` and an entry in
   `pyproject.toml`'s `include` list (so the new file ships with
   the wheel).
4. Optional: `_validate_provider_model_configs` /
   `_validate_default_model_priority_list` if the new field has
   an invariant.

The hash-equality parallel rule in
`testing_conventions.md` does NOT apply here — `ModelRegistry`
is not part of the cache key, so it does not appear in
`hash_serializer` or `is_query_record_equal`. Skip that step for
registry-shape changes.

### 8.6 `ProxDashConnection.get_model_registry` is best-effort

The fetch happens during `ProxAIClient.__init__` and **never
raises** — every failure mode (HTTP error, decode error,
degenerate payload) is logged at `ERROR` level and returns `None`.
This means `px.connect()` always succeeds even when ProxDash is
unreachable; the client falls back to the bundled BUILT_IN
registry. If you are debugging why a newly-shipped model is not
visible, check the proxdash log first — the reason is in there.

---

## 9. Where the registry sits in the layered import graph

```
Layer 0: types.py                                    # ModelRegistry lives here
                                                      #   (no internal deps)
Layer 0: serializers/type_serializer.py              # encode/decode (depends on types)

Layer 2: connectors/model_configs.py                 # ModelConfigs — owner / mutator
                                                      # (depends on types, type_serializer,
                                                      #  state_controller)

Layer 2: connections/proxdash.py                     # get_model_registry — remote source
                                                      # (depends on types, type_serializer)

Layer 4: connections/available_models.py             # consumes via model_configs_instance
                                                      # (also imports the model_registry
                                                      #  helper module — distinct artifact)

Layer 5: client.py / proxai.py                       # orchestrates _maybe_fetch...
```

Adding a new validator or a new mutation method belongs on
`ModelConfigs` (Layer 2). Adding a new field to the registry
belongs in `types.py` (Layer 0) and the matching serializer
function. Crossing layers — e.g., having a provider executor read
the registry directly — is a smell; the read path should always
go through `model_configs_instance.get_provider_model_config(...)`.
