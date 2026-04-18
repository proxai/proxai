# `px.Client` Control Parameters — Migration Plan

Implementation plan for `px_client_controls_proposal.md`. That document
describes the *what* and *why*; this document describes the *how*.

Source of truth for the migration: if this document disagrees with the
proposal, check which was updated more recently.

---

## 0. Summary

Replace five loose client-level parameters with three typed option objects,
propagated as whole units through the entire StateControlled hierarchy.

```
BEFORE (flat)                          AFTER (grouped)
──────────────────────                 ──────────────────────
feature_mapping_strategy ──┐           provider_call_options: ProviderCallOptions
suppress_provider_errors ──┘             ├── feature_mapping_strategy
                                         └── suppress_provider_errors
allow_multiprocessing ─────┐
model_test_timeout ────────┘           model_probe_options: ModelProbeOptions
                                         ├── allow_multiprocessing
keep_raw_provider_response ──          └── timeout

                                       debug_options: DebugOptions
                                         └── keep_raw_provider_response
```

---

## 1. Design decisions

### 1.1 Group at every level, not just the client

Option objects are stored as whole units in state containers at **every level**
of the hierarchy — `ProxAIClientState`, `AvailableModelsState`, `ProviderState`.
They are propagated as whole objects to children, never unpacked into individual
fields.

Rationale:
- Matches the existing pattern: `logging_options` and `proxdash_options` are
  already passed as whole objects to children.
- The loose fields (`feature_mapping_strategy`, `allow_multiprocessing`, etc.)
  were only loose because they lacked a grouping object. Now they have one.
- Passing whole objects means less bug-prone propagation code, better
  encapsulation, and children automatically get new fields added to the group.
- Children may not use every field in an option group — that's fine.
  `logging_options` already works this way (not every child uses
  `hide_sensitive_content`).

### 1.2 Plain dataclasses, not StateContainer subclasses

The new types follow the exact pattern of `CacheOptions`, `LoggingOptions`,
`ProxDashOptions` — plain `@dataclasses.dataclass`. This means:
- Use `get_property_value` / `set_property_value` (not state_controlled variants)
- No deserializer methods needed
- `load_state` handles them automatically (they're not `StateContainer` instances)

### 1.3 Rename `model_test_timeout` → `timeout`

Inside `ModelProbeOptions`, the context is already "model probing." The
`model_test_` prefix is redundant. `timeout` is consistent with how other
Python libraries name timeouts inside scoped option objects.

---

## 2. Propagation map (before → after)

### Before: individual fields at every level

```
ProxAIClient
  ├── self.feature_mapping_strategy          (property → ProxAIClientState)
  ├── self.suppress_provider_errors          (property → ProxAIClientState)
  ├── self.allow_multiprocessing             (property → ProxAIClientState)
  ├── self.model_test_timeout                (property → ProxAIClientState)
  └── self.keep_raw_provider_response        (property → ProxAIClientState)
        │
        │  unpacked into AvailableModelsParams:
        ▼
AvailableModels
  ├── self.feature_mapping_strategy          (property → AvailableModelsState)
  ├── self.allow_multiprocessing             (property → AvailableModelsState)
  ├── self.model_test_timeout                (property → AvailableModelsState)
  └── self.keep_raw_provider_response        (property → AvailableModelsState)
        │
        │  unpacked into ProviderConnectorParams:
        ▼
ProviderConnector
  ├── self.feature_mapping_strategy          (property → ProviderState)
  └── self.keep_raw_provider_response        (property → ProviderState)
```

### After: whole option objects at every level

```
ProxAIClient
  ├── self.provider_call_options             (property → ProxAIClientState)
  ├── self.model_probe_options               (property → ProxAIClientState)
  └── self.debug_options                     (property → ProxAIClientState)
        │
        │  passed as whole objects into AvailableModelsParams:
        ▼
AvailableModels
  ├── self.provider_call_options             (property → AvailableModelsState)
  ├── self.model_probe_options               (property → AvailableModelsState)
  └── self.debug_options                     (property → AvailableModelsState)
        │
        │  passed as whole objects into ProviderConnectorParams:
        ▼
ProviderConnector
  ├── self.provider_call_options             (property → ProviderState)
  └── self.debug_options                     (property → ProviderState)
```

---

## 3. Files to modify

| File | What changes |
|------|-------------|
| `src/proxai/types.py` | New types; `ProxAIClientState`, `AvailableModelsState`, `ProviderState`, `RunOptions` |
| `src/proxai/client.py` | `ProxAIClientParams`, `__init__`, properties, `get_current_options`, `_set_default_values`, all internal references |
| `src/proxai/connections/available_models.py` | `AvailableModelsParams`, properties, `get_model_connector()` propagation, all consumption sites |
| `src/proxai/connectors/provider_connector.py` | `ProviderConnectorParams`, properties, all consumption sites |
| `src/proxai/proxai.py` | `connect()`, `check_health()` |
| `src/proxai/__init__.py` | Exports |
| `src/proxai/serializers/type_serializer.py` | `encode_run_options()` + new encoder functions |

---

## 4. Step-by-step migration

### Step 1: Define new option types in `src/proxai/types.py`

Add three plain dataclasses after `ProxDashOptions` (~line 398), before
`RunOptions`:

```python
@dataclasses.dataclass
class ProviderCallOptions:
  """Client-wide defaults for provider call behaviour."""
  feature_mapping_strategy: FeatureMappingStrategy = (
      FeatureMappingStrategy.BEST_EFFORT)
  suppress_provider_errors: bool = False

@dataclasses.dataclass
class ModelProbeOptions:
  """Configuration for model probing (health checks, model discovery)."""
  allow_multiprocessing: bool = True
  timeout: int = 25

@dataclasses.dataclass
class DebugOptions:
  """Developer-only diagnostic options."""
  keep_raw_provider_response: bool = False
```

### Step 2: Update state containers in `src/proxai/types.py`

**`ProxAIClientState`** — replace five loose fields (lines 858-862):

```python
# REMOVE:
feature_mapping_strategy: FeatureMappingStrategy | None = None
suppress_provider_errors: bool | None = None
keep_raw_provider_response: bool | None = None
allow_multiprocessing: bool | None = None
model_test_timeout: int | None = None

# ADD:
provider_call_options: ProviderCallOptions | None = None
model_probe_options: ModelProbeOptions | None = None
debug_options: DebugOptions | None = None
```

**`AvailableModelsState`** — replace four loose fields (lines 819, 826-828):

```python
# REMOVE:
feature_mapping_strategy: FeatureMappingStrategy | None = None
allow_multiprocessing: bool | None = None
model_test_timeout: int | None = None
keep_raw_provider_response: bool | None = None

# ADD:
provider_call_options: ProviderCallOptions | None = None
model_probe_options: ModelProbeOptions | None = None
debug_options: DebugOptions | None = None
```

**`ProviderState`** — replace two loose fields (lines 806, 811):

```python
# REMOVE:
feature_mapping_strategy: FeatureMappingStrategy | None = None
keep_raw_provider_response: bool | None = None

# ADD:
provider_call_options: ProviderCallOptions | None = None
debug_options: DebugOptions | None = None
```

(No `model_probe_options` — `ProviderConnector` doesn't probe models.)

**`RunOptions`** — replace five loose fields (lines 448-452):

```python
# REMOVE:
allow_multiprocessing: bool | None = None
model_test_timeout: int | None = None
feature_mapping_strategy: FeatureMappingStrategy | None = None
suppress_provider_errors: bool | None = None
keep_raw_provider_response: bool | None = None

# ADD:
provider_call_options: ProviderCallOptions | None = None
model_probe_options: ModelProbeOptions | None = None
debug_options: DebugOptions | None = None
```

### Step 3: Update `ProxAIClientParams` in `src/proxai/client.py`

Replace five loose fields (lines 472-478):

```python
@dataclasses.dataclass
class ProxAIClientParams:
  experiment_path: str | None = None
  cache_options: types.CacheOptions | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_options: types.ProxDashOptions | None = None
  provider_call_options: types.ProviderCallOptions | None = None
  model_probe_options: types.ModelProbeOptions | None = None
  debug_options: types.DebugOptions | None = None
```

### Step 4: Update `ProxAIClient` properties in `src/proxai/client.py`

Remove five individual properties (lines 930-972). Add three option-object
properties using `get_property_value` / `set_property_value`:

```python
@property
def provider_call_options(self) -> types.ProviderCallOptions:
  return self.get_property_value("provider_call_options")

@provider_call_options.setter
def provider_call_options(self, value: types.ProviderCallOptions | None):
  result = types.ProviderCallOptions()
  if value is not None:
    result.feature_mapping_strategy = value.feature_mapping_strategy
    result.suppress_provider_errors = value.suppress_provider_errors
  self.set_property_value("provider_call_options", result)

@property
def model_probe_options(self) -> types.ModelProbeOptions:
  return self.get_property_value("model_probe_options")

@model_probe_options.setter
def model_probe_options(self, value: types.ModelProbeOptions | None):
  result = types.ModelProbeOptions()
  if value is not None:
    result.allow_multiprocessing = value.allow_multiprocessing
    if value.timeout < 1:
      raise ValueError("ModelProbeOptions.timeout must be >= 1.")
    result.timeout = value.timeout
  self.set_property_value("model_probe_options", result)

@property
def debug_options(self) -> types.DebugOptions:
  return self.get_property_value("debug_options")

@debug_options.setter
def debug_options(self, value: types.DebugOptions | None):
  result = types.DebugOptions()
  if value is not None:
    result.keep_raw_provider_response = value.keep_raw_provider_response
  self.set_property_value("debug_options", result)
```

### Step 5: Update `ProxAIClient.__init__()` in `src/proxai/client.py`

**5a. Constructor signature** (~line 504): Replace five loose kwargs with three.

**5b. Guard clause** (~line 589): Check new option objects instead of old kwargs.

**5c. ProxAIClientParams construction** (~line 614): Use new fields.

**5d. Init from params** (~lines 636-646): Assign option objects directly:

```python
self.provider_call_options = init_from_params.provider_call_options
self.model_probe_options = init_from_params.model_probe_options
self.debug_options = init_from_params.debug_options
```

**5e. AvailableModelsParams construction** (~lines 660-671): Pass whole objects:

```python
available_models_params = available_models.AvailableModelsParams(
    run_type=self.run_type,
    provider_call_options=self.provider_call_options,
    model_configs_instance=self.model_configs_instance,
    model_cache_manager=self.model_cache_manager,
    query_cache_manager=self.query_cache_manager,
    logging_options=self.logging_options,
    proxdash_connection=self.proxdash_connection,
    model_probe_options=self.model_probe_options,
    debug_options=self.debug_options,
)
```

### Step 6: Update `_set_default_values()` in `src/proxai/client.py`

```python
# REMOVE:
self.feature_mapping_strategy = types.FeatureMappingStrategy.BEST_EFFORT
self.suppress_provider_errors = False
self.keep_raw_provider_response = False
self.allow_multiprocessing = True
self.model_test_timeout = 25

# ADD:
self.provider_call_options = None
self.model_probe_options = None
self.debug_options = None
```

### Step 7: Update all internal references in `src/proxai/client.py`

Grep for old property names and update:

- `self.keep_raw_provider_response` → `self.debug_options.keep_raw_provider_response`
- `self.suppress_provider_errors` → `self.provider_call_options.suppress_provider_errors`
- `self.feature_mapping_strategy` → `self.provider_call_options.feature_mapping_strategy`

### Step 8: Update `get_current_options()` in `src/proxai/client.py`

```python
run_options = types.RunOptions(
    ...,
    provider_call_options=self.provider_call_options,
    model_probe_options=self.model_probe_options,
    debug_options=self.debug_options,
)
```

### Step 9: Update `AvailableModelsParams` in `src/proxai/connections/available_models.py`

Replace individual fields with whole option objects:

```python
@dataclasses.dataclass
class AvailableModelsParams:
  run_type: types.RunType | None = None
  provider_call_options: types.ProviderCallOptions | None = None
  model_configs_instance: model_configs.ModelConfigs | None = None
  model_cache_manager: model_cache.ModelCacheManager | None = None
  query_cache_manager: query_cache.QueryCacheManager | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_connection: proxdash.ProxDashConnection | None = None
  model_probe_options: types.ModelProbeOptions | None = None
  debug_options: types.DebugOptions | None = None
```

### Step 10: Update `AvailableModels` class in `src/proxai/connections/available_models.py`

**10a.** Replace class attribute annotations.

**10b.** Replace five individual properties with three option-object properties.

**10c.** Update `get_model_connector()` to pass whole objects:

```python
init_from_params.provider_call_options = self.provider_call_options
init_from_params.debug_options = self.debug_options
```

**10d.** Update all consumption sites:

- `self.allow_multiprocessing` → `self.model_probe_options.allow_multiprocessing`
- `self.model_test_timeout` → `self.model_probe_options.timeout`
- `self.feature_mapping_strategy` → `self.provider_call_options.feature_mapping_strategy`
- `self.keep_raw_provider_response` → `self.debug_options.keep_raw_provider_response`

### Step 11: Update `ProviderConnectorParams` in `src/proxai/connectors/provider_connector.py`

Replace individual fields with whole option objects:

```python
@dataclasses.dataclass
class ProviderConnectorParams:
  run_type: types.RunType | None = None
  provider_call_options: types.ProviderCallOptions | None = None
  query_cache_manager: types.QueryCacheManagerState | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_connection: proxdash.ProxDashConnection | None = None
  provider_token_value_map: types.ProviderTokenValueMap | None = None
  debug_options: types.DebugOptions | None = None
```

### Step 12: Update `ProviderConnector` class in `src/proxai/connectors/provider_connector.py`

**12a.** Replace class attribute annotations.

**12b.** Replace two individual properties with two option-object properties.

**12c.** Update all consumption sites:

- `self.feature_mapping_strategy` → `self.provider_call_options.feature_mapping_strategy`
- `self.keep_raw_provider_response` → `self.debug_options.keep_raw_provider_response`

### Step 13: Update `px.connect()` in `src/proxai/proxai.py`

Replace five loose kwargs with three option objects in signature and body.

### Step 14: Update `px.check_health()` in `src/proxai/proxai.py`

Change to accept `model_probe_options: types.ModelProbeOptions | None = None`.
Update state manipulation to set `state.model_probe_options`.

### Step 15: Update serializer in `src/proxai/serializers/type_serializer.py`

Replace five individual field encodings in `encode_run_options()` with three
grouped encodings. Add `encode_provider_call_options()`,
`encode_model_probe_options()`, `encode_debug_options()` helpers.

### Step 16: Update `__init__.py` exports

Export `ProviderCallOptions`, `ModelProbeOptions`, `DebugOptions`.

### Step 17: Skip tests (deferred)

Tests will be fixed after the major refactoring.

### Step 18: Update documentation

Update option structure tree in `docs/development/px_client_analysis.md`.

---

## 5. Verification

1. `poetry run ruff check src` — lint
2. `poetry run yapf --diff -r src` — format check
3. Manual smoke test:
   ```python
   import proxai as px
   px.connect(
       provider_call_options=px.ProviderCallOptions(
           feature_mapping_strategy=px.FeatureMappingStrategy.STRICT,
       ),
       model_probe_options=px.ModelProbeOptions(
           allow_multiprocessing=False,
           timeout=30,
       ),
   )
   opts = px.get_current_options()
   assert opts.provider_call_options.feature_mapping_strategy == (
       px.FeatureMappingStrategy.STRICT)
   assert opts.model_probe_options.timeout == 30
   assert opts.debug_options.keep_raw_provider_response == False
   ```
