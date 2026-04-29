# State Propagation Problem — Analysis & Options

## 0. The problem

When a `ProxAIClientState` is cloned, modified, and used to reconstruct a
client via `init_from_state`, the modifications only affect the **top-level**
fields. Nested state objects (`AvailableModelsState`, `ProviderState`) inside
the clone still hold the **original** values from before the modification.

```
state = px.get_default_proxai_client().clone_state()
state.model_probe_options = ModelProbeOptions(timeout=1)  # ← top-level updated

# But state.available_models_instance.model_probe_options is still the OLD value.
# When ProxAIClient reconstructs from this state, AvailableModels loads from
# its own stale snapshot.
```

This is not new — the same bug existed before the option-grouping migration
with the old flat fields (`allow_multiprocessing`, `model_test_timeout`,
`feature_mapping_strategy`, `keep_raw_provider_response`).

---

## 1. Scope of the problem

### 1.1 Fields with dual copies (client + child)

Every field that exists at BOTH the client level and a child level is
affected. This is not limited to the three new option types.

**ProxAIClientState → AvailableModelsState** (9 shared fields):
- `provider_call_options`, `model_probe_options`, `debug_options`
- `logging_options`
- `run_type`
- `model_cache_manager`, `query_cache_manager`
- `model_configs_instance`
- `proxdash_connection`

**AvailableModelsState → ProviderState** (6 shared fields):
- `provider_call_options`, `debug_options`
- `logging_options`
- `run_type`
- `query_cache_manager`
- `proxdash_connection`

### 1.2 Call sites that trigger the problem

Only places that do clone-modify-reconstruct:
- `px.check_health()` — clones default client, overrides probe options
  and experiment_path, reconstructs
- `examples/alias_test.py` — clones and overrides model_probe_options
- Any user code that does the same pattern

Normal `px.connect()` and `px.Client(...)` are **not affected** — they use
`init_from_params` which builds children fresh from the client's values.

### 1.3 Why `init_from_params` works but `init_from_state` doesn't

With `init_from_params`:
```
ProxAIClient.__init__():
  self.model_probe_options = params.model_probe_options    # client stores it
  AvailableModelsParams(model_probe_options=self.model_probe_options)
  # ↑ child is built FROM the client's value — single source of truth
```

With `init_from_state`:
```
ProxAIClient.__init__():
  self.load_state(state)
  # ↑ deserializes BOTH self.model_probe_options (top-level)
  #   AND self.available_models_instance (from nested AvailableModelsState)
  #   These are independent snapshots — no link between them
```

---

## 2. Design options

### Option A: Propagate after load_state (current ad-hoc fix, generalized)

After `load_state`, push all shared fields from parent to children.

**Implementation**: Add `_propagate_options_to_children()` to every
StateControlled class that has children with shared fields:

```python
# In ProxAIClient, after load_state:
def _propagate_options_to_children(self):
  if self.available_models_instance is not None:
    self.available_models_instance.provider_call_options = self.provider_call_options
    self.available_models_instance.model_probe_options = self.model_probe_options
    self.available_models_instance.debug_options = self.debug_options
    self.available_models_instance.logging_options = self.logging_options
    # ... all 9 shared fields

# In AvailableModels, after load_state:
def _propagate_options_to_children(self):
  for connector in self.provider_connectors.values():
    connector.provider_call_options = self.provider_call_options
    connector.debug_options = self.debug_options
    connector.logging_options = self.logging_options
    # ... all 6 shared fields
```

**Pros:**
- Straightforward to implement
- No changes to StateControlled base class
- Fixes the problem completely for all shared fields

**Cons:**
- Manual maintenance burden — every time you add a shared field, you must
  update propagation in every parent class
- Easy to forget a field (as happened with `logging_options` in the current
  ad-hoc fix)
- Every StateControlled class with children needs its own propagation method
- Propagation logic is scattered across classes instead of centralized

### Option B: Modify clone_state to produce a "flat" state

Instead of deep-copying the entire nested state tree, `clone_state` returns
a state where nested children are `None`. Reconstruction via `init_from_state`
then rebuilds children fresh from the top-level values (like `init_from_params`
does).

**Implementation**: Override `clone_state` on `ProxAIClient`:

```python
def clone_state_for_override(self) -> types.ProxAIClientState:
  state = self.get_state()
  # Clear nested children — they'll be rebuilt from top-level values
  state.available_models_instance = None
  state.registered_model_connectors = None
  return state
```

Then modify `ProxAIClient.__init__` to handle the hybrid case:
when `init_from_state` has `available_models_instance = None`, build it
from the top-level options (like the `init_from_params` path does).

**Pros:**
- Single source of truth — children are always built from parent values
- No manual field-by-field propagation
- No forgotten fields
- Clean separation: state = configuration snapshot, not full object tree

**Cons:**
- Loses cached state in children (provider connectors, model cache results)
- Requires `__init__` to handle a third initialization mode (partial state)
- Not compatible with how `load_state` currently works (it expects either
  full state or nothing)
- `check_health` would need to rebuild more objects, which is slower

### Option C: Children don't store options — read from parent

Children never store their own copy of shared options. Instead, they
receive a reference to the parent's options (or a getter function) and
read from it on every access.

**Implementation**: Use the StateControlled getter function pattern:

```python
# In AvailableModels:
# Instead of: self.model_probe_options = params.model_probe_options
# Do: _get_model_probe_options = lambda: parent.model_probe_options

class AvailableModels(StateControlled):
  @property
  def model_probe_options(self):
    # Try internal value first (from load_state), fall back to getter
    internal = self.get_property_internal_value('model_probe_options')
    if internal is not None:
      return internal
    # Fall back to parent's value via getter function
    getter = getattr(self, '_get_model_probe_options', None)
    if getter:
      return getter()
    return types.ModelProbeOptions()
```

**Pros:**
- Single source of truth — parent owns the value
- No propagation needed ever
- Matches the StateControlled getter function pattern that already exists

**Cons:**
- Getter functions are not serializable — breaks `get_state()` and
  `clone_state()` for multiprocessing
- The StateControlled documentation explicitly warns: "Once a literal value
  is set, getter function is IGNORED forever"
- During `init_from_params`, the literal value IS set (from the parent's
  current value), so the getter is immediately dead
- Fundamental incompatibility with the state serialization model —
  multiprocessing sends state across processes, where parent references
  don't exist

### Option D: StateControlled base class auto-propagation

Add a mechanism to the StateControlled base class where a class can declare
which fields are "inherited from parent" and the framework automatically
propagates them after `load_state`.

**Implementation**: Add a class method:

```python
class AvailableModels(StateControlled):
  @classmethod
  def get_parent_propagated_fields(cls):
    return ['provider_call_options', 'model_probe_options',
            'debug_options', 'logging_options']
```

Then in `load_state`, after restoring all fields, check if the parent
has updated values and apply them. But `load_state` doesn't know about
the parent — it only sees the state container.

The real question: **who calls the propagation?** The parent must do it
after calling `load_state` on the child. This is essentially Option A
but with field names declared centrally.

**Pros:**
- Field list declared once per class (not scattered in propagation code)
- Framework can validate that all shared fields are declared
- Could auto-generate propagation from the declaration

**Cons:**
- Still requires the parent to call propagation after load_state
- Adds complexity to StateControlled base class for a problem that only
  affects the clone-modify-reconstruct pattern
- Over-engineering for the small number of call sites that trigger this

### Option E: Don't use clone-modify-reconstruct — use params instead

The fundamental insight: `init_from_state` was designed for
**serialization/deserialization** (multiprocessing, disk persistence), not
for "tweak a few settings and make a new client." For the latter,
`init_from_params` is the correct path.

**Implementation**: Change `check_health` to build params directly:

```python
def check_health(
    model_probe_options: types.ModelProbeOptions | None = None,
    ...
):
  default = get_default_proxai_client()
  params = client.ProxAIClientParams(
      experiment_path=experiment_path,
      cache_options=default.cache_options,
      logging_options=default.logging_options,
      proxdash_options=default.proxdash_options,
      provider_call_options=default.provider_call_options,
      model_probe_options=(
          model_probe_options or default.model_probe_options),
      debug_options=default.debug_options,
  )
  px_client = client.ProxAIClient(init_from_params=params)
```

**Pros:**
- Uses the initialization path that was designed for this exact purpose
- No propagation problem — children are built from top-level values
- No changes to StateControlled or any class
- Clear semantic separation: `init_from_params` for "create with options",
  `init_from_state` for "restore from serialized snapshot"

**Cons:**
- Loses internal state from the default client (model cache results,
  provider connectors, proxdash connection state). `check_health` would
  need to do fresh model discovery.
  - Actually: `check_health` already calls `clear_model_cache=True`, so
    losing cached state is fine here.
- Every "clone with overrides" call site needs to manually copy options
  from the default client. This is verbose but explicit.
- If a new option is added to `ProxAIClientParams`, every clone-with-
  overrides site needs updating. (Same maintenance burden as Option A,
  but at the call site instead of inside the class.)

---

## 3. Recommendation

**Option E for `check_health`** (and any similar "clone with overrides"
use case). It's the simplest, most correct approach for the small number
of call sites that need it.

**Reasoning:**

1. `init_from_state` was built for state restoration (multiprocessing,
   persistence), not for "tweak and clone." Using it for the latter is
   the root cause of the bug.

2. `check_health` is the only production call site using clone-modify-
   reconstruct. Fixing it to use `init_from_params` eliminates the
   problem without touching the framework.

3. The `px.Client()` constructor already enforces "set everything at
   construction time" — that's the intended design. clone-modify-
   reconstruct bypasses this design and creates the dual-copy problem.

4. Options A and D add permanent complexity to handle a pattern that
   shouldn't exist in the first place. Option C is fundamentally
   incompatible with serialization. Option B is viable but adds a
   third init mode.

5. If we later need a proper "clone with overrides" pattern, we can
   add a dedicated method like `client.clone(model_probe_options=...)`
   that internally uses `init_from_params` — keeping the semantics
   clean.

**Keep the ad-hoc `_propagate_options_to_children()` as a safety net**
for now, but generalize it to ALL shared fields (add `logging_options`,
and propagate to provider connectors too). Even if `check_health` is
fixed to use `init_from_params`, the safety net protects against any
future code that accidentally uses clone-modify-reconstruct.

---

## 4. Full list of affected fields for safety-net propagation

If we keep `_propagate_options_to_children()` as defense-in-depth:

**ProxAIClient → AvailableModels** (all shared fields):
- `provider_call_options`
- `model_probe_options`
- `debug_options`
- `logging_options`

(Not `run_type`, `model_cache_manager`, `query_cache_manager`,
`model_configs_instance`, `proxdash_connection` — these are heavyweight
objects that shouldn't be swapped post-construction. The propagation
should only cover lightweight option/config objects.)

**AvailableModels → ProviderConnectors** (all shared fields):
- `provider_call_options`
- `debug_options`
- `logging_options`

---

## 5. What about `client.debug_options.keep_raw = True` post-construction?

The user's design intent: all options are set at `px.Client()` time.
Post-construction mutation of options is not supported. This is enforced
by the property setters creating fresh copies (not exposing mutable
references). For example:

```python
client = px.Client(debug_options=px.DebugOptions(keep_raw_provider_response=True))
# This does NOT work:
client.debug_options.keep_raw_provider_response = False
# ↑ Mutates the returned copy, not the stored value. The stored value
#   is a separate instance created by the setter.
```

Actually — this IS a problem. `get_property_value` returns the internal
reference, not a copy. So mutating the returned object DOES change the
stored value on the client. But it does NOT propagate to children because
there's no setter call, no `handle_changes`, no notification.

**This is an existing design hole for ALL option types** (LoggingOptions,
CacheOptions, ProxDashOptions). Mutating `client.logging_options.stdout = True`
changes the client's value but not the child's copy. This was true before
our migration and is inherent to the "copy in, no propagation" design.

The mitigation is documentation: "Set options at construction time. Do not
mutate option objects after construction." The fact that the setters create
copies helps (re-assigning the whole object works correctly), but in-place
mutation is silently broken.
