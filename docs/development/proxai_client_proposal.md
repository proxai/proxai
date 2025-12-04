# ProxAIClient Architecture Proposal

## Executive Summary

This document proposes refactoring ProxAI's global state management from module-level globals to a `ProxAIClient` class that extends `StateControlled`. This change enables:

1. **Serialization** for multiprocessing/workers
2. **Multi-client** support for complex applications
3. **Thread safety** through explicit instances
4. **100% backward compatibility** with existing API

---

## Current State Assessment

### Existing Architecture

ProxAI currently has **two systems coexisting**:

1. **Module-level globals in `proxai.py`** (~20 global variables)
2. **Sophisticated `StateControlled` system** for sub-components

```
proxai.py (module globals: ~20 variables)
    ↓ passes getter functions to
StateControlled subclasses:
    - ModelCacheManager
    - QueryCacheManager
    - ProxDashConnection
    - ProviderModelConnector
    - AvailableModels
    - ModelConfigs
```

### Current Global Variables

```python
# In proxai.py
_RUN_TYPE: types.RunType
_HIDDEN_RUN_KEY: str
_EXPERIMENT_PATH: Optional[str]
_ROOT_LOGGING_PATH: Optional[str]
_DEFAULT_MODEL_CACHE_PATH: Optional[tempfile.TemporaryDirectory]
_PLATFORM_USED_FOR_DEFAULT_MODEL_CACHE: bool

_LOGGING_OPTIONS: types.LoggingOptions
_CACHE_OPTIONS: types.CacheOptions
_PROXDASH_OPTIONS: types.ProxDashOptions

_MODEL_CONFIGS: Optional[model_configs.ModelConfigs]
_MODEL_CONFIGS_REQUESTED_FROM_PROXDASH: bool

_REGISTERED_MODEL_CONNECTORS: Dict[types.CallType, model_connector.ProviderModelConnector]
_MODEL_CONNECTORS: Dict[types.ProviderModelType, model_connector.ProviderModelConnector]
_DEFAULT_MODEL_CACHE_MANAGER: Optional[model_cache.ModelCacheManager]
_MODEL_CACHE_MANAGER: Optional[model_cache.ModelCacheManager]
_QUERY_CACHE_MANAGER: Optional[query_cache.QueryCacheManager]
_PROXDASH_CONNECTION: Optional[proxdash.ProxDashConnection]

_STRICT_FEATURE_TEST: bool
_SUPPRESS_PROVIDER_ERRORS: bool
_ALLOW_MULTIPROCESSING: bool
_MODEL_TEST_TIMEOUT: Optional[int]

_STATS: Dict[stat_types.GlobalStatType, stat_types.RunStats]
_AVAILABLE_MODELS: Optional[available_models.AvailableModels]
```

### The Gap

The sub-components properly use `StateControlled`, but the top-level `proxai.py` uses raw module globals. This creates a disconnect where:

- Sub-components can be serialized, but the whole system cannot
- State is scattered across module globals instead of unified
- Thread safety is impossible without locks on every global
- Multi-client scenarios are not supported

---

## Proposed Architecture

### Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ProxAIClient (StateControlled)                │
│  ┌─────────────────────┐     ┌─────────────────────────────┐   │
│  │  ProxAIClientState  │     │  Nested StateControlled:    │   │
│  │  - run_type         │     │  - model_cache_manager      │   │
│  │  - cache_options    │     │  - query_cache_manager      │   │
│  │  - logging_options  │     │  - proxdash_connection      │   │
│  │  - proxdash_options │     │  - model_configs            │   │
│  │  - experiment_path  │     │  - available_models         │   │
│  │  - ...              │     │  - model_connectors (dict)  │   │
│  └─────────────────────┘     └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
           ┌─────────────────┐   ┌─────────────────┐
           │  Default Client │   │  Custom Clients │
           │  (_DEFAULT)     │   │  (user-created) │
           └─────────────────┘   └─────────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Module-level    │
           │ API functions   │
           │ (px.connect,    │
           │  px.generate_*) │
           └─────────────────┘
```

### Design Goals

1. **Backward Compatible**: `px.connect()` and `px.generate_text()` work exactly as before
2. **New Capabilities**: Users can create multiple clients
3. **Serializable**: State can be exported/imported for workers
4. **Thread-Safe Option**: Each thread can have its own client

---

## Implementation Details

### Phase 1: State Container Definition

**File**: `src/proxai/types.py`

```python
@dataclasses.dataclass
class ProxAIClientState(StateContainer):
    """Complete state for a ProxAI client instance."""

    # Core settings
    run_type: Optional[RunType] = None
    hidden_run_key: Optional[str] = None
    experiment_path: Optional[str] = None
    root_logging_path: Optional[str] = None

    # Configuration options
    logging_options: Optional[LoggingOptions] = None
    cache_options: Optional[CacheOptions] = None
    proxdash_options: Optional[ProxDashOptions] = None

    # Nested StateControlled objects (stored as their State types)
    model_configs: Optional[ModelConfigsState] = None
    model_cache_manager: Optional[ModelCacheManagerState] = None
    query_cache_manager: Optional[QueryCacheManagerState] = None
    proxdash_connection: Optional[ProxDashConnectionState] = None
    available_models: Optional[AvailableModelsState] = None

    # Behavior flags
    strict_feature_test: Optional[bool] = False
    suppress_provider_errors: Optional[bool] = False
    allow_multiprocessing: Optional[bool] = True
    model_test_timeout: Optional[int] = 25
    model_configs_requested_from_proxdash: Optional[bool] = False

    # Runtime state (may not serialize perfectly)
    stats: Optional[Dict[str, Any]] = None
```

### Phase 2: ProxAIClient Class

**File**: `src/proxai/client.py` (new file)

```python
"""ProxAI Client - Main client class for ProxAI operations."""

import copy
from typing import Any, Dict, Optional, Union
import proxai.types as types
import proxai.state_controllers.state_controller as state_controller
import proxai.caching.model_cache as model_cache
import proxai.caching.query_cache as query_cache
import proxai.connections.proxdash as proxdash
import proxai.connections.available_models as available_models
import proxai.connectors.model_configs as model_configs
import proxai.connectors.model_connector as model_connector
import proxai.connectors.model_registry as model_registry
import proxai.stat_types as stat_types
import proxai.experiment.experiment as experiment

_PROXAI_CLIENT_STATE_PROPERTY = '_proxai_client_state'


class ProxAIClient(state_controller.StateControlled):
    """Main client for ProxAI operations.

    Can be used as:
    1. Singleton via px.connect() and px.generate_text()
    2. Explicit instance for multi-client scenarios

    Example (simple - uses default client):
        import proxai as px
        px.connect(cache_path="/tmp/cache")
        result = px.generate_text("Hello")

    Example (explicit client):
        from proxai import ProxAIClient
        client = ProxAIClient(cache_path="/tmp/cache")
        result = client.generate_text("Hello")

    Example (multiprocessing):
        # Main process
        client = ProxAIClient(cache_path="/tmp/cache")
        state = client.get_state()

        # Worker process
        worker_client = ProxAIClient(init_state=state)
        result = worker_client.generate_text("Hello from worker")
    """

    # Type hints for internal storage
    _run_type: Optional[types.RunType]
    _get_run_type: Optional[callable]
    _hidden_run_key: Optional[str]
    _experiment_path: Optional[str]
    _get_experiment_path: Optional[callable]
    _root_logging_path: Optional[str]

    _logging_options: Optional[types.LoggingOptions]
    _get_logging_options: Optional[callable]
    _cache_options: Optional[types.CacheOptions]
    _get_cache_options: Optional[callable]
    _proxdash_options: Optional[types.ProxDashOptions]
    _get_proxdash_options: Optional[callable]

    _model_configs: Optional[model_configs.ModelConfigs]
    _model_cache_manager: Optional[model_cache.ModelCacheManager]
    _query_cache_manager: Optional[query_cache.QueryCacheManager]
    _proxdash_connection: Optional[proxdash.ProxDashConnection]
    _available_models: Optional[available_models.AvailableModels]

    _strict_feature_test: Optional[bool]
    _suppress_provider_errors: Optional[bool]
    _allow_multiprocessing: Optional[bool]
    _model_test_timeout: Optional[int]
    _model_configs_requested_from_proxdash: Optional[bool]

    _stats: Optional[Dict[stat_types.GlobalStatType, stat_types.RunStats]]

    # Non-state runtime objects
    _model_connectors: Dict[types.ProviderModelType, model_connector.ProviderModelConnector]
    _registered_model_connectors: Dict[types.CallType, model_connector.ProviderModelConnector]
    _default_model_cache_manager: Optional[model_cache.ModelCacheManager]
    _default_model_cache_path: Optional[str]
    _platform_used_for_default_model_cache: bool

    _proxai_client_state: types.ProxAIClientState

    def __init__(
        self,
        experiment_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        cache_options: Optional[types.CacheOptions] = None,
        logging_path: Optional[str] = None,
        logging_options: Optional[types.LoggingOptions] = None,
        proxdash_options: Optional[types.ProxDashOptions] = None,
        allow_multiprocessing: Optional[bool] = True,
        model_test_timeout: Optional[int] = 25,
        strict_feature_test: Optional[bool] = False,
        suppress_provider_errors: Optional[bool] = False,
        run_type: Optional[types.RunType] = None,
        init_state: Optional[types.ProxAIClientState] = None
    ):
        """Initialize a ProxAI client.

        Args:
            experiment_path: Path for experiment tracking
            cache_path: Shorthand for cache_options.cache_path
            cache_options: Full cache configuration
            logging_path: Shorthand for logging_options.logging_path
            logging_options: Full logging configuration
            proxdash_options: ProxDash integration settings
            allow_multiprocessing: Enable multiprocessing for model tests
            model_test_timeout: Timeout for model connectivity tests
            strict_feature_test: Raise errors on unsupported features
            suppress_provider_errors: Return errors instead of raising
            run_type: PRODUCTION or TEST mode
            init_state: Load from serialized state (mutually exclusive with other args)
        """
        super().__init__(
            experiment_path=experiment_path,
            cache_options=cache_options,
            logging_options=logging_options,
            proxdash_options=proxdash_options,
            allow_multiprocessing=allow_multiprocessing,
            model_test_timeout=model_test_timeout,
            strict_feature_test=strict_feature_test,
            suppress_provider_errors=suppress_provider_errors,
            run_type=run_type,
            init_state=init_state
        )

        # Initialize non-state runtime objects
        self._model_connectors = {}
        self._registered_model_connectors = {}
        self._default_model_cache_manager = None
        self._default_model_cache_path = None
        self._platform_used_for_default_model_cache = False

        if init_state:
            self.load_state(init_state)
            self._init_default_model_cache_manager()
            self._init_runtime_objects()
        else:
            initial_state = self.get_state()

            # Set getter functions (none by default for client)
            self._get_run_type = None
            self._get_experiment_path = None
            self._get_logging_options = None
            self._get_cache_options = None
            self._get_proxdash_options = None

            # Initialize core settings
            self.run_type = run_type or types.RunType.PRODUCTION
            self.hidden_run_key = experiment.get_hidden_run_key()
            self.experiment_path = experiment_path

            # Process logging options
            self._set_logging_options(
                experiment_path=experiment_path,
                logging_path=logging_path,
                logging_options=logging_options
            )

            # Process cache options
            self._set_cache_options(
                cache_path=cache_path,
                cache_options=cache_options
            )

            # Set other options
            self.proxdash_options = proxdash_options or types.ProxDashOptions()
            self.allow_multiprocessing = allow_multiprocessing
            self.model_test_timeout = model_test_timeout
            self.strict_feature_test = strict_feature_test
            self.suppress_provider_errors = suppress_provider_errors
            self.model_configs_requested_from_proxdash = False

            # Initialize stats
            self.stats = {
                stat_types.GlobalStatType.RUN_TIME: stat_types.RunStats(),
                stat_types.GlobalStatType.SINCE_CONNECT: stat_types.RunStats()
            }

            # Initialize managers
            self._init_default_model_cache_manager()
            self._init_runtime_objects()

            self.handle_changes(initial_state, self.get_state())

    # === Abstract Method Implementations ===

    def get_internal_state_property_name(self):
        return _PROXAI_CLIENT_STATE_PROPERTY

    def get_internal_state_type(self):
        return types.ProxAIClientState

    def handle_changes(
        self,
        old_state: types.ProxAIClientState,
        current_state: types.ProxAIClientState
    ):
        """Handle state changes and validate configuration."""
        result_state = copy.deepcopy(old_state)

        # Merge changes
        if current_state.run_type is not None:
            result_state.run_type = current_state.run_type
        if current_state.experiment_path is not None:
            result_state.experiment_path = current_state.experiment_path
        if current_state.logging_options is not None:
            result_state.logging_options = current_state.logging_options
        if current_state.cache_options is not None:
            result_state.cache_options = current_state.cache_options
        if current_state.proxdash_options is not None:
            result_state.proxdash_options = current_state.proxdash_options

        # Validation (add as needed)
        if result_state.model_test_timeout is not None:
            if result_state.model_test_timeout < 1:
                raise ValueError('model_test_timeout must be greater than 0.')

    # === Property Definitions ===

    @property
    def run_type(self) -> types.RunType:
        return self.get_property_value('run_type')

    @run_type.setter
    def run_type(self, value: types.RunType):
        self.set_property_value('run_type', value)

    @property
    def hidden_run_key(self) -> str:
        return self.get_property_value('hidden_run_key')

    @hidden_run_key.setter
    def hidden_run_key(self, value: str):
        self.set_property_value('hidden_run_key', value)

    @property
    def experiment_path(self) -> Optional[str]:
        return self.get_property_value('experiment_path')

    @experiment_path.setter
    def experiment_path(self, value: Optional[str]):
        if value is not None:
            experiment.validate_experiment_path(value)
        self.set_property_value('experiment_path', value)

    @property
    def logging_options(self) -> types.LoggingOptions:
        return self.get_property_value('logging_options')

    @logging_options.setter
    def logging_options(self, value: types.LoggingOptions):
        self.set_property_value('logging_options', value)

    @property
    def cache_options(self) -> types.CacheOptions:
        return self.get_property_value('cache_options')

    @cache_options.setter
    def cache_options(self, value: types.CacheOptions):
        self.set_property_value('cache_options', value)

    @property
    def proxdash_options(self) -> types.ProxDashOptions:
        return self.get_property_value('proxdash_options')

    @proxdash_options.setter
    def proxdash_options(self, value: types.ProxDashOptions):
        self.set_property_value('proxdash_options', value)

    # ... additional properties follow same pattern ...

    # === Nested StateControlled Properties ===

    @property
    def model_cache_manager(self) -> model_cache.ModelCacheManager:
        return self.get_state_controlled_property_value('model_cache_manager')

    @model_cache_manager.setter
    def model_cache_manager(self, value):
        self.set_state_controlled_property_value('model_cache_manager', value)

    def model_cache_manager_deserializer(
        self,
        state_value: types.ModelCacheManagerState
    ) -> model_cache.ModelCacheManager:
        return model_cache.ModelCacheManager(init_state=state_value)

    # ... similar for query_cache_manager, proxdash_connection, etc. ...

    # === Public API Methods ===

    def generate_text(
        self,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[types.MessagesType] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[types.StopType] = None,
        provider_model: Optional[types.ProviderModelIdentifierType] = None,
        use_cache: Optional[bool] = None,
        unique_response_limit: Optional[int] = None,
        extensive_return: bool = False,
        suppress_provider_errors: Optional[bool] = None
    ) -> Union[str, types.LoggingRecord]:
        """Generate text using the configured model."""
        # Implementation moves here from proxai.py
        ...

    def set_model(
        self,
        provider_model: Optional[types.ProviderModelIdentifierType] = None,
        generate_text: Optional[types.ProviderModelIdentifierType] = None
    ):
        """Set the model to use for generation."""
        ...

    def check_health(
        self,
        verbose: bool = True,
        extensive_return: bool = False
    ) -> types.ModelStatus:
        """Check connectivity to all configured models."""
        ...

    def get_summary(
        self,
        run_time: bool = False,
        json: bool = False
    ) -> Union[stat_types.RunStats, Dict[str, Any]]:
        """Get usage statistics."""
        ...

    # === State Export/Import ===

    def export_state(self) -> types.ProxAIClientState:
        """Export current state for serialization.

        Use this to pass state to worker processes.
        """
        return self.get_state()

    @classmethod
    def from_state(cls, state: types.ProxAIClientState) -> 'ProxAIClient':
        """Create a client from exported state.

        Use this in worker processes to restore state.
        """
        return cls(init_state=state)

    # === Private Methods ===

    def _init_default_model_cache_manager(self):
        """Initialize the default model cache manager."""
        # Implementation from current _init_default_model_cache_manager()
        ...

    def _init_runtime_objects(self):
        """Initialize runtime objects after state is loaded."""
        # Initialize model_configs, managers, etc.
        ...

    def _set_logging_options(self, experiment_path, logging_path, logging_options):
        """Process and set logging options."""
        # Implementation from current _set_logging_options()
        ...

    def _set_cache_options(self, cache_path, cache_options):
        """Process and set cache options."""
        # Implementation from current _set_cache_options()
        ...
```

### Phase 3: Backward-Compatible Module API

**File**: `src/proxai/proxai.py` (refactored)

```python
"""ProxAI - Unified AI Integration Platform.

This module provides backward-compatible access to ProxAI functionality.
All functions delegate to a default ProxAIClient instance.

For advanced usage (multiprocessing, multi-client), use ProxAIClient directly:
    from proxai import ProxAIClient
    client = ProxAIClient(cache_path="/tmp/cache")
"""

from typing import Any, Dict, Optional, Union
import proxai.types as types
from proxai.client import ProxAIClient
import proxai.stat_types as stat_types

# Re-export types for backward compatibility
CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions

# Default client instance
_DEFAULT_CLIENT: Optional[ProxAIClient] = None


def _get_default_client() -> ProxAIClient:
    """Get or create the default client."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = ProxAIClient()
    return _DEFAULT_CLIENT


def connect(
    experiment_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    cache_options: Optional[CacheOptions] = None,
    logging_path: Optional[str] = None,
    logging_options: Optional[LoggingOptions] = None,
    proxdash_options: Optional[ProxDashOptions] = None,
    allow_multiprocessing: Optional[bool] = True,
    model_test_timeout: Optional[int] = 25,
    strict_feature_test: Optional[bool] = False,
    suppress_provider_errors: Optional[bool] = False
):
    """Configure the default ProxAI client.

    This creates a new default client with the specified configuration.
    All subsequent calls to generate_text(), etc. will use this client.
    """
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = ProxAIClient(
        experiment_path=experiment_path,
        cache_path=cache_path,
        cache_options=cache_options,
        logging_path=logging_path,
        logging_options=logging_options,
        proxdash_options=proxdash_options,
        allow_multiprocessing=allow_multiprocessing,
        model_test_timeout=model_test_timeout,
        strict_feature_test=strict_feature_test,
        suppress_provider_errors=suppress_provider_errors
    )


def generate_text(
    prompt: Optional[str] = None,
    system: Optional[str] = None,
    messages: Optional[types.MessagesType] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[types.StopType] = None,
    provider_model: Optional[types.ProviderModelIdentifierType] = None,
    use_cache: Optional[bool] = None,
    unique_response_limit: Optional[int] = None,
    extensive_return: bool = False,
    suppress_provider_errors: Optional[bool] = None
) -> Union[str, types.LoggingRecord]:
    """Generate text using the default client."""
    return _get_default_client().generate_text(
        prompt=prompt,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        provider_model=provider_model,
        use_cache=use_cache,
        unique_response_limit=unique_response_limit,
        extensive_return=extensive_return,
        suppress_provider_errors=suppress_provider_errors
    )


def set_model(
    provider_model: Optional[types.ProviderModelIdentifierType] = None,
    generate_text: Optional[types.ProviderModelIdentifierType] = None
):
    """Set the model for the default client."""
    _get_default_client().set_model(
        provider_model=provider_model,
        generate_text=generate_text
    )


def set_run_type(run_type: types.RunType):
    """Set the run type for the default client."""
    _get_default_client().run_type = run_type


def check_health(
    experiment_path: Optional[str] = None,
    verbose: bool = True,
    allow_multiprocessing: bool = True,
    model_test_timeout: int = 25,
    extensive_return: bool = False
) -> types.ModelStatus:
    """Check health using the default client."""
    return _get_default_client().check_health(
        verbose=verbose,
        extensive_return=extensive_return
    )


def get_summary(
    run_time: bool = False,
    json: bool = False
) -> Union[stat_types.RunStats, Dict[str, Any]]:
    """Get summary from the default client."""
    return _get_default_client().get_summary(run_time=run_time, json=json)


def get_available_models():
    """Get available models from the default client."""
    return _get_default_client().available_models


def get_current_options(json: bool = False):
    """Get current options from the default client."""
    return _get_default_client().get_current_options(json=json)


def reset_state():
    """Reset the default client to fresh state."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None:
        _DEFAULT_CLIENT.reset_platform_cache()
    _DEFAULT_CLIENT = None


def reset_platform_cache():
    """Reset platform cache for the default client."""
    client = _get_default_client()
    client.reset_platform_cache()


# === New Functions for Advanced Usage ===

def export_client_state() -> types.ProxAIClientState:
    """Export default client state for multiprocessing.

    Example:
        state = px.export_client_state()
        # Pass state to worker process

    In worker:
        px.import_client_state(state)
        px.generate_text("Hello from worker")
    """
    return _get_default_client().export_state()


def import_client_state(state: types.ProxAIClientState):
    """Import client state (typically in a worker process).

    This replaces the default client with one loaded from the given state.
    """
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = ProxAIClient.from_state(state)


def get_client() -> ProxAIClient:
    """Get the default client instance.

    Useful for advanced operations not exposed via module functions.
    """
    return _get_default_client()
```

---

## Usage Examples

### Simple Usage (Unchanged)

```python
import proxai as px

px.connect(cache_path="/tmp/cache")
result = px.generate_text("Hello, world!")
print(result)
```

### Explicit Client

```python
from proxai import ProxAIClient

client = ProxAIClient(
    cache_path="/tmp/cache",
    logging_path="/var/log/proxai"
)
result = client.generate_text("Hello, world!")
```

### Multi-Client Scenarios

```python
from proxai import ProxAIClient

# Different configurations for different use cases
production_client = ProxAIClient(
    cache_path="/production/cache",
    proxdash_options=ProxDashOptions(api_key="prod-key")
)

dev_client = ProxAIClient(
    cache_path="/tmp/dev-cache",
    run_type=RunType.TEST
)

# Use appropriate client
result = production_client.generate_text("Production query")
dev_result = dev_client.generate_text("Development query")
```

### Multiprocessing

```python
import proxai as px
from multiprocessing import Pool

# Configure in main process
px.connect(cache_path="/tmp/cache")
state = px.export_client_state()

def worker_task(prompt, client_state):
    px.import_client_state(client_state)
    return px.generate_text(prompt)

with Pool(4) as pool:
    results = pool.starmap(worker_task, [
        ("Query 1", state),
        ("Query 2", state),
        ("Query 3", state),
        ("Query 4", state),
    ])
```

### Thread-Per-Client Pattern

```python
from proxai import ProxAIClient
import threading

def worker_thread(thread_id, cache_path):
    # Each thread has its own client
    client = ProxAIClient(cache_path=cache_path)
    result = client.generate_text(f"Hello from thread {thread_id}")
    print(f"Thread {thread_id}: {result}")

threads = []
for i in range(4):
    t = threading.Thread(target=worker_thread, args=(i, f"/tmp/cache_{i}"))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

### Context Manager (Future Enhancement)

```python
import proxai as px

px.connect(cache_path="/default/cache")

# Temporary configuration change
with px.temporary_config(cache_path="/tmp/test-cache"):
    result = px.generate_text("Test query")
    # Uses /tmp/test-cache

# Back to /default/cache
result = px.generate_text("Normal query")
```

---

## Comparison: Current vs Proposed

| Aspect | Current (`proxai.py`) | Proposed (`ProxAIClient`) |
|--------|----------------------|---------------------------|
| State Location | ~20 module globals | Single StateControlled class |
| Serialization | Not possible | `client.get_state()` / `load_state()` |
| Multi-client | Not possible | Create multiple instances |
| Thread Safety | None | Each thread owns a client |
| Multiprocessing | Globals don't transfer | Export/import state |
| State Validation | In setter functions | Unified in `handle_changes()` |
| Backward Compat | N/A | 100% via module functions |
| Testability | Difficult (global state) | Easy (isolated instances) |

---

## Migration Path

### Step 1: Create Foundation (Low Risk)
- [ ] Add `ProxAIClientState` to `types.py`
- [ ] Create `client.py` with `ProxAIClient` skeleton
- [ ] Add basic properties and state management
- [ ] Write tests for state serialization

### Step 2: Move Logic (Medium Risk)
- [ ] Move `_set_logging_options`, `_set_cache_options`, etc. to `ProxAIClient`
- [ ] Move `generate_text`, `check_health` logic to `ProxAIClient`
- [ ] Move manager initialization logic
- [ ] Ensure all functionality works in `ProxAIClient`

### Step 3: Refactor Module API (Low Risk)
- [ ] Refactor `proxai.py` to delegate to default client
- [ ] Add `export_client_state`, `import_client_state`
- [ ] Add `get_client()` for advanced access
- [ ] Ensure all existing tests pass

### Step 4: Add Advanced Features
- [ ] Context manager for temporary config
- [ ] Improved thread safety documentation
- [ ] Performance optimization (lazy initialization)

### Step 5: Documentation
- [ ] Update user documentation
- [ ] Add multiprocessing examples
- [ ] Add multi-client examples

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Module API remains unchanged |
| Performance regression | Lazy initialization, same patterns |
| Complexity increase | Clear separation of concerns |
| Testing difficulty | Actually easier (isolated instances) |

---

## Thread Safety Recommendation

Based on industry patterns, we recommend the **"explicit instance per thread"** pattern rather than adding locks:

```python
# Recommended for multithreading
def worker_thread():
    client = ProxAIClient(cache_options=my_opts)  # Own instance
    client.generate_text("hello")
```

This is simpler and faster than lock-based approaches.

---

## Conclusion

The proposed `ProxAIClient` architecture:

1. **Leverages your existing `StateControlled` system** - no new patterns to learn
2. **Maintains 100% backward compatibility** - users don't need to change anything
3. **Enables new capabilities** - multiprocessing, multi-client, thread safety
4. **Follows industry best practices** - explicit client objects (Anthropic, modern OpenAI)
5. **Improves testability** - isolated instances are easier to test

---

*Last Updated: 2025-12-03*
