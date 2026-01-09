# Migration Guide: Module Globals to ProxAIClient

## Overview

This guide documents the step-by-step process for migrating ProxAI from module-level globals to the `ProxAIClient` StateControlled architecture.

---

## Phase 1: Foundation Setup

### 1.1 Add ProxAIClientState to types.py

**File**: `src/proxai/types.py`

Add the following after existing StateContainer definitions:

```python
@dataclasses.dataclass
class ProxAIClientState(StateContainer):
    """Complete state for a ProxAI client instance.

    This state container holds all configuration and nested states
    for a ProxAI client, enabling serialization for multiprocessing.
    """

    # Core settings
    run_type: Optional[RunType] = None
    hidden_run_key: Optional[str] = None
    experiment_path: Optional[str] = None
    root_logging_path: Optional[str] = None

    # Configuration options (simple dataclasses)
    logging_options: Optional[LoggingOptions] = None
    cache_options: Optional[CacheOptions] = None
    proxdash_options: Optional[ProxDashOptions] = None

    # Nested StateControlled states
    model_configs: Optional['ModelConfigsState'] = None
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

    # Runtime state
    stats: Optional[Dict[str, Any]] = None
```

**Note**: If `ModelConfigsState` doesn't exist, we need to create it first in the connectors module.

### 1.2 Check/Create Missing State Types

Verify these state types exist:
- [x] `ModelCacheManagerState` - exists in types.py
- [x] `QueryCacheManagerState` - exists in types.py
- [x] `ProxDashConnectionState` - exists in types.py
- [x] `AvailableModelsState` - exists in types.py
- [ ] `ModelConfigsState` - may need to be created

If `ModelConfigsState` doesn't exist, add it:

```python
@dataclasses.dataclass
class ModelConfigsState(StateContainer):
    """State for ModelConfigs."""
    model_configs_schema: Optional[Dict[str, Any]] = None
    # Add other fields as needed from ModelConfigs class
```

### 1.3 Create client.py Skeleton

**File**: `src/proxai/client.py` (new file)

```python
"""ProxAI Client - Main client class for ProxAI operations.

This module provides the ProxAIClient class which encapsulates all ProxAI
functionality in a StateControlled class, enabling:
- State serialization for multiprocessing
- Multiple client instances
- Thread-safe operation (via separate instances)
"""

import copy
import os
import tempfile
from typing import Any, Callable, Dict, Optional, Union

import platformdirs

import proxai.types as types
import proxai.type_utils as type_utils
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
import proxai.serializers.type_serializer as type_serializer
import proxai.logging.utils as logging_utils

_PROXAI_CLIENT_STATE_PROPERTY = '_proxai_client_state'


class ProxAIClient(state_controller.StateControlled):
    """Main client for ProxAI operations.

    This class encapsulates all ProxAI functionality, replacing module-level
    globals with a proper StateControlled implementation.

    Basic Usage:
        client = ProxAIClient(cache_path="/tmp/cache")
        result = client.generate_text("Hello, world!")

    Multiprocessing:
        # Main process
        client = ProxAIClient(cache_path="/tmp/cache")
        state = client.get_state()

        # Worker process
        worker_client = ProxAIClient(init_state=state)
        result = worker_client.generate_text("Hello from worker")
    """

    # === Type Hints for Internal Storage ===

    # Core settings
    _run_type: Optional[types.RunType]
    _hidden_run_key: Optional[str]
    _experiment_path: Optional[str]
    _root_logging_path: Optional[str]

    # Getter functions
    _get_run_type: Optional[Callable[[], types.RunType]]
    _get_experiment_path: Optional[Callable[[], str]]
    _get_logging_options: Optional[Callable[[], types.LoggingOptions]]
    _get_cache_options: Optional[Callable[[], types.CacheOptions]]
    _get_proxdash_options: Optional[Callable[[], types.ProxDashOptions]]

    # Configuration options
    _logging_options: Optional[types.LoggingOptions]
    _cache_options: Optional[types.CacheOptions]
    _proxdash_options: Optional[types.ProxDashOptions]

    # Nested StateControlled managers
    _model_configs: Optional[model_configs.ModelConfigs]
    _model_cache_manager: Optional[model_cache.ModelCacheManager]
    _query_cache_manager: Optional[query_cache.QueryCacheManager]
    _proxdash_connection: Optional[proxdash.ProxDashConnection]
    _available_models: Optional[available_models.AvailableModels]

    # Behavior flags
    _strict_feature_test: Optional[bool]
    _suppress_provider_errors: Optional[bool]
    _allow_multiprocessing: Optional[bool]
    _model_test_timeout: Optional[int]
    _model_configs_requested_from_proxdash: Optional[bool]

    # Stats
    _stats: Optional[Dict[stat_types.GlobalStatType, stat_types.RunStats]]

    # Internal state container
    _proxai_client_state: types.ProxAIClientState

    # === Non-State Runtime Objects ===
    # These are NOT serialized; recreated on load
    _model_connectors: Dict[types.ProviderModelType, model_connector.ProviderModelConnector]
    _registered_model_connectors: Dict[types.CallType, model_connector.ProviderModelConnector]
    _default_model_cache_manager: Optional[model_cache.ModelCacheManager]
    _default_model_cache_path: Optional[str]
    _platform_used_for_default_model_cache: bool

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
            model_test_timeout: Timeout for model connectivity tests (seconds)
            strict_feature_test: Raise errors on unsupported features
            suppress_provider_errors: Return errors instead of raising exceptions
            run_type: PRODUCTION or TEST mode
            init_state: Load from serialized state (exclusive with other args)

        Raises:
            ValueError: If init_state is provided with other arguments
            ValueError: If both cache_path and cache_options.cache_path are set
            ValueError: If both logging_path and logging_options.logging_path are set
        """
        # Parent validates init_state exclusivity
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
            self._post_load_init()
        else:
            initial_state = self.get_state()

            # Clear getter functions (client doesn't use external getters)
            self._get_run_type = None
            self._get_experiment_path = None
            self._get_logging_options = None
            self._get_cache_options = None
            self._get_proxdash_options = None

            # Initialize core settings
            self.run_type = run_type or types.RunType.PRODUCTION
            self.hidden_run_key = experiment.get_hidden_run_key()

            # Process experiment path
            self._process_experiment_path(experiment_path)

            # Process logging options
            self._process_logging_options(
                experiment_path=experiment_path,
                logging_path=logging_path,
                logging_options=logging_options
            )

            # Process cache options
            self._process_cache_options(
                cache_path=cache_path,
                cache_options=cache_options
            )

            # Set remaining options
            self.proxdash_options = proxdash_options or types.ProxDashOptions()
            self.allow_multiprocessing = allow_multiprocessing
            self.model_test_timeout = model_test_timeout
            self.strict_feature_test = strict_feature_test
            self.suppress_provider_errors = suppress_provider_errors
            self.model_configs_requested_from_proxdash = False

            # Initialize stats
            self._stats = {
                stat_types.GlobalStatType.RUN_TIME: stat_types.RunStats(),
                stat_types.GlobalStatType.SINCE_CONNECT: stat_types.RunStats()
            }
            self.set_property_internal_state_value('stats', None)  # Stats not in state

            # Initialize managers
            self._init_default_model_cache_manager()
            self._init_model_configs()

            self.handle_changes(initial_state, self.get_state())

    # === Abstract Method Implementations ===

    def get_internal_state_property_name(self) -> str:
        return _PROXAI_CLIENT_STATE_PROPERTY

    def get_internal_state_type(self):
        return types.ProxAIClientState

    def handle_changes(
        self,
        old_state: types.ProxAIClientState,
        current_state: types.ProxAIClientState
    ):
        """Handle state changes and validate configuration.

        This method is called after state changes to:
        1. Validate the new state
        2. Apply derived changes
        3. Raise errors for invalid configurations
        """
        result_state = copy.deepcopy(old_state)

        # Merge changes from current_state
        for field in [
            'run_type', 'hidden_run_key', 'experiment_path', 'root_logging_path',
            'logging_options', 'cache_options', 'proxdash_options',
            'strict_feature_test', 'suppress_provider_errors',
            'allow_multiprocessing', 'model_test_timeout',
            'model_configs_requested_from_proxdash'
        ]:
            value = getattr(current_state, field, None)
            if value is not None:
                setattr(result_state, field, value)

        # Validation
        if result_state.model_test_timeout is not None:
            if result_state.model_test_timeout < 1:
                raise ValueError('model_test_timeout must be greater than 0.')

    # === Properties ===

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
    def root_logging_path(self) -> Optional[str]:
        return self.get_property_value('root_logging_path')

    @root_logging_path.setter
    def root_logging_path(self, value: Optional[str]):
        self.set_property_value('root_logging_path', value)

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

    @property
    def strict_feature_test(self) -> bool:
        return self.get_property_value('strict_feature_test')

    @strict_feature_test.setter
    def strict_feature_test(self, value: bool):
        self.set_property_value('strict_feature_test', value)

    @property
    def suppress_provider_errors(self) -> bool:
        return self.get_property_value('suppress_provider_errors')

    @suppress_provider_errors.setter
    def suppress_provider_errors(self, value: bool):
        self.set_property_value('suppress_provider_errors', value)

    @property
    def allow_multiprocessing(self) -> bool:
        return self.get_property_value('allow_multiprocessing')

    @allow_multiprocessing.setter
    def allow_multiprocessing(self, value: bool):
        self.set_property_value('allow_multiprocessing', value)

    @property
    def model_test_timeout(self) -> int:
        return self.get_property_value('model_test_timeout')

    @model_test_timeout.setter
    def model_test_timeout(self, value: int):
        self.set_property_value('model_test_timeout', value)

    @property
    def model_configs_requested_from_proxdash(self) -> bool:
        return self.get_property_value('model_configs_requested_from_proxdash')

    @model_configs_requested_from_proxdash.setter
    def model_configs_requested_from_proxdash(self, value: bool):
        self.set_property_value('model_configs_requested_from_proxdash', value)

    # === Stats (special - not in state) ===

    @property
    def stats(self) -> Dict[stat_types.GlobalStatType, stat_types.RunStats]:
        if self._stats is None:
            self._stats = {
                stat_types.GlobalStatType.RUN_TIME: stat_types.RunStats(),
                stat_types.GlobalStatType.SINCE_CONNECT: stat_types.RunStats()
            }
        return self._stats

    # === Nested StateControlled Properties ===

    # TODO: Add properties for:
    # - model_cache_manager
    # - query_cache_manager
    # - proxdash_connection
    # - available_models
    # - model_configs
    # Each needs:
    # 1. @property getter using get_state_controlled_property_value
    # 2. @setter using set_state_controlled_property_value
    # 3. Deserializer method

    # === Public API Methods ===

    # TODO: Implement these by moving logic from proxai.py:
    # - generate_text()
    # - set_model()
    # - check_health()
    # - get_summary()
    # - get_available_models()
    # - get_current_options()
    # - reset_platform_cache()

    # === State Export/Import ===

    def export_state(self) -> types.ProxAIClientState:
        """Export current state for serialization.

        Use this to pass state to worker processes.

        Returns:
            ProxAIClientState that can be serialized and passed to workers.
        """
        return self.get_state()

    @classmethod
    def from_state(cls, state: types.ProxAIClientState) -> 'ProxAIClient':
        """Create a client from exported state.

        Use this in worker processes to restore state.

        Args:
            state: State previously exported via export_state()

        Returns:
            New ProxAIClient instance with restored state.
        """
        return cls(init_state=state)

    # === Private Helper Methods ===

    def _init_default_model_cache_manager(self):
        """Initialize the default model cache manager.

        Uses platform-specific cache directory when possible,
        falls back to temporary directory otherwise.
        """
        try:
            app_dirs = platformdirs.PlatformDirs(
                appname="proxai", appauthor="proxai")
            self._default_model_cache_path = app_dirs.user_cache_dir
            os.makedirs(self._default_model_cache_path, exist_ok=True)
            self._default_model_cache_manager = model_cache.ModelCacheManager(
                cache_options=types.CacheOptions(
                    cache_path=self._default_model_cache_path,
                    model_cache_duration=60 * 60 * 4  # 4 hours
                )
            )
            self._platform_used_for_default_model_cache = True
        except Exception:
            temp_dir = tempfile.TemporaryDirectory()
            self._default_model_cache_path = temp_dir.name
            self._default_model_cache_manager = model_cache.ModelCacheManager(
                cache_options=types.CacheOptions(
                    cache_path=self._default_model_cache_path
                )
            )
            self._platform_used_for_default_model_cache = False

    def _init_model_configs(self):
        """Initialize model configurations."""
        self._model_configs = model_configs.ModelConfigs()

    def _process_experiment_path(self, experiment_path: Optional[str]):
        """Process and validate experiment path."""
        if experiment_path is not None:
            experiment.validate_experiment_path(experiment_path)
        self.experiment_path = experiment_path

    def _process_logging_options(
        self,
        experiment_path: Optional[str],
        logging_path: Optional[str],
        logging_options: Optional[types.LoggingOptions]
    ):
        """Process logging options, handling path conflicts."""
        # Check for conflicting paths
        if (logging_path is not None and
            logging_options is not None and
            logging_options.logging_path is not None):
            raise ValueError(
                'logging_path and logging_options.logging_path are both set. '
                'Either set logging_path or logging_options.logging_path, not both.'
            )

        # Determine root logging path
        root_path = None
        if logging_path:
            root_path = logging_path
        elif logging_options and logging_options.logging_path:
            root_path = logging_options.logging_path

        self.root_logging_path = root_path

        # Build result logging options
        result = types.LoggingOptions()

        if root_path is not None:
            if not os.path.exists(root_path):
                raise ValueError(f'Root logging path does not exist: {root_path}')

            if experiment_path is not None:
                result.logging_path = os.path.join(root_path, experiment_path)
            else:
                result.logging_path = root_path

            if not os.path.exists(result.logging_path):
                os.makedirs(result.logging_path, exist_ok=True)

        if logging_options is not None:
            result.stdout = logging_options.stdout
            result.hide_sensitive_content = logging_options.hide_sensitive_content

        self.logging_options = result

    def _process_cache_options(
        self,
        cache_path: Optional[str],
        cache_options: Optional[types.CacheOptions]
    ):
        """Process cache options, handling path conflicts."""
        # Check for conflicting paths
        if (cache_path is not None and
            cache_options is not None and
            cache_options.cache_path is not None):
            raise ValueError(
                'cache_path and cache_options.cache_path are both set. '
                'Either set cache_path or cache_options.cache_path, not both.'
            )

        result = types.CacheOptions()

        if cache_path:
            result.cache_path = cache_path

        if cache_options:
            if cache_options.cache_path:
                result.cache_path = cache_options.cache_path
            result.unique_response_limit = cache_options.unique_response_limit
            result.retry_if_error_cached = cache_options.retry_if_error_cached
            result.clear_query_cache_on_connect = cache_options.clear_query_cache_on_connect
            result.disable_model_cache = cache_options.disable_model_cache
            result.clear_model_cache_on_connect = cache_options.clear_model_cache_on_connect
            result.model_cache_duration = cache_options.model_cache_duration

        self.cache_options = result

    def _post_load_init(self):
        """Initialize runtime objects after loading state.

        Called after load_state() to set up objects that aren't serialized.
        """
        self._init_model_configs()
        # Additional initialization as needed
```

---

## Phase 2: Moving Logic from proxai.py

### 2.1 Method Migration Checklist

For each method in `proxai.py`, follow this pattern:

1. Copy method to `ProxAIClient`
2. Replace `global _VAR` with `self._var`
3. Replace `_get_*()` with `self.*` property access
4. Replace `_set_*()` with property assignment
5. Update any cross-references
6. Add tests

### 2.2 Methods to Migrate

| Current Function | New Location | Notes |
|-----------------|--------------|-------|
| `_init_globals()` | `__init__` | Split into init logic |
| `_init_default_model_cache_manager()` | `_init_default_model_cache_manager()` | Already migrated |
| `_set_experiment_path()` | `_process_experiment_path()` | Already migrated |
| `_set_logging_options()` | `_process_logging_options()` | Already migrated |
| `_set_cache_options()` | `_process_cache_options()` | Already migrated |
| `_set_proxdash_options()` | Property setter | Direct assignment |
| `_get_model_connector()` | `_get_model_connector()` | TODO |
| `_get_registered_model_connector()` | `_get_registered_model_connector()` | TODO |
| `_get_model_cache_manager()` | Property | TODO |
| `_get_query_cache_manager()` | Property | TODO |
| `_get_proxdash_connection()` | Property | TODO |
| `connect()` | Module function delegates | See Phase 3 |
| `generate_text()` | `generate_text()` | TODO |
| `set_model()` | `set_model()` | TODO |
| `check_health()` | `check_health()` | TODO |
| `get_summary()` | `get_summary()` | TODO |
| `get_available_models()` | Property | TODO |
| `get_current_options()` | `get_current_options()` | TODO |
| `reset_state()` | Module function | Resets default client |
| `reset_platform_cache()` | `reset_platform_cache()` | TODO |

### 2.3 Example: Migrating generate_text()

**Before (in proxai.py):**
```python
def generate_text(
    prompt: Optional[str] = None,
    # ... params
) -> Union[str, types.LoggingRecord]:
    # ... validation ...

    if provider_model is not None:
        model_connector = _get_model_connector(
            provider_model_identifier=provider_model)
    else:
        model_connector = _get_registered_model_connector(
            call_type=types.CallType.GENERATE_TEXT)

    logging_record = model_connector.generate_text(
        prompt=prompt,
        # ... params
    )
    # ... error handling ...
    return logging_record.response_record.response
```

**After (in ProxAIClient):**
```python
def generate_text(
    self,
    prompt: Optional[str] = None,
    # ... params
) -> Union[str, types.LoggingRecord]:
    # ... validation ...

    if provider_model is not None:
        connector = self._get_model_connector(
            provider_model_identifier=provider_model)
    else:
        connector = self._get_registered_model_connector(
            call_type=types.CallType.GENERATE_TEXT)

    logging_record = connector.generate_text(
        prompt=prompt,
        # ... params
    )
    # ... error handling ...
    return logging_record.response_record.response
```

**Key Changes:**
1. Add `self` parameter
2. Replace `_get_*` with `self._get_*`
3. All global variable access becomes `self.*`

---

## Phase 3: Refactoring proxai.py

### 3.1 Final proxai.py Structure

```python
"""ProxAI - Unified AI Integration Platform.

Provides backward-compatible module-level API delegating to ProxAIClient.
"""

from typing import Any, Dict, Optional, Union
import proxai.types as types
from proxai.client import ProxAIClient
import proxai.stat_types as stat_types

# Re-export for backward compatibility
CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions

_DEFAULT_CLIENT: Optional[ProxAIClient] = None


def _get_default_client() -> ProxAIClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = ProxAIClient()
    return _DEFAULT_CLIENT


def connect(**kwargs):
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = ProxAIClient(**kwargs)


def generate_text(**kwargs):
    return _get_default_client().generate_text(**kwargs)


def set_model(**kwargs):
    _get_default_client().set_model(**kwargs)


def set_run_type(run_type: types.RunType):
    _get_default_client().run_type = run_type


def check_health(**kwargs):
    return _get_default_client().check_health(**kwargs)


def get_summary(**kwargs):
    return _get_default_client().get_summary(**kwargs)


def get_available_models():
    return _get_default_client().available_models


def get_current_options(**kwargs):
    return _get_default_client().get_current_options(**kwargs)


def reset_state():
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None:
        _DEFAULT_CLIENT.reset_platform_cache()
    _DEFAULT_CLIENT = None


def reset_platform_cache():
    _get_default_client().reset_platform_cache()


# === New Functions ===

def export_client_state() -> types.ProxAIClientState:
    """Export state for multiprocessing."""
    return _get_default_client().export_state()


def import_client_state(state: types.ProxAIClientState):
    """Import state in worker process."""
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = ProxAIClient.from_state(state)


def get_client() -> ProxAIClient:
    """Get the default client instance."""
    return _get_default_client()
```

---

## Phase 4: Testing

### 4.1 Test Categories

1. **Unit Tests for ProxAIClient**
   - State serialization round-trip
   - Property access
   - Configuration validation

2. **Integration Tests**
   - Module API backward compatibility
   - Multiprocessing state transfer
   - Multi-client isolation

3. **Migration Tests**
   - All existing tests still pass
   - No API changes detected

### 4.2 Sample Tests

```python
# tests/test_client.py

import pytest
from proxai import ProxAIClient
import proxai.types as types


class TestProxAIClientState:
    def test_state_serialization_round_trip(self):
        """State can be exported and imported."""
        client = ProxAIClient(
            cache_path="/tmp/test-cache",
            strict_feature_test=True
        )

        state = client.get_state()
        new_client = ProxAIClient(init_state=state)

        assert new_client.strict_feature_test == True
        assert new_client.cache_options.cache_path == "/tmp/test-cache"

    def test_multi_client_isolation(self):
        """Multiple clients don't interfere with each other."""
        client1 = ProxAIClient(strict_feature_test=True)
        client2 = ProxAIClient(strict_feature_test=False)

        assert client1.strict_feature_test == True
        assert client2.strict_feature_test == False

        client1.strict_feature_test = False
        assert client2.strict_feature_test == False  # Unchanged


class TestBackwardCompatibility:
    def test_module_api_unchanged(self):
        """Module-level API works as before."""
        import proxai as px

        px.connect(cache_path="/tmp/test")
        # Should not raise

        px.reset_state()
```

---

## Rollback Plan

If issues arise during migration:

1. **Keep old proxai.py**: Rename to `proxai_legacy.py`
2. **Feature flag**: Add `PROXAI_USE_LEGACY=1` env var
3. **Gradual rollout**: Test in staging before production

```python
# In proxai.py __init__
import os
if os.environ.get('PROXAI_USE_LEGACY') == '1':
    from proxai.proxai_legacy import *
else:
    from proxai.proxai_new import *
```

---

## Timeline

| Phase | Estimated Effort | Risk Level |
|-------|-----------------|------------|
| Phase 1: Foundation | 2-3 hours | Low |
| Phase 2: Move Logic | 4-6 hours | Medium |
| Phase 3: Refactor API | 1-2 hours | Low |
| Phase 4: Testing | 2-4 hours | Low |
| **Total** | **9-15 hours** | - |

---

*Last Updated: 2025-12-03*
