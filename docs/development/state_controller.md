# StateControlled System - Developer Guide

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Creating a New StateControlled Class](#creating-a-new-statecontrolled-class)
4. [Corner Cases and Critical Considerations](#corner-cases-and-critical-considerations)
5. [Existing Implementations](#existing-implementations)
6. [State Change Handling Patterns](#state-change-handling-patterns)
7. [Known Issues and Bugs](#known-issues-and-bugs)

---

## Overview

### Purpose and Intentions

The `StateControlled` system is a sophisticated state management framework designed to:

1. **Centralize State Management**: Keep all object state in a single, serializable container
2. **Enable Serialization**: Allow objects to be easily saved, transferred across processes/threads, or cached
3. **Track State Changes**: Monitor and react to state modifications through the `handle_changes` callback
4. **Support Dynamic Properties**: Properties can be set directly or via getter functions that compute values on-the-fly
5. **Handle Nested State**: Manage hierarchical state relationships where one StateControlled object contains another

### Key Benefits

- **Multiprocessing**: Objects can be serialized and sent to other processes
- **Caching**: State can be saved and restored from disk
- **Validation**: State changes can be validated in `handle_changes` before taking effect
- **Flexibility**: Properties support both literal values and dynamic getter functions
- **Type Safety**: Strong typing through dataclasses ensures state integrity

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    StateControlled Class                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐        ┌──────────────────┐           │
│  │ Internal Storage │        │  Internal State  │           │
│  │                  │        │                  │           │
│  │ _property_name   │◄──────►│ _state.property  │           │
│  │ _get_property    │        │                  │           │
│  └──────────────────┘        └──────────────────┘           │
│           │                            │                    │
│           ▼                            ▼                    │
│  ┌──────────────────┐        ┌──────────────────┐           │
│  │  Property        │        │  Serialization   │           │
│  │  Getter/Setter   │        │  get_state()     │           │
│  └──────────────────┘        │  load_state()    │           │
│                              └──────────────────┘           │
│                                        │                    │
│                                        ▼                    │
│                               ┌──────────────────┐          │
│                               │ handle_changes() │          │
│                               │  (Validation)    │          │
│                               └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. StateContainer

**Definition**: An abstract base class that marks dataclasses as valid state containers.

**Location**: `src/proxai/types.py`

```python
class StateContainer(ABC):
    """Base class for all state objects in the system."""
    pass
```

**Purpose**: Acts as a marker interface for type checking and ensures all state objects are dataclasses.

**Example State Types**:
```python
@dataclasses.dataclass
class MyState(StateContainer):
    property_1: Optional[str] = None
    property_2: Optional[int] = None
    nested_state: Optional[OtherState] = None
```

### 2. Internal State Property

**Naming Pattern**: `_<state_name>` (e.g., `_state`, `_provider_model_state`)

**Purpose**: Stores the current state snapshot as a StateContainer instance.

**Key Points**:
- Must be defined by `get_internal_state_property_name()`
- Automatically updated when properties change
- Used for serialization via `get_state()`

### 3. Property Storage

Each property has up to **three** associated attributes:

#### a. Internal Value Storage
**Pattern**: `_<property_name>` (e.g., `_logging_options`)

**Purpose**: Stores the literal value when set directly.

#### b. Internal Getter Function
**Pattern**: `_get_<property_name>` (e.g., `_get_logging_options`)

**Purpose**: Stores a function that dynamically computes the property value.

#### c. Public Property
**Pattern**: `<property_name>` (e.g., `logging_options`)

**Purpose**: Python property (with `@property` decorator) that users interact with.

### 4. Property Access Methods

#### `get_property_value(property_name)`
- Retrieves the value from internal storage OR getter function
- Updates internal state automatically
- **Triggers the property getter** (important for side effects)

#### `set_property_value(property_name, value)`
- Sets the internal value
- **Calls the property getter** to get the final value
- Updates internal state
- Most common setter method

#### `get_property_internal_state_value(property_name)`
- Gets the value directly from internal state container (`_state.property_name`)
- **Does NOT** call getter functions or property getters
- Returns what's actually stored in the serializable state
- Useful for checking what will be serialized via `get_state()`

#### `set_property_internal_state_value(property_name, value)`
- Sets the value directly in internal state container (`_state.property_name`)
- **Does NOT** call property getters/setters or update `_<property_name>` storage
- **Important**: Deep copies non-primitive values to prevent external mutations
- Used for internal state synchronization
- Performance note: Deep copying can be expensive for large objects

#### `get_state_controlled_property_value(property_name)`
- Special getter for nested StateControlled objects
- Extracts the state from nested object: `nested_obj.get_state()`
- Stores extracted state in internal state

#### `set_state_controlled_property_value(property_name, value)`
- Special setter for nested StateControlled objects
- Accepts: StateControlled object OR StateContainer
- If StateContainer, calls deserializer: `<property_name>_deserializer(state)`
- Updates internal state with nested state

#### `set_property_value_without_triggering_getters(property_name, value)`
- Sets both internal value AND internal state
- **Does NOT call property getter**
- Used during initialization and state loading
- Avoids unwanted side effects

### 5. Deserializer Methods

**Pattern**: `<property_name>_deserializer(state_container) -> StateControlled`

**Purpose**: Convert a StateContainer back into a StateControlled object.

**Example**:
```python
def query_cache_manager_deserializer(
    self,
    state_value: types.QueryCacheManagerState
) -> query_cache.QueryCacheManager:
    return query_cache.QueryCacheManager(init_state=state_value)
```

**When Required**: For every property that holds a nested StateControlled object.

### 6. Abstract Methods

All subclasses must implement:

#### `get_internal_state_property_name(self) -> str`
Returns the name of the internal state attribute (e.g., `'_state'`).

#### `get_internal_state_type(self) -> Type[StateContainer]`
Returns the StateContainer dataclass type for this class.

#### `handle_changes(self, old_state, current_state) -> None`
Called whenever state changes. Used for:
- Validation
- Computing derived values
- Triggering side effects
- Enforcing invariants

---

## Creating a New StateControlled Class

### What's Automatic (Handled by Parent Class)

The `StateControlled` base class automatically handles these common initialization tasks:

✅ **Exclusivity Validation**: Ensures `init_state` isn't mixed with other parameters \
✅ **Property Name Validation**: Verifies all kwargs match fields in your state type \
✅ **State Initialization**: Calls `init_state()` to set up the state structure

### Step 1: Define the State Container

**File**: `src/proxai/types.py`

```python
@dataclasses.dataclass
class MyClassState(StateContainer):
    # Simple properties
    name: Optional[str] = None
    count: Optional[int] = 0

    # Complex properties (dicts, lists, etc.)
    config: Optional[Dict[str, Any]] = None

    # Nested StateControlled objects (store as StateContainer)
    cache_manager: Optional[QueryCacheManagerState] = None
```

**Key Points**:
- Use `Optional[Type]` for all fields
- Provide default values (usually `None` or meaningful defaults)
- For nested StateControlled objects, use the nested object's StateContainer type

### Step 2: Create the Class Structure

```python
import proxai.state_controllers.state_controller as state_controller
from typing import Callable, Optional

_MY_CLASS_STATE_PROPERTY = '_my_class_state'

class MyClass(state_controller.StateControlled):
    # Type hints for all internal storage
    _name: Optional[str]
    _get_name: Optional[Callable[[], str]]
    _count: Optional[int]
    _config: Optional[Dict[str, Any]]
    _cache_manager: Optional[CacheManager]  # Actual object type
    _get_cache_manager: Optional[Callable[[], CacheManager]]
    _my_class_state: MyClassState  # Internal state storage
```

### Step 3: Implement __init__

**Note**: The parent `StateControlled.__init__()` automatically handles:
- ✅ Exclusivity validation (ensures `init_state` isn't mixed with other params)
- ✅ Property name validation (ensures all kwargs are valid properties)
- ✅ Calling `self.init_state()` to initialize the state structure

This eliminates boilerplate and ensures consistency across all StateControlled classes.

```python
def __init__(
    self,
    name: Optional[str] = None,
    get_name: Optional[Callable[[], str]] = None,
    count: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None,
    get_cache_manager: Optional[Callable[[], CacheManager]] = None,
    init_state: Optional[MyClassState] = None):

    # Call parent __init__ with all property parameters
    # Parent automatically validates exclusivity and property names
    super().__init__(
        name=name,
        get_name=get_name,
        count=count,
        cache_manager=cache_manager,
        get_cache_manager=get_cache_manager,
        init_state=init_state)

    # Branch 1: Loading from serialized state
    if init_state:
        self.load_state(init_state)
    # Branch 2: Fresh initialization
    else:
        initial_state = self.get_state()

        # Set internal getter functions
        self._get_name = get_name
        self._get_cache_manager = get_cache_manager

        # Set properties (triggers setters)
        self.name = name
        self.count = count
        self.cache_manager = cache_manager

        # Call handle_changes to validate initial state
        self.handle_changes(initial_state, self.get_state())
```

**What You Need to Implement**:
1. Call `super().__init__()` with all parameters including `init_state`
2. Branch on `if init_state:` to handle state loading vs fresh initialization
3. In the `if` branch: Call `load_state()` (and any extra setup if needed)
4. In the `else` branch: Set getter functions, set properties, call `handle_changes()`

### Step 4: Implement Required Abstract Methods

```python
def get_internal_state_property_name(self):
    return _MY_CLASS_STATE_PROPERTY

def get_internal_state_type(self):
    return MyClassState

def handle_changes(
    self,
    old_state: MyClassState,
    current_state: MyClassState):
    """Validate state changes and enforce invariants."""

    # Build result_state by merging old and new
    result_state = copy.deepcopy(old_state)
    if current_state.name is not None:
        result_state.name = current_state.name
    if current_state.count is not None:
        result_state.count = current_state.count

    # Validate: Example - name is required
    if result_state.name is None:
        raise ValueError('Name must be set')

    # Validate: Example - count must be positive
    if result_state.count < 0:
        raise ValueError('Count must be non-negative')

    # Apply computed/derived state changes if needed
    # (See examples in existing implementations)
```

### Step 5: Implement Property Getters and Setters

#### Simple Properties

```python
@property
def name(self) -> str:
    return self.get_property_value('name')

@name.setter
def name(self, value: str):
    self.set_property_value('name', value)

@property
def count(self) -> int:
    return self.get_property_value('count')

@count.setter
def count(self, value: int):
    self.set_property_value('count', value)
```

#### Nested StateControlled Properties

```python
@property
def cache_manager(self) -> CacheManager:
    return self.get_state_controlled_property_value('cache_manager')

@cache_manager.setter
def cache_manager(self, value: CacheManager):
    self.set_state_controlled_property_value('cache_manager', value)

# REQUIRED: Deserializer for nested objects
def cache_manager_deserializer(
    self,
    state_value: CacheManagerState
) -> CacheManager:
    return CacheManager(init_state=state_value)
```

#### Custom Getter Logic

Sometimes you need custom logic in the getter:

```python
@property
def experiment_path(self) -> str:
    # Custom retrieval logic
    internal_value = self.get_property_internal_value('experiment_path')
    internal_getter = self.get_property_func_getter('experiment_path')

    result = None
    if internal_value is not None and internal_value != '(not set)':
        result = internal_value
    elif internal_getter is not None:
        result = internal_getter()

    if result is None:
        result = '(not set)'

    # Update internal state
    self.set_property_internal_state_value('experiment_path', result)
    return result
```

### Step 6: Handle Special Cases in Setters

```python
@property
def status(self) -> StatusEnum:
    return self.get_property_value('status')

@status.setter
def status(self, value: StatusEnum):
    self.set_property_value('status', value)

    # Trigger side effects based on status
    if value == StatusEnum.CONNECTED:
        logging.info('Connected successfully')
    elif value == StatusEnum.ERROR:
        logging.error('Connection failed')
```

---

## Corner Cases and Critical Considerations

### 1. Deep Copying Non-Primitive Types

**Issue**: Non-primitive values (dicts, lists, custom objects) are deep-copied when stored in internal state.

**Location**: `state_controller.py:171-181`

```python
def set_property_internal_state_value(self, property_name: str, value: Any):
    if value is None:
        copied_value = None
    elif isinstance(value, (str, int, float, bool, type(None))):
        copied_value = value  # Primitives are safe
    else:
        copied_value = copy.deepcopy(value)  # Deep copy everything else
    setattr(
        getattr(self, self.get_internal_state_property_name()),
        property_name,
        copied_value)
```

**Why**: Prevents external modifications from affecting internal state.

**Performance Impact**: Deep copying can be expensive for large objects.

**Best Practice**: Keep state objects small and focused. Avoid storing large data structures directly in state.

### 2. Property Getter Priority

**Rule**: Internal literal value (`_property`) takes priority over getter function (`_get_property`).

**Location**: `state_controller.py:133-149`

```python
def get_property_internal_value_or_from_getter(self, property_name: str) -> Any:
    literal_value = self.get_property_internal_value(property_name)
    if literal_value is not None:
        return literal_value  # Literal wins

    func = self.get_property_func_getter(property_name)
    if func is not None:
        return func()  # Getter is fallback

    return None
```

**Implications**:
- Once you set a literal value, the getter function is ignored
- Setting a property to `None` does NOT revert to the getter function
- To switch back to the getter, you must explicitly set `_property = None`

**Example**:
```python
obj = MyClass(get_count=lambda: 42)
print(obj.count)  # 42 (from getter)
obj.count = 100
print(obj.count)  # 100 (literal wins)
obj._count = None
print(obj.count)  # 42 (back to getter)
```

### 3. Initialization Parameter Conflicts

**Rule**: Cannot set both `property` and `get_property` in `__init__`.

**Validation**: `state_controller.py:72-78`

```python
if (property_name in kwargs and
    kwargs[property_name] is not None and
    property_getter_name in kwargs and
    kwargs[property_getter_name] is not None):
    raise ValueError(
        f'Only one of {property_name} or {property_getter_name} '
        'should be set while initializing.')
```

**Why**: Ambiguous which value should take priority.

**Best Practice**: Choose one initialization pattern per property instance.

### 4. The `init_state` Exclusivity

**Rule**: If `init_state` is provided, ALL other parameters must be `None`.

**Why**: State loading should be atomic—either restore complete state OR initialize fresh.

**Enforcement**: ✅ **Automatically handled by parent class** (`state_controller.py:47-53`)

```python
class StateControlled:
    def __init__(self, init_state=None, **kwargs):
        # Automatic validation - no need to implement in child classes
        if init_state is not None:
            for key, value in kwargs.items():
                if value is not None:
                    raise ValueError(
                        f'init_state and other parameters cannot be set at the same time. '
                        f'Found non-None parameter: {key}={value}')
```

**What This Means**:
- ✅ You don't need to write this validation in child classes
- ✅ All StateControlled classes get consistent error messages
- ✅ Eliminates 5-15 lines of boilerplate per class

**Best Practice**: Simply pass all parameters including `init_state` to `super().__init__()` and the parent handles validation.

### 5. State Validation in handle_changes

**Critical Pattern**: Always build a `result_state` by merging old and new states.

**Why**: `current_state` might be partial (only changed fields). You need complete state for validation.

**Example** (`model_connector.py:127-159`):
```python
def handle_changes(self, old_state, current_state):
    result_state = copy.deepcopy(old_state)

    # Merge changes
    if current_state.provider_model is not None:
        result_state.provider_model = current_state.provider_model
    if current_state.logging_options is not None:
        result_state.logging_options = current_state.logging_options

    # Validate complete state
    if result_state.provider_model is None:
        raise ValueError('Provider model must be set')
    if result_state.logging_options is None:
        raise ValueError('Logging options must be set')
```

**Anti-pattern** ❌:
```python
def handle_changes(self, old_state, current_state):
    # Don't validate current_state directly—it might be partial!
    if current_state.provider_model is None:
        raise ValueError('Provider model must be set')
```

### 6. Setting Properties Without Triggering Getters

**Use Case**: During initialization and state loading, you want to set state without side effects.

**Method**: `set_property_value_without_triggering_getters(property_name, value)`

**Example** (`state_controller.py:258-263`):
```python
def init_state(self):
    setattr(self, self.get_internal_state_property_name(),
            self.get_internal_state_type()())

    for field in dataclasses.fields(self.get_internal_state_type()):
        # Don't trigger getters during init
        self.set_property_value_without_triggering_getters(
            field.name, field.default)
```

**When to Use**:
- During `init_state()`
- During `load_state()`
- When you want to bypass property getter logic

**When NOT to Use**:
- During normal operation
- When you need validation/side effects to run

### 7. Nested StateControlled Objects

**Challenge**: Nested objects need special handling for serialization.

**Solution**: Three-part pattern:

1. **State type references nested state**:
```python
@dataclasses.dataclass
class ParentState(StateContainer):
    nested: Optional[NestedState] = None  # StateContainer type
```

2. **Use specialized getter/setter**:
```python
@property
def nested(self):
    return self.get_state_controlled_property_value('nested')

@nested.setter
def nested(self, value):
    self.set_state_controlled_property_value('nested', value)
```

3. **Implement deserializer**:
```python
def nested_deserializer(self, state_value: NestedState) -> NestedObject:
    return NestedObject(init_state=state_value)
```

**What Happens**:
- Getter extracts state: `nested_obj.get_state()` → stores in internal state
- Setter accepts both `NestedObject` OR `NestedState`
- If `NestedState` provided, calls deserializer to reconstruct object

### 8. Circular Dependencies

**Problem**: Two StateControlled classes reference each other.

**Example**:
```python
class A(StateControlled):
    b: Optional[B]  # Needs B

class B(StateControlled):
    a: Optional[A]  # Needs A
```

**Solution**: Use `Optional[StateContainer]` for state types, not actual classes.

```python
@dataclasses.dataclass
class AState(StateContainer):
    b: Optional[BState] = None  # Forward reference OK

@dataclasses.dataclass
class BState(StateContainer):
    a: Optional[AState] = None  # Forward reference OK
```

**Best Practice**: Keep state types in `types.py` separate from implementation.

### 9. External State Changes

**Concept**: Properties computed by getter functions can change externally.

**Detection**: `get_external_state_changes()` compares current getter values with internal state.

**Example** (`test_state_controller.py:411-437`):
```python
dynamic_value = 'value_1'
obj = MyClass(get_property=lambda: dynamic_value)

# State is initially 'value_1'
assert obj.get_internal_state().property == 'value_1'

# External change
dynamic_value = 'value_2'

# Detect the change
changes = obj.get_external_state_changes()
assert changes.property == 'value_2'

# Apply the changes
obj.apply_external_state_changes()
```

**When to Use**:
- When getter functions depend on external state
- Before critical operations where state must be fresh
- In long-running processes where external state changes

### 10. Status Properties with Side Effects

**Pattern**: Status changes often trigger logging or other side effects.

**Example** (`proxdash.py:307-357`):
```python
@status.setter
def status(self, status: ProxDashConnectionStatus):
    self.set_property_value('status', status)

    # Log different messages based on status
    if status == ProxDashConnectionStatus.CONNECTED:
        logging_utils.log_proxdash_message(...)
    elif status == ProxDashConnectionStatus.DISABLED:
        logging_utils.log_proxdash_message(...)
```

**Best Practice**: Keep side effects in setters, not in `handle_changes`, when they're status-specific.

---

## Existing Implementations

### Summary Table

| Class | State Type | Nested StateControlled | Key Features |
|-------|-----------|------------------------|--------------|
| `AvailableModels` | `AvailableModelsState` | `model_cache_manager`, `proxdash_connection` | Model availability tracking, multiprocessing |
| `ProviderModelConnector` | `ProviderModelState` | `query_cache_manager`, `proxdash_connection` | Provider API integration, caching |
| `ProxDashConnection` | `ProxDashConnectionState` | None | API key validation, experiment tracking |
| `QueryCacheManager` | `QueryCacheManagerState` | None | Shard-based cache, LRU eviction |
| `ModelCacheManager` | `ModelCacheManagerState` | None | Model status caching, TTL support |

### 1. AvailableModels

**File**: `src/proxai/connections/available_models.py`

**Purpose**: Manages available AI models across providers, tests model connectivity.

#### State Definition
```python
@dataclasses.dataclass
class AvailableModelsState(StateContainer):
    run_type: Optional[RunType] = None
    model_cache_manager: Optional[ModelCacheManagerState] = None
    logging_options: Optional[LoggingOptions] = None
    proxdash_connection: Optional[ProxDashConnectionState] = None
    allow_multiprocessing: Optional[bool] = None
    model_test_timeout: Optional[int] = None
    providers_with_key: Optional[Set[str]] = None
    has_fetched_all_models: Optional[bool] = None
    latest_model_cache_path_used_for_update: Optional[str] = None
```

#### Dependencies
- **Nested Objects**: `model_cache_manager`, `proxdash_connection`
- **Deserializers**:
  - `model_cache_manager_deserializer` (implicit, handled by parent)
  - `proxdash_connection_deserializer` (implicit)

#### State Change Handling
**Location**: `available_models.py:127-131`

```python
def handle_changes(
    self,
    old_state: AvailableModelsState,
    current_state: AvailableModelsState):
    pass  # No validation currently
```

**Analysis**: Empty implementation—no validation. This is acceptable when all properties are optional and have no constraints.

#### Key Properties

**Simple Properties**:
- `run_type`: Execution mode (production/test)
- `allow_multiprocessing`: Whether to use multiprocessing for model tests
- `model_test_timeout`: Timeout for model connectivity tests
- `providers_with_key`: Set of providers with valid API keys
- `latest_model_cache_path_used_for_update`: Last cache path used

**Complex Properties**:
- `logging_options`: Configuration for logging
- `model_cache_manager`: Nested StateControlled for caching model status
- `proxdash_connection`: Nested StateControlled for ProxDash integration

#### Special Patterns

**1. Provider Key Loading** (`available_models.py:133-142`):
```python
def _load_provider_keys(self):
    self.providers_with_key = set()
    for provider, provider_key_name in PROVIDER_KEY_MAP.items():
        provider_flag = True
        for key_name in provider_key_name:
            if key_name not in os.environ:
                provider_flag = False
                break
        if provider_flag:
            self.providers_with_key.add(provider)
```
Called in `handle_changes` to update available providers based on environment.

**2. Getter Function Pattern** (`available_models.py:99-119`):
```python
self._get_run_type = get_run_type
self._get_model_cache_manager = get_model_cache_manager
self._get_logging_options = get_logging_options
self._get_proxdash_connection = get_proxdash_connection
self._get_allow_multiprocessing = get_allow_multiprocessing
self._get_model_test_timeout = get_model_test_timeout
self._get_model_connector = get_model_connector
```
Extensive use of getter functions for dynamic property resolution.

### 2. ProviderModelConnector

**File**: `src/proxai/connectors/model_connector.py`

**Purpose**: Base class for provider-specific AI model connectors.

#### State Definition
```python
@dataclasses.dataclass
class ProviderModelState(StateContainer):
    provider_model: Optional[ProviderModelType] = None
    run_type: Optional[RunType] = None
    strict_feature_test: Optional[bool] = None
    query_cache_manager: Optional[QueryCacheManagerState] = None
    logging_options: Optional[LoggingOptions] = None
    proxdash_connection: Optional[ProxDashConnectionState] = None
```

#### Dependencies
- **Nested Objects**: `query_cache_manager`, `proxdash_connection`
- **Deserializers**:
  - `query_cache_manager_deserializer` (`model_connector.py:205-209`)
  - `proxdash_connection_deserializer` (`model_connector.py:227-231`)

#### State Change Handling
**Location**: `model_connector.py:123-159`

```python
def handle_changes(
    self,
    old_state: ProviderModelState,
    current_state: ProviderModelState):
    # Build complete result state
    result_state = copy.deepcopy(old_state)
    if current_state.provider_model is not None:
        result_state.provider_model = current_state.provider_model
    if current_state.run_type is not None:
        result_state.run_type = current_state.run_type
    if current_state.strict_feature_test is not None:
        result_state.strict_feature_test = current_state.strict_feature_test
    if current_state.logging_options is not None:
        result_state.logging_options = current_state.logging_options
    if current_state.proxdash_connection is not None:
        result_state.proxdash_connection = current_state.proxdash_connection

    # Validate required fields
    if result_state.provider_model is None:
        raise ValueError(
            'Provider model is not set for both old and new states. '
            'This creates an invalid state change.')

    if result_state.provider_model.provider != self.get_provider_name():
        raise ValueError(
            'Provider needs to be same with the class provider name.')

    if result_state.logging_options is None:
        raise ValueError('Logging options are not set...')
    if result_state.proxdash_connection is None:
        raise ValueError('ProxDash connection is not set...')
```

**Analysis**: **Strong validation**—ensures required fields and provider consistency.

#### Key Properties

**Simple Properties**:
- `provider_model`: Which AI model to use
- `run_type`: Production or test mode
- `strict_feature_test`: Whether to raise errors on unsupported features

**Nested StateControlled Properties**:
- `query_cache_manager`: Manages response caching
- `proxdash_connection`: Handles ProxDash uploads
- `logging_options`: Logging configuration

**Special Property** - `api` (`model_connector.py:160-171`):
```python
@property
def api(self):
    if not getattr(self, '_api', None):
        if self.run_type == types.RunType.PRODUCTION:
            self._api = self.init_model()
        else:
            self._api = self.init_mock_model()
    return self._api

@api.setter
def api(self, value):
    raise ValueError('api should not be set directly.')
```

**Analysis**: Lazy-loaded API client, not part of state (not serializable). Smart pattern for heavy resources.

#### Important Methods

**Feature Checking** (`model_connector.py:251-269`):
```python
def feature_check(self, query_record: QueryRecord) -> QueryRecord:
    # Checks if model supports requested features
    # Modifies query_record to remove unsupported features
    # Logs warnings or raises errors based on strict_feature_test
```

**Stats Tracking** (`model_connector.py:301-351`):
```python
def _update_stats(self, logging_record: LoggingRecord):
    # Updates internal statistics
    # NOT part of StateControlled—stats are runtime only
```

### 3. ProxDashConnection

**File**: `src/proxai/connections/proxdash.py`

**Purpose**: Manages connection to ProxDash logging service, validates API keys.

#### State Definition
```python
@dataclasses.dataclass
class ProxDashConnectionState(StateContainer):
    status: Optional[ProxDashConnectionStatus] = None
    hidden_run_key: Optional[str] = None
    experiment_path: Optional[str] = None
    logging_options: Optional[LoggingOptions] = None
    proxdash_options: Optional[ProxDashOptions] = None
    key_info_from_proxdash: Optional[Dict] = None
    connected_experiment_path: Optional[str] = None
```

#### Dependencies
- **Nested Objects**: None
- **External Services**: ProxDash API (REST calls)

#### State Change Handling
**Location**: `proxdash.py:82-207`

**Complexity**: **Most complex** `handle_changes` in the codebase.

**Flow**:
```
1. Check if ProxDash is disabled → set status DISABLED
2. Check if API key exists → set status API_KEY_NOT_FOUND
3. Detect if API key changed → validate API key
4. Validate API key with ProxDash API
5. Set status based on validation result
6. Raise errors for invalid states
7. Update connected_experiment_path
```

**Key Logic** (`proxdash.py:110-119`):
```python
if result_state.proxdash_options.disable_proxdash:
    self.status = ProxDashConnectionStatus.DISABLED
    self.key_info_from_proxdash = None
    # IMPORTANT: Use set_property_value_without_triggering_getters
    # to avoid logging a disconnect when we're just disabling
    self.set_property_value_without_triggering_getters(
        'connected_experiment_path', None)
    return
```

**Analysis**: Uses `set_property_value_without_triggering_getters` to avoid side effects when changing state internally.

**API Key Validation** (`proxdash.py:157-162`):
```python
if api_key_query_required:
    validation_status, key_info = self._check_api_key_validity(
        base_url=result_state.proxdash_options.base_url,
        api_key=result_state.proxdash_options.api_key)
    result_state.status = validation_status
    result_state.key_info_from_proxdash = key_info
```

**Analysis**: Synchronous API call in `handle_changes`—blocking but necessary for validation.

#### Key Properties

**Custom Getter Pattern** - `experiment_path` (`proxdash.py:240-268`):
```python
@property
def experiment_path(self) -> str:
    internal_experiment_path = self.get_property_internal_value(
        'experiment_path')
    internal_get_experiment_path = self.get_property_func_getter(
        'experiment_path')

    experiment_path = None
    if (internal_experiment_path is not None and
        internal_experiment_path != _NOT_SET_EXPERIMENT_PATH_VALUE):
        experiment_path = internal_experiment_path
    elif internal_get_experiment_path is not None:
        experiment_path = internal_get_experiment_path()

    if experiment_path is None:
        experiment_path = _NOT_SET_EXPERIMENT_PATH_VALUE

    self.set_property_internal_state_value(
        'experiment_path', experiment_path)
    return experiment_path
```

**Analysis**: Custom logic to handle "not set" sentinel value. Good pattern for optional properties with special defaults.

**Status with Side Effects** (`proxdash.py:310-357`):
```python
@status.setter
def status(self, status: ProxDashConnectionStatus):
    self.set_property_value('status', status)
    # Log different messages for each status
    if status == ProxDashConnectionStatus.INITIALIZING:
        logging_utils.log_proxdash_message(...)
    elif status == ProxDashConnectionStatus.DISABLED:
        logging_utils.log_proxdash_message(...)
    # ... etc for each status
```

**Analysis**: Status changes are highly visible—logged to user immediately.

### 4. QueryCacheManager

**File**: `src/proxai/caching/query_cache.py`

**Purpose**: Manages disk-based caching of AI model responses with sharding and LRU eviction.

#### State Definition
```python
@dataclasses.dataclass
class QueryCacheManagerState(StateContainer):
    status: Optional[QueryCacheManagerStatus] = None
    cache_options: Optional[CacheOptions] = None
    shard_count: Optional[int] = 800
    response_per_file: Optional[int] = 200
    cache_response_size: Optional[int] = 40000
```

#### Dependencies
- **Nested Objects**: None
- **Internal Managers**: `ShardManager`, `HeapManager` (not StateControlled)

#### State Change Handling
**Location**: `query_cache.py:484-512`

```python
def handle_changes(
    self,
    old_state: QueryCacheManagerState,
    current_state: QueryCacheManagerState):
    # Validate cache options exist
    if current_state.cache_options is None:
        self.status = QueryCacheManagerStatus.CACHE_OPTIONS_NOT_FOUND
        return

    # Validate cache path exists
    if current_state.cache_options.cache_path is None:
        self.status = QueryCacheManagerStatus.CACHE_PATH_NOT_FOUND
        return

    # Validate cache path is writable
    if not os.access(current_state.cache_options.cache_path, os.W_OK):
        self.status = QueryCacheManagerStatus.CACHE_PATH_NOT_WRITABLE
        return

    # Initialize directory if cache path changed
    if (old_state.cache_options is None or
        old_state.cache_options.cache_path !=
        current_state.cache_options.cache_path):
        self._init_dir()

    # Reinitialize managers if any relevant option changed
    if (old_state.cache_options != current_state.cache_options or
        old_state.shard_count != current_state.shard_count or
        old_state.response_per_file != current_state.response_per_file or
        old_state.cache_response_size != current_state.cache_response_size):
        self._init_managers()

    self.status = QueryCacheManagerStatus.WORKING
```

**Analysis**: Progressive validation with early returns. Sets status at each failure point.

#### Key Patterns

**Status-Driven Initialization** (`query_cache.py:458-476`):
```python
def __init__(self, ...):
    self.init_state()
    self.set_property_value(
        'status', types.QueryCacheManagerStatus.INITIALIZING)

    if init_state:
        self.load_state(init_state)
        self._init_dir()  # Must init after loading state
        self._init_managers()
    else:
        initial_state = self.get_state()
        self._get_cache_options = get_cache_options
        self.cache_options = cache_options
        # ... set other properties
        self.handle_changes(initial_state, self.get_state())
```

**Analysis**: When loading from `init_state`, managers must be initialized explicitly since `handle_changes` isn't called.

**Computed Property** - `cache_path` (`query_cache.py:94-102`):
```python
@property
def cache_path(self) -> str:
    if self.cache_options is None or self.cache_options.cache_path is None:
        return None
    return os.path.join(self.cache_options.cache_path, CACHE_DIR)

@cache_path.setter
def cache_path(self, value: str):
    raise ValueError('cache_path needs to be set through cache_options.')
```

**Analysis**: Read-only derived property. Prevents direct setting—forces changes through `cache_options`.

### 5. ModelCacheManager

**File**: `src/proxai/caching/model_cache.py`

**Purpose**: Caches model availability status to avoid repeated connectivity tests.

#### State Definition
```python
@dataclasses.dataclass
class ModelCacheManagerState(StateContainer):
    status: Optional[ModelCacheManagerStatus] = None
    cache_options: Optional[CacheOptions] = None
```

#### Dependencies
- **Nested Objects**: None
- **Internal State**: `model_status_by_call_type` (not in state—runtime only)

#### State Change Handling
**Location**: `model_cache.py:53-92`

```python
def handle_changes(
    self,
    old_state: ModelCacheManagerState,
    current_state: ModelCacheManagerState):

    # Validate cache options
    if current_state.cache_options is None:
        self.status = ModelCacheManagerStatus.CACHE_OPTIONS_NOT_FOUND
        self.model_status_by_call_type = None
        return

    # Check if caching is disabled
    if current_state.cache_options.disable_model_cache == True:
        self.status = ModelCacheManagerStatus.DISABLED
        self.model_status_by_call_type = None
        return

    # Validate cache path
    if current_state.cache_options.cache_path is None:
        self.status = ModelCacheManagerStatus.CACHE_PATH_NOT_FOUND
        self.model_status_by_call_type = None
        return

    # Validate writable
    if not os.access(current_state.cache_options.cache_path, os.W_OK):
        self.status = ModelCacheManagerStatus.CACHE_PATH_NOT_WRITABLE
        self.model_status_by_call_type = None
        return

    # No change? Skip reload
    if current_state == old_state:
        return

    # Reload cache if relevant options changed
    if (old_state.cache_options is None or
        old_state.cache_options.cache_path !=
        current_state.cache_options.cache_path or
        old_state.cache_options.disable_model_cache !=
        current_state.cache_options.disable_model_cache or
        old_state.cache_options.clear_model_cache_on_connect !=
        current_state.cache_options.clear_model_cache_on_connect or
        old_state.cache_options.model_cache_duration !=
        current_state.cache_options.model_cache_duration):
        self._load_from_cache_path()

    self.status = ModelCacheManagerStatus.WORKING
```

**Analysis**: Most thorough validation chain. Clears internal state on failure.

#### Important Patterns

**Runtime-Only State** (`model_cache.py:120-128`):
```python
@property
def model_status_by_call_type(self) -> ModelStatusByCallType:
    if getattr(self, '_model_status_by_call_type', None) is None:
        self._model_status_by_call_type = {}
    return self._model_status_by_call_type

@model_status_by_call_type.setter
def model_status_by_call_type(self, value: ModelStatusByCallType):
    self._model_status_by_call_type = value
```

**Analysis**: `model_status_by_call_type` is NOT in StateContainer—it's large and changes frequently. Only cached to disk, not serialized in state.

**Read-Only Computed Property** (`model_cache.py:94-102`):
```python
@property
def cache_path(self) -> str:
    if self.cache_options is None or self.cache_options.cache_path is None:
        return None
    return os.path.join(self.cache_options.cache_path, AVAILABLE_MODELS_PATH)

@cache_path.setter
def cache_path(self, value: str):
    raise ValueError('cache_path needs to be set through cache_options.')
```

**Analysis**: Derived property pattern—prevents direct setting.

---

## State Change Handling Patterns

### Pattern 1: No Validation (Empty Handler)

**When to Use**: All properties are optional with no constraints.

**Example**: `AvailableModels`

```python
def handle_changes(self, old_state, current_state):
    pass
```

**Pros**: Simple, no overhead.

**Cons**: No protection against invalid states.

### Pattern 2: Progressive Validation with Early Returns

**When to Use**: Multiple validation steps, each with different failure modes.

**Example**: `QueryCacheManager`, `ModelCacheManager`

```python
def handle_changes(self, old_state, current_state):
    if current_state.cache_options is None:
        self.status = Status.OPTIONS_NOT_FOUND
        return  # Stop here

    if current_state.cache_options.cache_path is None:
        self.status = Status.PATH_NOT_FOUND
        return  # Stop here

    if not os.access(current_state.cache_options.cache_path, os.W_OK):
        self.status = Status.PATH_NOT_WRITABLE
        return  # Stop here

    # All validations passed
    self._init_managers()
    self.status = Status.WORKING
```

**Pros**: Clear failure points, status reflects exact problem.

**Cons**: State might be partially updated.

### Pattern 3: Merge-Then-Validate

**When to Use**: Need complete state for validation, properties are interdependent.

**Example**: `ProviderModelConnector`

```python
def handle_changes(self, old_state, current_state):
    # Build complete state
    result_state = copy.deepcopy(old_state)
    if current_state.provider_model is not None:
        result_state.provider_model = current_state.provider_model
    if current_state.logging_options is not None:
        result_state.logging_options = current_state.logging_options

    # Validate complete state
    if result_state.provider_model is None:
        raise ValueError('Provider model required')
    if result_state.logging_options is None:
        raise ValueError('Logging options required')

    # Apply final state (if needed)
```

**Pros**: Validates complete state, prevents partial updates.

**Cons**: More verbose, requires all properties to be considered.

### Pattern 4: State-Driven Side Effects

**When to Use**: State changes trigger external effects (API calls, logging, file I/O).

**Example**: `ProxDashConnection`

```python
def handle_changes(self, old_state, current_state):
    result_state = self._build_result_state(old_state, current_state)

    # Check if API key changed
    if old_state.api_key != result_state.api_key:
        # Side effect: API call
        status, key_info = self._check_api_key_validity(
            result_state.api_key)
        result_state.status = status
        result_state.key_info = key_info

    # Validate final state
    if result_state.status == Status.INVALID:
        raise ValueError('Invalid API key')

    # Apply state (triggers more side effects via setters)
    self.status = result_state.status
    self.key_info = result_state.key_info
```

**Pros**: Centralized side effect logic, atomic state changes.

**Cons**: Can be slow (blocking API calls), complex error handling.

### Pattern 5: Incremental State Updates

**When to Use**: Some state changes require expensive operations (file I/O, initialization).

**Example**: `QueryCacheManager`

```python
def handle_changes(self, old_state, current_state):
    # ... validation ...

    # Only reinitialize if cache path changed
    if (old_state.cache_options is None or
        old_state.cache_options.cache_path !=
        current_state.cache_options.cache_path):
        self._init_dir()  # Expensive: file system operation

    # Only reload if relevant options changed
    if (old_state.cache_options != current_state.cache_options or
        old_state.shard_count != current_state.shard_count or
        ...):
        self._init_managers()  # Expensive: reload cache

    self.status = Status.WORKING
```

**Pros**: Avoids unnecessary work, optimized performance.

**Cons**: Requires careful change detection.

### Pattern 6: Status-Based Control Flow

**When to Use**: Object behavior changes based on state status.

**Example**: All cache managers

```python
def handle_changes(self, old_state, current_state):
    # Set status based on validation
    if not self._validate():
        self.status = Status.ERROR
        return
    self.status = Status.READY

# Later, in methods:
def do_work(self):
    if self.status != Status.READY:
        raise ValueError(f'Cannot work, status is {self.status}')
    # ... do work ...
```

**Pros**: Clear operational state, easy to debug.

**Cons**: Status must be kept in sync.

---

## Known Issues and Bugs

### 1. Potential Issue: handle_changes Not Called on load_state

**Location**: `state_controller.py:294-308`

**Code**:
```python
def load_state(self, state: Any):
    if type(state) != self.get_internal_state_type():
        raise ValueError('Invalid state type')

    for field in dataclasses.fields(self.get_internal_state_type()):
        value = getattr(state, field.name, None)
        if value is None:
            continue
        if isinstance(value, types.StateContainer):
            value = getattr(
                self,
                self.get_state_controlled_deserializer_name(field.name))(value)
        self.set_property_value_without_triggering_getters(field.name, value)
    # Note: handle_changes is NOT called here!
```

**Issue**: `load_state` doesn't call `handle_changes`, so validation/side effects don't run.

**Current Workaround**: Classes call `handle_changes` manually after `load_state` in `__init__`.

**Example**: `proxdash.py:62-74`
```python
if init_state:
    self.load_state(init_state)
    # Caller must trigger handle_changes manually if needed
```

**Recommendation**: Consider adding optional `trigger_handle_changes` parameter to `load_state`:
```python
def load_state(self, state: Any, trigger_handle_changes: bool = False):
    old_state = self.get_internal_state()
    # ... load state ...
    if trigger_handle_changes:
        self.handle_changes(old_state, self.get_internal_state())
```

### 2. Inconsistent Initialization Patterns

**Issue**: Some classes call `handle_changes` after `load_state`, some don't.

**Examples**:

**ProxDashConnection** (no explicit call):
```python
if init_state:
    self.load_state(init_state)
    # handle_changes NOT called
```

**ProviderModelConnector** (explicit call):
```python
if init_state:
    self.load_state(init_state)
    # No explicit handle_changes call
```

**AvailableModels** (calls indirectly):
```python
else:  # not init_state
    # ... set properties ...
    self.handle_changes(initial_state, self.get_state())
```

**Recommendation**: Standardize pattern—either:
- Always call `handle_changes` after `load_state`, OR
- Make `load_state` call it automatically with a flag

### 3. Deep Copy Performance

**Location**: `state_controller.py:176`

**Code**:
```python
else:
    copied_value = copy.deepcopy(value)
```

**Issue**: Deep copying can be slow for large objects (e.g., large dicts, lists).

**Impact**: Property setters can become bottlenecks.

**Recommendation**: Consider:
- Lazy copying (copy-on-write)
- Configurable copy depth
- Document that state objects should be kept small

### 4. Getter Functions and None Values

**Issue**: Once a property is set to `None`, the getter function is ignored.

**Example**:
```python
obj = MyClass(get_count=lambda: 42)
obj.count = None  # Sets _count to None
# Now obj.count returns None, getter is ignored forever
```

**Expected Behavior**: Unclear if this is intentional.

**Recommendation**: Document this behavior clearly, or provide a way to "reset" to getter function:
```python
def reset_property(self, property_name: str):
    """Reset property to use getter function if available."""
    self.set_property_internal_value(property_name, None)
```

### 5. apply_state_changes Doesn't Return Anything

**Location**: `state_controller.py:310-322`

**Code**:
```python
def apply_state_changes(self, changes: Optional[Any] = None):
    if changes is None:
        changes = self.get_internal_state_type()()
    # ... validate and apply ...
    self.handle_changes(old_state, self.get_internal_state())
    # No return value
```

**Issue**: Caller doesn't know if changes were applied successfully.

**Recommendation**: Consider returning status or raising specific exceptions:
```python
def apply_state_changes(self, changes: Optional[Any] = None) -> bool:
    # ...
    self.handle_changes(old_state, self.get_internal_state())
    return True  # Or False if validation failed
```

### 6. StateContainer Is Empty

**Location**: `types.py:306-308`

**Code**:
```python
class StateContainer(ABC):
    """Base class for all state objects in the system."""
    pass
```

**Issue**: No enforcement that subclasses are dataclasses.

**Recommendation**: Add runtime check or use `typing.Protocol`:
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class StateContainer(Protocol):
    """Protocol for state containers."""
    __dataclass_fields__: dict  # Required by dataclasses
```

Or add validation in `StateControlled.__init__`:
```python
def __init__(self, **kwargs):
    state_type = self.get_internal_state_type()
    if not dataclasses.is_dataclass(state_type):
        raise TypeError(f'{state_type} must be a dataclass')
```

### 7. No Validation in Parent __init__

**Location**: `state_controller.py:46-79`

**Issue**: `StateControlled.__init__` validates property names but doesn't validate values.

**Example**:
```python
# This passes validation even if provider_model is wrong type
super().__init__(provider_model="invalid_type")
```

**Recommendation**: Add optional type checking:
```python
if value is not None and field.type != type(value):
    raise TypeError(f'Expected {field.type}, got {type(value)}')
```

### 8. Circular Reference Not Prevented

**Issue**: StateControlled objects can create circular references:
```python
a = ClassA()
b = ClassB()
a.nested_b = b
b.nested_a = a
# Now serialization might fail or recurse infinitely
```

**Recommendation**: Add cycle detection in `get_state`:
```python
def get_state(self, _visited=None) -> Any:
    if _visited is None:
        _visited = set()
    if id(self) in _visited:
        raise ValueError('Circular reference detected')
    _visited.add(id(self))
    # ... rest of method ...
```

---

## Best Practices Summary

### DO ✅

1. **Always implement handle_changes** – Even if empty, it's clearer than leaving it out.
2. **Merge old and new states** – Build complete `result_state` for validation.
3. **Use descriptive state property names** – E.g., `_provider_model_state` not `_state`.
4. **Keep state objects small** – Large objects slow down deep copying.
5. **Provide deserializers for nested objects** – Required for `StateContainer` → `StateControlled`.
6. **Call handle_changes after initialization** – Validates initial state.
7. **Use status enums** – Makes state machine explicit and debuggable.
8. **Document side effects** – Especially in handle_changes and property setters.
9. **Test serialization round-trips** – `obj → get_state() → load_state() → obj`
10. **Use set_property_value_without_triggering_getters carefully** – Only during init and load.

### DON'T ❌

1. **Don't validate current_state directly** – It might be partial; validate `result_state`.
2. **Don't forget to pass init_state to super().__init__()** – Parent needs it for validation.
3. **Don't forget type hints** – Helps with IDE support and debugging.
4. **Don't store non-serializable objects in state** – E.g., file handles, API clients, locks.
5. **Don't have side effects in property getters** – Keep getters pure; side effects go in setters or handle_changes.
6. **Don't set both property and get_property in __init__** – Parent validates and raises ValueError.
7. **Don't mutate state directly** – Always use property setters.
8. **Don't forget to deep copy in handle_changes** – `result_state = copy.deepcopy(old_state)`
9. **Don't block indefinitely in handle_changes** – Avoid long-running operations.
10. **Don't create circular references** – StateControlled objects shouldn't reference each other cyclically.

---

## Appendix: Property Naming Conventions

| Naming Pattern | Purpose | Example |
|----------------|---------|---------|
| `_<property>` | Internal value storage | `_logging_options` |
| `_get_<property>` | Internal getter function | `_get_logging_options` |
| `<property>` | Public property (with `@property`) | `logging_options` |
| `_<state_name>` | Internal state container | `_provider_model_state` |
| `<property>_deserializer` | StateContainer → StateControlled | `proxdash_connection_deserializer` |

---

## Quick Reference Card

```python
# Creating a StateControlled class (Simplified with automatic parent handling):
class MyClass(StateControlled):
    # 1. Define internal attributes
    _property: Optional[Type]
    _get_property: Optional[Callable[[], Type]]
    _state: MyState

    # 2. Implement __init__
    def __init__(self, property=None, get_property=None, init_state=None):
        # Call parent (automatically handles validation and init_state())
        super().__init__(
            property=property,
            get_property=get_property,
            init_state=init_state)

        # Branch on init_state
        if init_state:
            self.load_state(init_state)
        else:
            initial_state = self.get_state()
            self._get_property = get_property
            self.property = property
            self.handle_changes(initial_state, self.get_state())

    # 3. Implement required methods
    def get_internal_state_property_name(self):
        return '_state'

    def get_internal_state_type(self):
        return MyState

    def handle_changes(self, old_state: MyState, current_state: MyState):
        result_state = copy.deepcopy(old_state)
        # Merge changes
        if current_state.property is not None:
            result_state.property = current_state.property
        # Validate
        if result_state.property is None:
            raise ValueError('Property required')

    # 4. Define properties
    @property
    def property(self):
        return self.get_property_value('property')

    @property.setter
    def property(self, value):
        self.set_property_value('property', value)
```

**What Changed**:
- ❌ Removed manual exclusivity check (~5-10 lines)
- ❌ Removed manual `init_state()` call (~1 line)
- ✅ Parent handles everything automatically
- ✅ ~30% less boilerplate

---

**End of Documentation**

*Last Updated: 2025-11-12*

*This documentation is based on ProxAI codebase commit `360799a`*
