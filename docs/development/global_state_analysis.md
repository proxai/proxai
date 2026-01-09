# Global State Management Analysis

## Overview

This document provides a comprehensive analysis of how major Python libraries handle global state management, with recommendations for ProxAI's architecture evolution.

---

## Industry Patterns Comparison

### Summary Table

| Library | Pattern | Thread Safety | Multiprocessing | API Style |
|---------|---------|--------------|-----------------|-----------|
| **TensorFlow** | Context singleton | Partial (immutable after init) | Re-init on fork | `tf.config.set_*()` |
| **PyTorch** | Thread-local + locks | Yes (double-check locks) | Fork detection | `torch.cuda.device()` context |
| **JAX** | Functional/immutable | Perfect (no shared state) | N/A | Pure functions |
| **Pandas** | Hierarchical dict | NO | None | `pd.set_option()` |
| **Requests** | Explicit Session | NO (per-thread session) | Fresh per process | `session.get()` |
| **SQLAlchemy** | Factory pattern | NO (session per request) | Session per fork | `Session()` per unit |
| **OpenAI** | Dual mode | NO | None | `openai.api_key` OR `OpenAI()` |
| **Anthropic** | Explicit client only | NO | None | `Anthropic()` only |
| **LangChain** | Request-scoped config | Improved (v0.0.153+) | None | `chain.with_config()` |

---

## Detailed Library Analysis

### 1. TensorFlow/Keras

**Pattern: Context-Based Centralized State**

TensorFlow uses a `context.context()` object as the single source of truth for all runtime configuration.

**Key Characteristics:**
- **Immutability After Initialization**: Once TensorFlow's runtime initializes, attempting to modify configuration raises `RuntimeError`
- **Getter/Setter Pairs**: Functions like `get_visible_devices()` and `set_visible_devices()` operate on the global context
- **Lazy Initialization**: Configuration can be set before runtime initialization; some operations trigger initialization
- **Categories**: Threading (intra/inter-op), device management, execution control, determinism

**Code Pattern:**
```python
# Module level
_context = None

def get_context():
    global _context
    if _context is None:
        _context = _Context()
    return _context

tf.config.set_visible_devices([gpu0, gpu1])  # Modifies global context
devices = tf.config.get_visible_devices()     # Reads from global context
```

**Thread Safety**: NOT explicitly thread-safe for concurrent configuration changes; relies on immutability post-initialization

**Serialization/Multiprocessing**: Handled via lazy re-initialization in forked processes

---

### 2. PyTorch

**Pattern: Thread-Safe Lazy Initialization with Device Guards**

PyTorch employs careful initialization locking and thread-local storage for CUDA state.

**Key Characteristics:**
- **Lazy Initialization**: `_lazy_init()` defers CUDA setup until first use
- **Thread Safety**: `_initialization_lock` ensures only one thread initializes CUDA
- **Fork Detection**: `_is_in_bad_fork()` prevents reinitialization in forked subprocesses
- **Context Managers**: `device()` and `stream()` context managers for temporary state changes
- **Thread-Local Storage**: `_tls` for per-thread state like `is_initializing`

**Code Pattern:**
```python
_initialized: bool = False
_tls = threading.local()
_initialization_lock = threading.Lock()

def _lazy_init():
    global _initialized
    with _initialization_lock:
        if not _initialized and not _is_in_bad_fork():
            # Initialize CUDA
            _initialized = True

# Context manager for temporary device selection
with torch.cuda.device(1):
    model = model.cuda()  # Uses device 1 in this block
```

**Multiprocessing**: Explicitly handles fork detection to avoid state corruption

---

### 3. JAX

**Pattern: Functional Approach with Explicit State Management**

JAX avoids global state by design, making all state explicit and immutable.

**Key Characteristics:**
- **Pure Functions Required**: All functions passed to JAX transformations must be pure
- **No Hidden Global State**: JAX explicitly rejects global state
- **PRNG Keys as State**: Random state is encapsulated in immutable key objects
- **Configuration Options**: `jax.config` for system-level settings only
- **Pytree Approach**: State propagated through function inputs/outputs

**Philosophy Quote**:
> "JAX transformations like jit(), vmap(), grad() require the functions they wrap to be pure: functions whose outputs depend solely on the inputs, and which have no side effects such as updating global state."

**Code Pattern:**
```python
# State is NOT mutated; functional approach
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)  # Returns new key, doesn't mutate
result = jax.random.normal(subkey)   # Pure function
```

**Thread Safety**: Perfect; no shared state means no threading issues

---

### 4. Pandas

**Pattern: Hierarchical Dictionary with Validation and Callbacks**

Pandas uses nested dictionaries with three-tier organization.

**Key Characteristics:**
- **Three-Tier System**:
  - `_global_config`: Current option values
  - `_registered_options`: Metadata (defaults, validators, callbacks)
  - `_deprecated_options`: Migration paths
- **Dot Notation**: `pd.set_option("display.max_columns", 10)`
- **Validators**: Optional validation functions before setting values
- **Callbacks**: Trigger on option changes
- **Context Manager**: `pd.option_context()` for temporary scoped changes

**Code Pattern:**
```python
# Register an option
register_option("display.max_columns",
                default_value=80,
                validator=validate_int,
                callback=on_option_change)

# Context manager for scoped changes
with pd.option_context('display.max_rows', 10):
    print(large_df)  # Uses max_rows=10
    # Automatically restores when exiting
```

**Thread Safety**: NO - Documentation warns: "Pandas is not 100% thread safe"

---

### 5. NumPy

**Pattern: Module-Level Configuration via Context Managers**

**Code Pattern:**
```python
# Global setting
np.set_printoptions(precision=2, suppress=True)

# Context manager for scoped changes
with np.printoptions(precision=3):
    print(arr)  # Uses precision=3
    # Restores on exit
```

**Thread Safety**: Limited; relies on GIL

---

### 6. Requests/HTTPX

**Pattern: Explicit Session Objects with Connection Pooling**

Requests avoids global state for connections, using explicit Session objects.

**Key Characteristics:**
- **Session as Explicit Object**: Users create `requests.Session()` to maintain state
- **No Global Singleton**: Module-level functions create temporary sessions
- **Connection Pooling**: Each Session maintains HTTPAdapter pools
- **Keep-Alive Automatic**: 100% automatic within a session

**Code Pattern:**
```python
# Bad: Creates new session for each request (no pooling)
requests.get("http://example.com")
requests.get("http://example.com")  # New connection

# Good: Reuses connections within session
session = requests.Session()
session.get("http://example.com")
session.get("http://example.com")  # Reuses connection
```

**Thread Safety**: NO - Sessions are not thread-safe; use separate sessions per thread

---

### 7. SQLAlchemy

**Pattern: Factory Pattern with Explicit Scope Management**

SQLAlchemy uses the `sessionmaker` factory pattern with explicit lifecycle management.

**Key Characteristics:**
- **Sessionmaker Factory**: Single module-level factory creates Session instances
- **NOT a Global Registry**: Session is NOT a global singleton
- **Session Per Unit of Work**: Create session for each logical operation
- **Context Manager**: `with Session(engine) as session:` handles cleanup

**Code Pattern:**
```python
# Create factory at module level
engine = create_engine("postgresql://...")
SessionLocal = sessionmaker(bind=engine)

# Web application: session per request
def handle_request():
    with SessionLocal() as session:
        user = session.query(User).first()
        return user
```

**Thread Safety**: Explicit rule - "An instance of Session cannot be shared among concurrent threads"

**Anti-Patterns**:
- Don't embed session management in data access functions
- Don't use Session as global registry
- Don't share Session instances across threads

---

### 8. OpenAI Python SDK

**Pattern: Dual Mode - Explicit Objects + Module-Level Globals**

**Key Characteristics:**
- **Module-Level Globals**: `api_key`, `organization`, `base_url`, etc.
- **Dynamic Client Loading**: `_load_client()` creates appropriate client
- **Lazy Initialization**: Client created only when first needed
- **Environment Variable Primary**: `OPENAI_API_KEY`

**Code Pattern:**
```python
# Option 1: Environment variable (simple)
client = OpenAI()  # Uses OPENAI_API_KEY env var

# Option 2: Explicit client (production)
client = OpenAI(api_key="sk-...", timeout=30.0)

# Option 3: Custom configuration
custom_client = OpenAI(
    base_url="https://api.example.com",
    api_key="custom-key",
    default_headers={"X-Custom": "value"}
)
```

---

### 9. Anthropic Python SDK

**Pattern: Explicit Client Objects with Environment Variable Fallback**

**Key Characteristics:**
- **Environment Variable Primary**: `ANTHROPIC_API_KEY`
- **Constructor Arguments**: Explicit configuration in client creation
- **No Module-Level Globals**: Unlike OpenAI, doesn't expose settable module variables
- **Timeout Configuration**: Global or per-request
- **Automatic Retries**: 2 times for connection errors, 408, 429, 5xx

**Code Pattern:**
```python
from anthropic import Anthropic

# Environment variable (recommended)
client = Anthropic()  # Uses ANTHROPIC_API_KEY

# Explicit configuration
client = Anthropic(
    api_key="sk-ant-...",
    timeout=60.0,
    max_retries=3
)
```

---

### 10. LangChain

**Pattern: Callback Manager with Request-Scoped Configuration**

**Key Characteristics:**
- **CallbackManager**: Central object for callbacks
- **with_config()**: Binds configuration that propagates to children
- **Callback Modes**: Inheritable vs Local
- **Breaking Change in 0.0.153+**: Removed global SharedCallbackManager

**Code Pattern:**
```python
# With_config pattern for scoped callbacks
configured_chain = chain.with_config(
    callbacks=[StreamingStdOutCallbackHandler()],
    tags=["important"],
    metadata={"user_id": "123"}
)
result = configured_chain.invoke({"input": "Hello"})
```

**Philosophy**: Moving away from global shared state toward explicit, request-scoped configuration

---

## Key Industry Insights

### 1. The Industry is Moving Away from Module-Level Globals

Modern SDKs prefer explicit client objects:

```python
# OLD pattern (deprecated)
import openai
openai.api_key = "..."
openai.ChatCompletion.create(...)

# NEW pattern (modern)
from openai import OpenAI
client = OpenAI(api_key="...")
client.chat.completions.create(...)
```

### 2. Dual-Mode is a Good Transition Strategy

OpenAI maintains both patterns for backward compatibility:
- Simple scripts: Use environment variables
- Production: Use explicit client objects

### 3. Thread Safety is Often NOT Provided

Most libraries document that they are NOT thread-safe and recommend:
- One client/session per thread
- Explicit instance creation per worker

### 4. Serialization for Multiprocessing is Rare

Most libraries handle multiprocessing via:
- Fresh initialization in child processes
- Fork detection (PyTorch)
- Documentation recommending process-local state

### 5. Context Managers are Universal

Every major library provides context managers for:
- Temporary configuration changes
- Resource cleanup
- Scoped state modifications

---

## Sources

- [Stateful computations — JAX documentation](https://docs.jax.dev/en/latest/stateful-computations.html)
- [JAX Common Gotchas](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [Session Basics — SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- [Advanced Usage — Requests 2.32.5 documentation](https://requests.readthedocs.io/en/latest/user/advanced/)
- [GitHub - openai/openai-python](https://github.com/openai/openai-python)
- [How to attach callbacks to a runnable — LangChain](https://python.langchain.com/docs/how_to/callbacks_attach/)
- [Frequently Asked Questions (FAQ) — pandas documentation](https://pandas.pydata.org/docs/dev/user_guide/gotchas.html)

---

*Last Updated: 2025-12-03*
