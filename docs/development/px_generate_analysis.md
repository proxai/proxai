# `px.generate()` — API Use Cases

User-facing companion to `call_record_analysis.md` and
`px_client_analysis.md`. Those docs cover the full `CallRecord` structure
and client configuration; this doc covers how to **call** `px.generate()`
and its convenience wrappers, what you pass in, what you get back, and
common patterns. Source of truth: `src/proxai/proxai.py` (module-level
functions), `src/proxai/client.py` (`ProxAIClient` methods), and
`src/proxai/types.py` (parameter and record types).

---

## 0. `px.generate()` at a glance

```
px.generate(                                    → CallRecord
│
│   # What to say
├── prompt: str | None                          # simple text input
├── messages: dict | list[dict] | Chat | None   # multi-turn conversation
├── system_prompt: str | None                   # system instruction (prompt-only)
│
│   # Which model
├── provider_model: tuple[str, str] | ProviderModelType | None
│                                               # ("openai", "gpt-4o")
│
│   # How to generate
├── parameters: ParameterType | None
│   ├── temperature: float | None               # 0.0–2.0
│   ├── max_tokens: int | None                  # output length cap
│   ├── stop: str | list[str] | None            # stop sequences
│   ├── n: int | None                           # number of choices
│   └── thinking: ThinkingType | None           # LOW | MEDIUM | HIGH
│
│   # What tools to use
├── tools: list[Tools] | None                   # [Tools.WEB_SEARCH]
│
│   # What format to return
├── response_format: str | type[BaseModel] | ResponseFormat | None
│   │                                           # "text", "json", MyModel,
│   │                                           # or ResponseFormat(...)
│   ├── type: ResponseFormatType                # TEXT | IMAGE | AUDIO | VIDEO
│   │                                           # | JSON | PYDANTIC | MULTI_MODAL
│   ├── pydantic_class: type[BaseModel] | None
│   ├── pydantic_class_name: str | None
│   └── pydantic_class_json_schema: dict | None
│
│   # Per-call behaviour overrides
└── connection_options: ConnectionOptions | None
    ├── fallback_models: list[ProviderModelType] | None
    ├── suppress_provider_errors: bool | None   # None = inherit client default
    ├── endpoint: str | None                    # force a specific endpoint
    ├── skip_cache: bool | None                 # bypass cache entirely
    └── override_cache_value: bool | None       # call provider, update cache
)
```

### Convenience wrappers

```
px.generate_text(...)     → str               # same args, minus response_format
px.generate_json(...)     → dict              # same args, minus response_format
px.generate_pydantic(...) → pydantic.BaseModel # same args, keeps response_format
```

On error with `suppress_provider_errors=True`: wrappers return the error
message as `str`; `generate()` returns a `CallRecord` with
`result.status == FAILED`.

---

## 1. The four generation functions

ProxAI exposes four ways to generate content. All four exist as both
module-level functions (`px.generate(...)`) and instance methods
(`client.generate(...)`).

```
px.generate()          → CallRecord        # full control, full record
px.generate_text()     → str               # text shortcut
px.generate_json()     → dict              # JSON shortcut
px.generate_pydantic() → pydantic.BaseModel # structured output shortcut
```

The three convenience wrappers (`generate_text`, `generate_json`,
`generate_pydantic`) call `generate()` internally and unwrap the result
for you. Use `generate()` when you need the full `CallRecord` (usage
stats, timestamps, cache metadata, fallback info); use the wrappers when
you just want the output value.

### 1.1 Signature overview

```python
px.generate(
    prompt: str | None = None,
    messages: dict | list[dict] | Chat | None = None,
    system_prompt: str | None = None,
    provider_model: tuple[str, str] | ProviderModelType | None = None,
    parameters: ParameterType | None = None,
    tools: list[Tools] | None = None,
    response_format: str | type[BaseModel] | ResponseFormat | None = None,
    connection_options: ConnectionOptions | None = None,
) -> CallRecord
```

`generate_text`, `generate_json` have the same signature minus
`response_format` (they set it for you). `generate_pydantic` keeps
`response_format` — that is where you pass the pydantic class.

### 1.2 Return types on error

When `suppress_provider_errors=True` and the provider fails:

| Function | Returns on error |
|----------|-----------------|
| `generate()` | `CallRecord` with `result.status == FAILED` |
| `generate_text()` | The error message as `str` |
| `generate_json()` | The error message as `str` |
| `generate_pydantic()` | The error message as `str` |

When `suppress_provider_errors=False` (default), all four functions
**raise** the provider exception directly.

---

## 2. Input parameters

### 2.1 `prompt` — simple text input

A plain string. The most common input for single-turn requests.

```python
rec = px.generate(prompt="What is the capital of France?")
```

Cannot be combined with `messages`.

### 2.2 `messages` — multi-turn conversations

Accepts three forms:

```python
# Form 1: Single dict (one message)
px.generate(messages={"role": "user", "content": "Hello"})

# Form 2: List of dicts (multi-turn)
px.generate(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "What about 2+3?"},
])

# Form 3: Chat object (from px.Chat)
chat = px.Chat(system_prompt="You are a math tutor.")
chat.add_message(role="user", content="What is 2+2?")
px.generate(messages=chat)
```

Cannot be combined with `prompt` or `system_prompt` (use a `"system"`
role message inside `messages` instead).

### 2.3 `system_prompt` — system instruction

Sets the model's behaviour context. Only works with `prompt`, not with
`messages`.

```python
px.generate(
    prompt="Translate to French: Hello",
    system_prompt="You are a professional translator.",
)
```

When using `messages`, include the system prompt as a system-role
message in the messages list.

### 2.4 `provider_model` — which model to use

Accepts a `(provider, model)` tuple or a `ProviderModelType` instance.

```python
# Tuple form (most common)
px.generate(prompt="Hello", provider_model=("openai", "gpt-4o"))

# ProviderModelType form (when you need the exact identifier)
px.generate(
    prompt="Hello",
    provider_model=px.ProviderModelType(
        provider="anthropic",
        model="claude-3-5-sonnet",
        provider_model_identifier="claude-3-5-sonnet-20241022",
    ),
)
```

If omitted, the client uses whatever was set via `px.set_model()`, or
falls back to automatic model selection from the default priority list
(see `px_client_analysis.md` §5.5).

### 2.5 `parameters` — generation controls

```python
px.generate(
    prompt="Write a haiku about winter.",
    parameters=px.ParameterType(
        temperature=0.2,      # 0.0–2.0, lower = more deterministic
        max_tokens=50,        # cap on output length
        stop=["\n\n", "---"], # stop sequences
        n=3,                  # number of choices (see §4.4)
        thinking=px.ThinkingType.HIGH,  # reasoning mode (see §4.5)
    ),
)
```

All fields are optional. Unsupported parameters on a given model are
handled according to `feature_mapping_strategy` (BEST_EFFORT silently
adapts, STRICT raises).

### 2.6 `tools` — external tool access

Currently supports web search:

```python
rec = px.generate(
    prompt="What are the latest developments in fusion energy?",
    tools=[px.Tools.WEB_SEARCH],
)
# Citations available at:
#   rec.result.tool_usage.web_search_citations  → list[str]
#   rec.result.content  → includes TOOL blocks with rich Citation objects
```

### 2.7 `response_format` — output format control

Three accepted forms:

```python
# String: "text" or "json"
px.generate(prompt="...", response_format="json")

# Pydantic class directly
px.generate(prompt="...", response_format=MovieReview)

# ResponseFormat object (full control)
px.generate(
    prompt="...",
    response_format=px.ResponseFormat(
        type=px.ResponseFormatType.PYDANTIC,
        pydantic_class=MovieReview,
    ),
)
```

The convenience wrappers set this for you:
- `generate_text()` → `ResponseFormatType.TEXT`
- `generate_json()` → `ResponseFormatType.JSON`
- `generate_pydantic(response_format=MyModel)` →
  `ResponseFormatType.PYDANTIC`

### 2.8 `connection_options` — per-call behaviour

Override client defaults for a single call. See `px_client_analysis.md`
§3 for full details.

```python
px.generate(
    prompt="...",
    connection_options=px.ConnectionOptions(
        fallback_models=[("anthropic", "claude-3-5-sonnet")],
        skip_cache=True,
        override_cache_value=True,
        suppress_provider_errors=True,
        endpoint="responses.create",
    ),
)
```

---

## 3. Reading the result

`px.generate()` returns a `CallRecord`. Here is a quick-reference for
the fields you will use most often.

### 3.1 Getting the output

```python
rec = px.generate(prompt="What is 2+2?")

# Text output
rec.result.output_text          # → "4"

# Full content (always a list of MessageContent blocks)
rec.result.content              # → [MessageContent(type=TEXT, text="4")]

# JSON output (when response_format is JSON)
rec.result.output_json          # → {"answer": 4}

# Pydantic output (when response_format is a pydantic class)
rec.result.output_pydantic      # → MyModel(answer=4)
```

### 3.2 Checking success or failure

```python
from proxai.types import ResultStatusType

if rec.result.status == ResultStatusType.SUCCESS:
    print(rec.result.output_text)
else:
    print("Error:", rec.result.error)
```

Only relevant when `suppress_provider_errors=True`. Otherwise failures
raise exceptions and you never see a `FAILED` status.

### 3.3 Usage and cost

```python
rec.result.usage.input_tokens    # → 12
rec.result.usage.output_tokens   # → 8
rec.result.usage.total_tokens    # → 20
rec.result.usage.estimated_cost  # → 1500  (micro-dollars: $0.0015)
```

### 3.4 Timing

```python
rec.result.timestamp.response_time        # → timedelta(seconds=1.2)
rec.result.timestamp.cache_response_time  # → timedelta(ms=5) or None
```

### 3.5 Cache and connection info

```python
from proxai.types import ResultSource

rec.connection.result_source     # → CACHE or PROVIDER
rec.connection.endpoint_used     # → "chat.completions.create" or None
rec.connection.cache_look_fail_reason  # → why cache missed (or None)
```

### 3.6 Which model actually answered

```python
rec.query.provider_model.provider  # → "anthropic"
rec.query.provider_model.model     # → "claude-3-5-sonnet"
```

Especially useful with fallback chains — the returned model may differ
from the one you requested.

---

## 4. Common patterns

### 4.1 Quick text generation

```python
import proxai as px

# One-liner (auto-selects a working model)
text = px.generate_text(prompt="Explain quantum computing in one sentence.")

# With a specific model
text = px.generate_text(
    prompt="Explain quantum computing in one sentence.",
    provider_model=("openai", "gpt-4o"),
)
```

### 4.2 JSON output

```python
data = px.generate_json(
    prompt="List 3 European capitals as a JSON array.",
    provider_model=("openai", "gpt-4o"),
)
# data → {"capitals": ["Paris", "Berlin", "Rome"]}
```

### 4.3 Structured output with Pydantic

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

review = px.generate_pydantic(
    prompt="Review the movie Inception.",
    response_format=MovieReview,
    provider_model=("openai", "gpt-4o"),
)
print(review.title)    # → "Inception"
print(review.rating)   # → 9.2
```

### 4.4 Multiple choices (`n > 1`)

```python
rec = px.generate(
    prompt="Give me a creative name for a cat.",
    parameters=px.ParameterType(n=3, temperature=0.9),
)

# First choice
print(rec.result.output_text)            # → "Whiskers McFluffington"

# Remaining choices
for choice in rec.result.choices:
    print(choice.output_text)            # → "Sir Purrs-a-Lot", "Luna Moonbeam"
```

The first choice is always in `result.content` / `result.output_text`.
Additional choices are in `result.choices[0..n-2]`, each with its own
`output_text`, `output_json`, `output_pydantic`, etc.

### 4.5 Thinking / reasoning

```python
rec = px.generate(
    prompt="What is 127 * 389?",
    parameters=px.ParameterType(thinking=px.ThinkingType.HIGH),
    provider_model=("openai", "o1"),
)

# The final answer
print(rec.result.output_text)  # → "49,403"

# The reasoning trace (if you need it)
for block in rec.result.content:
    if block.type == px.ContentType.THINKING:
        print("Thinking:", block.text)
```

Thinking blocks appear before the answer in `result.content`. The
`output_text` shortcut skips them and gives you just the final answer.

### 4.6 Multi-turn conversation

```python
rec = px.generate(
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What about 2+3?"},
    ],
    provider_model=("anthropic", "claude-3-5-sonnet"),
)
print(rec.result.output_text)  # → "2 + 3 = 5"
```

### 4.7 Web search

```python
rec = px.generate(
    prompt="What happened in tech news today?",
    tools=[px.Tools.WEB_SEARCH],
)

print(rec.result.output_text)

# Flat URL list
print(rec.result.tool_usage.web_search_citations)

# Rich citations (title + URL)
for block in rec.result.content:
    if block.type == px.ContentType.TOOL:
        for citation in block.tool_content.citations:
            print(f"  {citation.title}: {citation.url}")
```

### 4.8 Fallback chain

Try the primary model; if it fails, fall back to alternatives in order.

```python
rec = px.generate(
    prompt="Explain quantum entanglement.",
    provider_model=("openai", "gpt-4"),
    connection_options=px.ConnectionOptions(
        fallback_models=[
            ("anthropic", "claude-3-5-sonnet"),
            ("google", "gemini-1.5-pro"),
        ],
    ),
)

# Which model actually answered?
print(rec.query.provider_model)

# Which models failed before it?
if rec.connection.failed_fallback_models:
    print("Failed models:", rec.connection.failed_fallback_models)
```

You always get exactly one `CallRecord` back, regardless of how many
models were tried.

### 4.9 Skipping or refreshing the cache

```python
# Skip cache entirely (no read, no write)
rec = px.generate(
    prompt="...",
    connection_options=px.ConnectionOptions(skip_cache=True),
)

# Force-refresh: ignore cached value, call provider, update cache
rec = px.generate(
    prompt="...",
    connection_options=px.ConnectionOptions(override_cache_value=True),
)
```

### 4.10 Suppressing errors for a single call

```python
rec = px.generate(
    prompt="Ping",
    connection_options=px.ConnectionOptions(suppress_provider_errors=True),
)
if rec.result.status == ResultStatusType.FAILED:
    print("Provider error:", rec.result.error)
else:
    print(rec.result.output_text)
```

### 4.11 Image generation

```python
rec = px.generate(
    prompt="Generate an image of a sunset over mountains.",
    provider_model=("openai", "dall-e-3"),
    response_format=px.ResponseFormat(type=px.ResponseFormatType.IMAGE),
)
# Image URL or data in content
image_block = rec.result.content[0]
print(image_block.source)  # → URL string
```

### 4.12 Multi-modal input (image in chat)

```python
from proxai.chat.message_content import MessageContent, ContentType

rec = px.generate(
    messages=[{
        "role": "user",
        "content": [
            MessageContent(type=ContentType.TEXT, text="What's in this image?"),
            MessageContent(
                type=ContentType.IMAGE,
                source="https://example.com/photo.jpg",
                media_type="image/jpeg",
            ),
        ],
    }],
    provider_model=("openai", "gpt-4o"),
)
print(rec.result.output_text)
```

### 4.13 Using the full `CallRecord` for logging

```python
rec = px.generate(prompt="Hello world")

log_entry = {
    "model": f"{rec.query.provider_model.provider}/{rec.query.provider_model.model}",
    "status": rec.result.status.value,
    "tokens": rec.result.usage.total_tokens,
    "cost_usd": rec.result.usage.estimated_cost / 1_000_000,
    "latency_s": rec.result.timestamp.response_time.total_seconds(),
    "cached": rec.connection.result_source.value == "CACHE",
    "endpoint": rec.connection.endpoint_used,
}
```

---

## 5. Validation errors

These are raised synchronously before any provider call. They are
programmer errors and are never routed through `suppress_provider_errors`.

| Trigger | Error |
|---------|-------|
| Both `prompt` and `messages` set | `prompt and messages cannot be used together` |
| Both `system_prompt` and `messages` set | `system_prompt and messages cannot be used together...` |
| `fallback_models` + `suppress_provider_errors` on same `ConnectionOptions` | `suppress_provider_errors and fallback_models cannot be used together` |
| `fallback_models` + `endpoint` on same `ConnectionOptions` | `endpoint and fallback_models cannot be used together` |
| `override_cache_value=True` but no query cache configured | `override_cache_value is True but query cache is not configured` |
| Forced endpoint not supported by model | `endpoint <name> is not supported` |
| No compatible endpoint for the request | `No compatible endpoint found for the query record...` |

---

## 6. Module-level vs client instance

Both calling styles support the same functions with the same signatures:

```python
# Module-level (uses hidden default client)
import proxai as px
px.connect(cache_options=px.CacheOptions(cache_path="/tmp/cache"))
px.set_model(provider_model=("openai", "gpt-4o"))
rec = px.generate(prompt="Hello")

# Client instance (independent client)
client = px.Client(
    cache_options=px.CacheOptions(cache_path="/tmp/cache"),
)
client.set_model(provider_model=("openai", "gpt-4o"))
rec = client.generate(prompt="Hello")
```

These are fully isolated — two caches, two log files, two ProxDash
connections. See `px_client_analysis.md` §5.1 for details.
