# `px.generate()` API Comprehensive Use Case Analysis

Source of truth: `src/proxai/proxai.py` (module-level functions),
`src/proxai/client.py` (`ProxAIClient` methods), and
`src/proxai/types.py` (parameter and record types). If this document
disagrees with those files, the files win — update this document.

This is the definitive reference for how to call `px.generate()` and
its six convenience wrappers — what you pass in, what you get back, and
the common patterns you will hit. Read this before adding a new
wrapper, a new generation parameter, or a new output format.

See also: `call_record_analysis.md` (the `CallRecord` shape and request
pipeline) and `px_client_analysis.md` (client construction and
options).

---

## 1. `px.generate()` structure (current)

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
├── output_format: str | type[BaseModel] | OutputFormat | None
│   │                                           # "text", "json", MyModel,
│   │                                           # or OutputFormat(...)
│   ├── type: OutputFormatType                  # TEXT | IMAGE | AUDIO | VIDEO
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

### 1.1 Convenience wrappers

```
px.generate_text(...)     → str              # text shortcut
px.generate_json(...)     → dict             # JSON shortcut
px.generate_pydantic(...) → pydantic.BaseModel # structured output
px.generate_image(...)    → MessageContent   # image shortcut
px.generate_audio(...)    → MessageContent   # audio shortcut
px.generate_video(...)    → MessageContent   # video shortcut
```

On error with `suppress_provider_errors=True`: wrappers return the
error message as `str`; `generate()` returns a `CallRecord` with
`result.status == FAILED`.

---

## 2. The seven generation functions

ProxAI exposes seven ways to generate content. All seven exist as both
module-level functions (`px.generate(...)`) and instance methods
(`client.generate(...)`).

```
px.generate()          → CallRecord        # full control, full record
px.generate_text()     → str               # text shortcut
px.generate_json()     → dict              # JSON shortcut
px.generate_pydantic() → pydantic.BaseModel # structured output shortcut
px.generate_image()    → MessageContent    # image shortcut (may return str on error)
px.generate_audio()    → MessageContent    # audio shortcut (may return str on error)
px.generate_video()    → MessageContent    # video shortcut (may return str on error)
```

The six convenience wrappers call `generate()` internally and unwrap
the result for you. Use `generate()` when you need the full
`CallRecord` (usage stats, timestamps, cache metadata, fallback info);
use the wrappers when you just want the output value.

### 2.1 Signature overview

```python
px.generate(
    prompt: str | None = None,
    messages: dict | list[dict] | Chat | None = None,
    system_prompt: str | None = None,
    provider_model: tuple[str, str] | ProviderModelType | None = None,
    parameters: ParameterType | None = None,
    tools: list[Tools] | None = None,
    output_format: str | type[BaseModel] | OutputFormat | None = None,
    connection_options: ConnectionOptions | None = None,
) -> CallRecord
```

All six wrappers drop `output_format` (they set it internally) —
except `generate_pydantic`, which keeps `output_format=` as the place
to pass the pydantic class. The image / audio / video wrappers each
pin their own `OutputFormatType` (IMAGE / AUDIO / VIDEO).

### 2.2 Return types on error

When `suppress_provider_errors=True` and the provider fails:

| Function | Returns on error |
|----------|-----------------|
| `generate()` | `CallRecord` with `result.status == FAILED` |
| `generate_text()` | The error message as `str` |
| `generate_json()` | The error message as `str` |
| `generate_pydantic()` | The error message as `str` |
| `generate_image()` | The error message as `str` (typed as `MessageContent \| str`) |
| `generate_audio()` | The error message as `str` (typed as `MessageContent \| str`) |
| `generate_video()` | The error message as `str` (typed as `MessageContent \| str`) |

When `suppress_provider_errors=False` (default), all seven functions
**raise** the provider exception directly.

---

## 3. Input parameters

### 3.1 `prompt` — simple text input

A plain string. The most common input for single-turn requests.

```python
rec = px.generate(prompt="What is the capital of France?")
```

Cannot be combined with `messages`.

### 3.2 `messages` — multi-turn conversations

Accepts three forms (see `type_utils.messages_param_to_chat`):

```python
# Form 1: List of role/content dicts (most common)
px.generate(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "What about 2+3?"},
])

# Form 2: Dict bundle with an optional system prompt + messages list.
# NOTE: a bare single-message dict like {"role": "user", "content": "..."}
# does NOT work here — the dict form is the bundle shape below.
px.generate(messages={
    "system": "You are a helpful assistant.",
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
    ],
})

# Form 3: Chat object (from px.Chat)
chat = px.Chat(system_prompt="You are a math tutor.")
chat.append({"role": "user", "content": "What is 2+2?"})
# Or with typed Message objects:
chat.append(px.Message(role="user", content="And 2+3?"))
px.generate(messages=chat)
```

Cannot be combined with `prompt` or `system_prompt` (use a `"system"`
role message inside the list form, or the `system` key in the bundle
form).

### 3.3 `system_prompt` — system instruction

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

### 3.4 `provider_model` — which model to use

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

### 3.5 `parameters` — generation controls

```python
px.generate(
    prompt="Write a haiku about winter.",
    parameters=px.ParameterType(
        temperature=0.2,      # 0.0–2.0, lower = more deterministic
        max_tokens=50,        # cap on output length
        stop=["\n\n", "---"], # stop sequences
        n=3,                  # number of choices (see §5.4)
        thinking=px.ThinkingType.HIGH,  # reasoning mode (see §5.5)
    ),
)
```

All fields are optional. Unsupported parameters on a given model are
handled according to `feature_mapping_strategy` (BEST_EFFORT silently
adapts, STRICT raises).

### 3.6 `tools` — external tool access

Currently supports web search:

```python
rec = px.generate(
    prompt="What are the latest developments in fusion energy?",
    tools=[px.Tools.WEB_SEARCH],
)
# Citations are surfaced as TOOL blocks inside `rec.result.content`.
# Each `MessageContent(type=TOOL, tool_content=ToolContent(citations=[
# Citation(title, url), ...]))` block carries the rich citations. See
# §5.7 for the idiomatic extraction pattern.
```

### 3.7 `output_format` — output format control

Three accepted forms (see `type_utils.output_format_param_to_output_format`):

```python
# 1. String shortcut: "text" | "json" | "image" | "audio" | "video"
px.generate(prompt="...", output_format="json")

# 2. Pydantic class directly (implies OutputFormatType.PYDANTIC)
px.generate(prompt="...", output_format=MovieReview)

# 3. OutputFormat object (full control, e.g. with schema overrides)
px.generate(
    prompt="...",
    output_format=px.OutputFormat(
        type=px.OutputFormatType.PYDANTIC,
        pydantic_class=MovieReview,
    ),
)
```

`MULTI_MODAL` has no string shortcut; use the `OutputFormat(type=...)`
form. If `output_format` is `None` / omitted, it defaults to `TEXT`.

The convenience wrappers set this for you:
- `generate_text()` → `OutputFormatType.TEXT`
- `generate_json()` → `OutputFormatType.JSON`
- `generate_pydantic(output_format=MyModel)` →
  `OutputFormatType.PYDANTIC`

### 3.8 `connection_options` — per-call behaviour

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

## 4. Reading the result

`px.generate()` returns a `CallRecord`. Here is a quick-reference for
the fields you will use most often.

### 4.1 Getting the output

```python
rec = px.generate(prompt="What is 2+2?")

# Text output
rec.result.output_text          # → "4"

# Full content (always a list of MessageContent blocks)
rec.result.content              # → [MessageContent(type=TEXT, text="4")]

# JSON output (when output_format is JSON)
rec.result.output_json          # → {"answer": 4}

# Pydantic output (when output_format is a pydantic class)
rec.result.output_pydantic      # → MyModel(answer=4)
```

### 4.2 Checking success or failure

```python
from proxai.types import ResultStatusType

if rec.result.status == ResultStatusType.SUCCESS:
    print(rec.result.output_text)
else:
    print("Error:", rec.result.error)
```

Only relevant when `suppress_provider_errors=True`. Otherwise failures
raise exceptions and you never see a `FAILED` status.

### 4.3 Usage and cost

```python
rec.result.usage.input_tokens    # → 12
rec.result.usage.output_tokens   # → 8
rec.result.usage.total_tokens    # → 20
rec.result.usage.estimated_cost  # → 1_500_000  (nano-USD: $0.0015)
#                                   see call_record_analysis.md §2.12
```

### 4.4 Timing

```python
rec.result.timestamp.response_time        # → timedelta(seconds=1.2)
#   Provider call latency. On a cache hit this is the ORIGINAL provider
#   latency that was captured at write time, not the lookup time.

rec.result.timestamp.cache_response_time  # → timedelta(ms=5) or None
#   Cache lookup latency. Populated only on cache hits
#   (result_source == CACHE); stays None on provider-path records.
```

### 4.5 Cache and connection info

```python
from proxai.types import ResultSource

rec.connection.result_source     # → CACHE or PROVIDER
rec.connection.endpoint_used     # → "chat.completions.create" or None
#                                    (None on the cache path)
rec.connection.cache_look_fail_reason
#   → CacheLookFailReason enum value or None. Survives onto the
#     returned record even when result_source == PROVIDER, so you can
#     see exactly why the cache did not serve. None when the cache was
#     bypassed entirely (skip_cache / override_cache_value) or when
#     the cache is disabled on the client.
```

### 4.6 Which model actually answered

```python
rec.query.provider_model.provider  # → "anthropic"
rec.query.provider_model.model     # → "claude-3-5-sonnet"
```

Especially useful with fallback chains — the returned model may differ
from the one you requested.

---

## 5. Common patterns

### 5.1 Quick text generation

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

### 5.2 JSON output

```python
data = px.generate_json(
    prompt="List 3 European capitals as a JSON array.",
    provider_model=("openai", "gpt-4o"),
)
# data → {"capitals": ["Paris", "Berlin", "Rome"]}
```

### 5.3 Structured output with Pydantic

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

review = px.generate_pydantic(
    prompt="Review the movie Inception.",
    output_format=MovieReview,
    provider_model=("openai", "gpt-4o"),
)
print(review.title)    # → "Inception"
print(review.rating)   # → 9.2
```

### 5.4 Multiple choices (`n > 1`)

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

### 5.5 Thinking / reasoning

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

### 5.6 Multi-turn conversation

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

### 5.7 Web search

```python
rec = px.generate(
    prompt="What happened in tech news today?",
    tools=[px.Tools.WEB_SEARCH],
)

print(rec.result.output_text)

# Citations live in TOOL blocks inside `content`. There is no flat
# URL sidecar — the TOOL block is the single source of truth.
for block in rec.result.content:
    if block.type == px.ContentType.TOOL:
        for citation in block.tool_content.citations:
            print(f"  {citation.title}: {citation.url}")

# If you just want a flat URL list, derive it:
urls = [
    c.url
    for block in rec.result.content
    if block.type == px.ContentType.TOOL and block.tool_content
    for c in (block.tool_content.citations or [])
]
```

### 5.8 Fallback chain

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

### 5.9 Skipping or refreshing the cache

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

### 5.10 Suppressing errors for a single call

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

### 5.11 Image generation

Via the `generate_image` wrapper (simplest):

```python
image_block = px.generate_image(
    prompt="Generate an image of a sunset over mountains.",
    provider_model=("openai", "dall-e-3"),
)
print(image_block.source)  # → URL string (or .data for inline bytes)
```

Or via `generate()` when you need the full `CallRecord`:

```python
rec = px.generate(
    prompt="Generate an image of a sunset over mountains.",
    provider_model=("openai", "dall-e-3"),
    output_format="image",   # or px.OutputFormat(type=px.OutputFormatType.IMAGE)
)
image_block = rec.result.output_image  # MessageContent block
print(image_block.source)
```

### 5.12 Multi-modal input (image in chat)

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

### 5.13 Using the full `CallRecord` for logging

```python
rec = px.generate(prompt="Hello world")

log_entry = {
    "model": f"{rec.query.provider_model.provider}/{rec.query.provider_model.model}",
    "status": rec.result.status.value,
    "tokens": rec.result.usage.total_tokens,
    # estimated_cost is integer nano-USD (see call_record_analysis.md §2.12).
    "cost_usd": rec.result.usage.estimated_cost / 1_000_000_000,
    "latency_s": rec.result.timestamp.response_time.total_seconds(),
    "cached": rec.connection.result_source.value == "CACHE",
    "endpoint": rec.connection.endpoint_used,
}
```

---

## 6. Validation errors

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

## 7. Module-level vs client instance

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

### 7.1 `set_model` variants

`set_model` accepts a default model per output format. The
`provider_model=` kwarg is a backward-compat alias that sets the
default TEXT model (`client.py:1528`):

```python
# Backward compat: set a single default for TEXT.
px.set_model(provider_model=("openai", "gpt-4o"))

# Per-format defaults — pick the best model for each generator.
px.set_model(
    generate_text=("openai", "gpt-4o"),
    generate_json=("openai", "gpt-4o"),
    generate_pydantic=("openai", "gpt-4o"),
    generate_image=("openai", "dall-e-3"),
    generate_audio=("openai", "tts-1"),
    generate_video=("openai", "sora-2"),
)
```

`provider_model=` and `generate_text=` cannot be set at the same time.
At least one of the seven kwargs must be provided — calling
`set_model()` with no args raises `ValueError`.
