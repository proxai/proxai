# CallRecord Comprehensive Use Case Analysis

Source of truth: `src/proxai/types.py` and
`src/proxai/connectors/provider_connector.py`. If this document disagrees with
those files, the files win — update this document.

This is the definitive reference for how a `CallRecord` is shaped, how it is
populated through the request pipeline, and which fields are authoritative
versus derived. Read this before touching the connector pipeline, the result
adapter, the query cache, or the ProxDash serializer.

---

## 1. CallRecord structure (current)

```
CallRecord
├── query: QueryRecord
│   ├── prompt: str | None
│   ├── chat: Chat | None
│   │   ├── system_prompt: str | None       # exposed as property
│   │   └── messages: list[Message]
│   │       ├── role: MessageRoleType       # USER | ASSISTANT
│   │       └── content: str | list[MessageContent]
│   ├── system_prompt: str | None
│   ├── provider_model: ProviderModelType   # top-level, NOT in connection_options
│   ├── parameters: ParameterType | None
│   │   ├── temperature: float | None
│   │   ├── max_tokens: int | None
│   │   ├── stop: str | list[str] | None
│   │   ├── n: int | None
│   │   └── thinking: ThinkingType | None   # LOW | MEDIUM | HIGH
│   ├── tools: list[Tools] | None           # Tools enum: WEB_SEARCH
│   ├── output_format: OutputFormat | None
│   │   ├── type: OutputFormatType        # TEXT | IMAGE | AUDIO | VIDEO
│   │   │                                   # | JSON | PYDANTIC | MULTI_MODAL
│   │   ├── pydantic_class: type[BaseModel] | None
│   │   ├── pydantic_class_name: str | None
│   │   └── pydantic_class_json_schema: dict | None
│   ├── connection_options: ConnectionOptions | None
│   │   ├── fallback_models: list[ProviderModelType] | None
│   │   ├── suppress_provider_errors: bool | None
│   │   ├── endpoint: str | None            # explicit endpoint override
│   │   ├── skip_cache: bool | None
│   │   └── override_cache_value: bool | None
│   └── hash_value: str | None
│
├── result: ResultRecord
│   ├── status: ResultStatusType            # SUCCESS | FAILED
│   ├── role: MessageRoleType               # ASSISTANT (always, for results)
│   │
│   │   # Canonical content (always populated on success)
│   ├── content: list[MessageContent] | None
│   ├── choices: list[ChoiceType] | None    # populated only when n > 1
│   │
│   │   # Convenience "output_*" fields — derived from `content` by the
│   │   # ResultAdapter. Treat these as read-only views into `content`;
│   │   # never write them directly from a connector executor.
│   ├── output_text: str | None
│   ├── output_image: MessageContent | None
│   ├── output_audio: MessageContent | None
│   ├── output_video: MessageContent | None
│   ├── output_json: dict | None
│   ├── output_pydantic: pydantic.BaseModel | None
│   │
│   │   # Error path
│   ├── error: str | Exception | None       # Exception pre-suppression,
│   │                                       # str once stringified
│   ├── error_traceback: str | None
│   │
│   │   # Bookkeeping
│   ├── usage: UsageType | None
│   │   ├── input_tokens: int | None
│   │   ├── output_tokens: int | None
│   │   ├── total_tokens: int | None
│   │   └── estimated_cost: int | None      # nano-USD (×1_000_000_000) — see §2.12
│   └── timestamp: TimeStampType | None
│       ├── start_utc_date: datetime
│       ├── end_utc_date: datetime
│       ├── local_time_offset_minute: int
│       ├── response_time: timedelta        # provider call latency
│       └── cache_response_time: timedelta | None
│                                           # cache lookup latency;
│                                           # set only on CACHE hits
│
├── connection: ConnectionMetadata          # (was named `cache` in old docs)
│   ├── result_source: ResultSource         # CACHE | PROVIDER
│   ├── cache_look_fail_reason: CacheLookFailReason | None
│   │                                       # set when a PROVIDER call was
│   │                                       # made because the cache could
│   │                                       # not serve (§3.10)
│   ├── endpoint_used: str | None           # set only on PROVIDER path
│   ├── failed_fallback_models: list[ProviderModelType] | None
│   └── feature_mapping_strategy: FeatureMappingStrategy | None
│
└── debug: DebugInfo | None                 # escape-hatch sidecar; never
                                            # serialized to cache or ProxDash
    └── raw_provider_response: Any | None   # only set when
                                            # debug_options.keep_raw_provider_response
```

### 1.1 `MessageContent` — the content block

All response content (and most input content) flows through
`MessageContent` blocks. Key fields by type:

| `type`                         | Required fields                                      |
|--------------------------------|------------------------------------------------------|
| `TEXT`                         | `text`                                               |
| `THINKING`                     | `text`                                               |
| `JSON`                         | `json` (`dict[str, Any]`)                            |
| `PYDANTIC_INSTANCE`            | `pydantic_content: PydanticContent`                  |
| `IMAGE` / `DOCUMENT` / `AUDIO` / `VIDEO` | one of `source` (URL), `data` (bytes), `path`; optional `media_type` |
| `TOOL`                         | `tool_content: ToolContent`                          |

`PydanticContent` carries `class_name`, `class_value` (the BaseModel
subclass), `instance_value` (the live pydantic instance), and
`instance_json_value` (the dict form used for cache round-trips).

`ToolContent` carries `name`, `kind` (`CALL` or `RESULT`), and
`citations: list[Citation]` — where each `Citation` has `title` and `url`.
This is the single source of truth for web-search citations (see §2.11).

---

## 2. Design decisions

These are implementation-level conventions that cannot be inferred from
`types.py` alone. They are enforced by the connector pipeline, the result
adapter, or runtime validation.

### 2.1 `ResultRecord.content` is always `list[MessageContent]`

Connectors must populate `content` as a list of `MessageContent` blocks —
even for plain text responses (a single `ContentType.TEXT` block). The
`content: str | ...` polymorphism only exists on input `Message.content`,
not on `ResultRecord.content`.

### 2.2 `output_*` fields are derived, not authoritative

`ResultAdapter._adapt_output_values()` is invoked once per content
container — once on `result` and once on each `ChoiceType` in
`result.choices`. It iterates `content` **forward** with no early
break, and applies the following per-block rules:

| Block type              | Effect on `output_text`                        | Effect on typed `output_*`                 |
|-------------------------|------------------------------------------------|--------------------------------------------|
| `TEXT`                  | `output_text += text` (no separator)           | —                                          |
| `IMAGE`                 | `output_text += "[image: <ref>]"`              | `output_image = block` (last wins)         |
| `AUDIO`                 | `output_text += "[audio: <ref>]"`              | `output_audio = block` (last wins)         |
| `VIDEO`                 | `output_text += "[video: <ref>]"`              | `output_video = block` (last wins)         |
| `DOCUMENT`              | `output_text += "[document: <ref>]"`           | — (no typed field on `ResultRecord`)       |
| `JSON`                  | skip                                           | `output_json = block.json` (last wins)     |
| `PYDANTIC_INSTANCE`     | skip                                           | `output_pydantic = instance` (last wins)   |
| `THINKING`              | skip                                           | —                                          |
| `TOOL`                  | skip                                           | —                                          |

`<ref>` is `source` if set, else `path`, else `"<data>"` (for inline
bytes). `output_text` stays `None` if nothing contributes to it; it
becomes `""` the first time a TEXT or media block is seen.

This shape supports mixed responses like
`[TEXT("Here: "), IMAGE(src=…), TEXT(" — good?")]`, which surface as
`output_text = "Here: [image: <src>] — good?"` with `output_image`
pointing at the block.

Consumers who need the full, correct response structure must read
`content`. Connector executors must NOT write `output_*` directly — the
adapter owns those fields. See `src/proxai/connectors/result_adapter.py`.

### 2.3 `QueryRecord.system_prompt` vs `Chat.system_prompt` — mutually exclusive

`QueryRecord.system_prompt` is set when the caller used the prompt API.
`Chat.system_prompt` (the property on the `Chat` object assigned to
`QueryRecord.chat`) is set when the caller used the messages API. Using
both at the same time is a validation error at the `generate()` boundary.

### 2.4 Structured output returns typed `MessageContent` blocks

For `OutputFormatType.JSON`, the result content is a
`MessageContent(type=JSON, json={...})` block. For
`OutputFormatType.PYDANTIC`, it is a
`MessageContent(type=PYDANTIC_INSTANCE, pydantic_content=PydanticContent(...))`
block. The `ResultAdapter` is responsible for converting from provider
text output when the endpoint reports `BEST_EFFORT` support (parses JSON
from a text block, revalidates into pydantic, etc.).

`pydantic_content.instance_json_value` is the serialization-friendly dict
used by the query cache; `pydantic_content.class_value.model_validate(
instance_json_value)` is how the instance is reconstructed on cache read.

### 2.5 Thinking content is inlined in `content`

Model reasoning/thinking appears as
`MessageContent(type=THINKING, text="...")` blocks in the same `content`
list, ordered before the final answer blocks. Some providers
(e.g. OpenAI responses endpoint) emit multiple thinking summaries; each
becomes its own block. Cache replay preserves them. `Chat.export(
omit_thinking=True)` strips them for re-submission.

### 2.6 Only pure JSON or Pydantic — no legacy "JSON schema" mode

`OutputFormat` exposes `pydantic_class` + `pydantic_class_name` +
`pydantic_class_json_schema` for pydantic mode, or nothing extra for plain
JSON mode. There is no "JSON with user-supplied schema but no pydantic
class" mode — callers who want schema-constrained output must define a
`BaseModel`.

### 2.7 Timestamps reflect the original provider call

On a cache hit, `timestamp.response_time` stays equal to the original
provider latency that was cached, and `timestamp.start_utc_date` /
`end_utc_date` are rebased to the current time (see `_get_cached_result`
in `provider_connector.py`). `timestamp.cache_response_time` records
the wall-clock duration of the cache lookup itself (measured with
`time.perf_counter`) and is set only on cache hits; on provider-path
records it stays `None`. Cost-attribution and latency metrics should
branch on `connection.result_source`.

### 2.8 Fallback chains: one record per attempt internally, one record returned to the caller

At the connector layer, each attempt against a different model is an
independent `CallRecord` with its own `query.provider_model`, and each
such record is logged to ProxDash and the query cache in isolation. The
client-level `ProxAIClient.generate()` wrapper expands the call into
`[primary] + connection_options.fallback_models`, then returns **only
one** `CallRecord` to the caller: the first success, or the last
failure if every model failed.

Implementation detail worth knowing: `ProxAIClient.generate()`
`copy.copy`'s the caller's `ConnectionOptions` before touching it, so
the caller's instance is never mutated. On the internal copy it forces
`suppress_provider_errors=True` and clears `fallback_models` to `None`
before dispatching to the connector. Both the internal per-attempt
`CallRecord`s and the returned record therefore have
`query.connection_options.fallback_models is None`. The intended chain
is reconstructible from `ConnectionMetadata.failed_fallback_models`
(accumulated on the returned record) plus `query.provider_model`.

Practical consequence: telemetry backends (ProxDash, the cache) see every
attempt, but application code calling `px.generate()` / `client.generate()`
only ever receives a single `CallRecord` per call.

### 2.9 `content` holds the first choice, `choices` holds the remaining n-1

When `parameters.n > 1`, the connector puts the first provider choice
into `result.content` and the remaining `n-1` choices into
`result.choices` as `ChoiceType` entries — so `choices[0]` is the
*second* provider choice, `choices[n-2]` is the *last*, and
`len(choices) == n - 1`. A single-choice response leaves `choices` as
`None`. There is no duplication between `content` and `choices`. This
contract is enforced per connector; see the `TestMultiChoiceShape`
test class for the locked cases.

### 2.10 `ChoiceType` mirrors `ResultRecord` for content shape

Each `ChoiceType` has its own `content: list[MessageContent]` plus the
same `output_text` / `output_image` / `output_audio` / `output_video` /
`output_json` / `output_pydantic` derived fields. The `ResultAdapter`
runs `_adapt_output_values()` against each choice independently, so any
shortcut that works on `result` also works on `result.choices[i]`.

### 2.11 Tool info lives exclusively in `MessageContent(TOOL)` blocks

The sole source of truth for tool calls and results (including
web-search citations) is a `MessageContent(type=TOOL,
tool_content=ToolContent(kind=RESULT, name="web_search",
citations=[Citation(title, url), ...]))` block inlined into
`result.content`. Every connector populates this directly (see
`openai.py`, `claude.py`, `gemini.py`, `mistral.py`, `grok.py`).

One representation, no drift. Because `content` round-trips through
`Chat.append(response)`, the conversation history also carries the
tool info forward for follow-up calls — no separate "tool usage"
sidecar needs to be threaded through.

The previous `ResultRecord.tool_usage: ToolUsageType` field and the
`ToolUsageType` dataclass have been removed entirely. The
deserializer silently ignores a legacy `tool_usage` key in older
cached records so historical data still loads; derive any flat URL
list you need from the `TOOL` blocks themselves.

### 2.12 `estimated_cost` is integer nano-USD

Unit contract — **everything on the cost path is int**:

- `ProviderModelPricingType.input_token_cost_nano_usd_per_token` and
  `output_token_cost_nano_usd_per_token` are typed `int | None` and
  quoted in **nano-USD per token** (1 nano-USD = 10⁻⁹ USD). For Claude
  Haiku at $0.80 per 1M input tokens → $0.0000008 per token → `800`.
  No fractional costs; nano-USD is fine enough to represent every real
  provider tier as an integer, and the integer typing keeps
  floating-point drift out of the cache and ProxDash.
- `UsageType.input_tokens` / `output_tokens` are `int`.
- `UsageType.estimated_cost` is `int`, in **nano-USD total**.
  `get_estimated_cost` computes
  `math.floor(input_tokens * input_token_cost_nano_usd_per_token +
  output_tokens * output_token_cost_nano_usd_per_token)`. With all int
  inputs the product is already int, so the `math.floor` is a defensive
  no-op; the unit is preserved because the scalars are already nano-USD
  per token. For the Haiku example above, 1,000,000 input tokens × 800
  = `800_000_000` nano-USD = $0.80.
- To display: USD = `estimated_cost / 1_000_000_000`;
  µ-USD = `estimated_cost / 1_000`.

Nano-USD (rather than µ-USD) is precise enough for per-token accounting
even on the cheapest models — at µ-USD scale a $0.30/1M token would
round to zero per-token.

### 2.13 Error payload lifecycle

`_safe_provider_query` captures the raw `Exception` instance into
`result.error` and stores `traceback.format_exc()` in
`result.error_traceback`. At the end of `generate()`:

- if `connection_options.suppress_provider_errors` is truthy, `error` is
  stringified (`str(exc)`) and the `CallRecord` is returned with
  `status=FAILED`;
- otherwise the exception is re-raised and no `CallRecord` reaches the
  caller.

So a persisted `CallRecord` with `status=FAILED` always has `error` as a
`str`; an in-flight one may briefly hold the raw exception object.
`timestamp` is still populated for failed calls.

### 2.14 `endpoint_used` is set on the provider path only

On cache hits, `ConnectionMetadata.endpoint_used` is left unset — the
cached result does not carry forward the endpoint that originally
produced it. Any downstream code that reads `endpoint_used` should
tolerate `None` alongside `result_source == CACHE`.

### 2.15 `cache_look_fail_reason` survives to the returned record

When `_get_cached_result` returns a `CacheLookFailReason` (cache miss),
the connector stores it on `connection_metadata.cache_look_fail_reason`
and then runs the provider. The field is **not** cleared before the
record is returned, so consumers (dashboards, debugging, triage) can
see on a returned `CallRecord` that `result_source == PROVIDER` *and*
`cache_look_fail_reason == CACHE_NOT_FOUND` (or whichever reason the
cache reported) together — telling the full story of why the provider
was hit.

When `connection_options.skip_cache=True` or `override_cache_value=True`
is set, `_get_cached_result` returns `None` without consulting the
cache, so no reason is produced in the first place —
`cache_look_fail_reason` stays `None` on the returned record. Use the
`ConnectionOptions` fields on the query side to distinguish that path.

---

## 3. Examples

Every example below assumes the imports:

```python
import datetime as dt
import pydantic
import proxai as px
from proxai.types import (
    CallRecord, QueryRecord, ResultRecord, ConnectionMetadata,
    ConnectionOptions, ProviderModelType, ParameterType, ThinkingType,
    OutputFormat, OutputFormatType, Tools,
    UsageType, TimeStampType,
    ResultStatusType, ResultSource, CacheLookFailReason,
    ChoiceType, MessageRoleType,
)
from proxai.chat.chat_session import Chat
from proxai.chat.message import Message
from proxai.chat.message_content import (
    MessageContent, ContentType, PydanticContent,
    ToolContent, ToolKind, Citation,
)
```

### 3.1 Simple text prompt

The minimal success case. Note that `content` is a single-element list,
not a bare string, and `output_text` is filled in by the adapter.

```python
CallRecord(
    query=QueryRecord(
        prompt="What is the capital of France?",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        hash_value="abc123",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.TEXT,
                text="The capital of France is Paris.",
            ),
        ],
        output_text="The capital of France is Paris.",
        usage=UsageType(
            input_tokens=12, output_tokens=8, total_tokens=20,
            estimated_cost=1500,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 1),
            response_time=dt.timedelta(seconds=1),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="chat.completions.create",
        feature_mapping_strategy=px.FeatureMappingStrategy.BEST_EFFORT,
    ),
)
```

### 3.2 Multi-turn chat

`Chat.system_prompt` is used (not `QueryRecord.system_prompt`). Input
messages may use plain `str` for `content`; output content is always
`list[MessageContent]`.

```python
CallRecord(
    query=QueryRecord(
        chat=Chat(
            system_prompt="You are a helpful math tutor.",
            messages=[
                Message(role="user", content="What is 2+2?"),
                Message(role="assistant", content="4"),
                Message(role="user", content="What about 2+3?"),
            ],
        ),
        provider_model=ProviderModelType(
            provider="anthropic", model="claude-3-5-sonnet",
            provider_model_identifier="claude-3-5-sonnet-20241022",
        ),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        hash_value="def456",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[MessageContent(type=ContentType.TEXT, text="2 + 3 = 5")],
        output_text="2 + 3 = 5",
        usage=UsageType(input_tokens=45, output_tokens=6, total_tokens=51),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 2),
            response_time=dt.timedelta(seconds=2),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="messages.create",
    ),
)
```

### 3.3 Error response (suppressed)

When `suppress_provider_errors=True`, the `CallRecord` survives with
`status=FAILED` and `error` stringified. `timestamp` is still set; `usage`
and `content` are not.

```python
CallRecord(
    query=QueryRecord(
        prompt="Tell me a joke.",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4",
            provider_model_identifier="gpt-4-0613",
        ),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        connection_options=ConnectionOptions(suppress_provider_errors=True),
        hash_value="err789",
    ),
    result=ResultRecord(
        status=ResultStatusType.FAILED,
        role=MessageRoleType.ASSISTANT,
        error="Rate limit exceeded",
        error_traceback="Traceback (most recent call last):\n  File ...",
        usage=UsageType(input_tokens=0, output_tokens=0, total_tokens=0),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0, 500000),
            response_time=dt.timedelta(milliseconds=500),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="chat.completions.create",
    ),
)
```

### 3.4 Fallback chain (telemetry view)

The caller of `client.generate()` receives exactly one `CallRecord` — the
successful one (here, the second attempt). The `CallRecord` for the first
attempt shown below is what ProxDash and the query cache observe
internally; it is not returned to user code. The returned record's
`connection.failed_fallback_models` lists the models that were tried and
failed before it.

Per §2.8, `ProxAIClient.generate()` copies the caller's
`ConnectionOptions` before it touches it, then on that copy forces
`suppress_provider_errors=True` and clears `fallback_models` to `None`
before dispatching to the connector. The caller's instance is
unchanged; both records below have
`query.connection_options.fallback_models is None`.

```python
# Attempt 1: openai/gpt-4 fails (internal — logged to ProxDash, not returned)
CallRecord(
    query=QueryRecord(
        prompt="Explain quantum entanglement.",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4",
            provider_model_identifier="gpt-4-0613",
        ),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        connection_options=ConnectionOptions(
            suppress_provider_errors=True,
        ),
        hash_value="fb1234",
    ),
    result=ResultRecord(
        status=ResultStatusType.FAILED,
        role=MessageRoleType.ASSISTANT,
        error="API key invalid",
        error_traceback="Traceback ...",
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 1),
            response_time=dt.timedelta(seconds=1),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="chat.completions.create",
    ),
)

# Attempt 2: anthropic succeeds
CallRecord(
    query=QueryRecord(
        prompt="Explain quantum entanglement.",
        provider_model=ProviderModelType(
            provider="anthropic", model="claude-3-5-sonnet",
            provider_model_identifier="claude-3-5-sonnet-20241022",
        ),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        connection_options=ConnectionOptions(
            suppress_provider_errors=True,
        ),
        hash_value="fb1234",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.TEXT,
                text="Quantum entanglement is a phenomenon...",
            ),
        ],
        output_text="Quantum entanglement is a phenomenon...",
        usage=UsageType(
            input_tokens=8, output_tokens=150, total_tokens=158,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 1),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 4),
            response_time=dt.timedelta(seconds=3),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="messages.create",
        failed_fallback_models=[
            ProviderModelType(
                provider="openai", model="gpt-4",
                provider_model_identifier="gpt-4-0613",
            ),
        ],
    ),
)
```

### 3.5 Multiple choices (`n > 1`)

Per §2.9, `content` holds the first choice and `choices` holds the
remaining n-1 — no duplication. For n=3, `choices` has 2 entries. The
adapter fills the `output_text` shortcut on both `result` and each
`choice`.

```python
CallRecord(
    query=QueryRecord(
        prompt="Give me a creative name for a cat.",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        parameters=ParameterType(n=3, temperature=0.9),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        hash_value="multi1",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(type=ContentType.TEXT, text="Whiskers McFluffington"),
        ],
        output_text="Whiskers McFluffington",
        choices=[
            ChoiceType(
                content=[
                    MessageContent(
                        type=ContentType.TEXT, text="Sir Purrs-a-Lot",
                    ),
                ],
                output_text="Sir Purrs-a-Lot",
            ),
            ChoiceType(
                content=[
                    MessageContent(
                        type=ContentType.TEXT, text="Luna Moonbeam",
                    ),
                ],
                output_text="Luna Moonbeam",
            ),
        ],
        usage=UsageType(
            input_tokens=10, output_tokens=15, total_tokens=25,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 3),
            response_time=dt.timedelta(seconds=3),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="chat.completions.create",
    ),
)
```

### 3.6 Pydantic structured output

`content[0]` is a `PYDANTIC_INSTANCE` block. `instance_json_value` is what
the query cache persists; `class_value.model_validate(instance_json_value)`
is how the instance is reconstructed on cache read. The `output_pydantic`
shortcut holds the live instance.

```python
class MovieReview(pydantic.BaseModel):
  title: str
  rating: float
  summary: str

review = MovieReview(
    title="Inception", rating=9.2,
    summary="A mind-bending masterpiece...",
)

CallRecord(
    query=QueryRecord(
        prompt="Review the movie Inception.",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        output_format=OutputFormat(
            type=OutputFormatType.PYDANTIC,
            pydantic_class=MovieReview,
            pydantic_class_name="MovieReview",
            pydantic_class_json_schema=MovieReview.model_json_schema(),
        ),
        hash_value="pyd001",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.PYDANTIC_INSTANCE,
                pydantic_content=PydanticContent(
                    class_name="MovieReview",
                    class_value=MovieReview,
                    instance_value=review,
                    instance_json_value={
                        "title": "Inception", "rating": 9.2,
                        "summary": "A mind-bending masterpiece...",
                    },
                ),
            ),
        ],
        output_pydantic=review,
        usage=UsageType(
            input_tokens=15, output_tokens=45, total_tokens=60,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 2),
            response_time=dt.timedelta(seconds=2),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="beta.chat.completions.parse",
    ),
)
```

### 3.7 JSON output

No schema enforcement. The adapter emits a `JSON` block; for
`BEST_EFFORT` endpoints it parses JSON out of a text response.

```python
CallRecord(
    query=QueryRecord(
        prompt="List 3 countries and their capitals as JSON.",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        output_format=OutputFormat(type=OutputFormatType.JSON),
        hash_value="json001",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.JSON,
                json={"countries": [
                    {"name": "France", "capital": "Paris"},
                    {"name": "Japan", "capital": "Tokyo"},
                    {"name": "Brazil", "capital": "Brasilia"},
                ]},
            ),
        ],
        output_json={"countries": [
            {"name": "France", "capital": "Paris"},
            {"name": "Japan", "capital": "Tokyo"},
            {"name": "Brazil", "capital": "Brasilia"},
        ]},
        usage=UsageType(
            input_tokens=12, output_tokens=80, total_tokens=92,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 2),
            response_time=dt.timedelta(seconds=2),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="chat.completions.create",
    ),
)
```

### 3.8 Web search with rich citations

Citations live in a `TOOL` block inside `content` — that's the single
source of truth (see §2.11). Consumers that want a flat URL list
derive it from the `TOOL` block's `citations`.

```python
CallRecord(
    query=QueryRecord(
        prompt="What are the latest developments in fusion energy?",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        tools=[Tools.WEB_SEARCH],
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        hash_value="web001",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.TOOL,
                tool_content=ToolContent(
                    name="web_search",
                    kind=ToolKind.RESULT,
                    citations=[
                        Citation(
                            title="Fusion Energy Breakthroughs 2026",
                            url="https://example.com/fusion-energy-2026",
                        ),
                        Citation(
                            title="Nature: Fusion Breakthrough",
                            url="https://nature.com/articles/fusion-breakthrough",
                        ),
                        Citation(
                            title="Reuters: Fusion Update",
                            url="https://reuters.com/energy/fusion-update",
                        ),
                    ],
                ),
            ),
            MessageContent(
                type=ContentType.TEXT,
                text="Recent developments in fusion energy include...",
            ),
        ],
        output_text="Recent developments in fusion energy include...",
        usage=UsageType(
            input_tokens=10, output_tokens=200, total_tokens=210,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 5),
            response_time=dt.timedelta(seconds=5),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="responses.create",
    ),
)
```

### 3.9 Cache hit

`result_source == CACHE`. `response_time` is the original provider
latency that was cached; `cache_response_time` is how long the cache
lookup itself took (measured via `time.perf_counter`). `endpoint_used`
is left `None` on the cache path. `start_utc_date` / `end_utc_date` are
rebased to the current time so downstream consumers see a "now"
timestamp with the original provider latency.

```python
CallRecord(
    query=QueryRecord(
        prompt="What is 2+2?",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        hash_value="cached001",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[MessageContent(type=ContentType.TEXT, text="4")],
        output_text="4",
        usage=UsageType(
            input_tokens=8, output_tokens=1, total_tokens=9,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 9, 59, 59),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            response_time=dt.timedelta(seconds=1),
            cache_response_time=dt.timedelta(milliseconds=5),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.CACHE,
    ),
)
```

### 3.10 Cache miss reasons

Every non-hit that went through the cache subsystem carries a reason
which survives onto the returned `CallRecord` alongside
`result_source = PROVIDER` (see §2.15). The reasons map one-to-one to
the `CacheLookFailReason` enum.

```python
# No cache entry for this query hash
ConnectionMetadata(
    result_source=ResultSource.PROVIDER,
    cache_look_fail_reason=CacheLookFailReason.CACHE_NOT_FOUND,
)

# Cache entry exists but the query fingerprint didn't match
ConnectionMetadata(
    result_source=ResultSource.PROVIDER,
    cache_look_fail_reason=CacheLookFailReason.CACHE_NOT_MATCHED,
)

# Collecting diverse responses — unique_response_limit not yet hit
ConnectionMetadata(
    result_source=ResultSource.PROVIDER,
    cache_look_fail_reason=(
        CacheLookFailReason.UNIQUE_RESPONSE_LIMIT_NOT_REACHED
    ),
)

# Cached result was an error; retry_if_error_cached forced a retry
ConnectionMetadata(
    result_source=ResultSource.PROVIDER,
    cache_look_fail_reason=CacheLookFailReason.PROVIDER_ERROR_CACHED,
)

# Cache manager is configured but not in the WORKING state
ConnectionMetadata(
    result_source=ResultSource.PROVIDER,
    cache_look_fail_reason=CacheLookFailReason.CACHE_UNAVAILABLE,
)
```

When `connection_options.skip_cache=True` or `override_cache_value=True`
is set, the cache layer is bypassed entirely and
`cache_look_fail_reason` stays `None` — use the `ConnectionOptions`
fields on the query side to distinguish that path.

### 3.11 Custom parameters

```python
CallRecord(
    query=QueryRecord(
        prompt="Write a haiku about winter.",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        parameters=ParameterType(
            temperature=0.2,
            max_tokens=50,
            stop=["\n\n", "---"],
        ),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        hash_value="params1",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.TEXT,
                text=(
                    "Silent snow descends\n"
                    "Blankets the sleeping forest\n"
                    "Winter breathes in white"
                ),
            ),
        ],
        output_text=(
            "Silent snow descends\n"
            "Blankets the sleeping forest\n"
            "Winter breathes in white"
        ),
        usage=UsageType(
            input_tokens=8, output_tokens=18, total_tokens=26,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 1),
            response_time=dt.timedelta(seconds=1),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="chat.completions.create",
    ),
)
```

### 3.12 Thinking / reasoning

Thinking traces and the final answer share a single `content` list.
Providers with multiple reasoning summaries emit multiple `THINKING`
blocks.

```python
CallRecord(
    query=QueryRecord(
        prompt="What is 127 * 389?",
        provider_model=ProviderModelType(
            provider="openai", model="o1",
            provider_model_identifier="o1-2024-12-17",
        ),
        parameters=ParameterType(thinking=ThinkingType.HIGH),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        hash_value="think001",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.THINKING,
                text=(
                    "127 * 389 = 127 * 400 - 127 * 11 "
                    "= 50800 - 1397 = 49403"
                ),
            ),
            MessageContent(type=ContentType.TEXT, text="49,403"),
        ],
        output_text="49,403",
        usage=UsageType(
            input_tokens=10, output_tokens=85, total_tokens=95,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 4),
            response_time=dt.timedelta(seconds=4),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="responses.create",
    ),
)
```

### 3.13 Image response

`output_image` is set by the adapter to the `MessageContent` block
itself — typed `MessageContent | None` on both `ResultRecord` and
`ChoiceType`. Read `content` if you need the canonical view;
`output_image` is a convenience shortcut to the last IMAGE block.

```python
CallRecord(
    query=QueryRecord(
        prompt="Generate an image of a sunset over mountains.",
        provider_model=ProviderModelType(
            provider="openai", model="dall-e-3",
            provider_model_identifier="dall-e-3",
        ),
        output_format=OutputFormat(type=OutputFormatType.IMAGE),
        hash_value="img001",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.IMAGE,
                source="https://oaidalleapiprodscus.blob.core.windows.net/...",
                media_type="image/png",
            ),
        ],
        usage=UsageType(
            input_tokens=10, output_tokens=0, total_tokens=10,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 15),
            response_time=dt.timedelta(seconds=15),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="images.generate",
    ),
)
```

### 3.14 Multi-modal input (image in chat)

Input message content lists may mix string items (auto-wrapped to
`TEXT`) with `MessageContent` blocks.

```python
CallRecord(
    query=QueryRecord(
        chat=Chat(
            messages=[
                Message(
                    role="user",
                    content=[
                        MessageContent(
                            type=ContentType.TEXT,
                            text="What's in this image?",
                        ),
                        MessageContent(
                            type=ContentType.IMAGE,
                            source="https://example.com/photo.jpg",
                            media_type="image/jpeg",
                        ),
                    ],
                ),
            ],
        ),
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        output_format=OutputFormat(type=OutputFormatType.TEXT),
        hash_value="mmi001",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.TEXT,
                text="The image shows a golden retriever sitting in a park...",
            ),
        ],
        output_text="The image shows a golden retriever sitting in a park...",
        usage=UsageType(
            input_tokens=1200, output_tokens=30, total_tokens=1230,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 3),
            response_time=dt.timedelta(seconds=3),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="chat.completions.create",
    ),
)
```

Other media content types (`AUDIO`, `DOCUMENT`, `VIDEO`) follow the same
pattern — the `MessageContent` block carries exactly one of `source`
(URL), `data` (raw bytes), or `path` (local file), plus an optional
`media_type` validated against `SUPPORTED_MEDIA_TYPES`.

### 3.15 Multiple choices with Pydantic

The most intricate case: `n > 1` combined with structured output. Each
choice carries its own full `PydanticContent` block plus its own
`output_pydantic` shortcut.

```python
class CityInfo(pydantic.BaseModel):
  name: str
  population: int

tokyo = CityInfo(name="Tokyo", population=13960000)
lagos = CityInfo(name="Lagos", population=15400000)
berlin = CityInfo(name="Berlin", population=3750000)

CallRecord(
    query=QueryRecord(
        prompt="Give me a random city and its population.",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        parameters=ParameterType(n=3, temperature=1.0),
        output_format=OutputFormat(
            type=OutputFormatType.PYDANTIC,
            pydantic_class=CityInfo,
            pydantic_class_name="CityInfo",
            pydantic_class_json_schema=CityInfo.model_json_schema(),
        ),
        hash_value="mc_pyd",
    ),
    result=ResultRecord(
        status=ResultStatusType.SUCCESS,
        role=MessageRoleType.ASSISTANT,
        content=[
            MessageContent(
                type=ContentType.PYDANTIC_INSTANCE,
                pydantic_content=PydanticContent(
                    class_name="CityInfo",
                    class_value=CityInfo,
                    instance_value=tokyo,
                    instance_json_value={
                        "name": "Tokyo", "population": 13960000,
                    },
                ),
            ),
        ],
        output_pydantic=tokyo,
        choices=[
            ChoiceType(
                content=[
                    MessageContent(
                        type=ContentType.PYDANTIC_INSTANCE,
                        pydantic_content=PydanticContent(
                            class_name="CityInfo",
                            class_value=CityInfo,
                            instance_value=lagos,
                            instance_json_value={
                                "name": "Lagos", "population": 15400000,
                            },
                        ),
                    ),
                ],
                output_pydantic=lagos,
            ),
            ChoiceType(
                content=[
                    MessageContent(
                        type=ContentType.PYDANTIC_INSTANCE,
                        pydantic_content=PydanticContent(
                            class_name="CityInfo",
                            class_value=CityInfo,
                            instance_value=berlin,
                            instance_json_value={
                                "name": "Berlin", "population": 3750000,
                            },
                        ),
                    ),
                ],
                output_pydantic=berlin,
            ),
        ],
        usage=UsageType(
            input_tokens=12, output_tokens=60, total_tokens=72,
        ),
        timestamp=TimeStampType(
            start_utc_date=dt.datetime(2026, 2, 14, 10, 0, 0),
            end_utc_date=dt.datetime(2026, 2, 14, 10, 0, 3),
            response_time=dt.timedelta(seconds=3),
        ),
    ),
    connection=ConnectionMetadata(
        result_source=ResultSource.PROVIDER,
        endpoint_used="beta.chat.completions.parse",
    ),
)
```

---

## 4. Where each field is populated

For contributors debugging the pipeline: this is where each piece of the
`CallRecord` is written, in order, inside
`ProviderConnector.generate()`.

1. `query: QueryRecord` — constructed from `generate()` arguments.
   `query.provider_model` comes directly from the caller.
2. `_prepare_execution()` chooses an endpoint (honouring
   `connection_options.endpoint` if set) and runs
   `FeatureAdapter.adapt_query_record()` to produce the
   `modified_query_record` actually sent to the provider.
3. `_get_cached_result()` either returns a `ResultRecord` (cache hit,
   rebased timestamps) or a `CacheLookFailReason`. It returns `None` if
   `connection_options.skip_cache` / `override_cache_value` is set or
   no cache manager is configured.
4. On cache hit: `_reconstruct_pydantic_from_cache()` rehydrates
   `pydantic_content.instance_value` from `instance_json_value` for
   every pydantic block (cache stores only the JSON form). Then
   `connection.result_source = CACHE`, the `CallRecord` is returned
   immediately, and `endpoint_used` stays `None`.
5. On cache miss: `_execute_call()` invokes the chosen endpoint
   executor. Each executor wraps its SDK call in
   `_safe_provider_query()`, which catches exceptions into
   `result.error` / `result.error_traceback` and sets
   `status=FAILED`. On success, `_execute_call()` also runs
   `ResultAdapter.adapt_result_record()` to normalize `content` (and
   each `choice.content`) and fill the `output_*` shortcuts.
6. `_compute_usage()` populates `result.usage.input_tokens` /
   `output_tokens` / `total_tokens` using the connector's token-count
   estimator (character + whitespace heuristic, `math.ceil(max(len/4,
   words*1.3))`).
7. `_compute_timestamp()` sets `start_utc_date`, `end_utc_date`,
   `local_time_offset_minute`, and `response_time`.
   `cache_response_time` is populated only on the cache-hit branch
   (step 4) — on this provider-path branch it stays `None`.
8. `connection.endpoint_used` is set to the endpoint that actually ran
   and `result_source` is set to `PROVIDER`. Any
   `cache_look_fail_reason` set by step 3 survives onto the returned
   record (see §2.15).
9. `get_estimated_cost()` computes `result.usage.estimated_cost` in
   integer nano-USD using `provider_model_config.pricing`. The pricing
   scalars are already nano-USD per token (see §2.12), so the function
   is just `math.floor(input_tokens *
   input_token_cost_nano_usd_per_token + output_tokens *
   output_token_cost_nano_usd_per_token)` — no further scaling.
10. If `executor_result.raw_provider_response` is present and
    `debug_options.keep_raw_provider_response` is set,
    `call_record.debug = DebugInfo(raw_provider_response=...)`;
    otherwise `call_record.debug` stays `None`.
11. If `status == FAILED` and `suppress_provider_errors` is falsy, the
    captured exception is re-raised (the `CallRecord` is uploaded to
    ProxDash first). Otherwise it is stringified, the record is
    returned to the caller, and — on success only — `_update_cache()`
    persists it (skipped when `skip_cache` is set).

If you are adding a new field to `CallRecord` or a new provider
connector, touch this section as well — it is the only place that ties
the flow of the request pipeline to the shape of the record.
