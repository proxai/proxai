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
│   ├── response_format: ResponseFormat | None
│   │   ├── type: ResponseFormatType        # TEXT | IMAGE | AUDIO | VIDEO
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
│   │   └── estimated_cost: int | None      # µ-dollars (×1_000_000)
│   ├── tool_usage: ToolUsageType | None
│   │   ├── web_search_count: int | None
│   │   └── web_search_citations: list[str] | None
│   └── timestamp: TimeStampType | None
│       ├── start_utc_date: datetime
│       ├── end_utc_date: datetime
│       ├── local_time_offset_minute: int
│       ├── response_time: timedelta
│       └── cache_response_time: timedelta | None
│
└── connection: ConnectionMetadata          # (was named `cache` in old docs)
    ├── result_source: ResultSource         # CACHE | PROVIDER
    ├── cache_look_fail_reason: CacheLookFailReason | None
    ├── endpoint_used: str | None           # set only on PROVIDER path
    ├── failed_fallback_models: list[ProviderModelType] | None
    └── feature_mapping_strategy: FeatureMappingStrategy | None
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
This is how rich web-search citations are surfaced today; the legacy flat
`ToolUsageType.web_search_citations: list[str]` is a parallel URL-only list
kept for compatibility.

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

`ResultAdapter._adapt_output_values()` scans `result.content` (and each
`choice.content`) in reverse and projects the last block of each type into
the matching `output_*` field. Consumers who need the full, correct
response structure must read `content`; the `output_*` shortcuts exist for
ergonomic access to "the final text/json/pydantic value" and nothing more.

Connector executors must NOT write `output_*` directly — the adapter owns
those fields. See `src/proxai/connectors/result_adapter.py`.

### 2.3 `QueryRecord.system_prompt` vs `Chat.system_prompt` — mutually exclusive

`QueryRecord.system_prompt` is set when the caller used the prompt API.
`Chat.system_prompt` (the property on the `Chat` object assigned to
`QueryRecord.chat`) is set when the caller used the messages API. Using
both at the same time is a validation error at the `generate()` boundary.

### 2.4 Structured output returns typed `MessageContent` blocks

For `ResponseFormatType.JSON`, the result content is a
`MessageContent(type=JSON, json={...})` block. For
`ResponseFormatType.PYDANTIC`, it is a
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

`ResponseFormat` exposes `pydantic_class` + `pydantic_class_name` +
`pydantic_class_json_schema` for pydantic mode, or nothing extra for plain
JSON mode. There is no "JSON with user-supplied schema but no pydantic
class" mode — callers who want schema-constrained output must define a
`BaseModel`.

### 2.7 Timestamps reflect the original provider call

On a cache hit, `timestamp.response_time` stays equal to the original
provider latency that was cached. `timestamp.start_utc_date` and
`end_utc_date` are rebased to the current time (see
`_get_cached_result` in `provider_connector.py`), and
`timestamp.cache_response_time` records how long the cache lookup
itself took. Cost-attribution and latency metrics should branch on
`connection.result_source`.

### 2.8 Fallback chains: one record per attempt internally, one record returned to the caller

At the connector layer, each attempt against a different model is an
independent `CallRecord` with its own `query.provider_model`, and each
such record is logged to ProxDash and the query cache in isolation. The
client-level `ProxAIClient.generate()` wrapper, however, loops over
`[primary] + connection_options.fallback_models`, forces
`suppress_provider_errors=True` for the duration of the loop, and returns
**only one** `CallRecord` to the caller: the first success, or the last
failure if every model failed. `ConnectionOptions.fallback_models` lists
the intended chain; `ConnectionMetadata.failed_fallback_models`
accumulates the models that failed before the returned record.

Practical consequence: telemetry backends (ProxDash, the cache) see every
attempt, but application code calling `px.generate()` / `client.generate()`
only ever receives a single `CallRecord` per call.

### 2.9 `content` holds the first choice, `choices` holds the rest

When `parameters.n > 1`, the connector puts the first choice into
`result.content` and the remaining `n - 1` choices into
`result.choices[0..n-2]` as `ChoiceType` entries. A single-choice response
leaves `choices` as `None`.

### 2.10 `ChoiceType` mirrors `ResultRecord` for content shape

Each `ChoiceType` has its own `content: list[MessageContent]` plus the
same `output_text` / `output_image` / `output_audio` / `output_video` /
`output_json` / `output_pydantic` derived fields. The `ResultAdapter`
runs `_adapt_output_values()` against each choice independently, so any
shortcut that works on `result` also works on `result.choices[i]`.

### 2.11 Rich citations live in `MessageContent(TOOL)`, flat URLs live in `ToolUsageType`

The source-of-truth for web-search citations is a
`MessageContent(type=TOOL, tool_content=ToolContent(kind=RESULT,
name="web_search", citations=[Citation(title, url), ...]))` block inlined
into `result.content`. `ToolUsageType.web_search_citations` is a
flat `list[str]` of URLs retained for quick access and compact logging;
consumers that need titles or ordering relative to the text must read the
`TOOL` blocks.

### 2.12 `estimated_cost` is integer micro-dollars

Stored as `int`, equal to the USD cost multiplied by 1,000,000 and floored.
`$0.0015` becomes `1500`. This avoids floating-point drift across the cache
and telemetry. See `get_estimated_cost` in
`connectors/provider_connector.py`.

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
    ResponseFormat, ResponseFormatType, Tools,
    UsageType, ToolUsageType, TimeStampType,
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
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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

```python
# Attempt 1: openai/gpt-4 fails (internal — logged to ProxDash, not returned)
CallRecord(
    query=QueryRecord(
        prompt="Explain quantum entanglement.",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4",
            provider_model_identifier="gpt-4-0613",
        ),
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
        connection_options=ConnectionOptions(
            fallback_models=[
                ProviderModelType(
                    provider="anthropic", model="claude-3-5-sonnet",
                    provider_model_identifier="claude-3-5-sonnet-20241022",
                ),
            ],
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
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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

First choice lives in `result.content`; the rest are `ChoiceType`
entries in `result.choices`, each with its own `content` list. The
adapter also fills the `output_text` shortcut on both `result` and each
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
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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
        response_format=ResponseFormat(
            type=ResponseFormatType.PYDANTIC,
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
        response_format=ResponseFormat(type=ResponseFormatType.JSON),
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

Citations live in a `TOOL` block inside `content`. The flat URL list on
`ToolUsageType.web_search_citations` is a secondary, compatibility view.

```python
CallRecord(
    query=QueryRecord(
        prompt="What are the latest developments in fusion energy?",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        tools=[Tools.WEB_SEARCH],
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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
        tool_usage=ToolUsageType(
            web_search_count=3,
            web_search_citations=[
                "https://example.com/fusion-energy-2026",
                "https://nature.com/articles/fusion-breakthrough",
                "https://reuters.com/energy/fusion-update",
            ],
        ),
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

`result_source == CACHE`. `response_time` is the
original provider latency; `cache_response_time` is how long the cache
lookup itself took. `endpoint_used` is left `None` on the cache path.
`start_utc_date` / `end_utc_date` are rebased to the current time so
downstream consumers see a "now" timestamp with the original latency.

```python
CallRecord(
    query=QueryRecord(
        prompt="What is 2+2?",
        provider_model=ProviderModelType(
            provider="openai", model="gpt-4o",
            provider_model_identifier="gpt-4o-2024-08-06",
        ),
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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

Every non-hit that still went through the cache subsystem carries a
reason. These map one-to-one to the `CacheLookFailReason` enum.

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

# Cached result was an error, retry_if_error_cached=True forced a retry
ConnectionMetadata(
    result_source=ResultSource.PROVIDER,
    cache_look_fail_reason=CacheLookFailReason.PROVIDER_ERROR_CACHED,
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
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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
        response_format=ResponseFormat(type=ResponseFormatType.IMAGE),
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
        response_format=ResponseFormat(type=ResponseFormatType.TEXT),
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
        response_format=ResponseFormat(
            type=ResponseFormatType.PYDANTIC,
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
   rebased timestamps) or a `CacheLookFailReason`.
4. On cache hit: `connection.result_source = CACHE`, and the
   `CallRecord` is returned immediately. `endpoint_used` stays `None`.
5. On cache miss: the executor runs inside `_safe_provider_query()`,
   which catches exceptions into `result.error` / `result.error_traceback`.
6. On success: `ResultAdapter.adapt_result_record()` normalizes
   `result.content` (and each `choice.content`) to the expected
   response format, then fills in the `output_*` shortcuts.
7. `_compute_usage()` populates `result.usage.input_tokens` /
   `output_tokens` / `total_tokens` using the connector's token-count
   estimator (character + whitespace heuristic, `math.ceil(max(len/4,
   words*1.3))`).
8. `_compute_timestamp()` sets `start_utc_date`, `end_utc_date`,
   `local_time_offset_minute`, and `response_time`.
9. `connection.endpoint_used` is set to the endpoint that actually ran,
   `result_source` is set to `PROVIDER`, and `cache_look_fail_reason`
   is cleared.
10. `get_estimated_cost()` computes `result.usage.estimated_cost` in
    µ-dollars, using `provider_model_config.pricing`.
11. If `status == FAILED` and `suppress_provider_errors` is falsy, the
    captured exception is re-raised. Otherwise it is stringified and the
    `CallRecord` is returned; on success, `_update_cache()` persists it
    (unless `skip_cache` is set).

If you are adding a new field to `CallRecord` or a new provider
connector, touch this section as well — it is the only place that ties
the flow of the request pipeline to the shape of the record.
