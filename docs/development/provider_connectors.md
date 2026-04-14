# Adding a New Provider Connector — Developer Guide

A provider connector is the class that adapts a third-party LLM SDK
(`openai`, `anthropic`, `google.genai`, …) to proxai's internal
`QueryRecord` → executor → `ResultRecord` pipeline. This guide is the
authoritative reference for writing one.

The three finalized reference connectors are:

- `src/proxai/connectors/providers/openai.py`
- `src/proxai/connectors/providers/gemini.py`
- `src/proxai/connectors/providers/claude.py`

Read at least one before starting. They are the source of truth for
patterns this doc summarizes.

## Table of Contents

1. [Overview](#1-overview)
2. [Files You Will Touch](#2-files-you-will-touch)
3. [The Required Contract](#3-the-required-contract)
4. [Naming Conventions](#4-naming-conventions)
5. [One Endpoint or Many?](#5-one-endpoint-or-many)
6. [`FeatureConfigType` Field Reference](#6-featureconfigtype-field-reference)
7. [Support Levels: SUPPORTED / BEST_EFFORT / NOT_SUPPORTED](#7-support-levels-supported--best_effort--not_supported)
8. [Writing an Executor](#8-writing-an-executor)
9. [Parameter Handling](#9-parameter-handling)
10. [Response Formats: TEXT / JSON / PYDANTIC](#10-response-formats-text--json--pydantic)
11. [Thinking Blocks](#11-thinking-blocks)
12. [Tools and Web Search Citations](#12-tools-and-web-search-citations)
13. [The Mock Model](#13-the-mock-model)
14. [Registering the Connector](#14-registering-the-connector)
15. [Verification Checklist and Pitfalls](#15-verification-checklist-and-pitfalls)

---

## 1. Overview

A connector is a Python class that inherits from
`proxai.connectors.model_connector.ProviderModelConnector`. The
framework drives it through this lifecycle (see
`model_connector.py:599-708` for `generate()`):

```
client.generate(prompt=..., parameters=..., response_format=...)
   └─ QueryRecord built from the request
      └─ FeatureAdapter.adapt_query_record()        # drops/injects per support level
         └─ chosen executor method called           # YOUR code
            └─ functools.partial(SDK call) built
            └─ self._safe_provider_query(partial)
            └─ result_record.content = [MessageContent, …]
         └─ ResultAdapter.adapt_result_record()     # transforms content per response_format
         └─ usage / cost / cache / timestamp computed
   └─ CallRecord returned to user
```

**Your job** is everything inside the executor: build the SDK call,
hand it to `_safe_provider_query`, parse the response into
`MessageContent` blocks. The framework handles everything around it.

---

## 2. Files You Will Touch

For a new provider named `widget`:

| Action | Path |
|---|---|
| Create | `src/proxai/connectors/providers/widget.py` |
| Create | `src/proxai/connectors/providers/widget_mock.py` |
| Edit   | `src/proxai/connectors/model_registry.py` (add to `_MODEL_CONNECTOR_MAP`) |
| Edit   | `src/proxai/connectors/model_configs.py` (add to `PROVIDER_KEY_MAP`) |

Section 14 covers the three edits in detail.

---

## 3. The Required Contract

Your class inherits from `ProviderModelConnector` and is validated at
**import time** by `__init_subclass__`
(`model_connector.py:97-127`). If any of the five class attributes
below are missing, or if the keys across `ENDPOINT_PRIORITY`,
`ENDPOINT_CONFIG`, and `ENDPOINT_EXECUTORS` don't match exactly,
import will raise.

### Required Class Attributes (all 5)

| Attribute | Type | Purpose |
|---|---|---|
| `PROVIDER_NAME` | `str` | Lowercase short id, e.g. `'openai'`. Must match the key in `_MODEL_CONNECTOR_MAP` and `PROVIDER_KEY_MAP`. |
| `PROVIDER_API_KEYS` | `list[str]` | Env var names the connector needs at runtime. The framework validates that every name is present in `provider_token_value_map` before instantiation (`model_connector.py:137-145`). Most providers have one (`['OPENAI_API_KEY']`); a few have two (Databricks: token + host). |
| `ENDPOINT_PRIORITY` | `list[str]` | Ordered endpoint keys. The framework iterates this list to find the first endpoint whose `ENDPOINT_CONFIG` advertises full support for the requested features. Order matters when you have multiple endpoints. |
| `ENDPOINT_CONFIG` | `dict[str, FeatureConfigType]` | One entry per endpoint key. Declares what each endpoint supports. See section 6. |
| `ENDPOINT_EXECUTORS` | `dict[str, str]` | Maps each endpoint key to the **string name** of the method that implements it. The framework calls `getattr(self, name)` at request time. |

### Required Methods (2 abstract + 1 per endpoint)

| Method | Signature | Purpose |
|---|---|---|
| `init_model()` | `() -> Any` | Returns the real SDK client (e.g., `anthropic.Anthropic(api_key=...)`). Called lazily on first access to `self.api` when `run_type == PRODUCTION`. |
| `init_mock_model()` | `() -> Any` | Returns the mock client. Called lazily when `run_type == TEST`. See section 13. |
| `_<endpoint>_executor` | `(self, query_record: types.QueryRecord) -> types.ResultRecord` | One per entry in `ENDPOINT_EXECUTORS`. See section 8. |

### Recommended (Not Required)

- A class-level constants block for any per-provider tunables (e.g.,
  Claude's `_THINKING_BUDGETS`, `_DEFAULT_MAX_TOKENS`,
  `STRUCTURED_OUTPUTS_BETA`). Keeps magic numbers out of executor
  bodies and makes them discoverable for tests.
- A private `_parse_content_blocks` helper if your response has more
  than 2-3 block types to walk (Claude has one; OpenAI inlines
  parsing per executor because each endpoint's response shape differs).

---

## 4. Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Connector file | `providers/{name}.py` | `providers/openai.py` |
| Connector class | `{Name}Connector` | `OpenAIConnector` |
| Mock file | `providers/{name}_mock.py` | `providers/openai_mock.py` |
| Mock class | `{Name}Mock` | `OpenAIMock` |
| `PROVIDER_NAME` | lowercase short id | `'openai'`, `'claude'`, `'gemini'` |
| Endpoint key | mirrors the literal SDK call path with dots | `'chat.completions.create'`, `'beta.messages.stream'` |
| Executor method | `_<endpoint key with dots → underscores>_executor` | `_chat_completions_create_executor`, `_beta_messages_stream_executor` |

**Endpoint keys should be literal**, not abstract capability names.
`'beta.messages.stream'` tells the reader which SDK call is invoked.
`'messages.create'` would be misleading if the body actually uses
`beta.messages.stream`. This is the convention the three finalized
connectors follow.

---

## 5. One Endpoint or Many?

Use **multiple endpoints** when:

- The SDK has structurally different APIs for different features
  (chat vs. images vs. videos vs. audio).
- Different feature subsets cannot be selected via flags on a single
  call (e.g., OpenAI's `beta.chat.completions.parse` only exists for
  native Pydantic structured-output parsing; `responses.create` is
  the only place that supports web search; images / audio / videos
  each live on their own SDK method).
- Response shapes differ enough that one parser would be a long
  if/elif chain over endpoint type — split it.

Use **one endpoint** when:

- A single SDK call covers everything you need via flags / config.
- Anthropic Claude is the canonical example: `beta.messages.stream`
  handles plain text, system prompts, tools, thinking, and Pydantic
  structured outputs all through one call. There is no second endpoint.

Reference counts:

| Provider | Endpoints | Reason for the split |
|---|---|---|
| openai  | 6 | chat / structured-parse / responses / images / audio / videos all have distinct SDK methods |
| gemini  | 2 | text+multimodal generation vs. async video generation (different lifecycle) |
| claude  | 1 | one streaming method handles everything |

If you're not sure, start with one. Splitting later is cheap; merging
is harder.

---

## 6. `FeatureConfigType` Field Reference

`FeatureConfigType` is defined in `types.py:160-171`. It declares
what one endpoint can do. Every field defaults to `None`, which the
framework treats as `NOT_SUPPORTED` via
`adapter_utils.resolve_support`.

### Top-Level Fields

| Field | Type | Required? | Notes |
|---|---|---|---|
| `prompt` | `FeatureSupportType` | Required if endpoint accepts a single user prompt | `SUPPORTED` for almost all text endpoints |
| `messages` | `FeatureSupportType` | Required if endpoint accepts chat history | `SUPPORTED` for chat endpoints |
| `system_prompt` | `FeatureSupportType` | Optional | Mark `SUPPORTED` if the SDK has a real system field |
| `add_system_to_messages` | `bool \| None` | Optional | Set `True` only when the SDK does **not** have a separate system field but accepts `[{role: 'system', content: ...}]` as the first message. The framework will prepend the system prompt to the message list automatically. OpenAI `chat.completions.create` uses this; Claude does not. |
| `parameters` | `ParameterConfigType` | Recommended | Nested config — see below |
| `tools` | `ToolConfigType` | Optional | Nested config — see below |
| `response_format` | `ResponseFormatConfigType` | Required (always validated by `_collect_response_format_level`) | Nested config — see below |

### `ParameterConfigType` (`types.py:129-138`)

| Field | What It Controls |
|---|---|
| `temperature` | Float sampling temperature |
| `max_tokens` | Max output token cap |
| `stop` | Stop sequences (string or list of strings) |
| `n` | Number of completions to return |
| `thinking` | Extended reasoning / thinking budget (`ThinkingType.LOW/MEDIUM/HIGH`) |

Mark unavailable parameters explicitly as `NOT_SUPPORTED` if a user
might reasonably try them. Anthropic doesn't have an `n` parameter,
so Claude declares `n=NOT_SUPPORTED` rather than omitting — the
framework needs to surface a clear error instead of silently dropping
the field.

### `ToolConfigType` (`types.py:140-145`)

| Field | What It Controls |
|---|---|
| `web_search` | Whether the endpoint can run a web search tool |

This is the only tool currently modeled. New tools require a new
field on `ToolConfigType` plus framework support — out of scope here.

### `ResponseFormatConfigType` (`types.py:147-158`)

| Field | What It Controls |
|---|---|
| `text` | Plain text output |
| `json` | Free-form JSON object output (no schema) |
| `pydantic` | Structured output to a Pydantic class |
| `image` | Image generation |
| `audio` | Audio generation / TTS |
| `video` | Video generation |
| `multi_modal` | Multi-modal output (text + image + audio in one response) |

Omit fields that are obviously irrelevant (e.g., don't declare `image`
on a text-only endpoint) — the convention across openai/gemini/claude
is to list only what's relevant rather than spelling out
`NOT_SUPPORTED` for every modality.

---

## 7. Support Levels: SUPPORTED / BEST_EFFORT / NOT_SUPPORTED

Every `FeatureSupportType` field in your config is one of three
values (`types.py:121-127`). The framework's behavior depends on the
level.

### `SUPPORTED`

The framework keeps the feature in the `query_record` as-is. **Your
executor must handle it.** If you mark a feature `SUPPORTED` and
forget to wire it into the SDK call, the framework won't catch it —
the request will silently miss the feature.

### `BEST_EFFORT`

The framework approximates the feature for you. For most features it
also **removes the field from the query_record before calling your
executor**, so the executor does not need to know about it. The one
exception is `response_format` — see the note below. What the
framework does:

- **`system_prompt = BEST_EFFORT`** — prepended to the first user
  message (if `chat`) or to the prompt string (if `prompt`), and
  `query_record.system_prompt` is cleared. Implemented in
  `feature_adapter.py:148-156, 200-201`.
- **`messages = BEST_EFFORT`** — chat history collapsed to a single
  string prompt via `chat.export(export_single_prompt=True)`. The
  executor receives `query_record.prompt` instead of
  `query_record.chat`.
- **`parameters.* = BEST_EFFORT`** — the field is set to `None` on
  the adapted query_record (`feature_adapter.py:281-307`). Effectively
  the parameter is dropped silently.
- **`response_format.json = BEST_EFFORT`** — appends `"You must
  respond with valid JSON."` to the prompt or system. **`query_record.
  response_format.type` is left as `JSON`** — the executor still sees
  it and may (optionally) enable a provider-side JSON mode on top of
  the prompt guidance. Either way, `result_adapter` will `json.loads`
  the returned TEXT block into a JSON block.
- **`response_format.pydantic = BEST_EFFORT`** — appends `"You must
  respond with valid JSON that follows this schema: {schema}"`.
  **`query_record.response_format.type` is left as `PYDANTIC`** and
  `pydantic_class` is still available. The executor can optionally
  enable a provider-side JSON mode. `result_adapter` parses the
  returned TEXT block through `json.loads` +
  `pydantic_class.model_validate` and produces the `PYDANTIC_INSTANCE`
  block automatically.

> **Note on `response_format`.** Unlike the other BEST_EFFORT features,
> `_adapt_response_format` (`feature_adapter.py:236-279`) does **not**
> clear `query_record.response_format.type` for `json` or `pydantic`.
> This is intentional: the executor needs the type so it can flip on a
> provider's native JSON mode (e.g. OpenAI's `response_format={'type':
> 'json_object'}`) to make the client-side parse reliable. Doing so is
> optional — if your SDK has no JSON mode, just return TEXT and let the
> framework parse it.

### `NOT_SUPPORTED`

The framework **raises `ValueError` at adaptation time** if the user
requests this feature. The executor never runs. Use this when:

- The capability is genuinely unavailable on the endpoint, AND
- A caller might reasonably try it (so they need a clear error
  instead of silent drop).

For features no one would ever try on this endpoint (e.g., image
generation on a text-only chat endpoint), omit the field rather than
spelling out `NOT_SUPPORTED`.

### The "No Best-Effort" Rule

`feature_adapter.py:19-21` defines:

```python
_NO_BEST_EFFORT_RESPONSE_FORMATS = (
    "text", "image", "audio", "video", "multi_modal",
)
```

You **cannot** mark `text` / `image` / `audio` / `video` /
`multi_modal` as `BEST_EFFORT`. The framework raises an Exception
if you do. The reason is semantic: there is no meaningful prompt
guidance for "best-effort generate an image." These five must be
either `SUPPORTED` or `NOT_SUPPORTED` (or omitted).

`json` and `pydantic` **can** be `BEST_EFFORT` because the framework
has prompt-injection fallbacks for them.

### Quick Rule Table

| Level | What framework does | What executor does |
|---|---|---|
| `SUPPORTED` | Keeps the feature, passes to executor | Wires it into the SDK call |
| `BEST_EFFORT` (`system_prompt` / `messages` / `parameters.*`) | Approximates and clears the field on the query_record | Nothing — doesn't see the feature |
| `BEST_EFFORT` (`response_format.json` / `response_format.pydantic`) | Injects prompt/system guidance; **leaves `response_format.type` set** | Optional: enable provider JSON mode. Otherwise return a TEXT block and let `result_adapter` parse it. |
| `NOT_SUPPORTED` | Raises `ValueError` if requested | Never sees the feature |
| Omitted (None) | Treated as `NOT_SUPPORTED` | Never sees the feature |

---

## 8. Writing an Executor

Every executor follows the same anatomy. Use the canonical pattern
below as your starting point.

```python
def _<endpoint>_executor(
    self,
    query_record: types.QueryRecord) -> types.ResultRecord:
  # 1. Bind the SDK method as a partial.
  call = functools.partial(self.api.<sdk_method>)
  call = functools.partial(call, model=(
      query_record.provider_model.provider_model_identifier
  ))

  # 2. Add input.
  if query_record.prompt is not None:
    call = functools.partial(
        call, messages=[{'role': 'user', 'content': query_record.prompt}])
  if query_record.chat is not None:
    call = functools.partial(call, messages=query_record.chat['messages'])
  if query_record.system_prompt is not None:
    call = functools.partial(call, system=query_record.system_prompt)

  # 3. Add parameters (only the fields the user actually set).
  if query_record.parameters is not None:
    if query_record.parameters.temperature is not None:
      call = functools.partial(
          call, temperature=query_record.parameters.temperature)
    # ... max_tokens, stop, n, thinking ...

  # 4. Add tools.
  if query_record.tools is not None:
    if types.Tools.WEB_SEARCH in query_record.tools:
      call = functools.partial(call, tools=[{...}])

  # 5. Add response format.
  # ... see section 10 ...

  # 6. Execute through the safe wrapper.
  response, result_record = self._safe_provider_query(call)
  if result_record.error is not None:
    return result_record

  # 7. Parse response into MessageContent blocks.
  result_record.content = [
      message_content.MessageContent(
          type=message_content.ContentType.TEXT,
          text=response.choices[0].message.content,
      )
  ]
  return result_record
```

### What `query_record` Carries (After FeatureAdapter)

The `query_record` you receive is a deep copy with best-effort
features already removed. You should treat it as your only source of
truth.

| Field | When set | Notes |
|---|---|---|
| `prompt` | When user passed `prompt=` (or BEST_EFFORT collapsed `chat`) | A plain string |
| `chat` | When user passed `messages=` and `messages` is `SUPPORTED` | A `Chat` object with `messages` list and optional `system_prompt` |
| `system_prompt` | When user passed `system_prompt=` and it's `SUPPORTED` (and not best-effort prepended) | A plain string |
| `parameters` | When user passed any parameter; `None` if all were dropped | A `ParameterType` with only the SUPPORTED fields populated |
| `tools` | When user passed any tool | A `list[Tools]` |
| `response_format` | Always set; defaults to `TEXT` | A `ResponseFormat` with `type` and optional `pydantic_class` |
| `provider_model` | Always set | `provider_model_identifier` is the SDK model id |

### What `_safe_provider_query` Expects

`model_connector.py:474-488`. It takes a **zero-argument callable**
that returns the SDK's response object. On success it returns
`(response, ResultRecord(status=SUCCESS, role=ASSISTANT))`; on
exception it returns `(None, ResultRecord(status=FAILED, error=...,
error_traceback=...))`.

Always check `result_record.error is not None` and short-circuit
before parsing — the response will be `None` on failure.

For SDK calls that need a context manager (e.g., Anthropic's
streaming surface), wrap the partial in a small helper. Claude does
this with `_run_stream`:

```python
def _run_stream(self, stream_partial):
  with stream_partial() as stream:
    return stream.get_final_message()

# at the call site:
response, result_record = self._safe_provider_query(
    functools.partial(self._run_stream, stream))
```

### What You Must Populate on `ResultRecord`

| Field | Required? | Notes |
|---|---|---|
| `content` | **Required** | `list[MessageContent]` with parsed response blocks |
| `choices` | Optional | `list[ChoiceType]` if your provider returns multiple choices (`n > 1`); each choice has its own `content` list |

Everything else (`status`, `role`, `error`, `error_traceback`,
`output_text`, `output_json`, `output_pydantic`, `usage`, `timestamp`,
`tool_usage`) is set by the framework. Don't touch them.

---

## 9. Parameter Handling

Parameters live on `query_record.parameters`. Map each one to your
SDK's name. Reference table for the three finalized providers:

| proxai param | OpenAI (chat.completions) | Gemini (generate_content) | Claude (beta.messages.stream) |
|---|---|---|---|
| `temperature` | `temperature` | `config.temperature` | `temperature` |
| `max_tokens` | `max_completion_tokens` | `config.max_output_tokens` | `max_tokens` |
| `stop` | `stop` | `config.stop_sequences` | `stop_sequences` |
| `n` | `n` | not supported | not supported |
| `thinking` | `reasoning_effort='low'/'medium'/'high'` | `config.thinking_config = ThinkingConfig(thinking_budget=…)` | `thinking={'type': 'enabled', 'budget_tokens': …}` |

For `thinking`, the proxai enum is `ThinkingType.LOW/MEDIUM/HIGH`. If
your SDK takes a token budget, define a constant mapping at the top
of your connector:

```python
_THINKING_BUDGETS = {
    types.ThinkingType.LOW: 1024,
    types.ThinkingType.MEDIUM: 8192,
    types.ThinkingType.HIGH: 24576,
}
```

If your SDK takes a string effort level, use `.value.lower()`:

```python
reasoning_effort=query_record.parameters.thinking.value.lower()
```

If thinking is enabled and your SDK requires `max_tokens > thinking
budget` (Anthropic does), validate it at request build time. Claude's
`_add_common_params` raises a `ValueError` with an explicit message
when the user-provided `max_tokens` is too small — better than a
cryptic API 400.

---

## 10. Response Formats: TEXT / JSON / PYDANTIC

This is the section to read carefully. The PYDANTIC and JSON paths
have a SUPPORTED-vs-BEST_EFFORT split that determines how much your
executor has to do.

### TEXT

Trivial. Don't set anything special on the SDK call. Return a
`MessageContent(type=ContentType.TEXT, text=...)` block per text
segment in the response. `text=SUPPORTED` is the only sensible level
for any text-generating endpoint.

### JSON

There are two patterns depending on whether your SDK has a free-JSON
mode (no schema required).

#### Pattern A — Native free-JSON mode → `json=SUPPORTED`

If your SDK has a flag like:

- OpenAI: `response_format={'type': 'json_object'}`
- Gemini: `config.response_mime_type = 'application/json'`

Then mark `json=SUPPORTED`. In the executor, set the flag when
`query_record.response_format.type == JSON`. Return a TEXT
`MessageContent` containing the JSON string — `result_adapter` will
call `json.loads` on it and convert to a JSON `MessageContent`
automatically.

#### Pattern B — No free-JSON mode → `json=BEST_EFFORT`

Anthropic is the canonical example. Anthropic's structured outputs
require a JSON schema **and** the SDK rewrites every object schema
to `additionalProperties: False`, which collapses any "open object"
type into "must be `{}`". You cannot express "any JSON object"
through Anthropic structured outputs.

For providers in this situation:

1. Mark `json=BEST_EFFORT`.
2. The framework will inject `"You must respond with valid JSON."`
   into the prompt automatically.
3. The framework **leaves `query_record.response_format.type` set to
   `JSON`** (see section 7 note), so your executor can still see it.
4. The model returns text — typically markdown-wrapped (`` ```json
   {…} ``` ``) or with prefatory natural language.
5. In your executor, after `_parse_content_blocks`, convert TEXT
   blocks to JSON blocks using `self._extract_json_from_text` (the
   base class helper at `model_connector.py:236-283`):

```python
needs_json = (
    query_record.response_format is not None
    and query_record.response_format.type == types.ResponseFormatType.JSON)

# … parse response normally …
result_record.content = self._parse_content_blocks(response)

if needs_json:
  result_record.content = [
      message_content.MessageContent(
          type=message_content.ContentType.JSON,
          json=self._extract_json_from_text(c.text),
      ) if c.type == message_content.ContentType.TEXT else c
      for c in result_record.content
  ]
```

`_extract_json_from_text` already handles markdown fences (with and
without language tag), surrounding prose, brace extraction, and
Python-style single-quoted dicts. It is tested in
`tests/connectors/test_model_connectors.py:TestExtractJsonFromText`.
**Don't write your own** — reuse it.

### PYDANTIC

Same two-pattern split.

#### Pattern A — Native structured-output parsing → `pydantic=SUPPORTED`

If your SDK accepts a Pydantic class for native parsing:

- OpenAI: `beta.chat.completions.parse(response_format=PydanticClass)`
- Anthropic: `beta.messages.stream(output_format=PydanticClass)`

Then mark `pydantic=SUPPORTED`. In the executor, when
`response_format.type == PYDANTIC`, pass the pydantic class to the
SDK and emit a `PYDANTIC_INSTANCE` `MessageContent` populated from
the SDK's already-parsed object:

```python
result_record.content = [
    message_content.MessageContent(
        type=message_content.ContentType.PYDANTIC_INSTANCE,
        pydantic_content=message_content.PydanticContent(
            class_name=query_record.response_format.pydantic_class.__name__,
            class_value=query_record.response_format.pydantic_class,
            instance_value=response.parsed_output,  # provider-specific field name
        ),
    )
]
```

`result_adapter` will pass `PYDANTIC_INSTANCE` blocks through
unchanged.

#### Pattern B — No native parsing → `pydantic=BEST_EFFORT`

Mark `pydantic=BEST_EFFORT`. The framework will:

1. Compute the JSON schema from the user's pydantic class via
   `pydantic_class.model_json_schema()`.
2. Append `"You must respond with valid JSON that follows this
   schema: {schema}"` to the prompt or system.
3. **Leave `query_record.response_format.type` set to `PYDANTIC`**
   (see section 7 note), so your executor can still see it.

Your executor returns a TEXT block.
`result_adapter._adapt_message_content` (`result_adapter.py:139-150`)
will:

1. `json.loads` the text.
2. Call `pydantic_class.model_validate(json_value)`.
3. Produce a `PYDANTIC_INSTANCE` block.

Because the type is still on the query_record, your executor
**may optionally** enable a provider-side JSON mode on top of the
prompt injection. OpenAI's `_chat_completions_create_executor` does
this — when it sees `type == PYDANTIC` it sets
`response_format={'type': 'json_object'}` so the model is forced to
emit parseable JSON, which makes the client-side
`json.loads` + `model_validate` step reliable. If your SDK has no
such mode, just return TEXT and the framework handles the rest.

### Why the Distinction Matters

| Level | Who handles parsing | Executor work |
|---|---|---|
| `SUPPORTED` | The SDK (server-side) | Pass the schema/class to the SDK; return `PYDANTIC_INSTANCE` from the parsed result |
| `BEST_EFFORT` | The framework + result_adapter (client-side) | Nothing — return TEXT, framework parses |

If you mark `pydantic=SUPPORTED` but forget to wire `output_format`
into the SDK call, the model gets no schema enforcement and no
prompt guidance (the framework skips injection for SUPPORTED). The
result will silently be wrong shapes. Mark accurately.

---

## 11. Thinking Blocks

When a model emits extended reasoning, parse it into a
`ContentType.THINKING` `MessageContent`. `result_adapter` passes
`THINKING` blocks through unchanged
(`result_adapter.py:122-130`), and `_adapt_output_values` ignores
them when populating `output_text` / `output_json` (it iterates in
reverse and stops at the first matching content type).

How each finalized provider exposes thinking:

| Provider | Detection | Text field |
|---|---|---|
| Anthropic Claude | `block.type == 'thinking'` | `block.thinking` |
| Google Gemini    | `part.thought` is True | `part.text` |
| OpenAI Responses | `output.type == 'reasoning'` | `output.summary[*].text` |

Emission pattern (Claude):

```python
if hasattr(block, 'type') and block.type == 'thinking':
  parsed.append(
      message_content.MessageContent(
          type=message_content.ContentType.THINKING,
          text=block.thinking,
      )
  )
```

When the user uses `parameters.thinking`, also add the corresponding
SDK config (see section 9).

---

## 12. Tools and Web Search Citations

`web_search` is the only tool currently modeled. Mark
`tools.web_search=SUPPORTED` if the SDK has a web search tool, then
in the executor:

```python
if query_record.tools is not None:
  if types.Tools.WEB_SEARCH in query_record.tools:
    call = functools.partial(call, tools=[{...provider tool spec...}])
```

### Citation Sources (Two Kinds)

Web search citations can arrive in **two distinct places** in a
response. Be thorough: surface both when both exist.

**1. Standalone tool result blocks.** One block per search call,
listing what the search engine returned. These represent "the
search ran and found these N results."

| Provider | How it appears |
|---|---|
| Claude | A content block with `block.type == 'web_search_tool_result'` and `block.content` as a list of result objects (each with `.title`, `.url`). The `content` field is a union — also handle the error variant. |

**2. Inline annotations on text blocks.** Per-text-segment references
to source URLs. These represent "this specific text came from this
source" — usually the more user-facing kind of citation.

| Provider | How it appears |
|---|---|
| OpenAI Responses | `content.annotations` list on each text block, each with `.title` and `.url` |
| Anthropic Claude | `block.citations` list on each text block; filter to entries where `citation.type == 'web_search_result_location'`, each with `.title` and `.url` |
| Google Gemini    | `candidate.grounding_metadata.grounding_chunks` list, each with `chunk.web.title` and `chunk.web.uri` |

### Emission Pattern

For each citation source, build one `TOOL` `MessageContent`:

```python
parsed.append(
    message_content.MessageContent(
        type=message_content.ContentType.TOOL,
        tool_content=message_content.ToolContent(
            name='web_search',
            kind=message_content.ToolKind.RESULT,
            citations=[
                message_content.Citation(title=…, url=…)
                for … in source
            ],
        ),
    )
)
```

`result_adapter` passes `TOOL` blocks through unchanged. The user's
test code typically iterates `result.content` looking for blocks
with `type == ContentType.TOOL`. Even if the citation list is empty,
emit the TOOL block — the presence of the block is itself
information ("the model ran a search"), and tests often assert at
least one TOOL block exists.

---

## 13. The Mock Model

When `run_type == TEST`, the framework uses `init_mock_model()`
instead of `init_model()`. The mock client is required because tests
exercise the executor without hitting real APIs.

### The Contract

The mock must mirror the **exact attribute path the executor uses**,
returning objects whose fields the executor reads. Nothing more.

Minimal viable shape:

- If your executor calls `self.api.foo.bar.create(**kwargs)`, your
  mock needs `api.foo.bar.create` to exist and return an object
  exposing the fields the executor accesses afterwards.
- If your executor uses a context manager
  (`with self.api.foo.stream(**kwargs) as s: s.get_final_message()`),
  the mock's `stream()` must return an object with `__enter__` /
  `__exit__` whose `__enter__` returns an object with
  `get_final_message()`.

### Template — Plain Method

```python
class _MockResponse:
  def __init__(self):
    # Whatever fields the executor reads on the SDK response
    self.choices = [_MockChoice()]


class _MockMethod:
  def create(self, **kwargs) -> _MockResponse:
    return _MockResponse()


class WidgetMock:
  def __init__(self):
    self.foo = _MockMethod()
```

### Template — Context Manager (e.g., Streaming)

```python
class _MockStream:
  def __enter__(self):
    return self
  def __exit__(self, exc_type, exc, tb):
    return False
  def get_final_message(self) -> _MockResponse:
    return _MockResponse()


class _MockMethod:
  def stream(self, **kwargs) -> _MockStream:
    return _MockStream()
```

### Tips

- The mock doesn't need to be smart. Returning generic dummy data
  (`'mock response'`, `parsed_output=None`) is fine. Tests that need
  specific responses can subclass or monkey-patch.
- The mock only needs the methods your executor actually calls. If
  you delete an executor or change which SDK method it calls, prune
  the mock to match. Dead mock surface accumulates.
- Reference the three finalized mocks
  (`openai_mock.py`, `gemini_mock.py`, `claude_mock.py`) for shape.

---

## 14. Registering the Connector

Three edits make your connector visible to the framework.

### 1. `model_registry.py` — add to `_MODEL_CONNECTOR_MAP`

`src/proxai/connectors/model_registry.py`:

```python
import proxai.connectors.providers.widget as widget_provider

_MODEL_CONNECTOR_MAP = {
    'openai': openai_provider.OpenAIConnector,
    'claude': claude_provider.ClaudeConnector,
    'gemini': gemini_provider.GeminiConnector,
    'widget': widget_provider.WidgetConnector,  # ← add
    ...
}
```

The map key must equal your `PROVIDER_NAME`.
`get_model_connector()` raises `"Provider not supported"` if the key
is missing.

### 2. `model_configs.py` — add to `PROVIDER_KEY_MAP`

`src/proxai/connectors/model_configs.py:20-33`:

```python
PROVIDER_KEY_MAP = MappingProxyType({
    'claude': ('ANTHROPIC_API_KEY',),
    'openai': ('OPENAI_API_KEY',),
    'gemini': ('GEMINI_API_KEY',),
    'widget': ('WIDGET_API_KEY',),  # ← add
    ...
})
```

Keys here must match your `PROVIDER_API_KEYS` class attribute. Most
providers have one env var; Databricks is an example with two
(`('DATABRICKS_TOKEN', 'DATABRICKS_HOST')`).

### 3. `model_configs_data/v1.2.0.json` — add model entries

`src/proxai/connectors/model_configs_data/v1.2.0.json`:

```json
{
  "provider_model_configs": {
    "widget": {
      "widget-large": {
        "provider_model": {
          "provider": "widget",
          "model": "widget-large",
          "provider_model_identifier": "widget-large-2025-01-01"
        },
        "pricing": {
          "input_token_cost": 0.0,
          "output_token_cost": 0.0
        },
        "features": {
          /* serialized FeatureConfigType — omit if it should inherit
             from the endpoint's ENDPOINT_CONFIG */
        },
        "metadata": {
          "call_type": "TEXT",
          "is_recommended": true,
          "model_size_tags": ["medium"]
        }
      }
    }
  }
}
```

`provider_model_identifier` is the literal model id the SDK accepts
(may differ from the user-facing `model` name). Multiple entries per
provider are fine — one per model.

---

## 15. Verification Checklist and Pitfalls

### Checklist

Run each item before declaring the connector done.

- [ ] **Imports cleanly.** `python -c "from proxai.connectors.providers.widget import WidgetConnector"` succeeds. If `__init_subclass__` rejects something, this is where it fires.
- [ ] **Mock imports.** `from proxai.connectors.providers.widget_mock import WidgetMock` succeeds and `WidgetMock()` instantiates.
- [ ] **TEST mode round-trip.** A `px.generate(provider_model=('widget', '...'))` call in `RunType.TEST` returns a `CallRecord` with non-None `result.output_text` and no errors.
- [ ] **PRODUCTION text.** A real API call with no special features returns a populated TEXT block.
- [ ] **PRODUCTION parameters.** `temperature`, `max_tokens`, `stop` each take effect. Verify by inspection or by stop-sequence behavior.
- [ ] **PRODUCTION response_format=JSON.** Returns valid `output_json`.
- [ ] **PRODUCTION response_format=PYDANTIC.** Returns a populated `output_pydantic` instance.
- [ ] **PRODUCTION thinking** (if declared `SUPPORTED`). The result contains at least one `THINKING` block.
- [ ] **PRODUCTION web_search** (if declared `SUPPORTED`). The result contains at least one `TOOL` block with non-empty citations.
- [ ] **`NOT_SUPPORTED` raises.** Requesting a feature you marked `NOT_SUPPORTED` raises `ValueError` from `FeatureAdapter`, not from the SDK.

### Common Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: WidgetConnector must define ENDPOINT_PRIORITY` at import | Missing one of the 5 required class attrs | Add it |
| `ValueError: WidgetConnector: ENDPOINT_PRIORITY and ENDPOINT_CONFIG keys do not match` | Typo in an endpoint key | Make all three dicts use identical keys |
| `Provider not supported. widget` | Not registered in `_MODEL_CONNECTOR_MAP` | Add to `model_registry.py` |
| `provider_token_value_map needs to contain WIDGET_API_KEY` | Missing from `PROVIDER_KEY_MAP` or the env var isn't set | Add to `model_configs.py:PROVIDER_KEY_MAP` and to your environment |
| `'<format>' response format config cannot be best effort` (e.g. `'text'`, `'image'`, `'audio'`, `'video'`, `'multi_modal'`) | You marked `text` / `image` / `audio` / `video` / `multi_modal` as `BEST_EFFORT` | Use `SUPPORTED` or `NOT_SUPPORTED` (or omit) for those five — they cannot be best-effort. `json` and `pydantic` are the only formats that support `BEST_EFFORT`. |
| `JSONDecodeError: Expecting value` from `result_adapter` | Marked `json=SUPPORTED` but the executor doesn't actually enforce JSON, so the model returned natural text | Either wire the SDK's JSON flag in the executor, or downgrade to `BEST_EFFORT` and use `_extract_json_from_text` |
| Pydantic returns wrong shape silently | Marked `pydantic=SUPPORTED` but didn't pass `output_format=PydanticClass` to the SDK | Wire it, or downgrade to `BEST_EFFORT` and let the framework inject schema guidance |
| Tests pass but live API calls fail with `AttributeError` | Mock has fields the real SDK doesn't, or vice versa | The mock should mirror only what the executor reads — keep them in sync |
| `n` parameter silently dropped on Anthropic | `parameters.n` left as `None` (defaults to `NOT_SUPPORTED`) but framework treats `None` as "not declared, drop silently" rather than raising | Set `n=NOT_SUPPORTED` explicitly so users get a clear error |
| Web search runs but tests find no `TOOL` block | Citations are on text-block annotations, not tool_result blocks (or vice versa) | Surface **both** sources — section 12 |

### When in Doubt

Read the finalized connectors. They are short:

- `src/proxai/connectors/providers/openai.py` — the multi-endpoint
  case, with a different parser per executor
- `src/proxai/connectors/providers/gemini.py` — the
  config-object-on-the-side pattern (Gemini puts most things on
  `GenerateContentConfig` rather than as kwargs)
- `src/proxai/connectors/providers/claude.py` — the consolidated
  single-endpoint case with structured outputs and streaming

Each one is under 300 lines. If your connector is much larger, you
are probably re-implementing something the framework already does.
