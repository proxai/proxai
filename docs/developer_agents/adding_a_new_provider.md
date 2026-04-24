# Adding a New Provider

Source of truth: `src/proxai/connectors/provider_connector.py`
(`ProviderConnector` base; `__init_subclass__` validates the contract
at import time), `src/proxai/connectors/feature_adapter.py` and
`src/proxai/connectors/result_adapter.py` (what runs before and after
your executor), `src/proxai/types.py` (`FeatureConfigType` /
`FeatureSupportType` / `OutputFormatConfigType` /
`InputFormatConfigType` / `ParameterConfigType` / `ToolConfigType` /
`ExecutorResult`), `src/proxai/connectors/model_registry.py` and
`model_configs.py` (two registration maps),
`model_configs_data/v1.3.0.json` (bundled model registry), and
`pyproject.toml`. If this document disagrees with those files, the
files win — update this document.

This is the self-contained end-to-end guide: probe the SDK, write the
connector class, declare `FeatureConfigType`, implement the executor,
register, test. Every piece of data a new-provider author needs is
inline — this doc is both the procedural recipe and the normative
contract (the older `provider_connector_contract.md` has been deleted
in favor of this single source).

See also:
[`feature_adapters_logic.md`](./feature_adapters_logic.md) (what the
adapter does to `query_record` on either side of the executor),
[`chat_export_logic.md`](./chat_export_logic.md) (how `Chat.export`
reshapes chat messages before your executor sees them),
[`files_internals.md`](./files_internals.md) §1-4 (File API dispatch,
auto-upload, `_to_*_part` consumers),
[`testing_conventions.md`](./testing_conventions.md) §3-4 (mock
contract, fixture patterns),
[`sanity_check_before_merge.md`](./sanity_check_before_merge.md) §8
(pre-merge audit for new-provider PRs).

---

## 1. Provider-addition flow (current)

```
Your task: add a provider "widget"
│
│   # Gate A — write and validate the connector
├── 0. scripts/probe_widget.py                              # bare SDK probes (throwaway)
├── 1. src/proxai/connectors/providers/widget.py            # real connector
├── 2. src/proxai/connectors/providers/widget_mock.py       # structural-duck-typed mock
├── 3. src/proxai/connectors/model_registry.py              # +1 import, +1 map entry
├── 4. src/proxai/connectors/model_configs.py               # +1 PROVIDER_KEY_MAP entry
├── 5. src/proxai/connectors/model_configs_data/v1.3.0.json # +N model entries
├── 6. pyproject.toml + poetry.lock                         # +dep if needed
│
│   # Gate B — optional, only if the provider has a File API
├── 7. src/proxai/connectors/file_helpers.py                # 4 dispatch tables + capability split
│
│   # Gate C — prove it works
├── 8. tests/connectors/providers/test_widget.py            # unit tests
└── 9. integration_tests/proxai_api_test.py                 # @integration_block(s)
```

Step 0 is throwaway — it never ships — but doing it first is how
you discover which `FeatureSupportType` each `FeatureConfigType`
field should carry. Steps 1-6 are the minimum for a
production-callable connector. Step 7 is only for providers with a
native File API. Steps 8-9 are what `sanity_check_before_merge.md`
§8 expects at PR review.

### 1.1 Reference connectors

| Provider | Connector | Mock | SDK | Template use |
|---|---|---|---|---|
| OpenAI | `openai.py` | `openai_mock.py` | `openai` | Multi-endpoint |
| Claude | `claude.py` | `claude_mock.py` | `anthropic` | Streaming, structured outputs, files, documents, thinking, web_search |
| Gemini | `gemini.py` | `gemini_mock.py` | `google-genai` | Config-object-on-the-side shape |
| Mistral | `mistral.py` | `mistral_mock.py` | `mistralai` | — |
| Cohere | `cohere.py` | `cohere_mock.py` | `cohere` | — |
| Grok | `grok.py` | `grok_mock.py` | `xai-sdk` | — |
| DeepSeek | `deepseek.py` | `deepseek_mock.py` | `openai` (base_url override) | **Simplest complete template** — one endpoint, Pattern-2 system, no File API |
| HuggingFace | `huggingface.py` | `huggingface_mock.py` | `huggingface_hub` | — |

Copy **`deepseek.py`** for an OpenAI-compatible HTTP provider.
Otherwise start from the closest SDK-family match.

---

## 2. Probe the SDK first (step 0)

**Before writing any connector code**, write a throwaway script that
calls the provider's SDK directly and tries every feature ProxAI
models: text, chat, system prompt, each parameter, JSON mode,
Pydantic parsing, web search, each multimodal input, streaming,
thinking. Without this you are guessing from provider docs, which
routinely claim support for features that silently no-op or 400 out.

### 2.1 Observation → `FeatureSupportType`

| Observed behavior | Encode as |
|---|---|
| SDK accepts the kwarg, response visibly reflects it | `SUPPORTED` |
| SDK accepts but silently ignores (output identical with/without) | `NOT_SUPPORTED` — never `SUPPORTED` |
| SDK rejects with 400 / validation error | `NOT_SUPPORTED` |
| No analogue, but framework's prompt-injection fallback is meaningful (only for `system_prompt`, `messages`, `json`, `pydantic`, parameters) | `BEST_EFFORT` |
| Provider has a close analogue, executor can translate | `BEST_EFFORT` with connector-side adaptation (see §7) |
| Feature only exists on a beta/structured endpoint | Split into a second endpoint, or keep one endpoint and gate via flags |

### 2.2 Script template

```python
# scripts/probe_widget.py — throwaway; do NOT commit.
# Run: poetry run python3 scripts/probe_widget.py
import os, pydantic
from widget_sdk import Widget

client = Widget(api_key=os.environ['WIDGET_API_KEY'])
MODEL = 'widget-pro-1'


def probe(label, fn):
  try:
    r = fn()
    print(f'[ OK ] {label}:', repr(r)[:160])
  except Exception as e:
    print(f'[FAIL] {label}: {type(e).__name__}: {e}')


# 1. Basic text
probe('text', lambda: client.chat.completions.create(
    model=MODEL, messages=[{'role': 'user', 'content': 'say "pong"'}]))

# 2. Multi-turn chat
probe('chat', lambda: client.chat.completions.create(model=MODEL, messages=[
    {'role': 'user', 'content': 'my name is Sam'},
    {'role': 'assistant', 'content': 'hi Sam'},
    {'role': 'user', 'content': 'what is my name?'}]))

# 3. System prompt — try both patterns, see which is idiomatic
probe('system kwarg', lambda: client.chat.completions.create(
    model=MODEL, system='reply in pig latin',
    messages=[{'role': 'user', 'content': 'hello'}]))
probe('system in messages', lambda: client.chat.completions.create(
    model=MODEL, messages=[
        {'role': 'system', 'content': 'reply in pig latin'},
        {'role': 'user', 'content': 'hello'}]))

# 4. Parameters — one at a time
probe('temperature', lambda: client.chat.completions.create(
    model=MODEL, temperature=0,
    messages=[{'role': 'user', 'content': 'pick a number 1..100'}]))
probe('max_tokens',  lambda: client.chat.completions.create(
    model=MODEL, max_tokens=5,
    messages=[{'role': 'user', 'content': 'count to 1000'}]))
probe('stop',        lambda: client.chat.completions.create(
    model=MODEL, stop=['DONE'],
    messages=[{'role': 'user', 'content': 'say hi DONE say bye'}]))
probe('n',           lambda: client.chat.completions.create(
    model=MODEL, n=3,
    messages=[{'role': 'user', 'content': 'random word'}]))
probe('thinking',    lambda: client.chat.completions.create(
    model=MODEL, reasoning_effort='high',   # whatever the SDK calls it
    messages=[{'role': 'user', 'content': 'prove sqrt(2) irrational'}]))

# 5. JSON mode (free JSON)
probe('json_object', lambda: client.chat.completions.create(
    model=MODEL, response_format={'type': 'json_object'},
    messages=[{'role': 'user', 'content': 'return {"a": 1}'}]))

# 6. Native Pydantic parse (if the SDK exposes it)
class Person(pydantic.BaseModel): name: str; age: int
probe('pydantic parse', lambda: client.chat.completions.parse(
    model=MODEL, response_format=Person,
    messages=[{'role': 'user', 'content': 'Alice, 30'}]))

# 7. Web search (if available)
probe('web_search', lambda: client.chat.completions.create(
    model=MODEL, tools=[{'type': 'web_search'}],
    messages=[{'role': 'user', 'content': 'latest news about X'}]))

# 8. Multimodal inputs — one per type
probe('image', lambda: client.chat.completions.create(
    model=MODEL, messages=[{'role': 'user', 'content': [
        {'type': 'text', 'text': 'describe'},
        {'type': 'image_url', 'image_url': {'url': 'https://...'}}]}]))
# repeat for document, audio, video, as applicable
```

Every line of output tells you which level to assign. **Keep the
script local; delete before opening the PR.** Probe scripts require
real keys, drift fast, and are not maintained.

---

## 3. The required contract

Your class inherits from `provider_connector.ProviderConnector` and
is validated at **import time** by `__init_subclass__`
(`provider_connector.py:93-124`). Import raises if any of the five
required attributes is missing, or if keys differ across
`ENDPOINT_PRIORITY`, `ENDPOINT_CONFIG`, `ENDPOINT_EXECUTORS`.

### 3.1 Five required class attributes

| Attribute | Type | Purpose |
|---|---|---|
| `PROVIDER_NAME` | `str` | Lowercase short id (`'widget'`). Must equal the key in `_MODEL_CONNECTOR_MAP`, `PROVIDER_KEY_MAP`, and the JSON registry. |
| `PROVIDER_API_KEYS` | `list[str]` | Env-var names the SDK needs. Validated against `provider_token_value_map` in `_validate_provider_token_value_map` (`provider_connector.py:134-142`). |
| `ENDPOINT_PRIORITY` | `list[str]` | Ordered endpoint keys. `_find_compatible_endpoint` iterates this, picks the first `SUPPORTED`; falls back to first `BEST_EFFORT` only when `feature_mapping_strategy=BEST_EFFORT`. |
| `ENDPOINT_CONFIG` | `dict[str, FeatureConfigType]` | One entry per endpoint. §4. |
| `ENDPOINT_EXECUTORS` | `dict[str, str]` | Map endpoint key → **string** name of the executor method. `_prepare_execution` calls `getattr(self, name)`. |

### 3.2 Two abstract methods + one executor per endpoint

| Method | Signature | Purpose |
|---|---|---|
| `init_model(self)` | `() -> Any` | Real SDK client. Called lazily on first `self.api` access when `run_type=PRODUCTION`. |
| `init_mock_model(self)` | `() -> Any` | Structural-duck-typed mock. Called when `run_type=TEST`. §11. |
| `_<endpoint_key>_executor(self, query_record)` | `(QueryRecord) -> ExecutorResult` | One per entry in `ENDPOINT_EXECUTORS`. §6. |

**Endpoint key convention:** mirror the literal SDK call path with
dots. **Executor method name:** replace dots with underscores, prefix
`_`, suffix `_executor`.

| Endpoint key | Executor method |
|---|---|
| `chat.completions.create` | `_chat_completions_create_executor` |
| `beta.messages.stream` | `_beta_messages_stream_executor` |
| `responses.create` | `_responses_create_executor` |

### 3.3 One endpoint or many?

Multiple endpoints when the SDK has structurally different methods
(chat vs image vs async video) or when feature subsets are not
expressible as flags on one method. One endpoint when a single SDK
call covers everything via flags/config. Reference counts: openai 6,
gemini 2, claude 1. Start with one — splitting later is cheap,
merging is not.

---

## 4. `FeatureConfigType` field reference

`FeatureConfigType` (`types.py:207-218`). Every field defaults to
`None`, treated as `NOT_SUPPORTED` by `adapter_utils.resolve_support`.

### 4.1 Top-level fields

| Field | Type | Notes |
|---|---|---|
| `prompt` | `FeatureSupportType` | Single user prompt string. Most text endpoints: `SUPPORTED`. |
| `messages` | `FeatureSupportType` | Chat history. Most text endpoints: `SUPPORTED`. |
| `system_prompt` | `FeatureSupportType` | §4.6 for the two patterns. |
| `add_system_to_messages` | `bool \| None` | Pattern-2 marker. §4.6. |
| `parameters` | `ParameterConfigType \| None` | §4.2 |
| `tools` | `ToolConfigType \| None` | §4.3 |
| `input_format` | `InputFormatConfigType \| None` | §4.4. Governs content-block types in chat messages. |
| `output_format` | `OutputFormatConfigType \| None` | §4.5. Required — `FeatureAdapter` raises if `output_format.type` is unset. |

### 4.2 `ParameterConfigType` (`types.py:164-171`)

| Field | Controls |
|---|---|
| `temperature` | Float sampling temperature |
| `max_tokens` | Max output token cap |
| `stop` | Stop sequences (`str` or `list[str]`) |
| `n` | Number of completions to return |
| `thinking` | `ThinkingType.LOW/MEDIUM/HIGH` — extended reasoning |

Mark a parameter explicitly `NOT_SUPPORTED` when a user might
reasonably try it and the provider has no analogue — Claude sets
`n=NOT_SUPPORTED` so the framework surfaces a clear error rather
than silently dropping.

### 4.3 `ToolConfigType` (`types.py:175-178`)

Only `web_search: FeatureSupportType`. Adding a tool requires a new
field *and* framework routing — out of scope here.

### 4.4 `InputFormatConfigType` (`types.py:194-204`)

Governs which content-block types may appear in chat messages the
executor receives. Adapter enforces before the executor runs:
`NOT_SUPPORTED` raises; `BEST_EFFORT` converts JSON/Pydantic blocks
to text or drops unsupported media (except `document`, which
passes through for connector-side extraction).

| Field | Controls |
|---|---|
| `text` | Text blocks. Always `SUPPORTED` for any text endpoint |
| `image` | Inline base64 or URL image blocks |
| `document` | PDF / docx / text documents. `BEST_EFFORT` passes through for connector-side extraction via `content_utils.read_text_document` / `read_pdf_document` |
| `audio` | Inline audio blocks |
| `video` | Inline video blocks |
| `json` | `ContentType.JSON` chat blocks. `BEST_EFFORT` → text serialization |
| `pydantic` | `ContentType.PYDANTIC_INSTANCE` chat blocks. `BEST_EFFORT` → text serialization |

`document` is the only `BEST_EFFORT` passthrough
(`feature_adapter.py:510`). Every other media type at `BEST_EFFORT`
without a converter is dropped.

### 4.5 `OutputFormatConfigType` (`types.py:181-191`)

| Field | Controls |
|---|---|
| `text` | Plain text output |
| `json` | Free-form JSON |
| `pydantic` | Structured output to a Pydantic class |
| `image` / `audio` / `video` | Media generation |
| `multi_modal` | Mixed modalities in one response |

**Only `json` and `pydantic` may be `BEST_EFFORT`.** The five
media-ish fields (`text`, `image`, `audio`, `video`, `multi_modal`)
must be `SUPPORTED`, `NOT_SUPPORTED`, or omitted — the adapter
raises at request time if one is `BEST_EFFORT`
(`feature_adapter.py:13-15, 408-414`). There is no meaningful prompt
fallback for "best-effort generate an image".

**Hard rule: if `text=SUPPORTED`, `json` and `pydantic` must be at
least `BEST_EFFORT`.** Any text-producing endpoint can emit JSON /
Pydantic via the framework's built-in conversion, so leaving them
unset is a silent gap that forces users to fall back to another
endpoint. `sanity_check_before_merge.md` §6 flags this anti-pattern.
Details in §8.

### 4.6 System prompts: two patterns

Every endpoint picks exactly one.

|  | Pattern 1 — Native `system` kwarg | Pattern 2 — First `messages` entry |
|---|---|---|
| When | SDK has a dedicated `system=` kwarg | SDK has none; expects `{'role':'system', ...}` as first `messages` entry |
| Config | `system_prompt=SUPPORTED`; **do not** set `add_system_to_messages` | `system_prompt=SUPPORTED` **and** `add_system_to_messages=True` |
| On entry to executor | `query_record.system_prompt` set if user passed one | `query_record.system_prompt is None`; adapter folded the system prompt into `messages` whether user passed `prompt=` or `messages=` (`feature_adapter.py:307-316`) |
| Executor reads | `query_record.system_prompt` | Only `query_record.chat['messages']` — never `system_prompt` |
| Examples | `claude.py` | `deepseek.py` |

**Flag rules:** `add_system_to_messages=True` is valid only with
`system_prompt=SUPPORTED`. Never pair it with a native `system=`
kwarg. Flag absence means Pattern 1 — it's a declaration, not a
default.

**Support-level rules (both patterns):**
- `SUPPORTED` → framework keeps the field (P1) or folds into
  messages (P2).
- `BEST_EFFORT` → framework merges into prompt / first user message
  and clears `system_prompt`. Executor sees nothing.
- `NOT_SUPPORTED` → framework raises before the executor runs.

---

## 5. Support levels

Every `FeatureSupportType` field resolves to one of three values
(`types.py:155-160`). The framework's behavior depends on the level.

### 5.1 `SUPPORTED`

Framework keeps the field on `query_record` as-is. **Executor must
wire it into the SDK call.** Marking `SUPPORTED` and forgetting to
wire it is the single most common connector bug — the framework does
not catch you; the feature silently vanishes.

### 5.2 `BEST_EFFORT`

Framework approximates; for most fields it also clears the field on
the query_record before the executor runs. The exception is
`output_format`, whose `type` stays set so the executor can
optionally enable a provider-side JSON mode on top of prompt
injection.

| Field at `BEST_EFFORT` | Adapter | Executor sees |
|---|---|---|
| `system_prompt` | Prepends to prompt / first user message; clears `system_prompt` (`feature_adapter.py:281-291, 323-336`) | Nothing |
| `messages` | Collapses chat history to a single string via `chat.export(export_single_prompt=True)`; populates `query_record.prompt`, clears `chat` | A prompt, no chat |
| `parameters.*` | Sets the field to `None` on the adapted record (`feature_adapter.py:432-458`). If all parameter fields end up `None`, `query_record.parameters` itself is set to `None` | Nothing |
| `tools.web_search` | **Cannot** be `BEST_EFFORT` — adapter raises (`feature_adapter.py:381-385`) | n/a |
| `input_format.json` / `.pydantic` | Converts in-chat JSON/Pydantic blocks to text | Text blocks only |
| `input_format.document` | Passes through — connector extracts text | Raw document block |
| `input_format.image/audio/video` | Drops the block silently | No block |
| `output_format.json` | Appends `"You must respond with valid JSON."` to prompt/system; **leaves `output_format.type == JSON`** | `type=JSON` — may optionally flip on native JSON mode |
| `output_format.pydantic` | Appends `"You must respond with valid JSON that follows this schema: {schema}"`; **leaves `output_format.type == PYDANTIC`** | `type=PYDANTIC`, `pydantic_class` populated — may optionally flip on native JSON mode |
| `output_format.text/image/audio/video/multi_modal` | Adapter raises "cannot be best effort" (`feature_adapter.py:408-414`) | n/a |

### 5.3 `NOT_SUPPORTED`

Adapter raises `ValueError` before the executor runs when the user
requests it. Use when the capability is genuinely unavailable *and*
a caller might reasonably try it. For features no caller would try
(image output on a text-only endpoint), omit — `None` resolves to
`NOT_SUPPORTED` anyway, and omitting reads as "not applicable".

### 5.4 Quick rule table

| Level | Adapter behavior | Executor responsibility |
|---|---|---|
| `SUPPORTED` | Keeps the field | Wire it into the SDK call |
| `BEST_EFFORT` (most fields) | Approximates; clears | Nothing |
| `BEST_EFFORT` (`output_format.json/pydantic`) | Injects prompt guidance; **leaves `output_format.type`** | Optional native JSON mode |
| `NOT_SUPPORTED` | Raises `ValueError` | Never sees it |
| Omitted (`None`) | `NOT_SUPPORTED` | Never sees it |

---

## 6. Writing an executor

Canonical anatomy:

```python
def _<endpoint>_executor(
    self,
    query_record: types.QueryRecord,
) -> types.ExecutorResult:
  call = functools.partial(self.api.<sdk_method>)
  call = functools.partial(
      call, model=query_record.provider_model.provider_model_identifier)

  # Input — exactly one is set, unless Pattern 2 folded prompt→chat.
  if query_record.prompt is not None:
    call = functools.partial(
        call, messages=[{'role': 'user', 'content': query_record.prompt}])
  if query_record.chat is not None:
    # `chat` is a dict after adaptation, NOT a Chat object.
    messages = self._convert_messages(query_record.chat['messages'])
    call = functools.partial(call, messages=messages)
    if 'system_prompt' in query_record.chat:
      call = functools.partial(
          call, system=query_record.chat['system_prompt'])

  # Pattern 1 prompt path only. Pattern 2 must not read this.
  if query_record.system_prompt is not None:
    call = functools.partial(call, system=query_record.system_prompt)

  # Parameters (only SUPPORTED ones survive adaptation).
  if query_record.parameters is not None:
    if query_record.parameters.temperature is not None:
      call = functools.partial(
          call, temperature=query_record.parameters.temperature)
    # ... max_tokens / stop / n / thinking (§7)

  # Tools.
  if query_record.tools is not None:
    if types.Tools.WEB_SEARCH in query_record.tools:
      call = functools.partial(call, tools=[{...}])

  # Output format — see §8.
  if query_record.output_format.type in (
      types.OutputFormatType.JSON, types.OutputFormatType.PYDANTIC):
    call = functools.partial(call, response_format={'type': 'json_object'})

  # Execute.
  response, result_record = self._safe_provider_query(call)
  if result_record.error is not None:
    return types.ExecutorResult(result_record=result_record)

  # Parse into MessageContent blocks.
  result_record.content = [message_content.MessageContent(
      type=message_content.ContentType.TEXT,
      text=response.choices[0].message.content)]

  return types.ExecutorResult(
      result_record=result_record, raw_provider_response=response)
```

### 6.1 What `query_record` carries on entry

`_prepare_execution` (`provider_connector.py:542-577`) hands your
executor a deep-copied, adapted `query_record`. Treat it as your only
source of truth.

| Field | When set | Shape |
|---|---|---|
| `prompt` | User passed `prompt=`, or `BEST_EFFORT messages` collapsed chat → prompt | `str` |
| `chat` | User passed `messages=`, or Pattern 2 folded a prompt+system into a chat | `dict` with `messages` always, `system_prompt` optionally. **Not** a `Chat` object |
| `system_prompt` | Pattern 1 via prompt path only. `None` under Pattern 2 or any BEST_EFFORT merge | `str` |
| `parameters` | At least one parameter was `SUPPORTED`; `None` if all were dropped | `ParameterType` with only `SUPPORTED` fields populated |
| `tools` | Passed and `SUPPORTED` | `list[Tools]` |
| `output_format` | Always set; defaults to `OutputFormat(type=TEXT)` | `type`, plus `pydantic_class` / `pydantic_class_name` / `pydantic_class_json_schema` when type is PYDANTIC |
| `provider_model` | Always | `provider_model_identifier` is the SDK model id |

### 6.2 `_safe_provider_query` and `ExecutorResult`

`_safe_provider_query(fn)` (`provider_connector.py:579-594`) takes a
zero-argument callable and returns
`(response_or_None, ResultRecord)`. On exception it returns
`(None, ResultRecord(status=FAILED, error=..., error_traceback=...))`.
Always short-circuit on `result_record.error is not None` before
parsing — the response is `None`.

For SDK calls needing a context manager (streaming), wrap the partial
in a helper and hand the helper-partial in. Claude's `_run_stream`:

```python
def _run_stream(self, stream_partial):
  with stream_partial() as stream:
    return stream.get_final_message()

response, result_record = self._safe_provider_query(
    functools.partial(self._run_stream, stream))
```

`ExecutorResult` (`types.py:644-649`):
- `result_record` — always. The framework will adapt content,
  compute usage, stamp timestamps, and upload to ProxDash.
- `raw_provider_response` — success only. Exposed through
  `CallRecord.debug.raw_provider_response` when
  `DebugOptions.keep_raw_provider_response=True`.

On error, return `ExecutorResult(result_record=result_record)` — omit
`raw_provider_response` because the response is `None`.

### 6.3 What to populate on `ResultRecord`

| Field | When | Shape |
|---|---|---|
| `content` | Always on success | `list[MessageContent]` — every block the response contains, in order |
| `choices` | Only for multi-choice (`n > 1`) | `list[ChoiceType]` — each with its own `content` |

The rest (`status`, `role`, `error`, `error_traceback`,
`output_text`, `output_json`, `output_pydantic`, `usage`,
`timestamp`) is populated by the framework (`provider_connector.py`
lines 596-960). Don't touch them.

---

## 7. Parameters per provider

| ProxAI | OpenAI (`chat.completions.create`) | Claude (`beta.messages.stream`) | Gemini (`generate_content`) |
|---|---|---|---|
| `temperature` | `temperature` | `temperature` | `config.temperature` |
| `max_tokens` | `max_completion_tokens` | `max_tokens` | `config.max_output_tokens` |
| `stop` | `stop` | `stop_sequences` (list) | `config.stop_sequences` |
| `n` | `n` | not supported | not supported |
| `thinking` | `reasoning_effort='low'/'medium'/'high'` | `thinking={'type':'enabled','budget_tokens': N}` | `config.thinking_config = ThinkingConfig(thinking_budget=N)` |

`thinking` is `ThinkingType.LOW/MEDIUM/HIGH`. For token-budget SDKs
define a constant map at the top of the connector (Claude's is at
`claude.py:44-48`). For string-effort SDKs use
`query_record.parameters.thinking.value.lower()`.

**Validate cross-parameter constraints inline.** Claude requires
`max_tokens > thinking.budget_tokens`; `_add_common_params`
(`claude.py:215-223`) raises a `ValueError` with an explicit message
if user-supplied `max_tokens` is too small. A ProxAI-side
`ValueError` pointing at the cause beats a cryptic provider 400.

---

## 8. Output formats: JSON and PYDANTIC

TEXT is trivial — return `MessageContent(type=ContentType.TEXT,
text=...)` per segment. JSON and PYDANTIC each have a two-pattern
split.

### 8.1 JSON

**Pattern A — native free-JSON mode → `json=SUPPORTED`.** Examples:
OpenAI `response_format={'type': 'json_object'}`, Gemini
`config.response_mime_type='application/json'`. Set the flag when
`output_format.type == JSON`. Return a TEXT block with the JSON
string; `result_adapter` does `json.loads` into a JSON block
(`result_adapter.py:119-125`).

**Pattern B — no native mode → `json=BEST_EFFORT`.** The adapter
injects `"You must respond with valid JSON."` and leaves
`output_format.type == JSON` on the query_record. The model returns
text (often markdown-wrapped or prose-prefixed). In the executor,
post-parse convert TEXT → JSON via `self._extract_json_from_text`:

```python
needs_json = (query_record.output_format.type ==
              types.OutputFormatType.JSON)
result_record.content = self._parse_content_blocks(response)
if needs_json:
  result_record.content = [message_content.MessageContent(
      type=message_content.ContentType.JSON,
      json=self._extract_json_from_text(c.text)
  ) if c.type == message_content.ContentType.TEXT else c
      for c in result_record.content]
```

`_extract_json_from_text` (`provider_connector.py:254-301`) handles
markdown fences, prose, brace extraction, and single-quoted Python
dict repr. Tested in
`tests/connectors/test_provider_connector.py:TestExtractJsonFromText`.
**Don't reinvent it.**

### 8.2 PYDANTIC

**Pattern A — native SDK parsing → `pydantic=SUPPORTED`.** Examples:
OpenAI `beta.chat.completions.parse(response_format=Cls)`, Claude
`beta.messages.stream(output_format=Cls, betas=[STRUCTURED_OUTPUTS_BETA])`.
Pass the class to the SDK and emit a `PYDANTIC_INSTANCE` block from
the parsed object:

```python
result_record.content = [message_content.MessageContent(
    type=message_content.ContentType.PYDANTIC_INSTANCE,
    pydantic_content=message_content.PydanticContent(
        class_name=query_record.output_format.pydantic_class.__name__,
        class_value=query_record.output_format.pydantic_class,
        instance_value=response.parsed_output,  # SDK-specific field
    ))]
```

**Pattern B — no native parsing → `pydantic=BEST_EFFORT`.** The
adapter computes `pydantic_class.model_json_schema()`, appends
`"You must respond with valid JSON that follows this schema: ..."`
to the prompt/system, and leaves `output_format.type == PYDANTIC`
with `pydantic_class` populated. Your executor returns TEXT;
`result_adapter._adapt_message_content` (`result_adapter.py:126-137`)
does `json.loads` + `model_validate` and emits a
`PYDANTIC_INSTANCE`. You *may optionally* enable a native JSON mode
on top — DeepSeek does (`deepseek.py:167-172`) — to make the
client-side parse reliable.

### 8.3 Why SUPPORTED vs BEST_EFFORT matters

| Level | Parsing | Executor work |
|---|---|---|
| `SUPPORTED` | SDK (server-side) | Pass schema/class to SDK; emit `PYDANTIC_INSTANCE` from parsed object |
| `BEST_EFFORT` | `result_adapter` (client-side) | Return TEXT; framework parses |

Trap: mark `pydantic=SUPPORTED` but forget to pass
`response_format=Cls` / `output_format=Cls` to the SDK → no
server-side enforcement *and* no prompt injection (framework skips
injection at SUPPORTED). Silently wrong shapes. Mark accurately.

---

## 9. Input formats: multimodal content blocks

Chat messages can carry `MessageContent` blocks of several types
(`ContentType.TEXT / IMAGE / DOCUMENT / AUDIO / VIDEO / JSON /
PYDANTIC_INSTANCE / THINKING / TOOL`). Adapter governs which reach
the executor via `input_format`; the executor still must translate
each block to the provider's native content-part shape.

### 9.1 The `_to_<provider>_part` + `_convert_messages` pair

Every multimodal connector implements two static helpers (see
`deepseek.py:82-132` and `claude.py:81-165`):

```python
@staticmethod
def _to_widget_part(part_dict):
  """Convert a ProxAI content block to a Widget content part.

  Returns None for unconvertible blocks — the caller drops them.
  """
  # File API ref has priority — §9.2
  file_ids = part_dict.get('provider_file_api_ids', {})
  if 'widget' in file_ids:
    ...
  t = part_dict.get('type')
  if t == 'text':
    return {'type': 'text', 'text': part_dict['text']}
  if t == 'image':
    # Prefer URL (part_dict['source']); else base64 from 'data' or 'path'
    ...
  if t == 'document':
    text = content_utils.read_text_document(part_dict)
    if text is not None:
      return {'type': 'text', 'text': text}
    # PDF via pypdf; docx/xlsx → drop or extract per provider
    ...
  return None

@staticmethod
def _convert_messages(messages):
  out = []
  for msg in messages:
    if isinstance(msg['content'], str):
      out.append(msg); continue
    if isinstance(msg['content'], list):
      parts = [p for p in (WidgetConnector._to_widget_part(b)
                           for b in msg['content']) if p is not None]
      out.append({**msg, 'content': parts})
    else:
      out.append(msg)
  return out
```

Executor calls `self._convert_messages(query_record.chat['messages'])`
before passing to the SDK.

### 9.2 File API references

If the provider has a File API and the connector wires it up (§13),
content blocks arrive with `provider_file_api_ids: {'widget': 'file_abc'}`
pre-populated by `_auto_upload_media`
(`provider_connector.py:752-823`). Prefer the file_id ref over
inline bytes:

```python
file_ids = part_dict.get('provider_file_api_ids', {})
if 'widget' in file_ids and part_dict.get('type') in (
    'image', 'document', 'audio', 'video'):
  return {'type': part_dict['type'],
          'source': {'type': 'file', 'file_id': file_ids['widget']}}
```

Concrete example: `claude.py:97-102`.

---

## 10. Thinking, tools, citations

### 10.1 Thinking

Parse reasoning segments into `MessageContent(type=ContentType.THINKING, text=...)`.
`result_adapter` passes THINKING blocks unchanged and skips them
when populating `output_text` (`result_adapter.py:191-196`), so
thinking never pollutes `output_text` / `output_json`.

| Provider | Detect | Text field |
|---|---|---|
| Claude | `block.type == 'thinking'` | `block.thinking` |
| Gemini | `part.thought is True` | `part.text` |
| OpenAI Responses | `output.type == 'reasoning'` | `output.summary[*].text` |
| DeepSeek-reasoner | `message.reasoning_content` attr | `message.reasoning_content` |

Wire the SDK thinking knob when `parameters.thinking` is set (§7).

### 10.2 Web search citations

Mark `tools.web_search=SUPPORTED` then pass the tool spec when
`types.Tools.WEB_SEARCH in query_record.tools`.

**Citations arrive in two distinct places — surface both.**

**1. Standalone tool-result blocks** (one per search):

| Provider | Appearance |
|---|---|
| Claude | `block.type == 'web_search_tool_result'`, `block.content` a list of results with `.title`, `.url` |

**2. Inline annotations on text blocks**:

| Provider | Appearance |
|---|---|
| OpenAI Responses | `content.annotations` with `.title`, `.url` |
| Claude | `block.citations`; filter `citation.type == 'web_search_result_location'` |
| Gemini | `candidate.grounding_metadata.grounding_chunks` with `chunk.web.title`, `chunk.web.uri` |

Emit one `TOOL` block per citation source:

```python
parsed.append(message_content.MessageContent(
    type=message_content.ContentType.TOOL,
    tool_content=message_content.ToolContent(
        name='web_search', kind=message_content.ToolKind.RESULT,
        citations=[message_content.Citation(title=..., url=...)
                   for ... in source])))
```

Even with an empty citation list, emit the TOOL block — its presence
signals "a search ran" and tests assert on it.

---

## 11. The mock model

`self.api` returns the real client when `run_type=PRODUCTION`, the
mock when `run_type=TEST` (`provider_connector.py:144-151`). The
mock **duck-types** the real SDK: accepts every method with
`*args, **kwargs`, returns objects whose attributes match what the
executor reads.

### 11.1 Template

```python
# src/proxai/connectors/providers/widget_mock.py

class _MockMessage:
  def __init__(self):
    self.content = 'mock response'

class _MockChoice:
  def __init__(self):
    self.message = _MockMessage()

class _MockResponse:
  def __init__(self):
    self.choices = [_MockChoice()]

class _MockCompletions:
  def create(self, **kwargs) -> _MockResponse:
    return _MockResponse()

class _MockChat:
  def __init__(self):
    self.completions = _MockCompletions()

class WidgetMock:
  def __init__(self):
    self.chat = _MockChat()
```

Context-manager (streaming) variant — mock's `stream(...)` returns
an object with `__enter__` / `__exit__` whose `__enter__` returns an
object with `get_final_message()`. See `claude_mock.py`.

### 11.2 Three rules

- **Every attribute path the executor reads on `self.api` must exist
  on the mock.** Missing path → `AttributeError` inside
  `_safe_provider_query` → `FAILED` result → test appears to pass
  but nothing exercised the intended code.
- **Accept `*args, **kwargs` on every method, return a fixed
  object.** The mock exercises flow, not behavior. Tests needing
  specific responses subclass or monkey-patch
  `_safe_provider_query`.
- **Plain classes with typed `__init__` attribute assignment.** Not
  dataclasses, not `SimpleNamespace`, not pydantic. Copy
  `deepseek_mock.py` or `claude_mock.py`.

Full contract: `testing_conventions.md` §3.1.

---

## 12. Registering the connector

All four registration sites must agree on the provider's short
lowercase name — mismatch across any pair is a runtime error.

### 12.1 `_MODEL_CONNECTOR_MAP` (`model_registry.py:16-29`)

```python
import proxai.connectors.providers.widget as widget_provider

_MODEL_CONNECTOR_MAP = {
    # ...
    'widget': widget_provider.WidgetConnector,
}
```

Key must equal `WidgetConnector.PROVIDER_NAME`. Mismatch →
`ValueError("Provider not supported. widget")` at runtime.

### 12.2 `PROVIDER_KEY_MAP` (`model_configs.py:19-32`)

```python
PROVIDER_KEY_MAP: dict[str, tuple[str]] = MappingProxyType({
    # ...
    'widget': ('WIDGET_API_KEY',),   # single-key: trailing-comma tuple
})
```

Every env-var name here must also appear in
`WidgetConnector.PROVIDER_API_KEYS`. Multi-key providers list every
required var (Databricks: `('DATABRICKS_TOKEN', 'DATABRICKS_HOST')`).

### 12.3 `model_configs_data/v1.3.0.json` — model entries

Loader is `ModelConfigs._load_model_registry_from_local_files`
(`model_configs.py:277-306`); version comes from
`LOCAL_CONFIG_VERSION` (currently `"v1.3.0"`). Add one entry per
model, matching the shape used by existing providers:

```json
{
  "provider_model_configs": {
    "widget": {
      "widget-pro-1": {
        "provider_model": {
          "provider": "widget",
          "model": "widget-pro-1",
          "provider_model_identifier": "widget-pro-1-2026-04"
        },
        "pricing": {"input_token_cost": 300, "output_token_cost": 1500},
        "features": {
          "prompt": "SUPPORTED",
          "messages": "SUPPORTED",
          "system_prompt": "SUPPORTED",
          "input_format": {"text": "SUPPORTED", "image": "NOT_SUPPORTED"},
          "output_format": {"text": "SUPPORTED", "json": "SUPPORTED",
                            "pydantic": "BEST_EFFORT"},
          "parameters": {"temperature": "SUPPORTED",
                         "max_tokens": "SUPPORTED",
                         "stop": "SUPPORTED",
                         "n": "NOT_SUPPORTED",
                         "thinking": "NOT_SUPPORTED"},
          "tools": {"web_search": "NOT_SUPPORTED"}
        },
        "metadata": {"is_recommended": false, "model_size_tags": ["medium"]}
      }
    }
  }
}
```

Three things to get right:

- **`pricing.*_token_cost` is integer nano-USD per token.** $0.80/M =
  `800`, not `0.0000008`. Unit convention on
  `ProviderModelPricingType` (`types.py:130-152`).
- **Per-model features can be narrower than the endpoint config.**
  The adapter merges `ENDPOINT_CONFIG` with per-model features via
  `adapter_utils.merge_feature_configs`, taking the minimum level.
  If `widget-lite-1` doesn't support `pydantic` but `widget-pro-1`
  does, encode it in the per-model features.
- **The `include` glob in `pyproject.toml`
  (`"src/proxai/connectors/model_configs_data/*.json"`) already
  covers new `v1.3.x.json` files** — no edit needed. Only touch
  `[tool.poetry] include` for a *new kind* of non-Python asset.

When rolling forward to `v1.4.0.json`, bump `LOCAL_CONFIG_VERSION`
in `model_configs.py` in the same PR.

### 12.4 `pyproject.toml` — SDK dependency

```toml
[tool.poetry.dependencies]
widget-sdk = "^1.2.0"          # ← add
```

Run `poetry lock` (not `poetry update`) and commit `poetry.lock`
in the same commit as `pyproject.toml`. A lock/manifest split
regenerates the lock in CI — at best slowing it, at worst resolving
different versions.

If the connector reuses an existing dep (DeepSeek and HuggingFace
reuse `openai`), no new dep is needed.

---

## 13. File API integration (optional)

**Trigger:** provider exposes a File API ProxAI should wire into
auto-upload. If it doesn't, skip — the connector's `_to_*_part`
always uses inline base64 or URL references.

Four edits in `src/proxai/connectors/file_helpers.py`:

```python
def upload_to_widget(file_path, file_data, filename, mime_type, token_map):
  ...  # return FileUploadMetadata

def remove_from_widget(file_id, token_map): ...
def list_from_widget(token_map, limit=100): ...
def download_from_widget(file_id, token_map): ...

UPLOAD_DISPATCH   = {..., 'widget': upload_to_widget}
REMOVE_DISPATCH   = {..., 'widget': remove_from_widget}
LIST_DISPATCH     = {..., 'widget': list_from_widget}
DOWNLOAD_DISPATCH = {..., 'widget': download_from_widget}

UPLOAD_SUPPORTED_MEDIA_TYPES['widget']    = frozenset({...})
REFERENCE_SUPPORTED_MEDIA_TYPES['widget'] = frozenset({...})

_MOCK_PROVIDERS = [..., 'widget']   # so run_type=TEST exercises uploads
```

Update `_to_widget_part` to emit a file_id reference when
`provider_file_api_ids['widget']` is set (§9.2).

**Capability-table split is load-bearing.** Every MIME in
`UPLOAD_SUPPORTED_MEDIA_TYPES['widget']` must also appear in
`REFERENCE_SUPPORTED_MEDIA_TYPES['widget']` — otherwise
`is_upload_supported` uploads a file the executor then silently
drops. Full contract: `files_internals.md` §1-4.

---

## 14. Tests and integration harness

### 14.1 Unit tests (`tests/connectors/providers/test_widget.py`)

```python
import pytest
import proxai.connectors.providers.widget as widget_provider
import proxai.types as types


@pytest.fixture
def connector(monkeypatch):
  monkeypatch.setenv('WIDGET_API_KEY', 'fake')
  return widget_provider.WidgetConnector(
      init_from_params=types.ProviderConnectorParams(
          run_type=types.RunType.TEST,
          provider_token_value_map={'WIDGET_API_KEY': 'fake'},
          # ... (see conftest.py for shared helpers)
      ))


class TestWidgetTextGeneration:
  def test_prompt_happy_path(self, connector): ...
  def test_chat_happy_path(self, connector): ...

class TestWidgetParameters:
  def test_temperature_passed_to_sdk(self, connector): ...
  def test_n_not_supported_raises(self, connector): ...

class TestWidgetOutputFormats:
  def test_json_mode_wires_response_format(self, connector): ...
  def test_pydantic_best_effort_injects_schema(self, connector): ...

class TestWidgetInputFormats:
  def test_image_block_translated_to_provider_part(self, connector): ...
  def test_unsupported_input_format_raises(self, connector): ...
```

Patterns:

- `run_type=types.RunType.TEST` on every connector-under-test.
- `conftest.py` already sets `PROVIDER_KEY_MAP` env vars to
  `'test_api_key'`; monkeypatch only when overriding.
- **Assert on the `functools.partial` call site** — "executor passed
  `temperature=0.7` to `self.api.chat.completions.create`" — not on
  the mock response. The mock is a fixture, not a contract.
- Every `raise` needs a test; every feature-support branch
  (SUPPORTED / BEST_EFFORT / NOT_SUPPORTED) needs a test.
  `testing_conventions.md` §4.

### 14.2 Integration harness (`integration_tests/proxai_api_test.py`)

Merge approval needs real-provider validation
(`sanity_check_before_merge.md` §11):

```python
@integration_block
def generate_text_widget(state_data):
  response = px.generate_text(
      'hello', provider_model=('widget', 'widget-pro-1'))
  print(response); assert response
  return state_data
```

Add blocks for every non-trivial `SUPPORTED` / `BEST_EFFORT` feature:
text, chat history, system prompts, parameters, JSON mode, Pydantic,
web search, multimodal inputs. Run once before declaring ready:

```bash
poetry run python3 integration_tests/proxai_api_test.py \
    --mode new --print-code
```

Full guide: `testing_conventions.md` §5.

---

## 15. Layer A documentation updates

The provider addition isn't complete until Layer A reflects it:

- [ ] `user_agents/api_guidelines/provider_feature_support_summary.md`
  — row in every capability table (endpoints, input formats, output
  formats, parameters, tools).
- [ ] `user_agents/api_guidelines/px_models_api.md` — check tree /
  examples if the default-priority list changed.
- [ ] `user_agents/api_guidelines/px_files_api.md` — support matrices
  if you added File API dispatch entries.
- [ ] This file §1.1 — add the new provider to the reference table
  if it's reference-quality.

Full audit: `sanity_check_before_merge.md` §13.

---

## 16. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: WidgetConnector must define ENDPOINT_PRIORITY` at import | Missing one of the 5 required class attrs | Add it |
| `ValueError: ENDPOINT_PRIORITY and ENDPOINT_CONFIG keys do not match` at import | Typo across the three endpoint dicts | Copy-paste the keys |
| `ValueError("Provider not supported. widget")` at runtime | Key mismatch across `PROVIDER_NAME` / `_MODEL_CONNECTOR_MAP` / `PROVIDER_KEY_MAP` / JSON | Make the short id identical in all four |
| `provider_token_value_map needs to contain WIDGET_API_KEY` | Missing from `PROVIDER_KEY_MAP` or env unset | Add the tuple and export the var |
| Test passes, production silently ignores `temperature` | `temperature=SUPPORTED` but executor forgot to wire `functools.partial(call, temperature=...)` | Add the branch; add an assert-on-partial test |
| `JSONDecodeError: Expecting value` from `result_adapter` | `json=SUPPORTED` but executor didn't flip native JSON mode, model returned prose | Wire the flag, or downgrade to `BEST_EFFORT` + `_extract_json_from_text` |
| Pydantic silently wrong shape | `pydantic=SUPPORTED` but didn't pass `response_format=Cls` / `output_format=Cls` to SDK | Wire it, or downgrade to `BEST_EFFORT` |
| `'text' output format config cannot be best effort` (or `image`/`audio`/`video`/`multi_modal`) | One of the five at `BEST_EFFORT` | Use `SUPPORTED`/`NOT_SUPPORTED`/omit — only `json`/`pydantic` may be `BEST_EFFORT` |
| Test-only `AttributeError` inside `_safe_provider_query` → `FAILED` result | Mock missing attribute path the executor reads | Grow the mock for paths your executor touches |
| System prompt ignored on chat but works on prompt | Pattern-2 executor reading `query_record.system_prompt` | Read only from `query_record.chat['messages']` (§4.6) |
| `AttributeError: 'dict' object has no attribute 'messages'` | Reading `query_record.chat.messages` | After adaptation `chat` is a dict — `query_record.chat['messages']` |
| `n` silently dropped on a single-choice provider | `parameters.n` left as `None` | Set `n=NOT_SUPPORTED` explicitly |
| `FileNotFoundError('Model config file "v1.3.1.json" not found')` | Added `v1.3.1.json` without bumping `LOCAL_CONFIG_VERSION` | Bump it in the same PR |
| Web search runs but no `TOOL` block | Citations on per-text annotations, not tool-result block (or vice versa) | Surface both sources (§10.2) |
| Executor returns `ResultRecord` | Legacy pattern | Return `ExecutorResult(result_record=..., raw_provider_response=...)` |

---

## 17. Where to read next

- [`feature_adapters_logic.md`](./feature_adapters_logic.md) — what
  happens to `query_record` before your executor runs, and to
  `result_record.content` after. Full per-content-type adaptation
  table.
- [`chat_export_logic.md`](./chat_export_logic.md) — how
  `Chat.export` reshapes chat messages (turns `query_record.chat`
  from a `Chat` object into the dict your executor reads).
- [`testing_conventions.md`](./testing_conventions.md) §3-4 — mock
  contract and fixture patterns.
- [`files_internals.md`](./files_internals.md) §1-4 — full File API
  integration contract.
- [`sanity_check_before_merge.md`](./sanity_check_before_merge.md)
  §8 — pre-merge audit checklist.
- Three closest reference connectors:
  - OpenAI-compatible HTTP → `deepseek.py` (cleanest template).
  - Anthropic-native → `claude.py` (streaming, structured outputs,
    thinking, web search, files, documents).
  - Google-native → `gemini.py` (config-object-on-the-side shape).
