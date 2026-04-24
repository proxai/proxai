# Feature and Result Adapters — Logic

Source of truth: `src/proxai/connectors/feature_adapter.py` (the
`FeatureAdapter` class — runs before the executor),
`src/proxai/connectors/result_adapter.py` (the `ResultAdapter` class
— runs after the executor), `src/proxai/connectors/adapter_utils.py`
(shared resolve/min/merge helpers), and `src/proxai/types.py` (the
`FeatureConfigType` / `FeatureSupportType` taxonomy). If this
document disagrees with those files, the files win — update this
document.

This is the definitive reference for what the two adapters do
between `client.generate()` and a provider executor — what gets
dropped, what gets injected, what raises, and which fields the
executor is allowed to read on entry. Read this before changing the
adapter pipeline, adding a new feature to `FeatureConfigType`,
introducing a new content type, or wiring a new output format.

See also: `adding_a_new_provider.md` (the executor-side
contract — required class attributes, support-level taxonomy from
the connector author's perspective, plus the full end-to-end
new-provider recipe), `chat_export_logic.md`
(how `Chat.export` reshapes messages — invoked by `_adapt_chat`),
and `state_controller.md` (StateControlled internals — separate
subsystem; adapters do not touch state).

---

## 1. Adapter pipeline (current)

```
ProviderConnector.generate(query_record)
│
│   # Pre-flight: pick an endpoint
├── _find_compatible_endpoint(query_record, provider_model_config)
│   │   # Iterates ENDPOINT_PRIORITY, builds a FeatureAdapter per endpoint
│   ├── FeatureAdapter(endpoint, ENDPOINT_CONFIG[endpoint], model_config)
│   │   ├── adapter_utils.merge_feature_configs(...)  # pairwise min over fields
│   │   └── self.feature_config = merged
│   ├── adapter.get_query_record_support_level(query_record)
│   │   └── min over: prompt | messages | system_prompt | input_format
│   │                 | parameters | tools | output_format
│   │   → SUPPORTED → use this endpoint
│   │   → BEST_EFFORT → buffer; pick later if no SUPPORTED found
│   │   → NOT_SUPPORTED → skip
│   └── If no endpoint: _raise_no_compatible_endpoint(...)
│       └── builds get_query_signature(...) + per-endpoint
│           get_query_support_details(...) for the ValueError text
│
│   # Pre-flight: STRICT-mode rejection of BEST_EFFORT
├── _check_endpoint_support_compatibility(endpoint, level, query_record)
│   └── raises if NOT_SUPPORTED (always) or BEST_EFFORT under STRICT
│
│   # Adapt the query (FeatureAdapter)
├── chosen_feature_adapter.adapt_query_record(query_record)
│   │   # Deep-copy first; never mutate the caller's record.
│   ├── _adapt_output_format(qr) → (json_guidance, pydantic_schema)
│   │   # Validates output_format level; raises NOT_SUPPORTED;
│   │   # returns flags telling chat/prompt branches what guidance
│   │   # to inject. Crucially does NOT clear qr.output_format.type.
│   ├── if qr.chat:    _adapt_chat(qr, json_guidance, pydantic_schema)
│   │                  _adapt_input_format(qr)
│   │   elif qr.prompt: _adapt_prompt(qr, json_guidance, pydantic_schema)
│   ├── if qr.tools:   _adapt_tools(qr)
│   └── if qr.parameters: _adapt_parameters(qr)
│       → returns the modified deep-copied query_record
│
│   # Executor runs (provider-specific)
├── chosen_executor(modified_query_record)
│   └── ResultRecord with .content = [MessageContent, ...]
│
│   # Adapt the result (ResultAdapter)
└── ResultAdapter(endpoint, ENDPOINT_CONFIG[endpoint], model_config)
    ├── adapt_result_record(query_record, result_record)
    │   ├── _adapt_content(qr, result.content)    # one pass
    │   │   └── _adapt_message_content(qr, item)  # per item
    │   │       → TEXT → JSON / PYDANTIC_INSTANCE conversion
    │   └── _adapt_output_values(result)          # populates output_*
    └── Same loop applied to each ChoiceType in result.choices
```

The two adapter classes are siblings — they share the constructor
shape, the merged `self.feature_config`, the
`get_feature_tags_support_level` API, and the `adapter_utils`
helpers. They differ in *what they care about*: `FeatureAdapter`
inspects every input feature on the query, while `ResultAdapter`
only cares about `output_format` (because it owns the post-call
TEXT → JSON / PYDANTIC conversion).

### 1.1 Construction contract (both adapters)

```python
FeatureAdapter(
    endpoint: str,
    endpoint_feature_config: FeatureConfigType | None = None,
    model_feature_config:    FeatureConfigType | None = None,
)
ResultAdapter(...)  # same signature
```

At least one of the two configs must be set. The constructor merges
them via `adapter_utils.merge_feature_configs`, taking the **lower**
support level field-by-field. The merged result becomes
`self.feature_config` and is the only config the rest of the class
reads. This matters because:

- An endpoint can declare `pydantic=SUPPORTED` while a specific
  model (per `model_configs_data/*.json`) declares
  `pydantic=BEST_EFFORT` — the merge yields `BEST_EFFORT` and the
  framework injects schema guidance instead of trusting native
  parsing for that model. Adapter behavior follows the merged
  config, never the endpoint config in isolation.
- `add_system_to_messages` is the one non-`FeatureSupportType`
  field on `FeatureConfigType`. It is merged with OR-semantics —
  if either side sets `True`, the merged value is `True`. Any
  other combination collapses to `None`.

`merge_feature_configs` (`adapter_utils.py:163-189`) is the
authoritative spec for the merge. Do not re-implement min-over-
support-level logic at call sites; reuse `adapter_utils.min_support`
or `merge_support_fields`.

### 1.2 Support-level resolution and ranking

```
adapter_utils.SUPPORT_RANK
│
├── NOT_SUPPORTED → 0
├── BEST_EFFORT   → 1
└── SUPPORTED     → 2
```

`resolve_support(None) → NOT_SUPPORTED`. Every accessor in
`adapter_utils` (`resolve_input_format_type_support`,
`resolve_output_format_type_support`, `resolve_tool_tag_support`,
`resolve_feature_tag_support`) goes through `resolve_support`, so
"field omitted from the config" and "field set to NOT_SUPPORTED"
behave identically downstream. The numeric `SUPPORT_RANK` exists
only so `min(...)` can rank levels — call sites should use the
`min_support` helper, not raw integer comparisons.

---

## 2. `FeatureAdapter` — what runs before the executor

`FeatureAdapter` does three jobs:

1. **Probe** the query against an endpoint config (used by
   `provider_connector` to *pick* an endpoint, before any adaptation).
2. **Adapt** the query to that endpoint's feature config (drop
   best-effort fields, raise on not-supported, inject guidance for
   JSON / Pydantic, fold system prompts, convert content blocks).
3. **Diagnose** failures — when no endpoint is compatible,
   `_raise_no_compatible_endpoint` builds the human-readable error
   from `get_query_signature` and `get_query_support_details`.

### 2.1 Probing (`get_query_record_support_level`)

Before the executor is chosen, `provider_connector` asks each
endpoint: "given this query, what is your *worst* support level
across the features the user actually invoked?" That number drives
endpoint selection.

```python
adapter = feature_adapter.FeatureAdapter(
    endpoint=endpoint,
    endpoint_feature_config=ENDPOINT_CONFIG[endpoint],
    model_feature_config=provider_model_config.features,
)
level = adapter.get_query_record_support_level(query_record)
# → SUPPORTED | BEST_EFFORT | NOT_SUPPORTED
```

The method walks every "set" feature on `query_record`:

| Feature set on query | Config field consulted |
|---|---|
| `prompt is not None` | `feature_config.prompt` |
| `chat is not None` | `feature_config.messages` |
| `chat.system_prompt is not None` | `feature_config.system_prompt` |
| Per content-block type in `chat.messages` (TEXT/IMAGE/DOCUMENT/AUDIO/VIDEO/JSON/PYDANTIC) | `feature_config.input_format.<type>` |
| `system_prompt is not None` (prompt-mode) | `feature_config.system_prompt` |
| `parameters.<temperature/max_tokens/stop/n/thinking>` | `feature_config.parameters.<name>` |
| `tools` containing `WEB_SEARCH` | `feature_config.tools.web_search` |
| `output_format.type` (always required — raises if missing) | `feature_config.output_format.<type>` |

The minimum across all collected levels wins.
`get_query_record_support_level` returns `SUPPORTED` if no features
are set (an empty-feature query is trivially supported), but in
practice `output_format.type` is required and validated up front,
so this branch is unreachable from the normal pipeline.

The endpoint-selection loop in `_find_compatible_endpoint`
(`provider_connector.py:516-540`) iterates `ENDPOINT_PRIORITY` and
returns the first `SUPPORTED` endpoint, falling back to the first
`BEST_EFFORT` endpoint only when `feature_mapping_strategy ==
BEST_EFFORT` (the default). Under
`feature_mapping_strategy == STRICT`, `BEST_EFFORT` is rejected by
`_check_endpoint_support_compatibility` and a `ValueError` is
raised — the caller wanted exact support and didn't get it.

### 2.2 `adapt_query_record` — the public adaptation entry point

Returns a deep-copied `QueryRecord` with all best-effort features
either approximated or removed. The caller's record is never
mutated.

```python
modified = adapter.adapt_query_record(query_record)
```

Order matters and is enforced by `adapt_query_record`
(`feature_adapter.py:556-587`):

1. **Deep copy** the query record. From here on out, mutation is
   safe.
2. **Reject the `prompt` + `chat` combination** with
   `ValueError("'prompt' and 'chat' cannot both be set.")`. The
   public API guards against this at parse time, but
   `adapt_query_record` defends against direct construction.
3. **`_adapt_output_format`** — validates `output_format.type` against
   the merged config, raises if `NOT_SUPPORTED`, raises if a
   "no-best-effort" type slipped through as `BEST_EFFORT` (see §2.6),
   and returns `(json_guidance, pydantic_schema)` flags for the
   downstream chat/prompt branches.
4. **`_adapt_chat` or `_adapt_prompt`** — depending on which input
   the caller passed. Both branches receive the guidance flags from
   step 3 so they can fold JSON guidance / Pydantic schema strings
   into the right slot (system, first user message, single-prompt
   tail). See §2.3 and §2.4.
5. **`_adapt_input_format`** — only on the chat branch. Walks the
   exported chat dict's content blocks and applies per-content-type
   support resolution (see §2.5).
6. **`_adapt_tools`** — only invoked if `query_record.tools` is
   non-empty. Validates each tool against the merged config and
   raises if `NOT_SUPPORTED`. `web_search` cannot be `BEST_EFFORT`
   (raises an internal `Exception` if it ever resolves to that — a
   bug-report path, not a normal one).
7. **`_adapt_parameters`** — drops every parameter whose merged
   support is `BEST_EFFORT`, raises on `NOT_SUPPORTED`, and zeroes
   out `query_record.parameters` itself if every field ends up
   `None`.

After return, the executor is allowed to read these fields without
re-checking support:

| Field | Guarantee |
|---|---|
| `prompt` | str, or `None` (if `chat` is set, or if `messages=BEST_EFFORT` collapsed to a single string) |
| `chat` | `dict` (the result of `Chat.export`, **not** the original `Chat` object), or `None` |
| `system_prompt` | Pattern-1 endpoints only (see §2.3); guaranteed `None` under Pattern 2 or after a BEST_EFFORT merge |
| `parameters` | `ParameterType` with only the SUPPORTED fields populated, or `None` if every field was dropped |
| `tools` | `list[Tools]` if the user passed any; never silently filtered |
| `output_format.type` | Always set; `JSON` / `PYDANTIC` are *not* cleared on BEST_EFFORT — see §2.6 |

If you find yourself asking "do I need to re-check this in the
executor?" the answer is no. Either the framework already removed
it, or the framework left it for you on purpose (the only such
case is `output_format.type` for JSON/Pydantic — see §2.6).

### 2.3 `_adapt_prompt` — system prompt folding

Three things can happen to a prompt-mode `system_prompt`:

```
system_prompt support  →  _adapt_prompt action
─────────────────────     ──────────────────────────────────────
SUPPORTED                 keep query_record.system_prompt as-is
                          (Pattern 1 endpoints)
BEST_EFFORT               concatenate into prompt, clear field:
                          "{system_prompt}\n\n{prompt}"
NOT_SUPPORTED             raise ValueError("Feature 'system_prompt'
                          is not supported by endpoint '<name>'.")
```

Then, regardless of system-prompt handling, the JSON / Pydantic
guidance from `_adapt_output_format` is appended to the prompt:

- `json_guidance=True`     → append `"You must respond with valid JSON."`
- `pydantic_schema is not None` → append `"You must respond with valid
  JSON that follows this schema:\n{schema}"` where `schema` is the
  Pydantic class's JSON schema serialized at indent=2.

Finally, **Pattern 2 folding** kicks in: if
`feature_config.add_system_to_messages` is `True` and a system
prompt is still present, the prompt is rewritten as a chat-shaped
dict and `query_record.prompt` and `query_record.system_prompt` are
both cleared:

```python
query_record.chat = {
    'messages': [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': prompt},
    ],
}
```

This means: from the executor's perspective, a Pattern 2 endpoint
*never* receives `query_record.system_prompt`, even on a prompt-mode
call. The system message is already inside `chat['messages']`. Read
only from there. (The provider connector contract calls this out
in `adding_a_new_provider.md` §4.6 "System prompts: two patterns".)

### 2.4 `_adapt_chat` — Chat → exported dict

For chat-mode queries, `_adapt_chat` resolves `system_prompt` and
`messages` support levels, exports the `Chat` object via
`Chat.export(...)`, and rebinds the result to either
`query_record.chat` (a dict) or `query_record.prompt` (a string —
when `messages=BEST_EFFORT` collapses the chat down to a single
turn).

```
              system_prompt level    →  _adapt_chat behavior
              ───────────────────       ───────────────────────────
SUPPORTED + add_system_to_messages=True   → exports with
                                            add_system_to_messages=True;
                                            system shows up as the first
                                            {'role': 'system'} entry in
                                            chat['messages']
SUPPORTED (Pattern 1, native system)       → exports with separate
                                            'system_prompt' key in the
                                            output dict; executor reads it
BEST_EFFORT                                → exports with
                                            add_system_to_first_user_message=True;
                                            system text is prepended to the
                                            first user message; chat.system_prompt
                                            is dropped
NOT_SUPPORTED                              → raise ValueError
```

`messages` itself follows the same three-state pattern, but
`BEST_EFFORT` for `messages` means "collapse the entire history to
one prompt string" via `Chat.export(export_single_prompt=True)`.
After collapse, `query_record.chat = None` and the string lands in
`query_record.prompt`. Executors receive the prompt branch instead
of the chat branch — they don't need to handle this case
specially because they already check both fields.

JSON / Pydantic guidance from `_adapt_output_format` is forwarded
into `Chat.export` as four flags:

- `add_json_guidance_to_system` and `add_json_schema_guidance_to_system`
  inject the guidance into the system prompt slot (used when a
  system prompt exists or is being added).
- `add_json_guidance_to_user_prompt` and
  `add_json_schema_guidance_to_user_prompt` inject the guidance into
  the last user message.

Both flags are passed simultaneously — `Chat.export` decides where
to place the guidance based on what's in the conversation. See
`chat_export_logic.md` for the export contract.

### 2.5 `_adapt_input_format` — per-content-block adaptation

Runs after `_adapt_chat`, on the exported chat dict's
`messages[*].content` lists (string-typed content is skipped — this
is plain TEXT and needs no adaptation). For each content block,
`_adapt_content_block` looks up the block's content type, resolves
the corresponding `input_format.<type>` support level, and acts:

```
input_format.<type> level  →  _adapt_content_block behavior
─────────────────────────     ─────────────────────────────────
SUPPORTED                     pass the block through unchanged
BEST_EFFORT                   ├── if a converter exists → run it
                              │   (text-out replacement; see below)
                              ├── if type ∈ _BEST_EFFORT_PASSTHROUGH_TYPES
                              │   → pass through (connector handles
                              │      conversion downstream)
                              └── else → drop the block (return None)
NOT_SUPPORTED                 raise ValueError("Input format '<type>'
                              is not supported by endpoint '<name>'.")
```

The converter table (`_CONTENT_TYPE_TO_INPUT_FORMAT` in
`feature_adapter.py:496-505`):

| Content block type | InputFormatType | Converter (BEST_EFFORT only) |
|---|---|---|
| `text` | `TEXT` | — (no converter; SUPPORTED is the only sensible level) |
| `image` | `IMAGE` | — (drop on BEST_EFFORT) |
| `document` | `DOCUMENT` | — (passthrough on BEST_EFFORT — connector converts) |
| `audio` | `AUDIO` | — (drop on BEST_EFFORT) |
| `video` | `VIDEO` | — (drop on BEST_EFFORT) |
| `json` | `JSON` | `_json_block_to_text` — emits `{type:'text', text: json.dumps(...)}` |
| `pydantic_instance` | `PYDANTIC` | `_pydantic_block_to_text` — emits a TEXT block with `class name: <name>` and `class value:\n<json>` |

The passthrough set (`_BEST_EFFORT_PASSTHROUGH_TYPES`) is
intentionally narrow — currently only `'document'`. The reason is
that document conversion (e.g., PDF → text extraction via
`content_utils`) is provider-specific and lives in the connector,
not in the adapter. Marking `document=BEST_EFFORT` says "let the
connector handle this"; marking `image=BEST_EFFORT` says "drop it,
it can't be approximated" because image-to-text would require an
upstream OCR step the framework does not own.

Dropping a content block is silent — there is no warning. If you
want a hard error, mark the input format `NOT_SUPPORTED`.

### 2.6 `_adapt_output_format` — validate and inject guidance

This is the only adapter step whose return value is consumed by
two other branches. It does **not** mutate
`query_record.output_format`; it returns
`(json_guidance: bool, pydantic_schema: dict | None)` for
`_adapt_prompt` / `_adapt_chat` to thread into the system or user
slot.

```
output_format.type level  →  _adapt_output_format action  +  return value
────────────────────────     ──────────────────────────────────────────────
TEXT  / IMAGE  / AUDIO  /    enforced "no best effort" — see below
VIDEO / MULTI_MODAL
  SUPPORTED                  → returns (False, None); executor handles natively
  BEST_EFFORT                → raises Exception (configuration bug)
  NOT_SUPPORTED              → raises ValueError

JSON
  SUPPORTED                  → returns (False, None); executor wires SDK
                               native JSON mode; result_adapter still
                               json.loads the returned text
  BEST_EFFORT                → returns (True, None); framework appends
                               "You must respond with valid JSON." to
                               prompt or system; type LEFT IN PLACE
  NOT_SUPPORTED              → raises ValueError

PYDANTIC
  SUPPORTED                  → returns (False, None); executor passes
                               pydantic class to native parsing; emits
                               PYDANTIC_INSTANCE block directly
  BEST_EFFORT                → returns (False, schema_dict); framework
                               appends schema guidance to prompt or
                               system; type LEFT IN PLACE
  NOT_SUPPORTED              → raises ValueError
```

Two non-obvious rules to internalize:

**The "no-best-effort" set.** `_NO_BEST_EFFORT_RESPONSE_FORMATS =
("text", "image", "audio", "video", "multi_modal")` defines five
formats that cannot be approximated via prompt injection — there
is no meaningful prompt for "best-effort generate an image."
Marking any of them `BEST_EFFORT` raises an internal `Exception`
("Code should never reach here") at `_adapt_output_format` time.
This is enforced even though `merge_feature_configs` could in
principle produce such a level — the rule lives in the adapter to
catch misconfigured `model_configs_data/*.json` entries early.

**`output_format.type` is intentionally not cleared.** For JSON and
Pydantic at `BEST_EFFORT`, the framework *both* injects prompt
guidance *and* leaves `query_record.output_format.type` set to
`JSON`/`PYDANTIC`. This lets the executor optionally enable a
provider-side JSON mode on top of the prompt injection — OpenAI's
chat connector does this. If the executor does nothing, the
framework parses the returned text via `result_adapter` (§3.2) and
the round-trip still works. Either path is correct; the executor
chooses based on what its SDK offers. `adding_a_new_provider.md`
§5 and §8 cover this from the executor author's side.

### 2.7 `_adapt_tools` and `_adapt_parameters` — drops and raises

**Tools.** Only `WEB_SEARCH` is currently modeled. Any other entry
in `query_record.tools` raises `ValueError("Unknown tool: ...")`.
`web_search` resolves against `feature_config.tools.web_search`:

```
web_search level   →  _adapt_tools action
──────────────────    ────────────────────────────
SUPPORTED             passes through; executor must wire the SDK
                      tool spec
BEST_EFFORT           raises Exception (web_search has no
                      best-effort approximation; this is a
                      bug-report path)
NOT_SUPPORTED         raises ValueError
```

**Parameters.** For each set field on `query_record.parameters`,
`_should_remove` returns `True` for `BEST_EFFORT` (drop) and raises
for `NOT_SUPPORTED`. `SUPPORTED` keeps the value. After all
parameters are processed, if the entire `ParameterType` is now
empty (every field is `None`), `query_record.parameters` itself is
set to `None` — this is the signal to the executor that no
parameter overrides are in effect.

The "drop on BEST_EFFORT" choice for parameters is deliberate. A
silent drop is the right approximation because parameters are
optional knobs (the model has reasonable defaults), and there is no
useful prompt-level approximation for "I wish you supported a
custom temperature." If a caller cared enough to need exact
support, they should run under
`feature_mapping_strategy=STRICT` — the strict-mode rejection
happens at endpoint selection, well before `_adapt_parameters`.

### 2.8 Diagnostic helpers

When endpoint selection fails, `_raise_no_compatible_endpoint`
(`provider_connector.py:486-514`) builds a structured error message
from two adapter helpers:

- **`get_query_signature(query_record)`** — a static method that
  emits a JSON-serializable dict summarizing what features the
  query *requested* (without any config-level information). Useful
  in error messages and tests because it doesn't require an adapter
  instance.
- **`get_query_support_details(query_record)`** — an instance
  method on the adapter that emits a per-feature support level
  dict for the *adapter's specific endpoint*. The error message
  loops over `ENDPOINT_PRIORITY` and prints one details dict per
  endpoint, so the caller can see exactly which endpoint had which
  shortfall.

Both methods are read-only and safe to call from anywhere — they
do not mutate the query record. They exist primarily for error
messages and for logging / debugging hooks; production code paths
go through `get_query_record_support_level`.

---

## 3. `ResultAdapter` — what runs after the executor

The executor returns a `ResultRecord` whose `content` is a list of
`MessageContent` blocks. `ResultAdapter` runs once per result (and
once per choice if `result.choices` is populated for `n > 1`),
doing two things in sequence:

```
adapt_result_record(query_record, result_record)
│
├── if result_record.content:
│   ├── content = _adapt_content(qr, content)        # block transform
│   └── _adapt_output_values(result_record)          # populate output_*
│
└── if result_record.choices:
    └── for choice in choices:
        ├── choice.content = _adapt_content(qr, choice.content)
        └── _adapt_output_values(choice)
```

There is no equivalent of `FeatureAdapter.adapt_query_record`'s
"raise on NOT_SUPPORTED" branch — by the time the result comes back,
the request has already been validated and executed. ResultAdapter
trusts the executor's content list and only transforms or summarizes
it.

### 3.1 Content vs. choices

`ResultRecord.content` and `ResultRecord.choices` are mutually
non-exclusive but typically only one is populated:

- Single-choice executors fill `result.content` and leave `choices`
  unset.
- Executors that exercise `parameters.n > 1` (currently only
  OpenAI) fill `result.choices` with `ChoiceType` entries (each
  with its own `content` list), and may also fill `result.content`
  with the first choice for callers that don't iterate.

`adapt_result_record` adapts both surfaces independently. The
identical `_adapt_content` and `_adapt_output_values` calls run on
each `ChoiceType` exactly as they do on the top-level result.

### 3.2 `_adapt_message_content` — TEXT → JSON / PYDANTIC

Per content block, `_adapt_message_content` decides whether to
transform it based on the block's current type and the requested
`output_format.type`. Pass-through types (`THINKING`, `IMAGE`,
`DOCUMENT`, `AUDIO`, `VIDEO`, `TOOL`) are returned unchanged — they
are not text and the JSON/Pydantic conversion does not apply.

The text-shaped conversion table:

| Block type in | output_format.type | Block type out |
|---|---|---|
| `TEXT` | `TEXT` | `TEXT` (unchanged) |
| `TEXT` | `JSON` | `JSON` — runs `json.loads(content_item.text)` |
| `TEXT` | `PYDANTIC` | `PYDANTIC_INSTANCE` — `json.loads` then `pydantic_class.model_validate(...)`; populates `class_name`, `class_value`, `instance_value`, `instance_json_value` |
| `JSON` | `JSON` | `JSON` (unchanged) |
| `JSON` | `PYDANTIC` | `PYDANTIC_INSTANCE` — uses `content_item.json` directly (no re-parse) |
| any other | any other | unchanged |

A few consequences worth knowing:

- **A bare `json.loads` is the default.** If the executor returned
  a TEXT block wrapped in markdown fences (`` ```json\n{...}\n``` ``)
  or with prefatory natural language, the parse will raise
  `JSONDecodeError`. The fix is **not** in `result_adapter` — it
  belongs in the executor, which can call
  `ProviderConnector._extract_json_from_text(text)` (a base-class
  helper) before constructing the TEXT block. Anthropic Claude and
  Mistral both do this. See `adding_a_new_provider.md` §8.1
  "Pattern B — no native mode".
- **Pydantic validation runs on every PYDANTIC request.** A model
  that returns syntactically valid JSON but the wrong shape will
  raise `pydantic.ValidationError` here. There is no recovery
  branch — the failure propagates as a normal exception and the
  caller sees it through `suppress_provider_errors` or directly.
- **Executors *can* emit `PYDANTIC_INSTANCE` blocks themselves**
  for the SUPPORTED path (server-side parsing). Such blocks fall
  through `_adapt_message_content` unchanged because the `(in,
  out) = (PYDANTIC_INSTANCE, PYDANTIC)` pair is not in the table —
  the default `return content_item` clause covers it.
- **Conversion is one-way.** A PYDANTIC instance is never
  converted back to TEXT or JSON; that direction is the caller's
  problem (they hold the live instance).

### 3.3 `_adapt_output_values` — populate output_text / output_*

After the per-block transform, `_adapt_output_values` walks
`result.content` (or `choice.content`) once forward and derives the
shortcut fields users typically read off of `CallRecord.result`:

```
content block type  →  effect on output_*
──────────────────     ───────────────────────────────────────────
TEXT                   appends content_item.text to output_text;
                       output_text starts at "" the first time TEXT
                       is seen (so empty-text responses report "")
IMAGE / AUDIO / VIDEO  appends "[<type>: <ref>]" to output_text
/ DOCUMENT              (where <ref> is content.source, content.path,
                       or "<data>"); also sets output_image /
                       output_audio / output_video to the LAST block
                       of that type (DOCUMENT has no typed output)
JSON                   sets output_json = content.json (LAST wins)
PYDANTIC_INSTANCE      sets output_pydantic =
                       content.pydantic_content.instance_value
                       (LAST wins)
THINKING / TOOL        skipped — they do not contribute to output_*
```

Two behaviors are subtly important:

- **`output_text` starts as `None`**, not `""`. It only becomes
  `""` the first time a TEXT or media block is seen. So a
  pure-thinking response (no TEXT, no media) ends up with
  `output_text=None`, which is distinct from "the model returned
  empty text" (`output_text=""`). Tests that assert on truthiness
  rely on this distinction.
- **JSON / Pydantic / media use last-wins.** A response with
  multiple JSON blocks surfaces the *last* one in `output_json`.
  This is intentional because providers that emit incremental
  parses or chain-of-tool-calls put the final answer at the end.
  If your code needs every block, iterate `result.content`
  yourself.

The previous `output_message: str | None` shortcut field has been
removed entirely. The deserializer silently ignores a legacy
`output_message` key in older cached records so historical data
still loads without error. New code should read `output_text`.

---

## 4. Inputs the adapters do not own

A handful of fields look like they should be the adapters' job but
aren't. Important to know so you don't add overlapping logic:

- **API-key validation** — handled by `ProviderConnector` at
  instantiation (`_check_provider_token_value_map`), not by either
  adapter. The adapters assume the connector exists.
- **Cache key construction** — `query_cache` builds keys off the
  *original* query record (and the response content), not the
  adapted one. Adapter changes do not change cache identity. See
  `cache_internals.md`.
- **`fallback_models`** — handled in `provider_connector.generate`
  before adapter logic is invoked; each fallback creates a fresh
  connector with its own adapter. No adapter ever sees more than
  one model at a time.
- **Multi-choice `n > 1`** — the adapters work per-choice (§3.1),
  but the executor is responsible for emitting the `choices` list
  and dispatching the correct `n` parameter to the SDK. The adapter
  does not validate `n`; if the endpoint doesn't support it,
  `_adapt_parameters` will have already dropped or raised.
- **Provider-side schema enforcement** — when `pydantic=SUPPORTED`,
  the executor passes the Pydantic class to the SDK. The adapter
  does not. `_adapt_output_format` returns
  `(False, None)` for SUPPORTED so the prompt-injection path is
  skipped, and `_adapt_message_content` short-circuits because
  `(PYDANTIC_INSTANCE, PYDANTIC)` is not in the conversion table.

If you find adapter logic doing one of these things, it's either
a refactor opportunity or a bug — these layers should stay
separate.

---

## 5. Where to read next

- `adding_a_new_provider.md` — the executor-side perspective on
  the same support levels and feature config, plus the full
  new-provider recipe. If you are *writing* a connector rather
  than modifying the adapters, start there.
- `chat_export_logic.md` — `Chat.export` is the workhorse
  `_adapt_chat` calls into. The flag matrix
  (`add_system_to_messages`, `add_system_to_first_user_message`,
  `add_json_guidance_to_*`, `export_single_prompt`) is documented
  there.
- `dependency_graph.md` — explains why the adapters import only
  `types` and `adapter_utils`. They sit at Layer 2 and must not
  pull from connectors, caches, or `proxai.py`.
- `tests/connectors/test_feature_adapter.py` /
  `tests/connectors/test_result_adapter.py` /
  `tests/connectors/test_adapter_utils.py` — the executable spec.
  When adapter behavior is ambiguous from the source, these tests
  resolve it.
