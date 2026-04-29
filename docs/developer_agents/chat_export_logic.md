# Chat Export Logic

Source of truth: `src/proxai/chat/chat_session.py` (the `Chat`
container and its `export` method plus ten private helpers),
`src/proxai/chat/message.py` (`Message` normalization in
`__post_init__`, `to_dict`, `from_dict`), and
`src/proxai/chat/message_content.py` (`MessageContent`, the
`ContentType` / `MessageRoleType` enums, and the per-type field
rules enforced in `__post_init__`). If this document disagrees
with those files, the files win — update this document.

This is the definitive reference for how a `Chat` is normalized,
serialized, and reshaped by `Chat.export`. Read this before
changing the export contract (adding a flag, reordering steps,
altering the dict / single-prompt output shape), adding a new
`ContentType` that needs export-time handling, modifying the
deep-copy invariant, or debugging a discrepancy between what the
caller passed and what an executor received on `query_record.chat`.

See also: `feature_adapters_logic.md` (how `_adapt_chat` invokes
`Chat.export` with the six guidance / placement flags derived from
the merged feature config — §2.4 there); the user-facing
counterpart `../user_agents/api_guidelines/px_chat_api.md`
(caller-level description of what flag combinations produce); and
`adding_a_new_provider.md` §4.6 and §6 (what shape executors
can rely on after `_adapt_chat` has exported, and the Pattern 1 vs.
Pattern 2 system-prompt split).

---

## 1. Chat → export pipeline (current)

```
Chat(messages=..., system_prompt=...)
│
│   # Construction (chat_session.py:54-69)
├── system_prompt setter               (str | None; raises TypeError otherwise)
└── messages: list[Message]
    │   # Each item normalized by _validate_message
    │   # Accepts: Message | dict | str | list[MessageContent | dict | str]
    │   # String / list shortcuts default to role=ASSISTANT
    └── Message(role, content)
        │   # __post_init__ coerces role → MessageRoleType
        │   # and each list item → MessageContent
        └── content: str | list[MessageContent]

Chat.export(...)                       # chat_session.py:187-299
│
│   # Order is enforced; do not reorder without reading §3.
├── 1. _validate_export_params(...)    # fail fast, before copy
├── 2. _normalize_allowed_types(...)   # str | ContentType → set[ContentType]
├── 3. _build_system_prompt(...)       # system + JSON guidance concat
├── 4. messages = [msg.copy() for msg in self.messages]   # deep copy
├── 5. if add_system_to_first_user_message:
│       _prepend_system_to_first_user(messages, system_prompt)
│       system_prompt = None
├── 6. if add_json_guidance_to_user_prompt or schema variant:
│       _append_guidance_to_last_user(messages, ...)
├── 7. if omit_thinking:
│       messages = _filter_thinking(messages)
├── 8. if allowed_set is not None:
│       _validate_allowed_types(messages, allowed_set)
├── 9. if merge_consecutive_roles:
│       messages = _merge_consecutive(messages)
└── 10. Build result:
        ├── if export_single_prompt:
        │     _validate_single_prompt_content(messages)
        │     return _format_as_single_prompt(system_prompt, messages)
        └── else:
              return _build_result_dict(
                  system_prompt, messages, add_system_to_messages,
              )
```

### 1.1 Why the order matters

Several of the steps cannot commute:

- **Deep copy before any mutation** (step 4). Every later helper
  mutates `messages` in place. Without the copy, a caller who
  exports the same `Chat` twice would see the second export
  operate on the first export's already-modified state. This is
  the single invariant that makes `Chat.export` side-effect-free
  from the caller's perspective.
- **System placement before user-message guidance** (5 before 6).
  If `add_system_to_first_user_message` is set, the first user
  message's text is prepended with the system prompt. If
  `add_json_guidance_to_user_prompt` is also set (with no user
  message), step 6 *creates* a user message at the end and
  appends the guidance. Reordering would attach JSON guidance to
  a message that is about to be turned into the system carrier.
- **`omit_thinking` before `allowed_types` validation** (7 before
  8). `THINKING` blocks are stripped first; the caller should be
  able to pass `allowed_types=["text"]` without having to also
  enumerate `"thinking"` when they're also omitting it. Tests in
  `tests/chat/test_chat_session.py::TestChatExportAllowedTypes`
  `test_thinking_omitted_before_allowed_types_check` lock this
  ordering in.
- **`merge_consecutive_roles` last** (9). Merging rewrites two
  same-role messages into one and normalizes string content into
  a `[MessageContent(type="text", ...)]` list along the way. Doing
  it before allowed-types validation would invisibly collapse the
  list and make the post-merge content harder to reason about in
  error messages.

Step numbers in the source's top-of-method docstring
(`chat_session.py:203-213`) use a 9-step numbering; this doc
splits step 1 into `_validate_export_params` + `_normalize_allowed_types`
to make the "fail fast" pair explicit, giving 10 items.

### 1.2 Construction and normalization

Before `export` ever runs, `Chat.__init__` routes every message
through `_validate_message` (`chat_session.py:71-105`). The four
accepted input shapes collapse to a single storage shape
(`list[Message]` with role-typed content):

| Input shape | Routed as | Default role |
|---|---|---|
| `Message` instance | returned as-is | — |
| `dict` | `Message.from_dict(dict)` | role from dict |
| `str` | `Message(ASSISTANT, [MessageContent(TEXT, text=str)])` | `ASSISTANT` |
| `list[MessageContent \| dict \| str]` | Message(ASSISTANT, [normalized list]) | `ASSISTANT` |

The `str` / `list` shortcuts default to `ASSISTANT` because the
common caller pattern is `chat.append(response.output_text)` after
a model reply. A user-role convenience shortcut would hide the
ambiguity in the wrong direction. `Message.__post_init__`
(`message.py:39-60`) does its own list-normalization pass,
coercing each `str` list item to `MessageContent(type="text",
text=item)`; the two normalization layers are deliberately
duplicated so a raw `Message(...)` constructed outside `Chat` still
gets a valid content list.

---

## 2. `Chat` container — mutation contract and helpers

`Chat` presents a full list-like interface over `self.messages`
(append / extend / insert / pop / clear / `__getitem__` /
`__setitem__` / `__delitem__` / `__iter__` / `__len__` / `__add__`
/ `__iadd__`). Every mutation path routes through
`_validate_message`, so you cannot put a raw dict or string in
`self.messages` by mistake — the list always contains `Message`
instances.

Fields worth calling out:

- `system_prompt` is stored as `_system_prompt: str | None` and
  guarded by a property setter (`chat_session.py:44-52`) that
  raises `TypeError` on non-str / non-None values. `export` reads
  via the property, so any logic that needs to see the system
  prompt must go through `self.system_prompt` rather than
  `self._system_prompt`.
- `self.messages` is a public list and the dataclass's primary
  field. Slicing (`chat[1:3]`) returns a new `Chat` that copies
  the system prompt — tested by
  `test_chat_session.py::TestChatListInterface::test_slice_returns_chat`.
  `copy()` is a `copy.deepcopy` over the whole instance
  (`chat_session.py:128-130`), so nested `MessageContent.data`
  bytes and `PydanticContent.instance_value` live models are
  duplicated wholesale.
- `__add__` deep-copies `self` then extends with `other.messages`
  — the combined chat keeps `self.system_prompt` and discards
  `other.system_prompt`. This is the documented behavior; do not
  "fix" it to merge or prompt for resolution.

`to_dict` / `from_dict` round-trip through
`Message.to_dict` / `Message.from_dict` / `MessageContent.to_dict`
/ `MessageContent.from_dict`. The round-trip is lossy for one
subclass of data only: `PydanticContent.class_value` and
`instance_value` (live Python handles to the Pydantic class and
instance) are dropped on serialization. `class_name` and
`instance_json_value` are kept as serializable proxies; see
`message_content.py:78-107` for the round-trip rule.
`MessageContent.data` bytes are base64-encoded in `to_dict` and
base64-decoded in `from_dict`, so JSON persistence is safe.

---

## 3. `export` — ten steps in detail

```python
chat.export(
    merge_consecutive_roles: bool = True,
    omit_thinking: bool = True,
    allowed_types: list[ContentType | str] | None = None,
    add_json_guidance_to_system: bool = False,
    add_json_schema_guidance_to_system: dict | str | None = None,
    add_json_guidance_to_user_prompt: bool = False,
    add_json_schema_guidance_to_user_prompt: dict | str | None = None,
    add_system_to_messages: bool = False,
    add_system_to_first_user_message: bool = False,
    export_single_prompt: bool = False,
) -> dict | str
```

The defaults (`merge_consecutive_roles=True`, `omit_thinking=True`,
everything else falsy / None) produce the tidy dict shape
`_adapt_chat` wants when the caller hasn't asked for special
behavior. The feature-adapter overrides every default that matters
per endpoint (see §4).

### 3.1 Step 1 — `_validate_export_params`

`chat_session.py:301-327`. Three mutually-exclusive pairs must not
both be truthy; each collision raises `ValueError` before anything
else happens (before the deep copy, even).

| Pair | Error message |
|---|---|
| `add_json_guidance_to_system` ∧ `add_json_schema_guidance_to_system` | `"add_json_guidance_to_system and add_json_schema_guidance_to_system are mutually exclusive."` |
| `add_json_guidance_to_user_prompt` ∧ `add_json_schema_guidance_to_user_prompt` | `"add_json_guidance_to_user_prompt and add_json_schema_guidance_to_user_prompt are mutually exclusive."` |
| `add_system_to_messages` ∧ `add_system_to_first_user_message` | `"add_system_to_messages and add_system_to_first_user_message are mutually exclusive."` |

The two JSON pairs are exclusive because "plain JSON" and "schema
JSON" are two spellings of the same suffix — picking both would
either duplicate guidance or silently overwrite. The two system
placements are exclusive because "emit as a {role:system} message"
and "prepend to the first user message" are two different
placements for the same text.

All three combinations are allowed across pairs: system-to-system
+ user-to-user guidance is a legal combination and happens in
practice when the endpoint wants JSON guidance in *both* slots (see
`test_can_combine_with_system_json_guidance` in
`test_chat_session.py`).

### 3.2 Step 2 — `_normalize_allowed_types`

`chat_session.py:434-453`. Takes `list[ContentType | str] | None`
and returns `set[ContentType] | None`. `None` (the default) means
"no filtering" and skips step 8 entirely. A list mixes free-form
strings (`"text"`) and enum members (`ContentType.TEXT`); strings
go through `ContentType(t)` which raises `ValueError` on an
unknown value, and any non-str / non-enum item raises `TypeError`.
The resulting set is what step 8 uses for membership checks — the
string-or-enum input polymorphism ends here.

### 3.3 Step 3 — `_build_system_prompt`

`chat_session.py:349-364` (with `_build_json_guidance_text` at
`chat_session.py:329-347`). Takes the current `self.system_prompt`
plus the two JSON-guidance-to-system flags and produces the
system-prompt string that the rest of the export pipeline treats as
authoritative.

The guidance suffix is deterministic:

| Flag | Suffix |
|---|---|
| `add_json_guidance_to_system=True` | `"You must respond with valid JSON."` |
| `add_json_schema_guidance_to_system=<dict>` | `"You must respond with valid JSON that follows this schema:\n" + json.dumps(dict, indent=2)` |
| `add_json_schema_guidance_to_system=<str>` | `"You must respond with valid JSON that follows this schema:\n" + str` (no re-serialization) |
| none | no suffix; return `self.system_prompt` unchanged (including `None`) |

Concatenation uses `"{system_prompt}\n\n{suffix}"`. If there is no
prior system prompt, the suffix becomes the system prompt.

Important: this is the *only* place that reads `self.system_prompt`.
Every downstream helper receives the result of this build step in
a local variable — once we are past step 3, `self.system_prompt`
is not consulted again and never mutated.

### 3.4 Step 4 — deep copy

`messages = [msg.copy() for msg in self.messages]`
(`chat_session.py:268`). `Message.copy` is `copy.deepcopy(self)`
(`message.py:85-87`), which recurses through every
`MessageContent`. From here on, in-place mutation of `messages[i]`
or `messages[i].content[j]` is safe and the caller's original
`Chat` is preserved.

This copy is also why every private helper below takes `messages`
as a parameter and mutates it in place — the public contract
(`export` returns a dict / string, never a `Chat`) is implemented
entirely on the local copy.

### 3.5 Step 5 — `_prepend_system_to_first_user`

`chat_session.py:394-414`. Runs only if `system_prompt is not None`
(after step 3) and `add_system_to_first_user_message=True`. It
folds the system prompt into the first USER message's content and
then clears the local `system_prompt` variable so subsequent steps
see no separate system slot.

The placement rules match the three content shapes a user message
can have:

| First user message shape | Action |
|---|---|
| `content: str` (bare string) | `content = f"{system_prompt}\n\n{content}"` |
| `content: list` starting with a `TEXT` block | prepend the system prompt to `content[0].text` with `\n\n` |
| `content: list` not starting with `TEXT` (e.g. image first) | `content.insert(0, MessageContent(type="text", text=system_prompt))` |
| No user message exists at all | `messages.insert(0, Message(role="user", content=system_prompt))` |

Each branch returns after its placement — only one user message is
ever modified. Tests in
`TestChatExportSystemToFirstUser::test_prepends_to_string_content`
/ `test_prepends_to_first_text_in_list` /
`test_inserts_text_when_first_item_is_not_text` /
`test_creates_user_message_when_none_exists` pin the four branches.

### 3.6 Step 6 — `_append_guidance_to_last_user`

`chat_session.py:366-392`. Runs only if
`add_json_guidance_to_user_prompt=True` or
`add_json_schema_guidance_to_user_prompt is not None`. Computes
the same JSON-guidance text as §3.3, then appends it to the *last*
USER message (scanning `messages` in reverse).

The placement rules mirror `_prepend_system_to_first_user` but act
on the tail rather than the head:

| Last user message shape | Action |
|---|---|
| `content: str` | `content = f"{content}\n\n{suffix}"` |
| `content: list` ending in a `TEXT` block | append to `content[-1].text` with `\n\n` |
| `content: list` ending in a non-`TEXT` block | `content.append(MessageContent(type="text", text=suffix))` |
| No user message exists | `messages.append(Message(role="user", content=suffix))` |

Two subtleties worth knowing:

- The append direction is **last** user, not first. This pairs
  correctly with multi-turn chats where the user's current turn
  (their last message) is the one that wants the "respond as
  JSON" instruction. Tests encode this in
  `test_appends_to_last_user_string_content` — a three-message
  chat gets guidance on the third message, not the first.
- The empty-chat / assistant-only-chat case appends a synthetic
  `Message(role="user", content=suffix)`. After this step a
  previously assistant-only chat has grown by one message. Treat
  this as a feature, not a bug — it makes `_adapt_chat`'s
  BEST_EFFORT JSON path work even on an empty chat.

### 3.7 Step 7 — `_filter_thinking`

`chat_session.py:416-432`. When `omit_thinking=True` (the default),
removes every content block whose `type == ContentType.THINKING`.
Two facts to internalize:

- Only list-shaped content is filtered. A message with `content:
  str` is text by definition and passes through unchanged.
- A message whose entire content list becomes empty after filtering
  is *dropped* from `messages` (the filter is a list
  comprehension followed by an `if msg.content` guard
  (`chat_session.py:428-431`)). A model that returned a single
  THINKING block and nothing else won't survive an export with the
  default `omit_thinking=True`.

The step returns a new list; it does not mutate in place. The
caller reassigns `messages = _filter_thinking(messages)`.

### 3.8 Step 8 — `_validate_allowed_types`

`chat_session.py:455-473`. Runs only if step 2 produced a
non-`None` set. The step is pure validation — it mutates nothing,
but raises `ValueError` on the first content block whose type is
outside the allowed set.

Two paths depending on content shape:

- `content: str` → validates against `ContentType.TEXT` (bare
  strings are always text).
- `content: list` → iterates and checks each block's `c.type`.

The check runs *after* `_filter_thinking`, so passing
`allowed_types=["text"]` with the default `omit_thinking=True`
works on a chat that contained thinking blocks — the thinking
blocks were already stripped by step 7.

### 3.9 Step 9 — `_merge_consecutive`

`chat_session.py:475-497`. Runs when
`merge_consecutive_roles=True` (the default). Two same-role
messages become one; content is concatenated by extending the
first message's content list with the second's items.

The merge normalizes string content into list form along the way:
if either side of a merge has a bare-string `content`, that side
is promoted to `[MessageContent(type="text", text=content)]`
before extension. The invariant after step 9 is:

- Consecutive messages always have different roles.
- A message created by merging has a list-shaped `content`, even if
  both inputs were bare strings.
- A message that was not merged keeps its original content shape.

This is the step most likely to surprise a caller who indexes
into `messages[*]["content"]` after export — `content` may be a
string on some messages and a list on others.

### 3.10 Step 10 — build the result

Two branches, driven by `export_single_prompt`.

**`export_single_prompt=True`** → string output.
`_validate_single_prompt_content` (`chat_session.py:499-517`)
first scans every list-shaped content block and raises `ValueError`
if any block has `data` or `path` set. Byte payloads and local file
paths have no meaningful string representation and will not be
silently dropped. `source` (URLs) is allowed; they render as the
URL string. Then `_format_as_single_prompt`
(`chat_session.py:538-560`) walks the messages and emits:

```
SYSTEM:
<system prompt text>

USER:
<user 1 content>

ASSISTANT:
<assistant 1 content>

USER:
<user 2 content>
```

with `\n\n` between blocks. List-shaped content flattens to
`\n`-joined `text` / `source` values in list order. Role labels
come from `role.value.upper()`, so `"user"` → `USER`, `"assistant"`
→ `ASSISTANT`.

The system prompt is included only if it survived step 5 (i.e.
`add_system_to_first_user_message` was not set). If it was set, the
system prompt is already part of a USER message's content.

**`export_single_prompt=False`** → dict output.
`_build_result_dict` (`chat_session.py:519-536`) collects
`[msg.to_dict() for msg in messages]`, then places the system
prompt:

- If `add_system_to_messages=True` and `system_prompt is not None`,
  insert `{"role": "system", "content": system_prompt}` at index 0
  of the messages list. The top-level `"system_prompt"` key is not
  emitted in this branch.
- Else, if `system_prompt is not None`, emit it as a top-level
  `"system_prompt"` key alongside `"messages"`.
- Else, emit only `"messages"`.

The `"messages"` key is always present — an empty `Chat` exports
as `{"messages": []}`.

---

## 4. Integration with `_adapt_chat`

`Chat.export` is called from exactly one hot path in the library:
`FeatureAdapter._adapt_chat` at
`src/proxai/connectors/feature_adapter.py:350-357`. Every flag
`_adapt_chat` sets has a direct mapping from feature-support
resolution to export flag.

```python
exported_chat = query_record.chat.export(
    add_system_to_first_user_message=system_best_effort,
    add_system_to_messages=add_system_to_messages,
    add_json_guidance_to_system=json_guidance,
    add_json_guidance_to_user_prompt=json_guidance,
    add_json_schema_guidance_to_system=pydantic_schema,
    add_json_schema_guidance_to_user_prompt=pydantic_schema,
    export_single_prompt=messages_best_effort)
```

| `_adapt_chat` local | Derived from | Flag on `export` |
|---|---|---|
| `system_best_effort` | `feature_config.system_prompt == BEST_EFFORT` | `add_system_to_first_user_message` |
| `add_system_to_messages` | `feature_config.system_prompt == SUPPORTED` ∧ `feature_config.add_system_to_messages` | `add_system_to_messages` |
| `json_guidance` | `_adapt_output_format` return (output format JSON at BEST_EFFORT) | both `add_json_guidance_to_system` and `add_json_guidance_to_user_prompt` |
| `pydantic_schema` | `_adapt_output_format` return (output format PYDANTIC at BEST_EFFORT) | both `add_json_schema_guidance_to_system` and `add_json_schema_guidance_to_user_prompt` |
| `messages_best_effort` | `feature_config.messages == BEST_EFFORT` | `export_single_prompt` |

Two behaviors the caller-side reader should know fall out of this
mapping:

- **JSON guidance is always passed on both axes.** `_adapt_chat`
  sets the same `json_guidance` flag for system and user prompt
  because `_build_system_prompt` / `_append_guidance_to_last_user`
  both act only when their branch's conditions are met. With
  `add_system_to_first_user_message=True`, the system-slot
  guidance ends up inside the user message (merged via step 5);
  the user-slot guidance appends to the tail. Both paths work;
  they do not conflict because step 5 runs before step 6.
- **`export_single_prompt` collapses into `query_record.prompt`,
  not `query_record.chat`.** When the endpoint has
  `messages=BEST_EFFORT`, `_adapt_chat` sets
  `query_record.prompt = exported_chat` (a string) and
  `query_record.chat = None`. Executors then route via the prompt
  branch and never see a chat dict. This is why
  `_validate_single_prompt_content` forbids `data` / `path`: a
  prompt-mode executor receiving a stringified chat cannot recover
  bytes or file paths from the serialized text.

The reverse dependency is also strict. `_adapt_chat` does **not**
override `merge_consecutive_roles`, `omit_thinking`, or
`allowed_types` — all three take their `export` defaults
(`True`, `True`, `None`). If you want per-endpoint merging or
thinking retention, that decision lives in `_adapt_chat`, not in
`export`; adding a flag to `FeatureConfigType` is the right
extension point.

---

## 5. Serialization (`to_dict` / `from_dict`) and the cache

`Chat.to_dict` / `Chat.from_dict` (`chat_session.py:179-185`,
`562-572`) are the persistence surface for message history — they
are *not* the same as `export`. The difference:

| Concern | `to_dict` / `from_dict` | `export` |
|---|---|---|
| Round-trip symmetry | yes; `from_dict(to_dict(chat)) == chat` (modulo Pydantic handles) | no; export is lossy by design |
| Normalization | none beyond `Message.to_dict` | system-prompt folding, thinking removal, merging |
| Shape | `{"system_prompt"?, "messages": [...]}` | dict or single-prompt string |
| Who calls it | callers persisting a conversation to disk / ProxDash | `_adapt_chat` preparing an executor payload |

The round-trip invariant lives in `tests/chat/test_chat_session.py::TestChatSerialization::test_round_trip`. Anything that breaks
it (e.g., a new `MessageContent` field that is not present on both
`to_dict` and `from_dict`) will surface there.

Interaction with caches:

- The query cache's key function serializes `QueryRecord.chat`
  (the exported dict, after `_adapt_chat`) as part of the key. So
  two calls that differ only by an export flag produce different
  cache keys. See `cache_internals.md` for the key construction.
- Persisted `CallRecord` entries on disk use the exported dict
  (not the original `Chat`). The deserializer reconstructs a dict
  on disk, not a `Chat`. If you need a `Chat` back, run
  `Chat.from_dict(...)` over the persisted dict explicitly.

---

## 6. Inputs export does not own

A few concerns that feel like they belong to `export` but live
elsewhere — keep the layering clean:

- **Support-level resolution.** `export` takes booleans and raw
  schema inputs. It never looks at `FeatureConfigType` or
  `FeatureSupportType`. Resolution is `_adapt_chat`'s job.
- **File upload / auto-upload.** `MessageContent` fields like
  `provider_file_api_ids` and `proxdash_file_id` round-trip
  through `MessageContent.to_dict` / `from_dict` untouched.
  Populating them is `FilesManager`'s job; see
  `files_internals.md`.
- **Role validation beyond USER / ASSISTANT.** `MessageRoleType`
  is a two-valued enum. Exporting a synthetic `{"role": "system",
  "content": ...}` dict entry when `add_system_to_messages=True`
  happens inside `_build_result_dict` and never constructs a
  `Message(role="system")` — because `MessageRoleType("system")`
  would raise. The system entry is a dict, not a `Message`, and
  lives only in the exported result.
- **Per-endpoint content-type adaptation.** Converting JSON or
  Pydantic content blocks to text, dropping images when an
  endpoint can't handle them — that's
  `_adapt_input_format` on the adapter, not `export`. See
  `feature_adapters_logic.md` §2.5.

If you find `export` doing one of these, it's a layering bug.

---

## 7. Errors raised by `export`

All synchronous; none are routed through provider-error handling
(`suppress_provider_errors`). Errors surface to the caller of
`export`, which in the hot path is `_adapt_chat` and by extension
`ProviderConnector.generate`.

| Where | Trigger | Error |
|---|---|---|
| `_validate_export_params` | `add_json_guidance_to_system` ∧ `add_json_schema_guidance_to_system` | `ValueError("... mutually exclusive.")` |
| `_validate_export_params` | `add_json_guidance_to_user_prompt` ∧ `add_json_schema_guidance_to_user_prompt` | `ValueError("... mutually exclusive.")` |
| `_validate_export_params` | `add_system_to_messages` ∧ `add_system_to_first_user_message` | `ValueError("... mutually exclusive.")` |
| `_normalize_allowed_types` | item not `str` / `ContentType` | `TypeError("allowed_types items must be str or ContentType, got ...")` |
| `_normalize_allowed_types` | string item not a known `ContentType` value | `ValueError` from `ContentType(str)` |
| `_validate_allowed_types` | bare-string content under `allowed_types` that excludes `TEXT` | `ValueError("Content type 'text' is not in allowed_types.")` |
| `_validate_allowed_types` | any content block's `.type` not in `allowed_set` | `ValueError("Content type '<t>' is not in allowed_types.")` |
| `_validate_single_prompt_content` | any content block has `data` set | `ValueError("export_single_prompt does not support 'data' field in message content.")` |
| `_validate_single_prompt_content` | any content block has `path` set | `ValueError("export_single_prompt does not support 'path' field in message content.")` |

`Chat.__init__` / `chat.append` / `Message(...)` /
`MessageContent(...)` errors (invalid role, bad type / media_type
combinations, missing required field per type) are raised at
construction, before `export` runs. The user-facing
`px_chat_api.md` §7 has the full construction-time error table.

---

## 8. Where to read next

- `feature_adapters_logic.md` §2.4 — how the six `export` flags
  are computed from the merged feature config and why the
  `messages_best_effort` branch moves the result into
  `query_record.prompt`.
- `adding_a_new_provider.md` §4.6 (system prompts) and §8
  (output formats) — the executor-side contract for what a
  post-export `query_record.chat` dict looks like.
- `../user_agents/api_guidelines/px_chat_api.md` — the caller's
  view of the same `export` surface, with worked examples of each
  flag combination and the type hierarchy laid out for user
  consumption.
- `tests/chat/test_chat_session.py` — the executable spec for
  every export path, broken into test classes per flag
  (`TestChatExportMergeConsecutiveRoles`,
  `TestChatExportOmitThinking`, `TestChatExportAllowedTypes`,
  `TestChatExportJsonGuidance`, `TestChatExportJsonSchemaGuidance`,
  `TestChatExportSystemToMessages`, `TestChatExportSystemToFirstUser`,
  `TestChatExportCombined`,
  `TestChatExportJsonGuidanceToUserPrompt`,
  `TestChatExportJsonSchemaGuidanceToUserPrompt`,
  `TestChatExportSinglePrompt`). When a behavior of `export` is
  ambiguous from reading the source, resolve it by finding the
  corresponding test.
