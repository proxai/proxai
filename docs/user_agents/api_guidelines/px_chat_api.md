# `px.Chat` API Comprehensive Use Case Analysis

Source of truth: `src/proxai/chat/chat_session.py` (the `Chat`
class), `src/proxai/chat/message.py` (`Message`), and
`src/proxai/chat/message_content.py` (`MessageContent` plus the
enums `ContentType`, `MessageRoleType`, `ToolKind`, and the nested
types `PydanticContent`, `ToolContent`, `Citation`). If this
document disagrees with those files, the files win — update this
document.

This is the definitive reference for building, mutating, and
serializing conversations with `px.Chat`. Read this when you need
to know what `px.Chat(...)` accepts, how to build a multi-modal
user message, how to attach a model's reply, how to compose two
chats, or how to emit a wire-shaped dict. Read it before adding a
new `ContentType`, changing what `Chat` accepts as an input, or
adding a new transform parameter to `Chat.export()`.

See also: `px_generate_analysis.md` (how a `Chat` plugs into
`generate()`), `px_files_analysis.md` (file-api metadata on
`MessageContent`), and `call_record_analysis.md` (how
`MessageContent` comes back on `ResultRecord.content`).

---

## 1. Type hierarchy (current)

```
Chat                                           # the conversation container
├── system_prompt: str | None                  # setter validates str-or-None
├── messages: list[Message]
│
└── Message                                    # one turn in the conversation
    ├── role: MessageRoleType | str            # "user" | "assistant"
    └── content: str | list[MessageContent]    # bare str = all-text content
        │
        └── MessageContent                     # one content block
            │
            │   # Block discriminator
            ├── type: ContentType | None
            │                                  # TEXT | THINKING | IMAGE | DOCUMENT
            │                                  # | AUDIO | VIDEO | JSON
            │                                  # | PYDANTIC_INSTANCE | TOOL
            │                                  # Inferred from media_type when omitted.
            │
            │   # Text-shape payload
            ├── text: str | None               # TEXT, THINKING
            │
            │   # Structured-output payloads
            ├── json: dict[str, Any] | None    # JSON
            ├── pydantic_content: PydanticContent | None
            │   ├── class_name: str | None
            │   ├── class_value: type[BaseModel] | None
            │   ├── instance_value: BaseModel | None
            │   └── instance_json_value: dict | None
            ├── tool_content: ToolContent | None
            │   ├── name: str | None
            │   ├── kind: ToolKind | None      # CALL | RESULT
            │   └── citations: list[Citation] | None
            │       └── Citation
            │           ├── title: str | None
            │           └── url: str | None
            │
            │   # Media payload (one of source / data / path;
            │   # or an existing provider_file_api_ids entry)
            ├── source: str | None             # URL
            ├── data: bytes | None             # inline bytes
            ├── path: str | None               # local filesystem path
            ├── media_type: str | None         # MIME ("image/png", "application/pdf")
            ├── filename: str | None
            │
            │   # File-api metadata (populated by px.files.upload /
            │   # auto-upload inside generate; see px_files_analysis.md)
            ├── provider_file_api_ids: dict[str, str] | None
            ├── provider_file_api_status: dict[str, FileUploadMetadata] | None
            ├── proxdash_file_id: str | None
            └── proxdash_file_status: ProxDashFileStatus | None
```

### 1.1 Per-type field rules

Only the fields a given `ContentType` needs (or forbids). Anything
else is optional.

| `ContentType` | Required | Notes |
|---|---|---|
| `TEXT`              | `text` | No `source` / `data` / `path` / `media_type` allowed. |
| `THINKING`          | `text` | Same as TEXT; represents a reasoning trace. |
| `JSON`              | `json` | |
| `PYDANTIC_INSTANCE` | `pydantic_content` | |
| `TOOL`              | `tool_content` | |
| `IMAGE`             | one of `source` / `data` / `path` (or a `provider_file_api_ids` entry) | |
| `DOCUMENT`          | one of `source` / `data` / `path` (or a `provider_file_api_ids` entry) | |
| `AUDIO`             | one of `source` / `data` / `path` (or a `provider_file_api_ids` entry) | |
| `VIDEO`             | one of `source` / `data` / `path` (or a `provider_file_api_ids` entry) | |

Extra rules worth knowing:

- `type` may be omitted when `media_type` is set — it's inferred
  (`image/*` → `IMAGE`, `audio/*` → `AUDIO`, `video/*` → `VIDEO`,
  document MIMEs → `DOCUMENT`).
- `media_type` must be one of the 25 MIME strings in
  `SUPPORTED_MEDIA_TYPES` (PNG / JPEG / GIF / WebP / HEIC / HEIF;
  PDF / DOCX / XLSX / CSV / plain-text / Markdown;
  MP3 / WAV / FLAC / AAC / OGG / AIFF; MP4 / WebM / MOV / AVI /
  MPEG / MKV). Unknown MIMEs raise.
- `MessageRoleType` has only `USER` and `ASSISTANT` — there is no
  `SYSTEM` role. The system prompt lives on `Chat.system_prompt`
  as a plain string.

---

## 2. Using `Chat`

### 2.1 Create a Chat

```python
import proxai as px

# Empty chat
chat = px.Chat()

# With a system prompt
chat = px.Chat(system_prompt="You are a helpful math tutor.")

# With initial messages (list of Message, dict, str, or list-of-content)
chat = px.Chat(
    system_prompt="You are a helpful math tutor.",
    messages=[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ],
)
```

The `messages` argument accepts a mixed list of four shapes — each
is coerced into a `Message`:

| Input shape | Becomes | Typical use |
|---|---|---|
| `Message(...)` | itself | Explicit construction. |
| `dict` `{"role": ..., "content": ...}` | `Message.from_dict(...)` | OpenAI-compatible shape. |
| bare `str` | `Message(role="assistant", content=[TEXT])` | Short-hand for an assistant reply. |
| `list` of `MessageContent` / `dict` / `str` | `Message(role="assistant", content=[...])` | Multi-block assistant reply. |

The bare `str` and bare `list` shortcuts **always produce an
ASSISTANT message** — for a user turn, pass a `Message(role="user",
...)` or a `{"role": "user", ...}` dict.

### 2.2 Add messages

```python
chat.append(px.Message(role="user", content="What is 2+2?"))
chat.append({"role": "assistant", "content": "4"})
chat.append("Quick follow-up thought")        # ASSISTANT shortcut
chat.append([
    px.MessageContent(type="text", text="What's in this image?"),
    px.MessageContent(type="image", path="photo.png", media_type="image/png"),
])                                             # ASSISTANT by shortcut — note below

chat.extend([
    px.Message(role="user", content="..."),
    {"role": "assistant", "content": "..."},
])

chat.insert(0, px.Message(role="user", content="prepended"))
chat.pop()                                     # removes and returns the last Message
chat.clear()                                   # remove all
```

**Gotcha.** `chat.append("text")` and `chat.append([...content...])`
create ASSISTANT-role messages. If you want a user turn, always
use the explicit `Message(role="user", ...)` or `{"role": "user",
...}` shape.

### 2.3 Access messages

```python
len(chat)                    # number of messages
chat[0]                      # first Message
chat[-1]                     # last Message
for msg in chat:             # iterate messages
    print(msg)

recent = chat[-4:]           # returns a NEW Chat with the same system_prompt
                             # — plug it straight into px.generate(messages=recent)
```

Slicing returns a fresh `Chat` (not a bare list) that carries the
same `system_prompt`, so it's drop-in for `px.generate()`.

### 2.4 Compose two chats

```python
a = px.Chat(system_prompt="A", messages=[...])
b = px.Chat(system_prompt="B", messages=[...])

merged = a + b       # new Chat; takes a.system_prompt ; b.messages appended
a += b               # in-place on a
```

The right-hand chat's `system_prompt` is **dropped** during
concatenation — only the left-hand system prompt survives.

### 2.5 Save and load

```python
import json

# Save
with open("conversation.json", "w") as f:
    json.dump(chat.to_dict(), f)

# Restore
with open("conversation.json") as f:
    chat = px.Chat.from_dict(json.load(f))
```

`to_dict` emits `{"system_prompt": ..., "messages": [...]}` (the
`system_prompt` key is omitted when it's `None`). Media `data`
bytes are base64-encoded on the wire, so binary payloads survive
JSON round-trips.

### 2.6 Equality and repr

```python
chat == other_chat           # compares system_prompt and messages
print(chat)                  # → "Chat(3 messages)"
```

---

## 3. Using `Message`

### 3.1 Create a Message

```python
# Simple text turn
px.Message(role="user", content="Hello!")
px.Message(role="assistant", content="Hi!")

# Multi-modal turn — content as a list of MessageContent blocks
px.Message(role="user", content=[
    px.MessageContent(type="text", text="What's in this image?"),
    px.MessageContent(type="image", path="photo.png", media_type="image/png"),
])
```

- `role` accepts `"user"` / `"assistant"` or the enum. Any other
  value raises.
- `content` can be a bare string (short-hand for one text block)
  or a list. In a list, bare `str` items are auto-wrapped into
  `MessageContent(type="text", text=str)`.

### 3.2 Serialize

```python
msg.to_dict()
# {"role": "user", "content": [{"type": "text", "text": "hi"}]}

px.Message.from_dict({"role": "user", "content": "hi"})
# Message(role='user', content=[MessageContent(type='text', text='hi')])

msg.copy()   # deep copy
```

---

## 4. Using `MessageContent`

A content block carries one payload — text, JSON, pydantic, tool
output, or media. Construct one per block.

### 4.1 Text

```python
px.MessageContent(type="text", text="Hello, world!")

# Thinking blocks surface in ResultRecord.content; you rarely build
# them by hand — they're emitted by reasoning models.
px.MessageContent(type="thinking", text="Working through step 1...")
```

### 4.2 Images, documents, audio, video

Provide exactly one of `source` (URL), `data` (bytes), or `path`
(local file). Set `media_type` for the MIME string.

```python
# From a URL
px.MessageContent(
    type="image",
    source="https://example.com/photo.jpg",
    media_type="image/jpeg",
)

# From a local file — read at request time
px.MessageContent(type="image", path="photo.png", media_type="image/png")

# From inline bytes
with open("photo.png", "rb") as f:
    img_bytes = f.read()
px.MessageContent(type="image", data=img_bytes, media_type="image/png")

# A document (PDF) — same pattern
px.MessageContent(type="document", path="report.pdf", media_type="application/pdf")

# Audio / video — same pattern
px.MessageContent(type="audio", path="recording.mp3", media_type="audio/mpeg")
px.MessageContent(type="video", source="https://example.com/clip.mp4",
                  media_type="video/mp4")
```

Two shortcuts:

- When `media_type` is set, `type` can be omitted — it's inferred
  from the MIME prefix (`image/*` → IMAGE, etc.).
- A media block can carry just a `provider_file_api_ids` entry
  (no local `source` / `data` / `path`) if you already uploaded
  the file via `px.files.upload()`. See `px_files_analysis.md`.

Supported MIME strings (the allow-list): see §1.1.

### 4.3 JSON

```python
px.MessageContent(type="json", json={"answer": 4, "confidence": 0.95})
```

JSON blocks mostly appear on the **result** side, when you used
`output_format="json"` in `px.generate()`. You rarely build one
by hand to *send*.

### 4.4 Pydantic instance

```python
import pydantic
from proxai.chat.message_content import PydanticContent

class MovieReview(pydantic.BaseModel):
    title: str
    rating: float

px.MessageContent(
    type="pydantic_instance",
    pydantic_content=PydanticContent(
        class_value=MovieReview,
        instance_value=MovieReview(title="Inception", rating=9.2),
    ),
)
```

You most often see these on the **result** side when you called
`px.generate(output_format=MovieReview)` — the returned
`rec.result.content[0]` is a `PYDANTIC_INSTANCE` block and
`rec.result.output_pydantic` is the already-parsed instance. You
rarely construct one by hand to send.

### 4.5 Tool output (with citations)

Tool blocks appear on result responses when the model invoked a
tool (e.g. web search). You don't build these by hand — they're
read off `ResultRecord.content`.

```python
# Reading tool output from a generate() result:
for block in rec.result.content:
    if block.type == px.ContentType.TOOL:
        for cite in block.tool_content.citations or []:
            print(cite.title, cite.url)
```

---

## 5. `Chat.export()` — shape it for the wire

`Chat.export()` returns a dict (the default) or a single string.
Its most-used parameters let you append JSON guidance to the
prompt when targeting providers without native structured output,
or relocate the system prompt for providers that don't accept a
system role.

```python
Chat.export(
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
) → dict | str
```

Parameter purposes:

- **`merge_consecutive_roles`** — merge runs of same-role
  messages into one. Default `True`.
- **`omit_thinking`** — strip `THINKING` blocks. Default `True`
  so round-trips through `Chat` don't re-send reasoning traces to
  the model.
- **`allowed_types`** — whitelist of content types. If any
  surviving block uses a disallowed type, raises. Useful as a
  pre-flight check before sending a chat to a text-only endpoint.
- **`add_json_guidance_to_system`** — append `"You must respond
  with valid JSON."` to the system prompt.
- **`add_json_schema_guidance_to_system`** — same, but with a
  schema block appended. Accepts a dict (JSON-serialized) or a
  pre-formatted string.
- **`add_json_guidance_to_user_prompt`** /
  **`add_json_schema_guidance_to_user_prompt`** — same, appended
  to the last user message instead of the system prompt.
- **`add_system_to_messages`** — emit the system prompt as a
  synthetic `{"role": "system", "content": ...}` first message
  (for providers that don't take a separate system field).
- **`add_system_to_first_user_message`** — prepend the system
  prompt to the first user message's text (for providers that
  don't accept a system role at all).
- **`export_single_prompt`** — return a plain string formatted as
  `"SYSTEM:\n...\n\nUSER:\n...\n\nASSISTANT:\n..."`. Useful for
  logs or completion-style providers. Binary `data` / `path`
  content raises — only `text` and `source` contribute.

The JSON-system / JSON-schema-system pair, the JSON-user / JSON-schema-user
pair, and the system-to-messages / system-to-first-user pair are
each mutually exclusive — setting both in a pair raises.

```python
chat = px.Chat(
    system_prompt="You are a movie review bot.",
    messages=[{"role": "user", "content": "Review Inception in one line."}],
)
wire = chat.export(
    add_json_schema_guidance_to_system={
        "type": "object",
        "properties": {"rating": {"type": "number"},
                       "summary": {"type": "string"}},
        "required": ["rating", "summary"],
    },
)
# wire["system_prompt"] now ends with:
#   "You must respond with valid JSON that follows this schema: {...}"
```

For structured output against providers that **do** support it
natively, just use `px.generate(output_format=...)` (see
`px_generate_analysis.md §2.7`) — you don't need to call `export()`
yourself.

---

## 6. Common patterns

### 6.1 Build a conversation turn-by-turn

```python
import proxai as px

chat = px.Chat(system_prompt="You are a helpful math tutor.")
chat.append(px.Message(role="user", content="What is 2+2?"))

rec = px.generate(messages=chat)
chat.append(rec.result.content)   # coerced to ASSISTANT message
print(rec.result.output_text)     # "4"

chat.append(px.Message(role="user", content="What about 2+3?"))
rec = px.generate(messages=chat)
print(rec.result.output_text)     # "5"
```

### 6.2 Multi-modal user message

```python
chat = px.Chat(system_prompt="You are a visual-reasoning assistant.")
chat.append(px.Message(role="user", content=[
    px.MessageContent(type="text", text="What's in this image?"),
    px.MessageContent(type="image", path="photo.png", media_type="image/png"),
]))

rec = px.generate(messages=chat, provider_model=("openai", "gpt-4o"))
print(rec.result.output_text)
```

`path` is read at request time; before the provider call,
`generate()` auto-uploads to the provider's file API when the
provider supports it (see `px_files_analysis.md §4`).

### 6.3 Dict-shaped messages (OpenAI-compatible)

```python
chat = px.Chat(messages=[
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What is 2+2?"},
])
# Mixing shapes on the same Chat is fine — some items can be dicts,
# others bare strings, others Message instances.
```

### 6.4 JSON-output guidance on providers without native support

```python
chat = px.Chat(
    system_prompt="You are a movie review bot.",
    messages=[{"role": "user", "content": "Review Inception in one line."}],
)
payload = chat.export(
    add_json_schema_guidance_to_system={
        "type": "object",
        "properties": {"rating": {"type": "number"},
                       "summary": {"type": "string"}},
        "required": ["rating", "summary"],
    },
)
# payload["system_prompt"] ends with the schema guidance.
```

### 6.5 Save and resume a conversation

```python
import json

# Save
with open("conversation.json", "w") as f:
    json.dump(chat.to_dict(), f)

# Restore (maybe in another process)
with open("conversation.json") as f:
    chat = px.Chat.from_dict(json.load(f))

# Continue
chat.append(px.Message(role="user", content="Continue where we left off."))
rec = px.generate(messages=chat)
```

### 6.6 Keep only the recent context

```python
full_chat = px.Chat.from_dict(stored)
recent = full_chat[-6:]              # new Chat, same system_prompt
rec = px.generate(messages=recent)   # drops straight into generate
```

### 6.7 Pre-flight check for a text-only endpoint

```python
try:
    chat.export(allowed_types=["text"])
except ValueError as e:
    print("Chat has non-text content — cannot send to a text-only endpoint.")
    print(e)
```

### 6.8 Compose two separate conversations

```python
tutorial = px.Chat(
    system_prompt="You are a tutor.",
    messages=[px.Message(role="user", content="Teach me Python lists.")],
)
followup = px.Chat(messages=[
    px.Message(role="user", content="Now teach me list comprehensions."),
])
combined = tutorial + followup
# → keeps tutorial.system_prompt; followup.system_prompt is dropped
```

### 6.9 Inspect the wire shape

```python
# Dict shape — what generate() would send internally
print(chat.export())

# Plain-text log
print(chat.export(export_single_prompt=True))
```

---

## 7. Errors

All synchronous — none are routed through
`suppress_provider_errors`.

| Where | Trigger | Error |
|---|---|---|
| `Chat.system_prompt = v` | non-str, non-None | `TypeError("system_prompt must be a string.")` |
| `Chat(messages=[item])` / `chat.append(item)` | `item` not `Message` / `dict` / `str` / `list` | `TypeError("Expected Message or dict, got ...")` |
| List-form content item | not `str` / `MessageContent` / `dict` | `TypeError("Content list items must be str or MessageContent, ...")` |
| `Message(role=...)` | invalid role string | `ValueError("Invalid role: ... Must be one of: ['user', 'assistant']")` |
| `MessageContent(...)` | type missing when `media_type` absent | `ValueError("'type' is required when 'media_type' is not provided.")` |
| `MessageContent(...)` | required field for a type not set (`text`, `json`, `pydantic_content`, `tool_content`, or one of `source/data/path/provider_file_api_ids` for media) | `ValueError` naming the required field |
| `MessageContent(type="text", source=..., ...)` | forbidden field on TEXT/THINKING | `ValueError` naming the forbidden fields |
| `MessageContent(media_type=...)` | MIME not in `SUPPORTED_MEDIA_TYPES` | `ValueError("Unsupported media_type: ... Supported [...]")` |
| `MessageContent(type="image", media_type="application/pdf")` | `type` / `media_type` mismatch | `ValueError("Content type '<t>' does not match media_type '<m>' ...")` |
| `Chat.export(...)` | mutually exclusive pair set together | `ValueError` naming the pair (see §5) |
| `Chat.export(allowed_types=[...])` | item not `str` / `ContentType` | `TypeError("allowed_types items must be str or ContentType, ...")` |
| `Chat.export(allowed_types=[...])` | some content has a type outside the list | `ValueError("Content type '<t>' is not in allowed_types.")` |
| `Chat.export(export_single_prompt=True)` | a content block has `data` or `path` | `ValueError("export_single_prompt does not support '...' field in message content.")` |
