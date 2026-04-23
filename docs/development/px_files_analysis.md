# `px.files` API Comprehensive Use Case Analysis

Source of truth: `src/proxai/connectors/files.py` (the `FilesManager`
class — both `px.files` and `client.files` delegate to a single
`FilesManager` on each client), `src/proxai/client.py` (the
`FileConnector` pass-through that exposes it), and
`src/proxai/chat/message_content.py` (`MessageContent`,
`FileUploadMetadata`, `FileUploadState`, `ProxDashFileStatus`). The
underlying provider / ProxDash operations live in
`src/proxai/connectors/file_helpers.py` (dispatch tables + MIME
allow-lists) and `src/proxai/connections/proxdash.py` (ProxDash file
endpoints). If this document disagrees with those files, the files
win — update this document.

This is the definitive reference for the file-management surface —
every method on `px.files`, how uploaded file identifiers are
carried on `MessageContent`, what changes when a ProxDash API key is
present, and the implicit auto-upload that happens inside
`generate()` before each provider call. Read this before adding a
new provider to the File API dispatch, a new MIME type to the
allow-list, or changing how `proxdash_file_id` is populated.

See also: `multimodal_large_file_analysis.md` (how media content
flows through `FeatureAdapter` and provider content converters),
`call_record_analysis.md` (CallRecord / `MessageContent` shape), and
`px_client_analysis.md` (where `allow_parallel_file_operations`
lives on `ProviderCallOptions`).

---

## 1. `px.files` structure (current)

```
px.files                                          # same on client.files
│
│   # Per-provider File API operations (real network I/O)
├── .upload(media, providers)          → MessageContent (mutated)
├── .download(media, provider=?, path=?) → MessageContent (mutated)
├── .list(providers=None,
│         limit_per_provider=100)      → list[MessageContent]
├── .remove(media, providers=None)     → MessageContent (mutated)
│
│   # Capability checks (no network)
├── .is_upload_supported(media, provider)   → bool
└── .is_download_supported(provider)        → bool
```

Every method operates on `MessageContent` media blocks (types IMAGE,
DOCUMENT, AUDIO, VIDEO). `list()` returns a list; the other
operational methods return the same `MessageContent` instance with
metadata mutated in place. Capability checks are pure predicates
against the dispatch tables in `file_helpers.py`.

Currently-supported File API providers: **gemini, openai, claude,
mistral**. Any other provider passed to these methods raises
`ValueError` at the public-method boundary
(`FilesManager._validate_provider_support`).

### 1.1 Fields mutated on `MessageContent`

A media `MessageContent` grows metadata through the file lifecycle.
All five file-related fields live on the `MessageContent` dataclass
at `message_content.py:280-284`:

```
MessageContent
│
│   # Local content — what the caller provides
├── path: str | None                             # local file path
├── data: bytes | None                           # raw bytes
├── media_type: str | None                       # MIME string
├── filename: str | None                         # filename for the provider
│
│   # Per-provider File API state — mutated by upload/remove/list
├── provider_file_api_ids: dict[str, str] | None
│   │                                           # {provider: file_id}
│   │                                           # e.g. {"openai": "file-abc123"}
│   │
├── provider_file_api_status: dict[str, FileUploadMetadata] | None
│   │                                           # per-provider rich metadata
│   │   ├── file_id: str                        # same as above
│   │   ├── provider: str
│   │   ├── filename: str | None
│   │   ├── size_bytes: int | None
│   │   ├── mime_type: str | None
│   │   ├── created_at: str | None              # ISO timestamp (provider side)
│   │   ├── expires_at: str | None              # when the provider will drop it
│   │   ├── uri: str | None                     # some providers return a URI
│   │   ├── state: FileUploadState | None       # PENDING | ACTIVE | FAILED
│   │   └── sha256_hash: str | None
│   │
│   # ProxDash-layer state — populated only when ProxDash is connected
├── proxdash_file_id: str | None                # persistent cross-provider ref
└── proxdash_file_status: ProxDashFileStatus | None
    ├── file_id: str
    ├── s3_key: str | None                      # ProxDash's S3 object key
    ├── upload_confirmed: bool                   # True once S3 PUT succeeded
    ├── source: str | None
    ├── created_at: str | None
    └── updated_at: str | None
```

`FileUploadState` values (`message_content.py:130-135`):

- `PENDING` — upload initiated, not yet confirmed active.
- `ACTIVE` — ready to be referenced by `file_id` in `generate()`.
- `FAILED` — upload failed; `file_id` on the status will be an
  empty string placeholder. The provider is recorded so callers can
  retry surgically.

### 1.2 Provider support matrices

The file-manager enforces three independent compatibility checks.
Data is cached in `file_helpers.py`.

#### 1.2.1 Which providers have a File API at all

From the `UPLOAD_DISPATCH` / `REMOVE_DISPATCH` / `LIST_DISPATCH` /
`DOWNLOAD_DISPATCH` tables (`file_helpers.py:144, 184, 280, 298`):

| Provider | upload | list | remove | download |
|---|---|---|---|---|
| `gemini`  | ✅ | ✅ | ✅ | ❌ |
| `openai`  | ✅ | ✅ | ✅ | ❌ |
| `claude`  | ✅ | ✅ | ✅ | ❌ |
| `mistral` | ✅ | ✅ | ✅ | ✅ |

Only Mistral lets you download a file you previously uploaded via
its File API (`DOWNLOAD_SUPPORTED_PROVIDERS` at line 428). Use
`is_download_supported()` before calling `download()` with a
specific provider, or let `download()` pick the best source
automatically (see §2.2).

#### 1.2.2 Upload MIME acceptance per provider

`UPLOAD_SUPPORTED_MEDIA_TYPES` (`file_helpers.py:352`). Gemini,
OpenAI, and Claude accept every ProxAI-recognised media type.
Mistral's File API accepts only documents (PDF, DOCX, XLSX, CSV,
TXT, Markdown) and images — no audio, no video.

`is_upload_supported(media, provider)` checks this.

#### 1.2.3 `file_id` reference acceptance in `generate()`

Even a provider whose File API accepts your upload may refuse to
reference it from a chat message. The adapter tables live in
`REFERENCE_SUPPORTED_MEDIA_TYPES` (`file_helpers.py:365-426`):

- **gemini** — full multi-modal: PDFs, Word, Excel, CSV, text,
  images (PNG/JPEG/GIF/WebP/HEIC/HEIF), audio
  (MPEG/WAV/FLAC/AAC/OGG/AIFF), video (MP4/WebM/MOV/AVI/MPEG/MKV).
- **claude** — PDFs, plain text, and images
  (PNG/JPEG/GIF/WebP/HEIC/HEIF). No audio, no video.
- **openai** — documents only (PDF, DOCX, XLSX, CSV, TXT,
  Markdown). Images are sent inline rather than by `file_id`.
- **mistral** — documents (PDF, TXT, MD, CSV) and images
  (PNG/JPEG/GIF/WebP/HEIC/HEIF).

`is_upload_supported(media, provider)` returns `True` only when
BOTH the upload MIME check AND the reference MIME check pass —
uploading a file that cannot be referenced is wasteful.

See `multimodal_large_file_analysis.md §8` for why the reference
check is stricter than the upload check.

---

## 2. File operations

Each method below validates arguments, runs the network I/O in
parallel when the client's
`provider_call_options.allow_parallel_file_operations` is `True`
(the default; see `px_client_analysis.md §2.5.3`), and — when
ProxDash is connected — fans the same operation into the ProxDash
layer in the same parallel pool. §3 covers the ProxDash side in
detail.

### 2.1 `upload()`

Upload a media block to one or more provider File APIs. Mutates
`media.provider_file_api_ids` and `media.provider_file_api_status`
in place; returns the same instance for convenience.

```python
px.files.upload(
    media: MessageContent,                       # required, IMAGE/DOCUMENT/AUDIO/VIDEO
    providers: list[str],                        # e.g. ['gemini', 'claude']
) → MessageContent
```

`media` must have at least one of `path` or `data` set. `providers`
may be an empty list **only** when ProxDash is connected — in that
case the upload goes to ProxDash only, with no per-provider
`file_id`s populated.

**Validation raises (`ValueError`) before any network call:**

- `media.type` is not one of IMAGE / DOCUMENT / AUDIO / VIDEO.
- `media.path` and `media.data` are both `None`.
- `providers` is empty and ProxDash is not connected.
- A named provider has no File API dispatch entry, or no API key
  configured for its provider keys.
- A named provider does not support this `media_type` for both
  upload and reference (checked via `is_upload_supported`).

**Partial-failure semantics.** Uploads to different providers are
independent. If any fails:

- Providers that succeeded have their entries populated on
  `media.provider_file_api_ids` and `provider_file_api_status`.
- Providers that failed get a placeholder status with
  `file_id=''` and `state=FileUploadState.FAILED` — preserving the
  record so callers can see which provider failed.
- A `FileUploadError` is raised at the end with
  `error.errors: dict[str, Exception]` and `error.media` — the
  caller can retry only the failed providers.

### 2.2 `download()`

Fetch file bytes back from a previous upload. On Mistral this hits
the provider's File API directly; when ProxDash is connected and
the media has a `proxdash_file_id`, ProxDash's S3 bucket is tried
first (cross-provider, no ephemeral expiry).

```python
px.files.download(
    media: MessageContent,
    provider: str | None = None,                 # default: auto
    path: str | None = None,                     # save to disk if set
) → MessageContent
```

**Resolution order when `provider=None`:**

1. ProxDash S3 (if `proxdash_file_id` is set and ProxDash is
   connected and the download succeeds).
2. Mistral (first in `_DOWNLOAD_PROVIDER_PRIORITY`) if present on
   `media.provider_file_api_ids`.
3. Any other provider present on the media — but only Mistral has
   an entry in `DOWNLOAD_DISPATCH`, so anything else raises.

When `path` is set, the bytes are written to disk and `media.path`
is updated. Otherwise the bytes are stored on `media.data`.

**Raises:**

- `ValueError` if an explicit `provider` is named but missing from
  `media.provider_file_api_ids`.
- `ValueError` if no upload source is reachable (no
  `proxdash_file_id`, no provider in `media.provider_file_api_ids`).
- `ValueError` if the resolved provider does not support download
  (`DOWNLOAD_SUPPORTED_PROVIDERS`).

### 2.3 `list()`

Enumerate files already uploaded to one or more provider File APIs,
with optional deduplication against ProxDash's central index. The
returned list contains a `MessageContent` per file, carrying only
metadata — `path` and `data` are `None`, so call `download()` to
fetch bytes.

```python
px.files.list(
    providers: list[str] | None = None,           # default: all with keys
    limit_per_provider: int = 100,
) → list[MessageContent]
```

When `providers=None`, every provider with an API key and a
`LIST_DISPATCH` entry is queried in parallel. An empty list for
`providers` raises `ValueError` — pass `None` to mean "all".

**ProxDash interaction (§3.1).** When connected:

1. ProxDash `/files` endpoint is queried first and returns
   `MessageContent`s with combined provider metadata already
   stitched together (these come first in the return list).
2. Each provider's `list` is still queried, but any file whose
   `(provider, file_id)` pair already appears in a ProxDash result
   is dropped — the ProxDash record is authoritative.
3. Provider-only files (unknown to ProxDash) follow in the list.

**Raises:**

- `ValueError` if `providers=[]` (empty list).
- `ValueError` if no provider in the resolved set has an API key.
- Per-provider list failures are **swallowed** — that provider
  contributes an empty slice. The call does not fail because one
  provider is down.

### 2.4 `remove()`

Delete uploaded files from one or more providers, mirroring
`upload()`.

```python
px.files.remove(
    media: MessageContent,
    providers: list[str] | None = None,          # default: all uploaded
) → MessageContent
```

When `providers=None`, every provider currently on
`media.provider_file_api_ids` is removed (and ProxDash if the
media has a `proxdash_file_id`). Empty list raises.

**Mutation semantics.** Successful removals clear their entries from
`provider_file_api_ids` / `provider_file_api_status` on the media
instance. Failed removals leave the entry in place, so retrying
with the same `media` targets only the still-uploaded providers.
If ProxDash's delete succeeded, `proxdash_file_id` and
`proxdash_file_status` are cleared too.

**Raises:**

- `ValueError` if `providers=[]`, if the media has no uploaded
  providers, or if a named provider has no `file_id` on this media.
- `FileRemoveError` with `error.errors: dict[str, Exception]` and
  `error.media` if one or more provider removals fail. ProxDash
  delete failure is **silent** — it never raises.

### 2.5 `is_upload_supported()`

```python
px.files.is_upload_supported(
    media: MessageContent,
    provider: str,
) → bool
```

Pure predicate against the two allow-list tables; no network call.
Checks both:

- The provider's File API accepts this MIME type
  (`UPLOAD_SUPPORTED_MEDIA_TYPES`).
- The provider's `generate()` endpoint accepts `file_id`
  references for this MIME type
  (`REFERENCE_SUPPORTED_MEDIA_TYPES`).

Returns `False` if `media.media_type` is `None`, if the provider
is not registered, or either allow-list check fails.

### 2.6 `is_download_supported()`

```python
px.files.is_download_supported(provider: str) → bool
```

Pure predicate. Returns `True` only for providers in
`DOWNLOAD_SUPPORTED_PROVIDERS` — today that is exactly
`{'mistral'}`.

---

## 3. ProxDash integration layer

ProxDash is ProxAI's hosted monitoring backend (see
`px_client_analysis.md §2.4` for the connection options). When a
client is connected to ProxDash, the file-manager automatically
fans every operational call into ProxDash alongside the provider
calls. When it's not connected, the file-manager reduces to a thin
multi-provider dispatcher — nothing crosses network boundaries other
than the providers you named.

The internal gate is
`FilesManager._proxdash_connected()`: `proxdash_connection is not
None and proxdash_connection.status == CONNECTED`. A connection
needs a valid `PROXDASH_API_KEY` and `disable_proxdash != True` on
the client's `proxdash_options` (see `px_client_analysis.md §5.4`).

### 3.1 What changes per method when ProxDash is connected

| Operation | Without ProxDash | With ProxDash connected |
|---|---|---|
| **upload** | Uploads to each named provider; mutates `provider_file_api_ids` / `provider_file_api_status`. | Same, **plus** uploads the bytes to ProxDash S3 in the same parallel pool. On success, `media.proxdash_file_id` and `proxdash_file_status` are set. After all provider uploads complete, provider metadata is pushed to ProxDash via `PATCH /files/{id}` so the central record reflects every provider's `file_id`. ProxDash failure is silent. |
| **download** | Only Mistral works — other providers raise. | When `provider=None` is passed, ProxDash is tried first via presigned S3 URL — works for any file with a `proxdash_file_id` regardless of original provider. Falls back to provider download if ProxDash fails. |
| **list** | Returns provider-side file listings only. | ProxDash `/files` is queried first (returns combined cross-provider metadata); providers are still queried and then **deduplicated** against the ProxDash results by `(provider, file_id)` pair. |
| **remove** | Removes from named providers only. | Same, **plus** deletes the ProxDash record (and the S3 object) in parallel. ProxDash delete failure is silent. On success, `proxdash_file_id` and `proxdash_file_status` are cleared. |
| **is_upload_supported / is_download_supported** | Unchanged — pure predicates. | Unchanged. |

### 3.2 Why ProxDash changes the lifecycle story

Provider File APIs are ephemeral:

- Gemini `files` — expire 48 hours after upload.
- OpenAI `files` — `user_data` files last ~30 days by default.
- Claude `beta.files` — retention policy varies; check Anthropic docs.
- Mistral `files` — persistent until explicit delete.

Provider `file_id`s are also scoped to one provider — a
`gemini` file-id is useless to `claude`. A fallback chain that
switches providers on failure would have to re-upload every
attachment.

ProxDash sits above the providers as the canonical copy:

- Bytes live in ProxDash's S3 bucket (no provider-side expiry).
- `proxdash_file_id` is persistent and carries the provider
  metadata for every provider you've uploaded to.
- `upload()` with `providers=[]` is legal when ProxDash is
  connected — the file goes to ProxDash only. Later, when you
  actually call `generate()` against some provider, the
  auto-upload path (§4) resolves from ProxDash back into that
  provider's File API.
- `download()` without naming a provider can always succeed
  because ProxDash is provider-agnostic.

### 3.3 ProxDash connection states

From `types.ProxDashConnectionStatus`:

- `NOT_INITIALIZED` — client has never tried to connect.
- `API_KEY_NOT_FOUND` — no `PROXDASH_API_KEY` env var and none
  passed in options. File manager behaves as "disconnected" —
  every operation is provider-only, all ProxDash fanout is
  skipped.
- `DISABLED` — `proxdash_options.disable_proxdash=True`. Same as
  above.
- `CONNECTED` — key validated. File manager fans out as §3.1.
- `PROXDASH_INVALID_RETURN` — connection attempted but the key
  failed verification. Same as disconnected for file operations.

Only `CONNECTED` triggers ProxDash fanout. Every other status is
"off" from the file manager's perspective.

---

## 4. Auto-upload in `generate()`

`ProviderConnector.generate()` inspects chat messages for media
blocks before dispatching to the provider. For every media
`MessageContent` that has local content (`path` or `data`) but no
`file_id` for the target provider, the connector calls
`FilesManager.upload()` with `providers=[self.PROVIDER_NAME]`
before the request goes out. This is the hook that makes
`px.generate(messages=[...images...])` "just work" — you never
have to call `px.files.upload()` by hand.

Reference: `provider_connector.py:_auto_upload_media` (line ~752).

### 4.1 When it fires

All of:

- `self.files_manager_instance is not None` (the client has a
  files manager wired in — it does by default).
- `query_record.chat is not None` (only the chat API is covered;
  the `prompt` API has no attachments).
- The message content has `type` in {IMAGE, DOCUMENT, AUDIO,
  VIDEO}.
- The media doesn't already carry a `file_id` for this provider
  (previous uploads are respected — no re-upload).
- The media has local content (`path` or `data`). Remote-only
  references (`source` URL with no local copy) are passed to the
  provider directly.
- `files_manager_instance.is_upload_supported(mc, provider)` is
  `True` (both MIME allow-lists).

### 4.2 What it does on match

Calls `FilesManager.upload(media=mc, providers=[provider_name])`
per qualifying media block. Parallelism is governed by
`provider_call_options.allow_parallel_file_operations` (default
`True`); with multiple media blocks a thread pool is used.

Failures in the auto-upload are suppressed (each `future.result()`
sits inside a `try: ... except: pass`). The media then falls
through to the provider's content converter, which will either use
inline base64 (if it can) or surface whatever error the provider
returns. The query is NOT aborted on upload failure — the
provider-side converter gets the last word.

### 4.3 TEST mode

When the client's `run_type == TEST` (used by `tests/` and by any
integration harness that wants to exercise the file-id code path
without real uploads), the dispatch tables route to the `_MOCK`
variants (`file_helpers.py:459-462`). `mock_upload` returns a
`FileUploadMetadata` with `file_id=f'mock-file-{uuid.uuid4()[:8]}'`
and `state=ACTIVE`. No network call happens, but the `file_id` is
populated so the generate pipeline exercises the file-reference
code path.

---

## 5. Common patterns

### 5.1 Upload and reuse within a session

```python
import proxai as px

doc = px.MessageContent(
    type="document", path="report.pdf", media_type="application/pdf",
)

# One upload, two providers.
px.files.upload(media=doc, providers=["gemini", "claude"])

# Each call below references the same file_id per provider.
px.generate(
    messages=[{"role": "user", "content": [doc, "Summarise."]}],
    provider_model=("gemini", "gemini-2.5-flash"),
)
px.generate(
    messages=[{"role": "user", "content": [doc, "Translate to French."]}],
    provider_model=("claude", "claude-sonnet-4-6"),
)
```

### 5.2 List and clean up

```python
# List everything currently uploaded to providers you have keys for.
files = px.files.list()
for f in files:
    print(f.filename, f.media_type, f.provider_file_api_ids)

# Remove any older than a week (provider side).
import datetime as dt
cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
for f in files:
    for provider, meta in (f.provider_file_api_status or {}).items():
        if meta.created_at and meta.created_at < cutoff.isoformat():
            px.files.remove(media=f, providers=[provider])
```

### 5.3 Download from Mistral

```python
files = px.files.list(providers=["mistral"])
px.files.download(media=files[0], path="/tmp/doc.pdf")
print(files[0].path)   # → "/tmp/doc.pdf"
```

### 5.4 Multi-provider upload for fallback chains

If your `generate()` call uses `fallback_models` across providers,
pre-upload media to every model you might fall back to so the
fallback loop doesn't pause for a cold upload:

```python
img = px.MessageContent(
    type="image", path="chart.png", media_type="image/png",
)
px.files.upload(media=img, providers=["openai", "claude", "gemini"])

rec = px.generate(
    messages=[{"role": "user", "content": [img, "Describe this chart."]}],
    provider_model=("openai", "gpt-4o"),
    connection_options=px.ConnectionOptions(
        fallback_models=[("claude", "claude-sonnet-4-6"),
                         ("gemini", "gemini-2.5-flash")],
    ),
)
```

### 5.5 Handling partial upload failures

```python
from proxai.connectors.files import FileUploadError

try:
    px.files.upload(media=doc, providers=["gemini", "claude", "openai"])
except FileUploadError as e:
    # Successes are already on `e.media`.
    print("failed providers:", list(e.errors.keys()))
    for provider, err in e.errors.items():
        print(f"  {provider}: {err}")
    # Retry only the failed ones when ready.
    failed = list(e.errors.keys())
    px.files.upload(media=e.media, providers=failed)
```

### 5.6 ProxDash-backed cross-session reuse

```python
# First session: upload, ProxDash persists the canonical copy.
import proxai as px
px.connect()   # PROXDASH_API_KEY must be set
doc = px.MessageContent(type="document", path="report.pdf",
                        media_type="application/pdf")
px.files.upload(media=doc, providers=["gemini"])
pid = doc.proxdash_file_id
print("proxdash id:", pid)

# Later session (different process): rehydrate from ProxDash,
# no re-upload needed.
same = px.files.list(providers=["gemini"])[0]  # carries the same proxdash_file_id
px.generate(
    messages=[{"role": "user", "content": [same, "One-line summary."]}],
    provider_model=("gemini", "gemini-2.5-flash"),
)
```

### 5.7 Skip explicit upload: auto-upload inside `generate()`

When the client has a files manager (the default), you never have
to call `upload()` at all for the common case:

```python
img = px.MessageContent(
    type="image", path="diagram.png", media_type="image/png",
)
# No upload() call. generate() uploads img to openai on your behalf
# before dispatching the chat request.
px.generate(
    messages=[{"role": "user", "content": [img, "What does this show?"]}],
    provider_model=("openai", "gpt-4o"),
)
print(img.provider_file_api_ids)  # → {'openai': 'file-xxx'}
```

### 5.8 Client-instance file management

Same API, scoped to an independent client (separate API keys,
separate ProxDash connection, separate log stream):

```python
client = px.Client(
    proxdash_options=px.ProxDashOptions(disable_proxdash=True),
)
client.files.upload(media=doc, providers=["openai"])
client.files.remove(media=doc)
```

---

## 6. Errors

| Method | Trigger | Error |
|---|---|---|
| `upload()` | `media.type` not a media type | `ValueError` (`upload() requires a media content type...`) |
| `upload()` | `media.path` and `media.data` both `None` | `ValueError` (`MessageContent must have 'path' or 'data' set for upload.`) |
| `upload()` | `providers=[]` and ProxDash not connected | `ValueError` (`No providers specified and ProxDash is not connected...`) |
| `upload()` | named provider not in `UPLOAD_DISPATCH` | `ValueError` (`Provider '<name>' does not support the File API. Supported: [...]`) |
| `upload()` | provider has no API key in env | `ValueError` (`No API key configured for provider '<name>'.`) |
| `upload()` | media type rejected by upload or reference allow-list | `ValueError` (`Media type '...' cannot be uploaded and referenced by file_id on provider '<name>'.`) |
| `upload()` | any provider upload raised | `FileUploadError` with `error.errors: dict[str, Exception]` and `error.media` — successes stay recorded on the media |
| `download()` | named provider missing from `media.provider_file_api_ids` | `ValueError` (`No file_id found for provider '<name>' on this media content.`) |
| `download()` | no provider named and media has no uploads | `ValueError` (`No uploaded providers found on this media content.`) |
| `download()` | resolved provider has no `DOWNLOAD_DISPATCH` entry | `ValueError` (`Provider '<name>' does not support file download. Supported: [...]`) |
| `download()` | provider has no API key in env | `ValueError` (`No API key configured for provider '<name>'.`) |
| `list()` | `providers=[]` (empty list) | `ValueError` (`'providers' must contain at least one provider name...`) |
| `list()` | `providers=None` and no providers with keys | `ValueError` (`No providers with API keys found for file listing.`) |
| `remove()` | `providers=[]` | `ValueError` (same "`'providers' must contain at least one provider name...`") |
| `remove()` | media has no uploads and no `proxdash_file_id` | `ValueError` (`No uploaded providers found on this media content.`) |
| `remove()` | named provider has no `file_id` on this media | `ValueError` (`No file_id found for provider '<name>' on this media content.`) |
| `remove()` | any provider removal raised | `FileRemoveError` with `error.errors` / `error.media` |

`FileUploadError` and `FileRemoveError` both subclass `Exception`
and are defined in `src/proxai/connectors/files.py:17-38`. Both
carry the `media` field so the caller can inspect partial state and
retry surgically.

**ProxDash failures never raise** — the ProxDash fanout in upload /
remove / list / download swallows every exception silently, so the
caller's success signal depends only on the named providers.

---

## 7. Module-level vs client instance

```python
# Module-level (uses the hidden default client)
import proxai as px
px.files.upload(media=doc, providers=["gemini"])

# Client instance (independent)
client = px.Client(...)
client.files.upload(media=doc, providers=["gemini"])
```

Same API, same methods, same parameters. Each client owns its own
`FilesManager` (`_files_manager_instance`), so the two do not share
ProxDash connection state or API-key state. See `px_client_analysis.md`
§5.1 for the default-vs-instance isolation rules.
