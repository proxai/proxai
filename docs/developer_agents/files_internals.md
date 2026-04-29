# Files — Internals

Source of truth: `src/proxai/connectors/files.py` (the
`FilesManager` class and its per-operation helpers),
`src/proxai/connectors/file_helpers.py` (the
`UPLOAD_DISPATCH` / `REMOVE_DISPATCH` / `LIST_DISPATCH` /
`DOWNLOAD_DISPATCH` tables, the `UPLOAD_SUPPORTED_MEDIA_TYPES` /
`REFERENCE_SUPPORTED_MEDIA_TYPES` / `DOWNLOAD_SUPPORTED_PROVIDERS`
capability tables, and the mock dispatches),
`src/proxai/connectors/provider_connector.py` (`_auto_upload_media`
and the `generate` hook that calls it), and
`src/proxai/connections/proxdash.py` (ProxDash file endpoints —
`upload_file`, `download_file`, `list_files`, `delete_file`,
`update_file`). If this document disagrees with those files, the
files win — update this document.

This is the definitive reference for how ProxAI turns a local
`MessageContent` into a provider-uploaded file, how it dispatches
per provider, and how the auto-upload hook inside
`ProviderConnector.generate` converts inline media into `file_id`
references before the executor runs. Read this before adding a
new provider File API, extending the supported MIME sets, editing
`_auto_upload_media`, wiring a new dispatch table (e.g., signed-URL
generation), or changing how `MessageContent.provider_file_api_ids`
is populated.

See also: `../user_agents/api_guidelines/px_files_api.md` (caller-
level view of the same surface, with per-method argument semantics
and error tables); `adding_a_new_provider.md` (executor-side
contract — what `query_record.chat` carries on entry and the
`_to_*_part` convention each connector follows; this doc covers
the auto-upload step that runs before the executor);
`state_controller.md` (`FilesManager` is a `StateControlled`
subclass, so its nested fields — `proxdash_connection`,
`api_key_manager` — follow the same state-propagation rules as the
rest of the client).

---

## 1. Files subsystem structure (current)

```
ProxAIClient                                         # src/proxai/client.py
│
├── _files_manager_instance: FilesManager            # StateControlled
│     │   # 1:1 with client; created when api_key_manager is ready
│     │   # (client.py:919-927). Nested state propagates via
│     │   # FilesManagerState (types.py:877-884).
│     │
│     ├── run_type                   → TEST | PRODUCTION
│     ├── logging_options            → LoggingOptions
│     ├── proxdash_connection        → ProxDashConnection (state-controlled)
│     ├── provider_call_options      → ProviderCallOptions
│     └── api_key_manager            → ApiKeyManager (state-controlled)
│
ProviderConnector                                    # src/proxai/connectors/provider_connector.py
│
├── files_manager_instance: FilesManager | None      # shared with client
└── _auto_upload_media(query_record)                 # called from generate()
      │   # Walks chat.messages[*].content for MEDIA blocks, uploads
      │   # via files_manager_instance.upload(...) to self.PROVIDER_NAME.
      │   # Mutates MessageContent in place.

file_helpers                                         # src/proxai/connectors/file_helpers.py
│
│   # Real-provider dispatch
├── UPLOAD_DISPATCH        = {'claude', 'openai', 'gemini', 'mistral'}
├── REMOVE_DISPATCH        = {'claude', 'openai', 'gemini', 'mistral'}
├── LIST_DISPATCH          = {'claude', 'openai', 'gemini', 'mistral'}
├── DOWNLOAD_DISPATCH      = {'mistral'}                # only Mistral
│
│   # Mock dispatch (run_type == TEST)
├── MOCK_UPLOAD_DISPATCH     # fake file_id per call, ACTIVE state
├── MOCK_REMOVE_DISPATCH     # no-op
├── MOCK_LIST_DISPATCH       # always []
├── MOCK_DOWNLOAD_DISPATCH   # returns b'mock-file-content'
│
│   # Capability tables (MIME-level)
├── UPLOAD_SUPPORTED_MEDIA_TYPES      # what each File API accepts
├── REFERENCE_SUPPORTED_MEDIA_TYPES   # what each generate endpoint accepts
│                                     # as a file_id (stricter than upload)
└── DOWNLOAD_SUPPORTED_PROVIDERS      # {'mistral'}

FilesManager operations (files.py:52-855)
│
├── upload(media, providers)          → FileUploadError on any failure
│   ├── _validate_upload_media          # require IMAGE/DOCUMENT/AUDIO/VIDEO + path|data
│   ├── _validate_provider_support      # dispatch ∋ provider ∧ api key set
│   ├── is_upload_supported             # MIME ∈ UPLOAD ∧ MIME ∈ REFERENCE
│   ├── _init_upload_fields             # allocate .provider_file_api_* dicts
│   ├── _execute_uploads                # ThreadPoolExecutor path or serial
│   └── _sync_provider_metadata_to_proxdash
│
├── download(media, provider?, path?)  → MessageContent (with path or data)
│   ├── _download_from_proxdash         # tried first when ProxDash connected
│   ├── _resolve_download_provider      # _DOWNLOAD_PROVIDER_PRIORITY = ['mistral']
│   └── _write_download_result          # writes to path or sets media.data
│
├── list(providers?, limit_per_provider=100)
│   ├── _resolve_list_providers         # None → all providers with API keys
│   ├── _execute_lists                  # ProxDash + providers in parallel
│   ├── _filter_proxdash_by_providers   # ProxDash dedupes provider-side results
│   └── _build_covered_file_ids         # skip provider files ProxDash already tracks
│
├── remove(media, providers?)          → FileRemoveError on any failure
│   ├── _resolve_remove_providers       # None → media.provider_file_api_ids.keys()
│   ├── _validate_remove_providers      # per provider: dispatch ∧ file_id exists
│   ├── _execute_removes                # ThreadPoolExecutor path or serial
│   └── del media.provider_file_api_ids[provider]  # on success
│
├── is_upload_supported(media, provider)  → bool
└── is_download_supported(provider)       → bool
```

The four dispatch tables (`UPLOAD_DISPATCH`, `REMOVE_DISPATCH`,
`LIST_DISPATCH`, `DOWNLOAD_DISPATCH`) are the single extension
point for new-provider support. Adding a new provider means (a)
writing four module-level functions in `file_helpers.py`, (b)
inserting them into the right dispatch maps, and (c) populating
the capability tables with the provider's MIME support.
Everything else — the auto-upload hook, the per-provider `_to_*_part`
consumers, the `px.files.*` façade — reads from these tables.

### 1.1 Dispatch table signatures

Every function in a dispatch table accepts a common keyword-only
shape so the dispatch site in `FilesManager` can call them
polymorphically. The signatures below are enforced by convention,
not a protocol — new providers must match them exactly.

| Dispatch | Function signature | Returns |
|---|---|---|
| `UPLOAD_DISPATCH[p]` | `(file_path: str \| None, file_data: bytes \| None, filename: str, mime_type: str, token_map: ProviderTokenValueMap) -> FileUploadMetadata` | `FileUploadMetadata` with `file_id`, `state=ACTIVE`/`PENDING`/`FAILED`, and provider-specific fields (`uri`, `expires_at`, `sha256_hash`, etc. where available). |
| `REMOVE_DISPATCH[p]` | `(file_id: str, token_map: ProviderTokenValueMap) -> None` | — |
| `LIST_DISPATCH[p]` | `(token_map: ProviderTokenValueMap, limit: int = 100) -> list[FileUploadMetadata]` | Provider-normalized list, one entry per file, MIME and expiry populated when the provider exposes them. |
| `DOWNLOAD_DISPATCH[p]` | `(file_id: str, token_map: ProviderTokenValueMap) -> bytes` | Raw file bytes. |

Uploads come in two input shapes — `file_path` (path on disk) or
`file_data` (in-memory bytes). Every upload helper handles both: if
`file_path` is set, the helper opens the file; if only `file_data`
is set, the helper wraps it in `io.BytesIO`. Exactly one must be
set — `FilesManager._validate_upload_media` enforces this before
dispatch.

The `token_map` is a dict of env-var-name → value pulled from
`ApiKeyManager.get_provider_keys(provider)`. This is the same map
each provider connector uses at SDK init — e.g.,
`upload_to_claude` reads `token_map['ANTHROPIC_API_KEY']`,
`upload_to_mistral` reads `token_map['MISTRAL_API_KEY']`.

### 1.2 Capability tables — upload vs. reference

Two MIME-level capability tables gate `is_upload_supported` and,
by extension, `_auto_upload_media`:

- **`UPLOAD_SUPPORTED_MEDIA_TYPES[provider]`** — the set of MIMEs
  the provider's File API will *accept on upload*.
- **`REFERENCE_SUPPORTED_MEDIA_TYPES[provider]`** — the set of
  MIMEs the provider's `generate`-side endpoints accept as a
  `file_id` reference. Often stricter than upload.

`is_upload_supported` returns `True` only when the MIME is in
both sets:

```python
def is_upload_supported(media, provider):
    if media.media_type is None: return False
    if provider not in UPLOAD_SUPPORTED_MEDIA_TYPES: return False
    if media.media_type not in UPLOAD_SUPPORTED_MEDIA_TYPES[provider]:
        return False
    if provider not in REFERENCE_SUPPORTED_MEDIA_TYPES: return False
    return media.media_type in REFERENCE_SUPPORTED_MEDIA_TYPES[provider]
```

The two-table split exists because a provider's File API is a
separate product surface from its chat endpoint. Mistral, for
example, accepts DOCX uploads via `client.files.upload(...,
purpose='ocr')`, but Mistral's chat endpoint does not take a DOCX
`file_id` as a content block — only PDF / text / CSV / image. If
`UPLOAD_SUPPORTED_MEDIA_TYPES` alone gated the auto-upload path,
we would upload a DOCX to Mistral, then silently drop the `file_id`
on the executor side because chat wouldn't accept it. The
intersection gate prevents that waste.

When you extend a provider's capability, update both tables (or
neither). Updating only one produces asymmetric behavior: uploads
work but the file_id never reaches the executor, or vice versa.

### 1.3 `_DOWNLOAD_PROVIDER_PRIORITY`

`FilesManager._DOWNLOAD_PROVIDER_PRIORITY = ['mistral']`
(`files.py:55`). When `download(media, provider=None)` is called
and ProxDash has no copy (or the ProxDash download fails), this
list is consulted: the first provider in it that appears in
`media.provider_file_api_ids` wins. Today, Mistral is the only
provider that exposes a stable download endpoint; Claude, OpenAI,
and Gemini all return "file not found" or refuse the request for
user-uploaded content. Do not add a provider to this list without
first verifying its download endpoint returns content suitable for
callers — see the "download from Mistral" notes in `px_files_api.md`
§2.2 for the behavioral contract the priority order promises.

---

## 2. `FilesManager` — state and lifecycle

`FilesManager` is a `StateControlled` subclass (see
`state_controller.md`) with five state-propagating fields:

```python
FilesManagerState:
  run_type:            RunType          # TEST / PRODUCTION dispatch selection
  logging_options:     LoggingOptions
  proxdash_connection: ProxDashConnectionState
  provider_call_options: ProviderCallOptions  # the parallelism flag
  api_key_manager:     ApiKeyManagerState     # token_map lookup
```

One instance exists per `ProxAIClient` — it is created in
`client.py:919-927` as soon as the `api_key_manager` and
`proxdash_connection` are both ready, and stored in
`self._files_manager_instance`. The client exposes it via a
property that goes through `StateControlled`'s state-controlled
getter/setter (`client.py:1359-1371`) so downstream objects
(notably `ProviderConnector`) receive a shared reference rather
than an independent copy.

Two of the fields — `proxdash_connection` and `api_key_manager` —
are themselves `StateControlled`. They are stored and set via
`set_state_controlled_property_value` and reconstituted on
deserialization through
`proxdash_connection_deserializer` / `api_key_manager_deserializer`
(`files.py:110-113, 133-136`). See `state_controller.md` §6 for the
nested-state contract these follow.

### 2.1 Run-type dispatch (TEST vs. PRODUCTION)

Every `_get_*_dispatch` helper (`files.py:164-182`) branches on
`self.run_type`:

```
run_type == TEST       → MOCK_UPLOAD_DISPATCH     / etc.
run_type == PRODUCTION → UPLOAD_DISPATCH          / etc.
```

Under TEST, `mock_upload` allocates a `file_id` of the form
`f'mock-file-{uuid.uuid4().hex[:8]}'` and marks the upload
`ACTIVE`. No real API is hit, but the `provider_file_api_ids` dict
ends up populated exactly as in production — so any executor path
that reads `provider_file_api_ids` exercises the `file_id` branch
of `_to_*_part` just like in a real call. This is the reason the
mock returns a non-empty fake id rather than an empty string: the
test path must behave identically to the production path at the
code-flow level.

The four mock dispatches (`MOCK_UPLOAD_DISPATCH`,
`MOCK_REMOVE_DISPATCH`, `MOCK_LIST_DISPATCH`,
`MOCK_DOWNLOAD_DISPATCH`) are keyed by `_MOCK_PROVIDERS =
['gemini', 'claude', 'openai', 'mistral']`. They deliberately list
only the four File-API-enabled providers, not every provider
ProxAI supports. Adding a new provider's mock upload means adding
it to `_MOCK_PROVIDERS` as well.

### 2.2 Parallelism — when `ThreadPoolExecutor` kicks in

`_use_parallel(providers)` (`files.py:145-149`) returns `True`
only when **all three** conditions hold:

1. `len(providers) > 1` — multi-provider upload / list / remove.
2. `self.provider_call_options is not None`.
3. `provider_call_options.allow_parallel_file_operations is True`
   (the default, per `types.py:459`).

When ProxDash is also connected, parallelism is forced on *even
for a single provider* because the ProxDash put-to-S3 call runs
alongside the provider upload (`files.py:326`). So a single-provider
upload with ProxDash connected uses a 2-worker pool; a
two-provider upload with ProxDash connected uses a 3-worker pool.

The pool's `max_workers` is exactly the number of parallel tasks
(`len(providers) + (1 if include_proxdash else 0)`), so there is no
overhead for idle threads. Callers disable parallel file ops by
setting `provider_call_options.allow_parallel_file_operations =
False` — useful when running inside an already-threaded test
harness or a constrained environment.

Failures are isolated per future: one provider's upload failing
does not cancel the others. The errors dict returned by
`_execute_uploads` collects every provider that raised, and
`upload` then raises a single `FileUploadError(errors=dict,
media=media)` at the end. The media object still has successful
uploads registered in `provider_file_api_ids` / `_status`, so a
partial-failure recovery path can inspect which providers
succeeded.

---

## 3. Upload flow — serial and parallel paths

`FilesManager.upload(media, providers)` is the entry point. The
order (`files.py:406-431`) is:

1. **Validate media type** — `_validate_upload_media`. Raises
   `ValueError` if the content type is not one of
   `IMAGE / DOCUMENT / AUDIO / VIDEO`, or if both `path` and
   `data` are `None`.
2. **Validate providers list.** The list may be empty only when
   ProxDash is connected (and the upload is ProxDash-only).
   Without ProxDash, an empty list raises.
3. **Per-provider validation** — for each provider:
   `_validate_provider_support` (provider ∈
   `UPLOAD_DISPATCH` and API key present) and
   `is_upload_supported` (MIME in both capability tables). A
   failure at step 3 raises before any upload fires — a whole-batch
   abort on invalid configuration.
4. **Resolve file info** — `_resolve_upload_file_info` extracts
   `file_path` / `file_data` / `filename` / `mime_type`. The
   default `filename` is `'file'` if `media.path` is None.
5. **Init upload fields** — `_init_upload_fields` allocates
   `media.provider_file_api_status = {}` and
   `provider_file_api_ids = {}` if either is None. Existing entries
   (from earlier uploads) are preserved.
6. **Execute uploads** — `_execute_uploads` dispatches
   sequentially or through a `ThreadPoolExecutor` per §2.2. Each
   successful provider populates `status[provider] =
   metadata` and `ids[provider] = metadata.file_id`. Each failure
   writes a `FileUploadState.FAILED` placeholder into `status` and
   records the exception in the local `errors` dict.
7. **Sync to ProxDash** — `_sync_provider_metadata_to_proxdash`
   patches the ProxDash file record with the aggregated provider
   metadata dict. Runs unconditionally after `_execute_uploads`,
   but only does work when ProxDash is connected and
   `media.proxdash_file_id is not None` (i.e., a ProxDash upload
   already completed earlier in this call). Failures are swallowed.
8. **Raise on partial failure.** If any provider errored, raise
   `FileUploadError(errors=..., media=media)`.

```python
# files_manager.upload(media, ['gemini', 'claude'])
# on a client with ProxDash connected:
#
#   parallel pool (3 workers):
#     - upload_to_gemini(...)   → status['gemini'] = FileUploadMetadata(...)
#     - upload_to_claude(...)   → status['claude'] = FileUploadMetadata(...)
#     - proxdash.upload_file(...) → sets media.proxdash_file_id
#   after pool drains:
#     proxdash.update_file(media)   # patch with provider metadata
#     # if errors dict non-empty, raise FileUploadError
```

The mutation-in-place contract is important: callers passing the
same `MessageContent` across multiple `upload` calls see the
successive provider entries accumulate, and downstream code
(`_auto_upload_media`, per-provider executors) reads from the
populated dicts without needing any additional lookups.

### 3.1 Gemini processing state

Gemini is the only provider whose File API has an asynchronous
processing phase. `upload_to_gemini` (`file_helpers.py:68-111`)
polls `client.files.get(name=...)` every `poll_interval` seconds
(default 2.0) until `state == ACTIVE`, stopping at
`max_poll_seconds` (default 300.0). The returned
`FileUploadMetadata.state` reflects reality:

- `ACTIVE` — file is ready for generate references.
- `PENDING` — still `PROCESSING` after `max_poll_seconds` elapsed.
- `FAILED` — the provider reported a non-ACTIVE, non-PROCESSING state.

`_auto_upload_media` does not re-check state after upload — if the
upload returned `PENDING`, the generate call will fire against an
unready file and the provider will return an error through
`_safe_provider_query`. Callers who need stricter semantics should
use `px.files.upload(...)` explicitly with a retry loop rather
than relying on auto-upload.

Claude / OpenAI / Mistral return `ACTIVE` immediately in their
SDKs; no polling is needed for those three.

---

## 4. The `_auto_upload_media` hook in `generate`

`ProviderConnector._auto_upload_media` (`provider_connector.py:752-823`)
runs once per `generate()` call, between `QueryRecord` construction
and `_prepare_execution` (line 883). It is the bridge between the
caller's "I just passed a local path in my chat" and the executor's
"I have a `file_id` to reference."

```python
# provider_connector.py:872-884
query_record = types.QueryRecord(
    prompt=prompt,
    chat=messages,
    system_prompt=system_prompt,
    provider_model=provider_model,
    parameters=parameters,
    tools=tools,
    output_format=output_format,
    connection_options=connection_options,
)
self._auto_upload_media(query_record)        # ← this line
(chosen_executor, chosen_endpoint, modified_query_record) = (
    self._prepare_execution(query_record=query_record, ...)
)
```

### 4.1 What it skips

The iteration guards (`provider_connector.py:782-799`) skip a
`MessageContent` whenever any of the following holds:

| Skip when | Rationale |
|---|---|
| `msg.content` is a string (bare-text message) | No media blocks to inspect |
| `mc` already seen (by `id(mc)`) | Same object referenced twice in the chat — upload once |
| `mc.type not in {IMAGE, DOCUMENT, AUDIO, VIDEO}` | Text / JSON / Pydantic / Tool blocks never upload |
| `provider in mc.provider_file_api_ids` already | Already uploaded for this provider — reuse |
| `mc.path is None and mc.data is None` | Remote-only reference (`source=URL`) — nothing to upload |
| `files_manager_instance.is_upload_supported(mc, provider) is False` | MIME not accepted by this provider's File API or generate endpoint |

The `id(mc)`-based deduplication matters when a caller constructs
a chat that includes the same `MessageContent` object twice —
without it, the second occurrence would trigger a second upload
and break the "one file_id per media, per provider" invariant the
executors rely on.

If `files_manager_instance is None` (no api-key manager available
yet, or a deserialized client without a files manager) the hook is
a no-op. If `query_record.chat is None` (pure-prompt call), the
hook is a no-op. Neither case is an error.

### 4.2 Upload batch

All media that pass the skip filter accumulate into a `pending`
list. If non-empty, the hook runs them through
`files_manager_instance.upload(...)`. Two code paths per the same
`allow_parallel_file_operations` flag §2.2 tracks:

- **Parallel** (default, `>1` pending): one `Future` per media,
  each calling `upload(media=mc, providers=[provider])`. Failures
  are swallowed (`pass` inside the future-result block) so a
  single media failing doesn't abort the generate call — the
  executor will see `provider_file_api_ids[provider]` absent for
  the failed media and fall back to inline-base64 conversion.
- **Serial** (`allow_parallel_file_operations=False` or single
  pending item): calls `upload(...)` directly in the order
  encountered. Same error-swallowing behavior — exceptions bubble
  up to the generate-level handler.

Note the error swallowing in the parallel branch (`except
Exception: pass`). This is intentional: an auto-upload failure is
not a fatal error — the media still has its `path` / `data` /
`source`, and the downstream `_to_*_part` converter will fall back
to inline base64 or URL reference. The caller's generate call
still completes. If you want hard-fail semantics, call
`px.files.upload(...)` explicitly before `px.generate(...)` and
catch `FileUploadError` yourself.

### 4.3 What the executors read

After `_auto_upload_media` returns, every per-provider
`_to_<provider>_part` converter checks `provider_file_api_ids`
first and emits a `file_id` reference when the current provider
appears:

| Connector | Converter | Check |
|---|---|---|
| Claude | `_to_claude_part` | `claude.py:97-102` — emits `{'type': 'document', 'source': {'type': 'file', 'file_id': ...}}` |
| OpenAI (chat.completions) | `_to_chat_completions_part` | `openai.py:209-213` — emits `{'type': 'file', 'file': {'file_id': ...}}` |
| OpenAI (responses) | `_to_responses_part` | `openai.py:273-275` — emits `{'type': 'input_file', 'file_id': ...}` |
| Gemini | `_content_dict_to_part` | `gemini.py:89-96` — emits `Part.from_uri(file_uri=status['gemini'].uri, mime_type=...)` (note: needs the URI from `provider_file_api_status`, not just the id) |
| Mistral | `_to_mistral_part` | `mistral.py:146-148` — emits `{'type': 'file', 'file_id': ...}` |

Every other connector (Cohere / DeepSeek / Grok / HuggingFace /
Databricks) does not check `provider_file_api_ids` because those
providers have no File API in our dispatch tables — there is
nothing for auto-upload to populate, and the converter falls
through to inline-base64 / URL / text-extraction paths.

Gemini is the one case that reads from `provider_file_api_status`
rather than `provider_file_api_ids` — it needs the URI (stored
under `status['gemini'].uri`) to call `Part.from_uri`, not the
bare file id. Keep this asymmetry in mind when editing the gemini
path: a populated `provider_file_api_ids['gemini']` with a missing
`status['gemini'].uri` will silently fall through to inline bytes.

---

## 5. ProxDash as a second persistence layer

When ProxDash is connected, every `FilesManager` operation fans
out to ProxDash in addition to the provider targets. ProxDash's
role is a central file directory backed by its own S3 bucket —
orthogonal to the provider File APIs, which are ephemeral
transport stages.

### 5.1 Per-operation ProxDash behavior

```
upload(media, providers)
│   ├── provider uploads run in parallel (or serial)
│   ├── proxdash.upload_file(media) runs alongside                 # uploads bytes to ProxDash S3
│   │   sets media.proxdash_file_id + media.proxdash_file_status
│   └── proxdash.update_file(media) patches with provider metadata  # runs after all uploads
│
list(providers, limit_per_provider)
│   ├── proxdash.list_files(limit) runs alongside provider list calls
│   ├── ProxDash results come first in the returned list
│   ├── _filter_proxdash_by_providers narrows by requested providers
│   └── _build_covered_file_ids — if a provider's file_id is in a
│         ProxDash record's provider_file_api_ids, the provider-side
│         duplicate is skipped
│
download(media, provider, path)
│   ├── if provider is None:
│   │     try proxdash.download_file(proxdash_file_id) first
│   │     if non-None bytes, return — never touches provider
│   └── fall back to provider download_dispatch (mistral-only today)
│
remove(media, providers)
│   ├── provider removes run in parallel (or serial)
│   ├── proxdash.delete_file(proxdash_file_id) runs alongside
│   └── on success, proxdash_file_id / proxdash_file_status cleared
```

ProxDash failures are non-fatal — every ProxDash call is wrapped
in a `try/except Exception: pass` so a ProxDash outage does not
break provider operations. The symmetric case (provider fails but
ProxDash succeeds) is represented via the per-provider
`FileUploadState.FAILED` entry in `provider_file_api_status` and
surfaces via `FileUploadError`.

### 5.2 ProxDash file IDs vs. provider file IDs

`MessageContent` carries two distinct identifier slots:

| Field | Scope | Lifetime |
|---|---|---|
| `proxdash_file_id` | ProxDash's S3 key | persistent (until `remove`) |
| `provider_file_api_ids: dict[provider, str]` | per-provider File API IDs | varies: Gemini expires in 48h; Claude persists; OpenAI configurable |

ProxDash acts as the **canonical copy** — its file id is the
long-term reference. Provider file ids are transport metadata that
rotate with each provider's retention policy. The
`_sync_provider_metadata_to_proxdash` call after every upload
keeps ProxDash's record of provider ids up to date so that, later,
`list()` can dedupe correctly.

---

## 6. MIME set semantics and the `_MIME_TO_CONTENT_TYPE` contract

`FilesManager._MEDIA_TYPES = (IMAGE, DOCUMENT, AUDIO, VIDEO)`
(`files.py:138-143`). These are the only content types that can
be uploaded; text / thinking / json / pydantic / tool are rejected
by `_validate_upload_media` with `ValueError`.

The capability tables in `file_helpers.py` use concrete MIME
strings (`'image/png'`, `'application/pdf'`, etc.) rather than the
higher-level `ContentType` enum. This is deliberate: a provider
may accept PDF uploads but reject DOCX, or accept JPEG images but
reject GIF — the enum-level grain is too coarse to express that.
When adding a provider, populate the MIME sets to match the
provider's real policy at both upload and reference time; do not
fall back to "all IMAGE types accepted" shortcuts.

The `MessageContent.__post_init__` check
(`message_content.py:374-375`) already validates that
`media_type` is in `SUPPORTED_MEDIA_TYPES` (the global frozenset
at `message_content.py:11-40`), so every MIME reaching
`is_upload_supported` is a known supported type. The capability
tables only *narrow* per provider — they do not introduce new
MIMEs.

---

## 7. Errors raised by the files subsystem

All synchronous; none are routed through `suppress_provider_errors`
(that flag only affects the `generate` path).

| Where | Trigger | Error |
|---|---|---|
| `_validate_upload_media` | `media.type` not `IMAGE / DOCUMENT / AUDIO / VIDEO` | `ValueError("upload() requires a media content type ...")` |
| `_validate_upload_media` | `path` and `data` both `None` | `ValueError("MessageContent must have 'path' or 'data' set ...")` |
| `_validate_provider_support` | provider not in dispatch | `ValueError("Provider '<p>' does not support the File API. Supported: [...]")` |
| `_validate_provider_support` | API key absent | `ValueError("No API key configured for provider '<p>'.")` |
| `upload()` top-level | `providers=[]` and ProxDash not connected | `ValueError("No providers specified and ProxDash is not connected ...")` |
| `upload()` top-level | MIME not in both capability tables | `ValueError("Media type '<m>' cannot be uploaded and referenced by file_id on provider '<p>'.")` |
| `upload()` (after dispatch) | any provider future raised | `FileUploadError(errors={provider: Exception}, media=media)` |
| `download()` | provider given but file_id absent | `ValueError("No file_id found for provider '<p>' on this media content.")` |
| `download()` | media has no uploaded providers and no ProxDash | `ValueError("No uploaded providers found on this media content.")` |
| `download()` | resolved provider not in `DOWNLOAD_DISPATCH` | `ValueError("Provider '<p>' does not support file download. Supported: [...]")` |
| `list()` | `providers=[]` | `ValueError("'providers' must contain at least one provider name ...")` |
| `list()` | `providers=None` and no API keys configured | `ValueError("No providers with API keys found for file listing.")` |
| `remove()` | `providers=[]` | `ValueError("'providers' must contain at least one provider name ...")` |
| `remove()` | media has no uploaded providers and no ProxDash id | `ValueError("No uploaded providers found on this media content.")` |
| `remove()` (after dispatch) | any provider future raised | `FileRemoveError(errors={provider: Exception}, media=media)` |

`FileUploadError` and `FileRemoveError` (defined at `files.py:17-38`)
both carry the full `errors` dict (provider-name → exception) and
the `media` object mid-mutation. Callers catching them can inspect
which providers succeeded (by reading the mutated
`provider_file_api_ids`) and which failed (by inspecting
`error.errors`). Tests in `tests/connectors/` use this pattern to
assert partial-success behavior.

---

## 8. Concerns the files subsystem does not own

Several things look like they should be in `FilesManager` but
aren't — keep the layering clean:

- **Inline base64 encoding on the executor side.** When a media
  block has no `provider_file_api_ids[provider]`, the per-provider
  `_to_*_part` converter does its own base64 + data-URI
  construction (`openai.py:179-180`, `claude.py:106-107`,
  `mistral.py`, etc.). `FilesManager` is not involved in the
  fallback path.
- **Content-type support resolution.** The
  `feature_config.input_format.<type>` levels
  (SUPPORTED / BEST_EFFORT / NOT_SUPPORTED) are resolved in
  `FeatureAdapter._adapt_input_format` — see
  `feature_adapters_logic.md` §2.5. `FilesManager` operates at the
  MIME level, not at the input-format-support level.
- **Cache-key construction for file content.** The query cache
  hashes `path`-based content via `os.stat()` metadata (mtime_ns,
  size) and hashes `data`-based content via the base64-encoded
  bytes in `to_dict()`. This happens in
  `serializers/hash_serializer.py`, not in `FilesManager`. See
  `cache_internals.md`. A notable consequence: in-place file
  edits invalidate the cache; replacing a file with identical
  mtime + size does not (documented on `MessageContent.path` in
  `message_content.py:248-253`).
- **PDF text extraction.** BEST_EFFORT document conversion uses
  `content_utils.py` (`read_text_document`, `read_pdf_document`)
  and is invoked by the per-provider converters, not by
  `FilesManager`. Providers without a File API still send PDFs —
  as text extractions — through the normal inline path.
- **Large-file streaming.** ProxAI today reads every file fully
  into memory before upload. The Gemini SDK accepts a path and
  streams internally, but `upload_to_gemini` still wraps bytes in
  `io.BytesIO` when `file_data` is set. Streaming for `file_data`
  uploads would live in the per-provider helpers, not in
  `FilesManager`.

If you find `FilesManager` doing one of these things, it's a
layering bug.

---

## 9. Where to read next

- `../user_agents/api_guidelines/px_files_api.md` — the caller's
  view of the same surface. Covers the `px.files.upload /
  download / list / remove / is_upload_supported /
  is_download_supported` façade with per-method argument
  semantics and the full error list for callers.
- `adding_a_new_provider.md` §6 and §9 — executor-side
  assumptions about `provider_file_api_ids` on entry to the
  executor (auto-upload has already run) and the `_to_*_part`
  convention for falling back to inline when auto-upload didn't
  populate anything.
- `state_controller.md` — `FilesManager` uses
  `set_state_controlled_property_value` for its nested
  `proxdash_connection` and `api_key_manager` fields, and
  `FilesManagerState` is a member of `ProxAIClientState` at
  `types.py:918`.
- `cache_internals.md` §5 — how `path`- and `data`-based content
  hashes into the query cache key (mtime_ns + size for `path`;
  full bytes for `data`) and why the excluded-fields list
  includes `provider_file_api_ids` / `provider_file_api_status` /
  `filename`, so an auto-upload after hashing does not
  invalidate the cache.
- `tests/connectors/test_files.py` plus the per-provider executor
  tests — the executable spec for the dispatch tables, parallel /
  serial paths, and partial-failure semantics. When the contract
  is ambiguous from reading the source, resolve it by finding the
  corresponding test.
