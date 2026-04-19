# Analysis: Multi-Modal & Large File Handling

## 1. Architecture Overview

ProxAI's multi-modal pipeline follows the same layered pattern as
text generation:

```
MessageContent (user-facing type)
  → Chat.export() serializes to dicts
    → FeatureAdapter checks support levels per content type
      → Provider connector converts to native API format
        → _safe_provider_query() executes and catches errors
          → ResultAdapter normalizes the response
```

### Core types

**`MessageContent`** (`src/proxai/chat/message_content.py:131`) is the
universal content block. It carries a `ContentType` enum (TEXT, THINKING,
IMAGE, DOCUMENT, AUDIO, VIDEO, JSON, PYDANTIC_INSTANCE, TOOL) and one
of three data sources for media:

- `source` — a URL the provider fetches directly
- `data` — raw bytes (base64-encoded on the wire)
- `path` — local file path, read at call time

Validation in `__post_init__` enforces that media types have at least
one of source/data/path, and that `media_type` is in
`SUPPORTED_MEDIA_TYPES`.

**`FeatureAdapter`** (`src/proxai/connectors/feature_adapter.py`)
introspects content types in chat messages and maps them to
`InputFormatType` via `_CONTENT_TYPE_TO_INPUT_FORMAT_TYPE` (line 222).
Each endpoint declares support per input format in
`InputFormatConfigType`. The adapter enforces:

- **SUPPORTED** → pass through unchanged
- **BEST_EFFORT** → convert to text (JSON/Pydantic) or passthrough
  for connectors to handle (documents, via
  `_BEST_EFFORT_PASSTHROUGH_TYPES`)
- **NOT_SUPPORTED** → raise `ValueError`

### Provider content converters

Each connector has a static method that converts `MessageContent` dicts
to the provider's native format:

| Provider | Converter | Location |
|----------|-----------|----------|
| Claude | `_to_claude_part()` | `claude.py:81` |
| OpenAI | `_to_chat_completions_part()` | `openai.py:185` |
| OpenAI | `_to_responses_part()` | `openai.py:248` |
| Gemini | `_content_dict_to_part()` | `gemini.py:85` |
| Mistral | `_to_mistral_part()` | `mistral.py:133` |
| Cohere | `_to_cohere_part()` | `cohere.py:102` |
| DeepSeek | `_to_deepseek_part()` | `deepseek.py:82` |
| Grok | `_to_grok_content()` | `grok.py:112` |
| HuggingFace | `_to_huggingface_part()` | `huggingface.py:93` |
| Databricks | `_to_databricks_part()` | `databricks.py` |

### Document handling strategies by provider

| Provider | Text docs (md/csv/txt) | PDF | DOCX/XLSX |
|----------|------------------------|-----|-----------|
| Claude | Text extraction → text block | Native document block (base64 or URL) | Dropped |
| OpenAI (chat.completions) | Text extraction → text block | Native file block (data URI) | Dropped |
| OpenAI (responses.create) | Native input_file block | Native input_file block | Native input_file block |
| Gemini | `Part.from_bytes()` with MIME | `Part.from_bytes()` with MIME | `Part.from_bytes()` with MIME |
| Mistral | Native `document_url` block | Native `document_url` block | Native `document_url` block |
| Cohere | Text extraction → text block | pypdf text extraction → text block | Dropped |
| DeepSeek | Text extraction → text block | pypdf text extraction → text block | Dropped |
| Grok | Text extraction → text block | pypdf text extraction → text block | Dropped |
| HuggingFace | Text extraction → text block | pypdf text extraction → text block | Dropped |
| Databricks | Text extraction → text block | pypdf text extraction → text block | Dropped |

Text extraction is handled by shared utilities in
`src/proxai/connectors/content_utils.py`:

- `read_text_document()` (line 31): reads md/csv/txt, prepends header
- `read_pdf_document()` (line 53): uses pypdf for text extraction

### Cache invalidation for file-based content

`src/proxai/serializers/hash_serializer.py` hashes `path`-based content
using `os.stat()` metadata (`st_mtime_ns` and `st_size`) rather than
file bytes. This is efficient — it avoids reading large files for hash
computation. An in-place edit invalidates the cache; replacing a file
with identical size and mtime will not (documented in MessageContent
docstring, lines 143–147).

For `data`-based content, the full bytes are base64-encoded and
JSON-serialized before hashing (via `content_item.to_dict()`), which is
correct but memory-intensive for large blobs.

---

## 2. Large File Concerns

### 2.1 All files are loaded fully into memory

Every provider connector reads entire files with `f.read()`, then
base64-encodes them (1.33× size expansion), then embeds inline in the
JSON payload. This happens even when the user provides a local `path`
— the file is deferred until call time but still fully loaded:

- `claude.py:106–107` — `f.read()` → base64 for images
- `claude.py:124–125` — `f.read()` → base64 for PDFs
- `openai.py:179–180` — `_build_data_uri()` → full read + base64
- `gemini.py:110–111` — `f.read()` → `Part.from_bytes()`
- `content_utils.py:45–46` — `f.read()` for text documents

A 50 MB PDF becomes ~67 MB as a base64 string held in a Python string,
embedded in a JSON request body. Under concurrency this compounds.

### 2.2 No file size validation or pre-flight checks

There is no size check anywhere in the pipeline. `MessageContent`
validates types and MIME types but not file size.
`FeatureAdapter` checks support levels but not size constraints.
Provider connectors read and encode blindly.

A user sending a 200 MB scanned PDF will:

1. Read it fully into memory
2. Base64-encode it (~267 MB string)
3. Serialize into the JSON request body
4. Send to the provider → receive a 413 or provider-specific size error
5. Error caught by `_safe_provider_query` but memory already allocated

### 2.3 Provider-specific size limits (see §3 for full breakdown)

Each provider has different constraints for inline content. ProxAI
currently sends everything inline, but several providers offer File
APIs that handle much larger payloads:

| Provider | Inline limit | File API limit | File API used? |
|----------|-------------|----------------|----------------|
| Claude | 32 MB request payload | 500 MB/file (beta) | **No** |
| OpenAI | 50 MB/request | 512 MB/file | **No** |
| Gemini | **20 MB request** | **2 GB/file** | **No** |
| Mistral | Context window bound | 512 MB/file | **No** |
| Cohere | Context window bound | N/A | N/A |
| DeepSeek | ~20 MB (OpenAI-compat) | N/A | N/A |
| Grok | ~20 MB (OpenAI-compat) | Mentioned, not impl. | **No** |
| HuggingFace | ~20 MB (OpenAI-compat) | N/A | N/A |
| Databricks | ~20 MB (OpenAI-compat) | N/A | N/A |

### 2.4 PDF text extraction doubles memory pressure

In `content_utils.py:65–66` (the BEST_EFFORT path for providers without
native PDF support):

```python
pdf_bytes = base64.b64decode(part_dict['data'])  # full decode
reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))  # another copy
```

A 50 MB PDF creates ~100 MB memory pressure. The `path` codepath
(line 68) is better — pypdf reads directly from file.

### 2.5 No error classification for size-related failures

`_safe_provider_query` (`provider_connector.py:540`) catches all
exceptions identically. A 413 (payload too large) or provider-specific
"file too large" error is treated the same as a transient network error.
If retry logic is added later, these should be classified as
non-retryable.

### 2.6 Cache stores full binary response blobs

Multi-modal output responses (generated images, audio, video) are cached
via the query cache. `MessageContent.to_dict()` base64-encodes the
`data` field and writes it to JSON shard files. A generated image could
be several MB, potentially bloating cache files.

---

## 3. Provider File Delivery Analysis

This section covers every delivery method each provider supports for
sending file content to their API, including methods ProxAI does not
yet use. This information is useful when evaluating which File APIs
to integrate.

### 3.1 Anthropic Claude

**SDK:** `anthropic` Python SDK

#### Delivery methods

| Method | Supported | ProxAI uses? | How it works |
|--------|-----------|--------------|--------------|
| Inline base64 | Yes | **Yes** | Base64 string in `source.data` field |
| URL reference | Yes | **Yes** | URL in `source.url` field, provider fetches |
| Files API (upload → file_id) | Yes (beta) | **No** | Upload via `POST /v1/files`, reference by `file_id` |
| Local file path | No (API-level) | Converted to base64 | SDK has no path parameter |

#### Files API details

- **Status:** Beta — requires header `anthropic-beta: files-api-2025-04-14`
- **Upload endpoint:** `POST /v1/files`
- **Per-file limit:** 500 MB
- **Organization storage:** 500 GB total
- **Retention:** Files persist until explicitly deleted (no auto-expiry)
- **Reference format:** `{"type": "document", "source": {"type": "file", "file_id": "file-abc123"}}`
- **SDK methods:** `client.files.create()`, `client.files.retrieve()`,
  `client.files.delete()`, `client.files.download()`

#### Size limits

| Constraint | Limit |
|------------|-------|
| Inline request payload (standard API) | 32 MB |
| Inline request payload (Vertex AI) | 30 MB |
| Inline request payload (Amazon Bedrock) | 20 MB |
| Files API per-file upload | 500 MB |
| Files API org-wide storage | 500 GB |
| Max images per request | 600 (100 for 200K-ctx models) |
| Max PDF pages per request | 600 (100 for 200K-ctx models) |

#### Supported formats

| Category | Formats |
|----------|---------|
| Images | JPEG, PNG, GIF, WebP |
| Documents | PDF (native vision-based, 1500–3000 tokens/page) |
| Text files | Not natively supported as document blocks — must be sent as text content |
| Audio | Not supported |
| DOCX/XLSX | Not supported — must be converted to text externally |

#### Model-specific notes

- **Opus 4.7:** max image resolution 2576px long edge
- **All other models:** max image resolution 1568px long edge
- **PDF support:** requires Claude 3.5+ (Sonnet 3.5, Haiku 3.5, Opus 4+)
- Image token cost: `width × height / 750`

---

### 3.2 OpenAI

**SDK:** `openai` Python SDK

#### Delivery methods

| Method | Supported | ProxAI uses? | How it works |
|--------|-----------|--------------|--------------|
| Inline base64 (data URI) | Yes | **Yes** | `data:<mime>;base64,...` in content block |
| URL reference | Responses API only | **No** | `file_url` in `input_file` block |
| Files API (upload → file_id) | Yes | **No** | Upload via `POST /v1/files`, reference by `file_id` |
| Local file path | No (API-level) | Converted to base64 | SDK has no path parameter |

#### Files API details

- **Status:** GA (stable)
- **Upload endpoint:** `POST /v1/files`
- **Per-file limit:** 512 MB
- **Project-wide storage:** 2.5 TB
- **Retention:** configurable 1 hour–30 days; batch files default 30 days;
  others persist until deleted
- **Purpose parameter:** use `"user_data"` for model input content
- **Reference in chat.completions:**
  `{"type": "file", "file": {"file_id": "file-abc123"}}`
- **Reference in responses.create:**
  `{"type": "input_file", "file_id": "file-abc123"}`

#### Size limits

| Constraint | Limit |
|------------|-------|
| Per-file inline (base64) | 50 MB |
| Combined per-request (all files) | 50 MB |
| Files API upload per file | 512 MB |
| Project-wide storage | 2.5 TB |
| Max images per request | 500 |

#### Supported formats

| Category | Formats |
|----------|---------|
| Images | PNG, JPEG, WebP, non-animated GIF |
| Documents (chat.completions) | PDF (+ expanded types as of 2026) |
| Documents (responses.create) | PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, CSV, TXT, MD, JSON, HTML, XML, 80+ code formats |
| Audio | Separate `input_audio` content type (WAV, MP3); not through file system |
| Spreadsheets | Parsed up to 1000 rows/sheet with auto-generated headers and summary |

#### Endpoint differences

**chat.completions.create:**
- Inline base64 and file_id references supported
- No URL-based file input
- PDF natively supported; other document types added in 2026

**responses.create (recommended by OpenAI for new projects):**
- All three methods: inline base64, file_id, and file URL
- Broadest format support (docx, pptx, xlsx, csv, etc.)
- Native `input_file` content block type

#### Model-specific notes

- PDF parsing requires vision-capable models (GPT-4o, GPT-4.1+)
- Non-PDF file types extract text only, no vision needed
- Image token costs: low detail = 85 tokens fixed; high detail =
  85 + 170 per 512×512 tile
- GPT-4.1-mini: image tokens × 1.62; GPT-4.1-nano: × 2.46
- PDF: ~1536 tokens/page (rendered as image + extracted text)

---

### 3.3 Google Gemini

**SDK:** `google-genai` Python SDK

#### Delivery methods

| Method | Supported | ProxAI uses? | How it works |
|--------|-----------|--------------|--------------|
| Inline bytes | Yes | **Yes** | `Part.from_bytes(data=bytes, mime_type=...)` |
| URL/URI reference | Yes | **Yes** (for `source`) | `Part.from_uri(file_uri=..., mime_type=...)` |
| Files API (upload → URI) | Yes | **No** | `client.files.upload(file=path)` → `Part.from_uri(file.uri)` |
| Local file path (via Files API) | Yes | **No** | Files API accepts `str | PathLike | IOBase` directly |

#### Files API details

- **Status:** GA (stable)
- **Upload method:** `client.files.upload(file=path, config={'mime_type': ...})`
- **Per-file limit:** 2 GB
- **Project storage:** 20 GB total
- **Retention:** Files auto-expire after **48 hours** (no extension)
- **Processing:** must poll `client.files.get(name=file.name)` until
  `state == 'ACTIVE'` before using
- **Key advantage:** SDK accepts local file path directly — handles
  streaming/chunking internally, avoids loading full file into
  Python process memory

#### Size limits

| Constraint | Limit |
|------------|-------|
| Inline total request payload | **20 MB** (all parts + prompt combined) |
| Files API per-file upload | 2 GB |
| Files API project storage | 20 GB |
| Max images per request | 3600 |
| PDF pages | 1000 pages or 50 MB |

#### Supported formats

| Category | Formats |
|----------|---------|
| Images | PNG, JPEG, WebP, HEIC, HEIF (no GIF) |
| Documents | PDF (native vision-based, up to 1000 pages) |
| Audio | WAV, MP3, AIFF, AAC, OGG, FLAC |
| Video | MP4, MPEG, MOV, AVI, FLV |
| Text | Plain text, Markdown, HTML, XML, CSV — passed as text |

#### Model-specific notes

- File size and format limits are **uniform across Gemini models**
  (2.5 Pro, 2.5 Flash, etc.)
- Differences between models are in context window, rate limits,
  and pricing — not file handling
- All models share the same 20 MB inline / 2 GB Files API limits

#### Current gap

The 20 MB inline limit is the strictest among major providers. Any
document over ~15 MB sent to Gemini will fail. The Files API path
(`client.files.upload(file=local_path)`) is the intended solution
and the SDK handles streaming internally.

---

### 3.4 Mistral

**SDK:** `mistralai` Python SDK

#### Delivery methods

| Method | Supported | ProxAI uses? | How it works |
|--------|-----------|--------------|--------------|
| Inline base64 (data URI) | Yes | **Yes** | `data:<mime>;base64,...` via `DocumentURLChunk` |
| URL reference | Yes | **Yes** (for `source`) | Direct URL via `DocumentURLChunk` |
| Files API (upload → file_id) | Yes | **No** | Upload via `client.files.upload()`, ref via `FileChunk` |
| Local file path | No (API-level) | Converted to base64 | SDK has no path parameter for chat |

#### Files API details

- **Status:** GA
- **Upload method:** `client.files.upload(file=...)`
- **Per-file limit:** 512 MB
- **Methods:** upload, list, retrieve, download, delete, get_signed_url
- **Visibility:** workspace (default) or user-level
- **Optional expiry:** configurable
- **Reference format:** `{"type": "file", "file_id": "file-abc123"}`
  (FileChunk) — currently used only in OCR endpoint

#### OCR endpoint (`/v1/ocr`)

Mistral has a dedicated OCR endpoint for document processing:
- Accepts `DocumentURLChunk`, `FileChunk`, or `ImageURLChunk`
- Max file size: 50 MB
- Max pages: 1000
- Output: structured Markdown per page with optional image extraction
- Cost: ~1000 pages per dollar

#### Size limits

| Constraint | Limit |
|------------|-------|
| Inline document (data URI) | Context window bound (~128K tokens) |
| Per-image | 20 MB |
| Document page limit (chat) | 64 pages default (configurable) |
| Document image limit (chat) | 8 images default (configurable) |
| Files API upload | 512 MB |
| OCR endpoint | 50 MB / 1000 pages |

#### Supported formats

| Category | Formats |
|----------|---------|
| Images | JPEG, PNG, WebP, GIF |
| Documents (chat) | PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, CSV, TXT, MD, RST, LaTeX, JSON, JSONL, XML, YAML, ODT, EPUB, RTF, EML, MSG, code files |
| Audio | Not supported |

#### Model-specific notes

- Document understanding supported by all Mistral models (not just
  Pixtral)
- Chat document processing uses vision-based page rendering internally

---

### 3.5 Cohere

**SDK:** `cohere` Python SDK (V2 client)

#### Delivery methods

| Method | Supported | ProxAI uses? | How it works |
|--------|-----------|--------------|--------------|
| Inline text (documents param) | Yes | **No** | `documents=[{"data": {"text": "..."}}]` in chat |
| Inline base64 (images) | Yes | **No** | Data URI in `image_url` content block |
| URL reference (images) | Yes | **No** | HTTP URL in `image_url` content block |
| Files API | **No** | N/A | Datasets API exists but not for chat |
| Local file path | No | Converted to text | Text extraction in connector |

#### Documents parameter

Cohere's chat API has a native `documents` parameter for RAG:
- Type: `list[str | Document]`
- `Document` has `data` (dict of key-value text pairs) and optional `id`
- Enables automatic **citation generation** in responses
- **Text-only** — no binary file content
- Recommended: under 300 words per document chunk
- Not compatible with `json_schema` structured output mode

**Note:** ProxAI's Cohere connector does not use the native `documents`
parameter — it embeds document text directly in message content.

#### Size limits

| Constraint | Limit |
|------------|-------|
| Documents | 128K token context window (shared input + output) |
| Per document | ~300 words recommended per chunk |
| Images (Command A Vision) | 20 images OR 20 MB total per request |
| File upload | N/A (no file API for chat) |

#### Supported formats

| Category | Formats |
|----------|---------|
| Images | PNG, JPEG, WebP, non-animated GIF (Command A Vision) |
| Documents | Text-only via `documents` param — no binary support |
| Audio | Not supported |

#### Model-specific notes

- Image support requires Command A Vision (July 2025+)
- Image token costs: 256 tokens/image (low detail); high detail
  divided into 512×512 tiles at 256 tokens each

---

### 3.6 DeepSeek

**SDK:** `openai` Python SDK with `base_url='https://api.deepseek.com'`

#### Delivery methods

| Method | Supported | ProxAI uses? | How it works |
|--------|-----------|--------------|--------------|
| Inline text (extracted) | Yes | **Yes** | Documents converted to text blocks |
| Inline base64 | No | N/A | DeepSeek chat has no native document/image block |
| URL reference | No | N/A | Not supported in chat endpoint |
| Files API | **No** | N/A | DeepSeek has no file upload API |
| Local file path | No (API-level) | Text extracted | pypdf / text read |

#### Size limits

| Constraint | Limit |
|------------|-------|
| Request payload | ~20 MB (OpenAI-compatible inherited) |
| Context window | 128K tokens |
| File upload | N/A |

#### Supported formats

| Category | Formats |
|----------|---------|
| Images | Not supported |
| Documents | Text-based only (md/csv/txt via extraction, PDF via pypdf) |
| Audio | Not supported |

#### Notes

DeepSeek's chat.completions endpoint is OpenAI-compatible but only
supports text content. All documents are extracted to plain text
by the ProxAI connector. Binary formats (DOCX, XLSX) and image-heavy
PDFs are silently dropped.

---

### 3.7 Grok (xAI)

**SDK:** `xai-sdk` (custom protobuf-based SDK)

#### Delivery methods

| Method | Supported | ProxAI uses? | How it works |
|--------|-----------|--------------|--------------|
| Inline base64 (images) | Yes | **Yes** | Data URI via `image()` protobuf helper |
| URL reference (images) | Yes | **Yes** | Direct URL via `image()` helper |
| Inline text (documents) | Yes | **Yes** | Text extraction → `text()` helper |
| File upload (file_id) | Mentioned in SDK | **No** | Comment in code: "SDK only supports file references by pre-uploaded file_id" |
| Local file path | No (API-level) | Converted to base64/text | Read + encode in connector |

#### Size limits

| Constraint | Limit |
|------------|-------|
| Inline | ~20 MB (estimated, OpenAI-compatible behavior) |
| File API | Unknown (not implemented) |
| Context window | Model-dependent |

#### Supported formats

| Category | Formats |
|----------|---------|
| Images | PNG, JPEG (via data URI or URL) |
| Documents | Text-based (md/csv/txt via extraction, PDF via pypdf) |
| Binary documents | Requires file_id upload (not implemented) |
| Audio | Not supported |

#### Current gap

The xAI SDK comments indicate a file upload mechanism exists for
binary document formats (DOCX, XLSX), but the ProxAI connector does
not implement the upload step. Binary documents are silently dropped.

---

### 3.8 HuggingFace

**SDK:** `huggingface_hub` (`InferenceClient` with OpenAI-compatible router)

#### Delivery methods

| Method | Supported | ProxAI uses? | How it works |
|--------|-----------|--------------|--------------|
| Inline base64 (images) | Yes | **Yes** | Data URI in `image_url` block |
| URL reference (images) | Yes | **Yes** | Direct URL in `image_url` block |
| Inline text (documents) | Yes | **Yes** | Text extraction → text block |
| Files API | **No** | N/A | HuggingFace router has no file API |
| Local file path | No (API-level) | Converted to base64/text | Read + encode in connector |

#### Size limits

| Constraint | Limit |
|------------|-------|
| Inline | ~20 MB (OpenAI-compatible router inherited) |
| File upload | N/A |
| Context window | Varies by routed model |

#### Supported formats

| Category | Formats |
|----------|---------|
| Images | Via routed model capabilities |
| Documents | Text-based only (md/csv/txt via extraction, PDF via pypdf) |
| Audio | Not supported |

#### Notes

HuggingFace acts as a router to various underlying models. Capabilities
depend on the specific model being routed to. The ProxAI connector
treats it as an OpenAI-compatible endpoint with document extraction
fallbacks.

---

### 3.9 Databricks

**SDK:** `databricks-sdk[openai]` (wraps OpenAI client via
`WorkspaceClient.serving_endpoints.get_open_ai_client()`)

#### Delivery methods

| Method | Supported | ProxAI uses? | How it works |
|--------|-----------|--------------|--------------|
| Inline base64 (images) | Yes | **Yes** | Data URI in `image_url` block |
| URL reference (images) | Yes | **Yes** | Direct URL in `image_url` block |
| Inline text (documents) | Yes | **Yes** | Text extraction → text block |
| Files API | **No** | N/A | No file upload for serving endpoints |
| Local file path | No (API-level) | Converted to base64/text | Read + encode in connector |

#### Size limits

| Constraint | Limit |
|------------|-------|
| Inline | ~20 MB (OpenAI-compatible inherited) |
| File upload | N/A |

#### Supported formats

| Category | Formats |
|----------|---------|
| Images | JPEG, PNG, WebP, GIF (OpenAI-compatible) |
| Documents | Text-based only (md/csv/txt via extraction, PDF via pypdf) |
| Audio | Not supported |

#### Notes

Databricks serves models via OpenAI-compatible endpoints but lacks the
native file block support that OpenAI itself has. All documents are
extracted to text. No File API equivalent exists for serving endpoints.

---

## 4. Cross-Provider Comparison Matrix

### File delivery methods

| Provider | Inline base64 | URL ref | File API | Path (SDK) | ProxAI uses File API? |
|----------|:---:|:---:|:---:|:---:|:---:|
| Claude | Yes | Yes | Yes (beta, 500 MB) | No | **No** |
| OpenAI | Yes | Responses only | Yes (GA, 512 MB) | No | **No** |
| Gemini | Yes | Yes | Yes (GA, 2 GB) | **Yes** (SDK accepts path) | **No** |
| Mistral | Yes | Yes | Yes (GA, 512 MB) | No | **No** |
| Cohere | Images only | Images only | No | No | N/A |
| DeepSeek | No | No | No | No | N/A |
| Grok | Images | Images | Mentioned | No | **No** |
| HuggingFace | Images | Images | No | No | N/A |
| Databricks | Images | Images | No | No | N/A |

### Inline size limits

| Provider | Inline limit | Notes |
|----------|-------------|-------|
| Gemini | **20 MB** | Strictest — total request including prompt |
| Databricks | ~20 MB | OpenAI-compatible inherited |
| DeepSeek | ~20 MB | OpenAI-compatible inherited |
| Grok | ~20 MB | Estimated |
| HuggingFace | ~20 MB | Router inherited |
| Claude (Bedrock) | 20 MB | Bedrock variant |
| Claude (Vertex) | 30 MB | Vertex variant |
| Claude (standard) | 32 MB | Standard API |
| OpenAI | 50 MB | Per-file and per-request |
| Mistral | Context bound | No hard byte limit documented |
| Cohere | 20 MB images | Context window for text documents |

### File API comparison (providers with File APIs)

| | Claude | OpenAI | Gemini | Mistral |
|---|---|---|---|---|
| Status | Beta | GA | GA | GA |
| Per-file limit | 500 MB | 512 MB | 2 GB | 512 MB |
| Total storage | 500 GB (org) | 2.5 TB (project) | 20 GB (project) | Not documented |
| Retention | Until deleted | 1h–30d configurable | **48h auto-expiry** | Configurable |
| SDK accepts path | No | No | **Yes** | No |
| Reference style | `file_id` | `file_id` | URI | `file_id` |
| Reuse across requests | Yes | Yes | Yes (within 48h) | Yes |

### Native document format support

| Provider | PDF | DOCX | XLSX | PPTX | CSV | TXT/MD |
|----------|:---:|:----:|:----:|:----:|:---:|:------:|
| Claude | Native | — | — | — | — | Text |
| OpenAI (responses) | Native | Native | Native | Native | Native | Native |
| OpenAI (chat.comp) | Native | 2026+ | 2026+ | — | — | Text |
| Gemini | Native | Native | Native | — | Text | Text |
| Mistral | Native | Native | Native | Native | Native | Native |
| Cohere | — | — | — | — | — | Text |
| DeepSeek | Text* | — | — | — | — | Text |
| Grok | Text* | — | — | — | — | Text |
| HuggingFace | Text* | — | — | — | — | Text |
| Databricks | Text* | — | — | — | — | Text |

*Text = pypdf text extraction, not native vision-based processing.

---

## 5. File API Code Examples

Bare-minimum examples showing the upload → reference → query flow
for each provider that offers a File API. These use the provider SDKs
directly; ProxAI does not currently integrate any of these paths.

### 5.1 Anthropic Claude (beta)

```python
import anthropic

client = anthropic.Anthropic()

# 1. Upload
uploaded = client.beta.files.upload(
    file=open("report.pdf", "rb"),
)
# uploaded.id  → "file-abc123..."

# 2. Use in message
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": uploaded.id,
                },
            },
            {
                "type": "text",
                "text": "Summarize this document.",
            },
        ],
    }],
    betas=["files-api-2025-04-14"],
)
print(response.content[0].text)

# 3. Cleanup (files persist until deleted)
client.beta.files.delete(uploaded.id)
```

**Notes:**
- Requires beta header `files-api-2025-04-14` (passed via `betas=`).
- Files persist until explicitly deleted — no auto-expiry.
- Per-file limit: 500 MB. Org storage: 500 GB.
- Same `file_id` can be reused across multiple requests.

#### File lifecycle management

Claude files have no auto-expiry and no configurable retention at
upload time. Files persist indefinitely until deleted. This makes
reuse straightforward but requires explicit cleanup.

```python
# List all uploaded files
files = client.beta.files.list()
for f in files.data:
    print(f.id, f.filename, f.size_bytes, f.created_at)

# Retrieve metadata for a specific file
meta = client.beta.files.retrieve_metadata(file_id="file-abc123")
print(meta.filename, meta.mime_type, meta.size_bytes)

# Download file content
content = client.beta.files.download(file_id="file-abc123")

# Delete when no longer needed
client.beta.files.delete(file_id="file-abc123")
```

### 5.2 OpenAI

```python
from openai import OpenAI

client = OpenAI()

# 1. Upload
uploaded = client.files.create(
    file=open("report.pdf", "rb"),
    purpose="user_data",
)
# uploaded.id  → "file-abc123..."

# 2a. Use in chat.completions
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {
                    "file_id": uploaded.id,
                },
            },
            {
                "type": "text",
                "text": "Summarize this document.",
            },
        ],
    }],
)
print(response.choices[0].message.content)

# 2b. Use in responses.create (alternative)
response = client.responses.create(
    model="gpt-4.1",
    input=[{
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_file",
                "file_id": uploaded.id,
            },
            {
                "type": "input_text",
                "text": "Summarize this document.",
            },
        ],
    }],
)
print(response.output_text)

# 3. Cleanup
client.files.delete(uploaded.id)
```

**Notes:**
- `purpose="user_data"` is required for model input content.
- Per-file limit: 512 MB. Project storage: 2.5 TB.
- Retention: configurable 1 hour–30 days, or until deleted.
- `responses.create` also supports `file_url` for remote files
  (no upload needed).
- Spreadsheets (xlsx, csv) are auto-parsed up to 1000 rows/sheet.

#### Upload with auto-expiry

OpenAI is the only provider with configurable retention at upload
time. The `expires_after` parameter sets a TTL relative to the
file's creation timestamp.

```python
# Upload with 1-hour expiry (3600 seconds)
uploaded = client.files.create(
    file=open("report.pdf", "rb"),
    purpose="user_data",
    expires_after={
        "anchor": "created_at",
        "seconds": 3600,        # 1 hour
    },
)
print(uploaded.id, uploaded.expires_at)  # unix timestamp

# Upload with 7-day expiry
uploaded_weekly = client.files.create(
    file=open("report.pdf", "rb"),
    purpose="user_data",
    expires_after={
        "anchor": "created_at",
        "seconds": 604800,      # 7 days
    },
)

# Upload without expiry (persists until deleted)
uploaded_permanent = client.files.create(
    file=open("report.pdf", "rb"),
    purpose="user_data",
    # omit expires_after — no auto-expiry
)
```

#### File lifecycle management

```python
# List files (filterable by purpose)
files = client.files.list(purpose="user_data")
for f in files.data:
    print(f.id, f.filename, f.bytes, f.status, f.expires_at)

# Check file status
meta = client.files.retrieve(file_id="file-abc123")
print(meta.status)       # "processed", "error", etc.
print(meta.expires_at)   # None if no expiry set

# Wait for processing (for large files)
processed = client.files.wait_for_processing(
    file_id="file-abc123",
    poll_interval=1.0,    # seconds between checks
    max_wait_seconds=300,
)

# Delete explicitly
client.files.delete(file_id="file-abc123")
```

### 5.3 Google Gemini

```python
from google import genai
from google.genai import types as genai_types
import time

client = genai.Client()

# 1. Upload (SDK accepts local path directly)
uploaded = client.files.upload(
    file="report.pdf",
    config={"mime_type": "application/pdf"},
)

# 2. Wait for processing
while uploaded.state == "PROCESSING":
    time.sleep(2)
    uploaded = client.files.get(name=uploaded.name)

# 3. Use in generate_content
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        genai_types.Content(
            role="user",
            parts=[
                genai_types.Part.from_uri(
                    file_uri=uploaded.uri,
                    mime_type=uploaded.mime_type,
                ),
                genai_types.Part(text="Summarize this document."),
            ],
        )
    ],
)
print(response.text)

# 4. Cleanup (optional — files auto-expire after 48 hours)
client.files.delete(name=uploaded.name)
```

**Notes:**
- `client.files.upload(file=...)` accepts `str | PathLike | IOBase`.
  When given a path, the SDK handles streaming internally — the file
  is **not loaded into Python process memory**.
- Must poll until `state == "ACTIVE"` before using.
- Per-file limit: 2 GB. Project storage: 20 GB.
- Files auto-expire after **48 hours** — track timestamps if reusing.
- This is the only provider where the SDK itself handles large file
  streaming from a local path.

#### File lifecycle and 48-hour expiry

Gemini files have a fixed 48-hour TTL with no way to extend it.
This requires tracking upload timestamps and re-uploading when files
expire.

```python
import time
from google import genai

client = genai.Client()

# Upload with display name for easier management
uploaded = client.files.upload(
    file="report.pdf",
    config={
        "mime_type": "application/pdf",
        "display_name": "Q4 Financial Report",
    },
)

# Check file state and expiry
file_info = client.files.get(name=uploaded.name)
print(file_info.name)             # "files/abc123..."
print(file_info.display_name)     # "Q4 Financial Report"
print(file_info.state)            # PROCESSING → ACTIVE
print(file_info.expiration_time)  # datetime, ~48h from upload
print(file_info.size_bytes)
print(file_info.sha256_hash)

# List all uploaded files
for f in client.files.list():
    print(f.name, f.display_name, f.state, f.expiration_time)

# Re-upload check: if expired or expiring soon, upload again
import datetime
if file_info.expiration_time < datetime.datetime.now(
    datetime.timezone.utc
) + datetime.timedelta(hours=1):
    uploaded = client.files.upload(
        file="report.pdf",
        config={"mime_type": "application/pdf"},
    )

# Delete early (optional — files auto-delete after 48h)
client.files.delete(name=uploaded.name)
```

### 5.4 Mistral

```python
from mistralai import Mistral
from mistralai.models import File as MistralFile

client = Mistral()

# 1. Upload
uploaded = client.files.upload(
    file=MistralFile(
        file_name="report.pdf",
        content=open("report.pdf", "rb"),
    ),
)
# uploaded.id  → "file-abc123..."

# 2. Use in chat (via FileChunk content type)
response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "file",
                "file_id": uploaded.id,
            },
            {
                "type": "text",
                "text": "Summarize this document.",
            },
        ],
    }],
)
print(response.choices[0].message.content)

# 3. Cleanup
client.files.delete(file_id=uploaded.id)
```

**Notes:**
- Per-file limit: 512 MB.
- Mistral also has a dedicated OCR endpoint (`/v1/ocr`) that
  accepts `FileChunk` references for document processing
  (50 MB / 1000 pages, structured Markdown output).
- Supports wide range of document formats natively: PDF, DOCX,
  PPTX, XLSX, CSV, TXT, MD, LaTeX, XML, YAML, EPUB, RTF, and more.

#### File lifecycle management

```python
from mistralai import Mistral
from mistralai.models import File as MistralFile

client = Mistral()

# Upload
uploaded = client.files.upload(
    file=MistralFile(
        file_name="report.pdf",
        content=open("report.pdf", "rb"),
    ),
)

# Retrieve metadata
meta = client.files.retrieve(file_id=uploaded.id)
print(meta.id, meta.filename, meta.size_bytes, meta.mimetype)

# List all files
files = client.files.list()
for f in files.data:
    print(f.id, f.filename, f.purpose, f.created_at)

# Get a signed download URL
signed = client.files.get_signed_url(file_id=uploaded.id)
print(signed.url)  # temporary download URL

# Download file content
content = client.files.download(file_id=uploaded.id)

# Delete
client.files.delete(file_id=uploaded.id)
```

#### Using with OCR endpoint

Mistral's OCR endpoint is a dedicated document processing pipeline
that returns structured Markdown. It can reference previously
uploaded files via `file_id`.

```python
from mistralai import Mistral
from mistralai.models import File as MistralFile

client = Mistral()

# Upload for OCR processing
uploaded = client.files.upload(
    file=MistralFile(
        file_name="scanned_contract.pdf",
        content=open("scanned_contract.pdf", "rb"),
    ),
    purpose="ocr",
)

# Run OCR using file reference
ocr_result = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "file",
        "file_id": uploaded.id,
    },
)
for page in ocr_result.pages:
    print(f"--- Page {page.index} ---")
    print(page.markdown)
```

---

## 6. File API Format Support by MIME Type

This section maps every MIME type declared in ProxAI's
`SUPPORTED_MEDIA_TYPES` (`src/proxai/chat/message_content.py:11`)
against what each provider's File Upload API accepts. This covers
**file upload + reference via file_id/URI only**, not inline base64
content.

Legend: **Yes** = documented as supported, **No** = explicitly
unsupported or requires conversion, **—** = not documented (may
work but no guarantee).

### Images

| MIME type | Extension | Claude | OpenAI | Gemini | Mistral |
|-----------|-----------|:------:|:------:|:------:|:-------:|
| `image/png` | .png | Yes | Yes | Yes | Yes (ocr) |
| `image/jpeg` | .jpg/.jpeg | Yes | Yes | Yes | Yes (ocr) |
| `image/gif` | .gif | Yes | Yes | — | — |
| `image/webp` | .webp | Yes | Yes | Yes | Yes (ocr) |
| `image/heic` | .heic | — | — | Yes | — |
| `image/heif` | .heif | — | — | Yes | — |

**Notes:**
- Gemini is the only provider that documents HEIC/HEIF support
  (Apple's native photo formats).
- OpenAI's GIF support is limited to non-animated GIFs.
- Mistral's image file upload is restricted to `purpose="ocr"`;
  for chat, images are sent inline (URL or base64), not via
  file_id.

### Documents

| MIME type | Extension | Claude | OpenAI | Gemini | Mistral |
|-----------|-----------|:------:|:------:|:------:|:-------:|
| `application/pdf` | .pdf | Yes | Yes | Yes | Yes (ocr) |
| `application/vnd...wordprocessingml.document` | .docx | No | Yes | — | — |
| `application/vnd...spreadsheetml.sheet` | .xlsx | No | Yes | — | — |
| `text/csv` | .csv | No | Yes | Yes | — |
| `text/plain` | .txt | Yes | Yes | Yes | — |
| `text/markdown` | .md | No | Yes | — | — |

**Notes:**
- OpenAI has the broadest document support — all 6 formats work
  via file_id. Spreadsheets (xlsx, csv) are auto-parsed up to
  1000 rows per sheet.
- Claude Files API only accepts PDF and plain text as document
  types. CSV, markdown, DOCX, XLSX must be converted to plain
  text before upload.
- Gemini supports PDF natively (vision-based, up to 1000 pages)
  and plain text / CSV as text content. DOCX and XLSX are not
  documented.
- Mistral Files API only documents PDF and images for
  `purpose="ocr"`. The Le Chat UI accepts more formats, but the
  API is more restricted.

### Audio

| MIME type | Extension | Claude | OpenAI | Gemini | Mistral |
|-----------|-----------|:------:|:------:|:------:|:-------:|
| `audio/mpeg` | .mp3 | — | — | Yes | — |
| `audio/wav` | .wav | — | — | Yes | — |
| `audio/flac` | .flac | — | — | Yes | — |
| `audio/aac` | .aac | — | — | Yes | — |
| `audio/ogg` | .ogg | — | — | Yes | — |
| `audio/aiff` | .aiff | — | — | Yes | — |

**Notes:**
- Gemini is the only provider with audio file upload support.
  All 6 formats work via the Files API + `Part.from_uri()`.
- OpenAI supports audio inline in chat completions via the
  `input_audio` content type (mp3, wav), but **not** via
  file_id reference from the Files API.
- Claude and Mistral have no documented audio support.
- Gemini uses `audio/mp3` internally — the standard MIME type
  `audio/mpeg` should still work.

### Video

| MIME type | Extension | Claude | OpenAI | Gemini | Mistral |
|-----------|-----------|:------:|:------:|:------:|:-------:|
| `video/mp4` | .mp4 | — | — | Yes | — |
| `video/webm` | .webm | — | — | Yes | — |
| `video/quicktime` | .mov | — | — | Yes | — |
| `video/x-msvideo` | .avi | — | — | — | — |
| `video/mpeg` | .mpeg | — | — | Yes | — |
| `video/x-matroska` | .mkv | — | — | — | — |

**Notes:**
- Gemini is the only provider with video file upload support.
- `video/x-msvideo` (.avi): Gemini documents `video/avi` but
  ProxAI uses `video/x-msvideo`. These refer to the same format
  but the MIME string differs — may need mapping.
- `video/x-matroska` (.mkv): not documented by any provider.
- Gemini also supports `video/x-flv`, `video/wmv`, `video/3gpp`
  which are not in ProxAI's `SUPPORTED_MEDIA_TYPES`.

### Summary

| Category | Claude | OpenAI | Gemini | Mistral |
|----------|:------:|:------:|:------:|:-------:|
| Images | 4 of 6 | 4 of 6 | 5 of 6 | 3 of 6 |
| Documents | 2 of 6 | **6 of 6** | 3 of 6 | 1 of 6 |
| Audio | 0 of 6 | 0 of 6 | **6 of 6** | 0 of 6 |
| Video | 0 of 6 | 0 of 6 | **4 of 6** | 0 of 6 |
| **Total** | **6 of 24** | **10 of 24** | **18 of 24** | **4 of 24** |

OpenAI is strongest for documents. Gemini is strongest overall and
the only option for audio/video file uploads. Claude and Mistral
are narrower — primarily PDF and common image formats.

---

## 7. File API Metadata and Identification

No provider supports custom metadata, tags, folders, or paths on
uploaded files. The generated ID is the only unique identifier —
filenames are not unique (uploading the same filename twice produces
two distinct IDs). Any mapping from local files to provider file
IDs must be managed externally.

### Returned metadata per provider

| Field | Claude | OpenAI | Gemini | Mistral |
|-------|:------:|:------:|:------:|:-------:|
| Unique ID | `id` | `id` | `name` | `id` |
| Filename | `filename` | `filename` | — | `filename` |
| Display name | — | — | `display_name` | — |
| Custom name | — | — | `name` | — |
| MIME type | `mime_type` | — | `mime_type` | `mimetype` |
| Size | `size_bytes` | `bytes` | `size_bytes` | `size_bytes` |
| Created at | `created_at` | `created_at` | `create_time` | `created_at` |
| Expiry | — | `expires_at` | `expiration_time` | — |
| Content hash | — | — | `sha256_hash` | `signature` |
| Processing status | — | `status` | `state` | — |
| Downloadable | `downloadable` | — | `download_uri` | — |
| Custom tags/labels | — | — | — | — |

### How filenames are set at upload

- **Claude** — auto-extracted from the file handle name (e.g.,
  `open("report.pdf", "rb")` → `filename="report.pdf"`). Can be
  overridden with the tuple form:
  `file=("custom_name.pdf", open(..., "rb"))`.
- **OpenAI** — same as Claude: auto-extracted from handle, tuple
  form for override:
  `file=("custom_name.pdf", open(..., "rb"))`.
- **Gemini** — no filename field. Uses `display_name` (human label)
  and `name` (resource ID), both set via `config=` at upload time.
- **Mistral** — `file_name` is an explicit required string set
  via `File(file_name="report.pdf", content=...)`.

### Uniqueness guarantees

- **IDs are unique** — each upload generates a new ID regardless
  of content or filename. Uploading the same file twice produces
  two distinct IDs.
- **Filenames are not unique** — multiple files can share the same
  filename. Filenames cannot be used for lookup.
- **Content hashes** — only Gemini (`sha256_hash`) and Mistral
  (`signature`) return content hashes. These can be used to detect
  duplicate uploads, but the providers do not deduplicate
  automatically — each upload consumes storage quota.
