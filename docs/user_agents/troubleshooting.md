# Troubleshooting

Source of truth: the Errors sections of the individual API docs
under `docs/user_agents/api_guidelines/` —
[`px_client_api.md`](./api_guidelines/px_client_api.md) §4,
[`px_generate_api.md`](./api_guidelines/px_generate_api.md),
[`px_models_api.md`](./api_guidelines/px_models_api.md) §6,
[`px_files_api.md`](./api_guidelines/px_files_api.md) §6,
[`px_chat_api.md`](./api_guidelines/px_chat_api.md) §7,
[`cache_behaviors.md`](./api_guidelines/cache_behaviors.md) §10,
[`raw_provider_response.md`](./api_guidelines/raw_provider_response.md) §5.
Each symptom here links back to the authoritative row in one of
those tables. If this document disagrees with those files, the
files win — update this document.

This is the symptom-first lookup for the errors and surprises
callers hit most often. Scan §1 for the shape of your problem,
then jump to the section it points at. Every section has a
concrete symptom, the likely cause, and a runnable fix. Read
this when something blew up and you want to know what happened
before you dig into stack traces.

See also:
[`overview.md`](./overview.md) (the landing page and decision
tree for library use),
[`recipes/best_practices.md`](./recipes/best_practices.md) (the
happy-path patterns whose absence often becomes a failure mode
documented here), and the per-API Errors tables linked above
for the normative list.

---

## 1. Symptom quick-jump

```
Your symptom                                          → §
│
│   # Setup / connection / environment
├── Nothing runs; "no active ProxAIClient" / imports fail    →  §2.1
├── ValueError: No API key configured for provider '<p>'     →  §2.2
├── ValueError: cache_path is required while setting ...     →  §2.3
├── ValueError: ModelProbeOptions.timeout must be >= 1       →  §2.3
│
│   # Cache
├── "I changed the prompt and still got the old response"    →  §3.1
├── Every call reports CACHE_UNAVAILABLE                     →  §3.2
├── override_cache_value=True raises ValueError              →  §3.3
├── keep_raw_provider_response=True raises ValueError        →  §3.4
│
│   # Feature support / endpoint picking
├── "endpoint <name> is not supported"                       →  §4.1
├── "endpoint <name> is not supported in STRICT mode"        →  §4.2
├── "No compatible endpoint found for provider '<p>'"        →  §4.3
├── "Feature '<f>' is not supported by endpoint '<e>'"       →  §4.4
├── "Input format '<t>' is not supported by endpoint '<e>'"  →  §4.5
│
│   # Models / health checks
├── get_working_model raises "API key missing"               →  §5.1
├── list_working_models is slow the first time               →  §5.2
├── ModelStatus shows models as FAILED I know work           →  §5.3
│
│   # Fallback chains
├── "suppress_provider_errors and fallback_models cannot..." →  §6.1
├── "endpoint and fallback_models cannot be used together"   →  §6.2
├── fallback_models tuple / list shape ValueError            →  §6.3
│
│   # Files / multi-modal
├── "Media type '...' cannot be uploaded and referenced..."  →  §7.1
├── FileUploadError with partial successes                   →  §7.2
├── Auto-upload in generate() "succeeded" but the provider
│   still got inline bytes                                   →  §7.3
│
│   # Multi-choice and output format
├── n > 1 silently behaves like n = 1                        →  §8.1
├── PYDANTIC output → pydantic.ValidationError at runtime    →  §8.2
├── JSON output → JSONDecodeError at runtime                 →  §8.3
│
│   # Chat / MessageContent construction
├── TypeError("Expected Message or dict, got ...")           →  §9.1
├── ValueError("Invalid content type: ...") at Chat build    →  §9.2
├── ValueError("Unsupported media_type: ...")                →  §9.3
│
│   # Provider error routing
└── generate() raised; I thought suppress_provider_errors
    would catch it                                           →  §10
```

Every row is a symptom you can grep a stack trace for. If your
error doesn't appear here, check the Errors table in the
corresponding `api_guidelines/<file>.md` first — this doc is a
cross-cut of those tables, not a superset.

---

## 2. Setup and API keys

### 2.1 Nothing runs — `RuntimeError: no active ProxAIClient` or imports fail

**Likely cause.** You called a module-level `px.*` method
(`px.generate_text`, `px.models.list_working_models`, …) without
first calling `px.connect(...)` or constructing a `px.Client(...)`.
ProxAI's module-level façade forwards to a singleton
`_DEFAULT_CLIENT` that only gets created by `px.connect` or the
first `px.generate` fallback path.

**Fix.**

```python
import proxai as px

# Either connect (recommended — centralizes options):
px.connect()                       # minimum: picks up env-var keys
response = px.generate_text('hi')

# Or construct a Client directly if you want a non-singleton:
client = px.Client()
response = client.generate_text('hi')
```

If imports fail (`ImportError: cannot import name 'X' from
partially initialized module 'proxai'`), it's almost always a
bad editable install — reinstall via `poetry install` and confirm
the venv is active. See
[`px_client_api.md`](./api_guidelines/px_client_api.md) §2 for
the full construction surface.

### 2.2 `ValueError: No API key configured for provider '<p>'`

**Likely cause.** You called a method that needs the provider's
API key (any `px.files.*` method, `px.models.list_provider_models(<p>)`,
`px.generate_text(provider_model=(<p>, ...))`) but the env var
for that provider is unset.

**Fix.** Set the env var before the call. The provider → env-var
mapping lives in `PROVIDER_KEY_MAP`
(`src/proxai/connectors/model_configs.py:19`); the top-level
names are:

| Provider | Env var(s) |
|---|---|
| openai | `OPENAI_API_KEY` |
| claude | `ANTHROPIC_API_KEY` |
| gemini | `GEMINI_API_KEY` |
| mistral | `MISTRAL_API_KEY` |
| cohere | `CO_API_KEY` |
| grok | `XAI_API_KEY` |
| deepseek | `DEEPSEEK_API_KEY` |
| huggingface | `HF_TOKEN` |
| databricks | `DATABRICKS_TOKEN`, `DATABRICKS_HOST` |

Providers that don't have a key set are silently excluded from
`px.models.list_providers()` / `list_working_providers()` — those
methods don't raise, they just return a smaller list.
See [`px_models_api.md`](./api_guidelines/px_models_api.md) §4.

### 2.3 `CacheOptions` / `ModelProbeOptions` validation errors at connect

**Likely causes and fixes.**

| Error | Cause | Fix |
|---|---|---|
| `cache_path is required while setting cache_options` | Passed `cache_options=CacheOptions(...)` with `cache_path=None` and `disable_model_cache=False`. | Either set `cache_path=...` or set `disable_model_cache=True`. See [`cache_behaviors.md`](./api_guidelines/cache_behaviors.md) §5. |
| `ModelProbeOptions.timeout must be >= 1.` | Passed `model_probe_options=ModelProbeOptions(timeout=0)` or negative. | Use `timeout >= 1` — values less than 1s mean "abort every probe". |
| `Root logging path does not exist: ...` | `logging_options.logging_path` parent directory can't be created. | Create the parent, or pass a path under `~` that the process can write. |

See [`px_client_api.md`](./api_guidelines/px_client_api.md) §4
for the full `ValueError` table.

---

## 3. Cache surprises

### 3.1 "I changed the prompt and still got the old response"

**Likely cause.** The query cache keys on an exact hash — leading
/ trailing whitespace, punctuation, case changes all count. If
the cached bucket's query still equals yours by structural compare
(`is_query_record_equal`), the cached result wins.

**Fix.** Three options, pick one:

```python
# 1. Skip the cache for this call (no read, no write)
px.generate_text('my prompt', connection_options=px.types.ConnectionOptions(
    skip_cache=True,
))

# 2. Override the cache bucket (still writes the new result,
#    so future calls see the fresh value)
px.generate_text('my prompt', connection_options=px.types.ConnectionOptions(
    override_cache_value=True,
))

# 3. Wipe the whole cache dir on your next connect()
px.connect(cache_options=px.types.CacheOptions(
    cache_path=my_cache_path, clear_query_cache_on_connect=True,
))
```

If you expected the prompt to hash differently but it didn't,
confirm `CallRecord.connection.cache_look_fail_reason` — a
`None` (cache hit) with your stale response means the bucket
really matched. See
[`cache_behaviors.md`](./api_guidelines/cache_behaviors.md) §4
and §6.

### 3.2 Every call reports `CACHE_UNAVAILABLE`

**Likely cause.** Your `cache_options.cache_path` points at a
non-writable directory (permissions, read-only volume, bad
mount). The cache degrades to "always miss" rather than
crashing the call — but every
`CallRecord.connection.cache_look_fail_reason` is
`CACHE_UNAVAILABLE`.

**Fix.** Check the log output for
`QueryCacheManager` warnings. The manager logs a message when
`os.access(cache_path, os.W_OK)` fails at construction. Fix the
permissions or point at a writable directory and reconnect. See
[`cache_behaviors.md`](./api_guidelines/cache_behaviors.md) §10.

### 3.3 `override_cache_value=True` raises `ValueError`

**Symptom.** `ValueError: override_cache_value is True but query
cache is not configured.`

**Likely cause.** You passed
`connection_options.override_cache_value=True` on a client that
has no query cache (`cache_options.cache_path` is `None`, or the
manager is in a non-`WORKING` state). `override_cache_value` is
only meaningful when there's a cache to override.

**Fix.** Either set `cache_options.cache_path` so the query cache
actually exists, or drop `override_cache_value` — on a cacheless
client every call already hits the provider. See
[`cache_behaviors.md`](./api_guidelines/cache_behaviors.md) §10.

### 3.4 `keep_raw_provider_response=True` raises at construction

**Symptom.** `ValueError:
keep_raw_provider_response=True is incompatible with cache_options. …`

**Likely cause.** Debug flag collides with the cache. Cached
results do not carry raw provider responses (the raw body isn't
serialized to disk), so the combination is forbidden at
construction rather than silently degraded.

**Fix.** Pick one:

```python
# Debug mode — no cache, raw response available
px.connect(debug_options=px.types.DebugOptions(
    keep_raw_provider_response=True,
))

# Production mode — cache on, no raw response
px.connect(cache_options=px.types.CacheOptions(
    cache_path='/tmp/proxai_cache',
))
```

See [`raw_provider_response.md`](./api_guidelines/raw_provider_response.md) §5.

---

## 4. Feature support / endpoint picking

### 4.1 `endpoint <name> is not supported`

**Likely cause.** You passed
`connection_options.endpoint='<name>'` forcing a specific
endpoint, and the resolved model does not advertise support for
the features your query uses (e.g., forcing
`beta.chat.completions.parse` on a query with a tool it can't
handle).

**Fix.** Drop the explicit `endpoint=` and let the framework
pick — or look up the endpoints the model actually supports via
`px.models.get_model_config(...)` and pick one whose feature
config fits your query. See
[`px_client_api.md`](./api_guidelines/px_client_api.md) §4.

### 4.2 `endpoint <name> is not supported in STRICT mode`

**Likely cause.** Your `provider_call_options.feature_mapping_strategy`
is `STRICT`, which rejects `BEST_EFFORT` support levels. Under
`STRICT`, a feature that would normally be approximated via
prompt injection instead raises. Common on pydantic output
against providers without a native parse endpoint.

**Fix.** Either drop `STRICT` (the default is `BEST_EFFORT`) or
pick a model that declares the feature `SUPPORTED` outright.
Query `px.models.list_working_models(...)` with a concrete
filter to find candidates. See
[`provider_feature_support_summary.md`](./api_guidelines/provider_feature_support_summary.md).

### 4.3 `No compatible endpoint found for provider '<p>'`

**Likely cause.** Your query invokes features the provider has
no endpoint for at all — e.g., requesting `output_format.type=IMAGE`
against DeepSeek, or passing a `provider_file_api_ids` for an
unsupported MIME. The framework walks `ENDPOINT_PRIORITY` and
every endpoint comes back `NOT_SUPPORTED`.

**Fix.** The error message lists what each endpoint saw. Read
`get_query_support_details` output per endpoint to see which
feature tripped each one; then drop / downgrade that feature, or
switch to a provider whose `list_models()` includes a model
advertising the capability.

### 4.4 `Feature '<f>' is not supported by endpoint '<e>'`

**Likely cause.** A specific feature on the query
(`system_prompt`, `messages`, `web_search`, etc.) is marked
`NOT_SUPPORTED` on the endpoint. Raised from
`FeatureAdapter.adapt_query_record` before the executor runs.

**Fix.** The same fix as §4.3 in miniature: drop the offending
feature, or pick a different model. `system_prompt=NOT_SUPPORTED`
on a chat endpoint typically means the provider has no equivalent
— your options are to concatenate the system guidance into the
first user message yourself, or use a different provider for this
call.

### 4.5 `Input format '<t>' is not supported by endpoint '<e>'`

**Likely cause.** You passed a content block of type `<t>` (IMAGE
/ DOCUMENT / AUDIO / VIDEO / JSON / PYDANTIC) in a chat message,
but the chosen endpoint's `input_format.<t>` is `NOT_SUPPORTED`.

**Fix.** Convert the content to a supported format before
passing it, or pick a model that supports the format. For
documents specifically, most OpenAI-compatible providers accept
PDF natively — if yours doesn't, ProxAI can extract text from
PDF / md / csv / txt documents when the format is
`BEST_EFFORT`. See
[`provider_feature_support_summary.md`](./api_guidelines/provider_feature_support_summary.md)
for the per-provider matrix.

---

## 5. Models and health checks

### 5.1 `get_working_model` raises "API key missing" or "health check failed"

**Likely causes and fixes.**

| Error | Cause | Fix |
|---|---|---|
| `API key missing` | Env var for the named provider unset | Set the env var. See §2.2. |
| Model not found | Provider/model name typo | Spell-check against `px.models.list_models()` output; both the `provider` and `model` strings must match registry entries exactly. |
| Health check failed | The provider actually returned an error on the probe | Run `px.models.check_health(provider, model, return_all=True)` and inspect the `ModelStatus.provider_queries[<model>]` entry — the `CallRecord.result.error` has the provider-specific reason. |

See [`px_models_api.md`](./api_guidelines/px_models_api.md) §6.

### 5.2 `list_working_models` is slow the first time

**Likely cause.** It hits every provider to run health probes in
parallel. The default 25-second timeout
(`model_probe_options.timeout`) is a ceiling, not a target — a
call may take close to that much wall-clock the first time.

**Fix.** This is expected. After the first call, results are
cached (default model cache is always-on; TTL 4 hours when
writable). Subsequent calls within the TTL return near-instantly.
If you need to force a fresh probe, pass `clear_model_cache=True`:

```python
fresh_status = px.models.list_working_models(
    clear_model_cache=True, return_all=True,
)
```

See [`cache_behaviors.md`](./api_guidelines/cache_behaviors.md)
§2.2 for the two-layer model cache design.

### 5.3 "Models I know work show as `FAILED` in `ModelStatus`"

**Likely causes.**

- **Cached probe is stale.** Your `cache_options.model_cache_duration`
  lapsed and a flaky probe got cached as FAILED. Run
  `clear_model_cache=True` once.
- **Health-probe `output_format` mismatch.** Probes are per
  `output_format`. A model that works for TEXT may have a cached
  FAILED state for JSON — `list_working_models(output_format=...)`
  only sees the one slice.
- **API key for that provider was unset at the time of probe.**
  The probe logs `FAILED` rather than raising, so it sticks
  until cleared.

**Fix.**

```python
status = px.models.list_working_models(
    clear_model_cache=True, return_all=True,
)
print(status.failed_models)
print(status.provider_queries[failed_model].result.error)
```

The `CallRecord.result.error` field tells you why that specific
model fell over.

---

## 6. Fallback chains

### 6.1 `suppress_provider_errors and fallback_models cannot be used together`

**Likely cause.** You set both
`connection_options.suppress_provider_errors=True` *and*
`connection_options.fallback_models=[...]`. The two serve
overlapping purposes — the first turns provider errors into
`FAILED` results, the second retries on error — so the
combination is rejected to force a choice.

**Fix.** Decide what you want:

```python
# Retry across providers, then raise on exhaustion
connection_options=px.types.ConnectionOptions(
    fallback_models=[('openai', 'gpt-4o-mini'), ('anthropic', 'claude-3-5-haiku')],
)

# Single call; errors surface as CallRecord.result.error instead of raising
connection_options=px.types.ConnectionOptions(
    suppress_provider_errors=True,
)
```

See [`px_client_api.md`](./api_guidelines/px_client_api.md) §4.

### 6.2 `endpoint and fallback_models cannot be used together`

**Likely cause.** You forced a specific `endpoint=` *and* a
`fallback_models=` list. An endpoint pin is per-model; a
fallback chain crosses models. The two combinations compose
to "force endpoint X on every fallback model", which almost
never makes sense.

**Fix.** Pick one. If you need different endpoints per model,
write the control flow yourself (catch the error, retry with the
next model explicitly).

### 6.3 `fallback_models` tuple / list shape rejected

**Likely cause.** You passed `fallback_models=['gpt-4o-mini', ...]`
or `fallback_models=[('openai', 'gpt-4o-mini', 'extra'), ...]`.
The framework expects a list of `(provider, model)` tuples OR
`ProviderModelType` instances.

**Fix.**

```python
fallback_models=[
    ('openai', 'gpt-4o-mini'),          # tuple form
    ('anthropic', 'claude-3-5-haiku'),
]

# or
fallback_models=[
    px.models.get_model('openai', 'gpt-4o-mini'),
    px.models.get_model('anthropic', 'claude-3-5-haiku'),
]
```

---

## 7. Files and auto-upload

### 7.1 `Media type '...' cannot be uploaded and referenced by file_id on provider '<p>'`

**Likely cause.** `px.files.upload(media, providers=[<p>])` was
called with a media whose MIME type is in the provider's upload
allow-list but **not** in its `file_id`-reference allow-list.
The framework refuses to do an upload whose resulting `file_id`
the executor would then silently drop. Classic case: DOCX
upload to Mistral (the File API accepts it, but Mistral's chat
endpoint won't take a DOCX `file_id`).

**Fix.** Check `px.files.is_upload_supported(media, provider)`
first — it returns `False` in exactly this case:

```python
for provider in ['gemini', 'claude', 'openai']:
    if px.files.is_upload_supported(media, provider):
        px.files.upload(media, providers=[provider])
        break
else:
    print('No provider can both upload and reference this MIME.')
```

See
[`px_files_api.md`](./api_guidelines/px_files_api.md) §2.5 /
§1.2.3 for the per-provider reference matrix.

### 7.2 `FileUploadError` with partial successes

**Likely cause.** You called `px.files.upload(media, providers=[p1, p2, ...])`
with multiple providers and at least one failed (e.g., rate
limit, invalid key). The raise reports every failure; the media
object still carries the successful provider IDs.

**Fix.** Catch the error, inspect `error.errors`, and decide per
provider:

```python
try:
    px.files.upload(media, providers=['gemini', 'claude', 'openai'])
except px.files.FileUploadError as e:
    print('Failed providers:', list(e.errors.keys()))
    print('Successful providers:',
          list(e.media.provider_file_api_ids or {}))
    # e.media still has successful uploads recorded — you can
    # proceed with just those, or retry the failed ones.
```

See [`px_files_api.md`](./api_guidelines/px_files_api.md) §6.

### 7.3 Auto-upload "succeeded" but the provider still received inline bytes

**Likely cause.** The auto-upload hook in `generate()` swallows
failures so a failed upload doesn't crash the call — the
executor then falls back to inline base64 / URL reference for
that media. Symptoms: you see a log entry about an upload
attempt, your `CallRecord.query.chat[*].content[*].provider_file_api_ids`
is empty for the provider you expected, and the request body is
larger than you expected.

**Fix.** Upload explicitly before `generate()`. Explicit uploads
raise on failure, so you can handle them rather than silently
degrading:

```python
px.files.upload(media, providers=['gemini'])
# Only after the upload succeeds does the media carry a gemini file_id.
response = px.generate_text(
    messages=px.Chat(messages=[px.Message(role='user', content=[media, 'summarize'])]),
    provider_model=('gemini', 'gemini-2.5-flash'),
)
```

See [`px_files_api.md`](./api_guidelines/px_files_api.md) §4.

---

## 8. Multi-choice and output format

### 8.1 `n > 1` silently behaves like `n = 1`

**Likely cause.** Only OpenAI's `chat.completions.create`
endpoint supports `n > 1` today. Every other provider declares
`parameters.n = NOT_SUPPORTED`, so the framework raises (if you
set it explicitly) or the adapter drops it silently if the
feature advertises `BEST_EFFORT` anywhere in the chain.

**Fix.** Use OpenAI for multi-choice, or issue `n` separate
calls yourself:

```python
# OpenAI only
rec = px.generate_text('...', provider_model=('openai', 'gpt-4o'),
                       parameters=px.types.Parameters(n=3))
choices = rec.result.choices   # list of ResultRecord

# Cross-provider equivalent
choices = [
    px.generate_text('...', provider_model=('claude', 'sonnet-4.5')).result
    for _ in range(3)
]
```

See
[`provider_feature_support_summary.md`](./api_guidelines/provider_feature_support_summary.md)
for the `n` column.

### 8.2 `PYDANTIC` output raises `pydantic.ValidationError`

**Likely cause.** The model returned syntactically valid JSON
but the wrong shape — missing fields, wrong types, wrong enum
value. The framework parses the TEXT block via `json.loads` and
then calls `pydantic_class.model_validate(...)`; the latter
raises.

**Fix.** This is a prompt-quality / model-capability issue, not
a framework bug. Try one of:

- Add a concrete example to the prompt showing the exact shape.
- Switch to a more capable model (pydantic with
  `feature_mapping_strategy=STRICT` forces a model that declares
  `pydantic=SUPPORTED` natively, which tends to hallucinate less).
- Catch and retry:

  ```python
  try:
      rec = px.generate_pydantic('...', pydantic_class=MySchema)
  except pydantic.ValidationError:
      rec = px.generate_pydantic('...', pydantic_class=MySchema,
                                 parameters=px.types.Parameters(temperature=0.0))
  ```

See
[`provider_feature_support_summary.md`](./api_guidelines/provider_feature_support_summary.md)
(pydantic column) for who supports what natively.

### 8.3 `JSON` output raises `json.JSONDecodeError`

**Likely cause.** The model returned markdown-wrapped JSON
(`` ```json\n...\n``` ``) or prose before the JSON. The
framework's `json.loads` only accepts pure JSON text. Some
provider connectors pre-strip fences (Anthropic, Mistral); if
yours doesn't, raw model output slips through.

**Fix.** Pick a provider whose connector handles this:
`claude` and `mistral` strip fences automatically. For other
providers, switch from `type=JSON` to a Pydantic class — the
Pydantic path uses the same JSON-extract helper on supporting
connectors. Failing that, extract the JSON yourself from the
raw `output_text`.

---

## 9. Chat / `MessageContent` construction

### 9.1 `TypeError("Expected Message or dict, got ...")`

**Likely cause.** You passed an unsupported type to `Chat(messages=...)`
or `chat.append(...)`. Accepted shapes: `Message`, `dict`, `str`
(becomes an assistant message), `list[MessageContent | dict | str]`
(becomes an assistant message).

**Fix.**

```python
# OK
chat.append('hello')                                    # str → assistant
chat.append({'role': 'user', 'content': 'hi'})         # dict
chat.append(px.Message(role='user', content='hi'))     # Message

# Not OK
chat.append(42)                                         # TypeError
```

### 9.2 `ValueError("Invalid content type: ...")` at Chat build

**Likely cause.** `MessageContent(type=<string>)` with a value
not in `ContentType` (the valid set is `text / thinking / image /
document / audio / video / json / pydantic_instance / tool`).
Typo, or using a legacy value that was renamed.

**Fix.** Use the enum to avoid typos:

```python
px.MessageContent(type=px.MessageContent.Type.TEXT, text='hi')
# or the string form, spelled correctly:
px.MessageContent(type='text', text='hi')
```

See [`px_chat_api.md`](./api_guidelines/px_chat_api.md) §1.1
for the per-type field rules.

### 9.3 `ValueError("Unsupported media_type: ...")`

**Likely cause.** A MIME string not in
`SUPPORTED_MEDIA_TYPES` (see
[`px_chat_api.md`](./api_guidelines/px_chat_api.md) §7). E.g.,
`application/json`, `image/svg+xml`, arbitrary non-standard
types.

**Fix.** Map to the closest supported MIME, or drop `media_type`
and set `type` explicitly:

```python
# Instead of media_type='image/svg+xml', convert the SVG to PNG first.
# Or if it's truly text:
px.MessageContent(type='text', text=svg_string)
```

The SUPPORTED_MEDIA_TYPES list in `src/proxai/chat/message_content.py`
is the authoritative set.

---

## 10. "Why didn't `suppress_provider_errors` catch this?"

`suppress_provider_errors` only swallows **provider-side errors**
surfaced through `_safe_provider_query` — rate limits, 5xx
responses, SDK exceptions raised inside the executor. Client-side
`ValueError` / `TypeError` / configuration errors are raised
synchronously and are **not** suppressed.

Rough guide:

| Error class | Suppressed by `suppress_provider_errors=True`? |
|---|---|
| Every `ValueError` from `FeatureAdapter` / `_check_endpoint_support_compatibility` / config validation | **No** — these are programmer errors |
| Every `TypeError` from `Chat` / `MessageContent` construction | **No** |
| Rate-limit / authentication errors from the provider SDK | **Yes** — translated to `CallRecord.result.error` |
| Timeout / network errors inside the executor | **Yes** |
| `pydantic.ValidationError` from post-call parsing | **No** — the call already succeeded at the provider layer |
| `json.JSONDecodeError` from post-call parsing | **No** — same reason |

See
[`px_generate_api.md`](./api_guidelines/px_generate_api.md) for
the exact routing rules and
[`px_client_api.md`](./api_guidelines/px_client_api.md) §4 for
the client-level error menu.

---

## 11. Still stuck?

- Run the call once with
  `debug_options=px.types.DebugOptions(keep_raw_provider_response=True)`
  and inspect `rec.debug.raw_provider_response` — it's the
  unmodified provider SDK response, which often makes the real
  failure obvious. Note that this flag is incompatible with
  `cache_options` (see §3.4). See
  [`raw_provider_response.md`](./api_guidelines/raw_provider_response.md).
- Check
  [`provider_feature_support_summary.md`](./api_guidelines/provider_feature_support_summary.md)
  for the capability matrix before assuming a feature works on a
  given provider.
- If the symptom doesn't appear in this doc's §1 table, the
  authoritative place is the per-API Errors section listed in
  the intro — those tables have every caller-facing raise in
  the library.
