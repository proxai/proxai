# Raw Provider Response — `keep_raw_provider_response`

Source of truth: `DebugOptions` in `src/proxai/types.py` and
`_validate_raw_provider_response_options` in `src/proxai/client.py`.
If this document disagrees with those files, the files win — update
this document.

**For local debugging only.** `keep_raw_provider_response` attaches
the provider SDK's raw response object to `CallRecord.debug` so you
can inspect provider-specific fields during development. It is not
part of ProxAI's stable contract, it is not serialized to the cache
or to ProxDash, and it can break at any provider SDK upgrade. Do
**not** rely on it in production code paths.

If a provider field you need is not exposed on ProxAI's typed
`CallRecord` surface, please open an issue at
<https://github.com/proxai/proxai/issues> so the team can model it
properly — don't build production logic on top of this hatch.

See also: `px_client_api.md` §2.7 (where `DebugOptions` lives),
`call_record.md` §1 (the `debug: DebugInfo | None` field).

---

## 1. Usage

```python
import proxai as px

px.connect(debug_options=px.DebugOptions(keep_raw_provider_response=True))

rec = px.generate_text("Hello")
raw = rec.debug.raw_provider_response
print(type(raw))
# e.g. <class 'openai.types.chat.chat_completion.ChatCompletion'>
```

The raw object is whatever the provider's Python SDK returned — an
`openai.ChatCompletion`, `anthropic.Message`, gemini
`GenerateContentResponse`, etc. Exact type varies by provider and by
endpoint; read the executor in
`src/proxai/connectors/providers/<provider>.py` for the specific
binding.

---

## 2. When the field is populated

`rec.debug.raw_provider_response` is non-`None` only when all three
conditions hold:

1. The client was built with `keep_raw_provider_response=True`.
2. The call was served from the provider — cache hits are impossible
   here because the flag is incompatible with `cache_options` (§3).
3. `rec.result.status == SUCCESS`. On `FAILED`, `rec.debug` stays
   `None`.

Guard before accessing:

```python
if rec.debug is not None and rec.debug.raw_provider_response is not None:
    inspect(rec.debug.raw_provider_response)
```

On a fallback chain, the returned record corresponds to the model
that actually answered — `rec.query.provider_model` identifies it.

---

## 3. Mutually exclusive with `cache_options`

Passing any non-`None` `cache_options` together with
`keep_raw_provider_response=True` raises `ValueError` at client
construction. The query cache cannot reconstruct live SDK objects on
cache hits, so the combination would silently return `None` for
cached calls — the validator refuses it up front.

If you need both a cached production path and raw-response
inspection, use two clients:

```python
prod = px.Client(cache_options=px.CacheOptions(cache_path="/var/cache/proxai"))
debug = px.Client(debug_options=px.DebugOptions(keep_raw_provider_response=True))
```

---

## 4. Not serialized

`CallRecord.debug` is dropped by the encoder in
`src/proxai/serializers/type_serializer.py`. It never reaches the
query cache, ProxDash, or any JSON round-trip through ProxAI's own
serializers. Hold the reference while the process is alive; don't
expect it to survive persistence.

---

## 5. Errors

| Trigger | Error |
|---|---|
| Building a client with both `cache_options` and `keep_raw_provider_response=True` | `ValueError: keep_raw_provider_response=True is incompatible with cache_options. …` |
| Reading `rec.debug.raw_provider_response` when the flag is off | `AttributeError` — `rec.debug` is `None`. |
| Reading `rec.debug.raw_provider_response` on a `FAILED` call | Returns `None` — no exception. |

Constructing with the flag on also emits a one-time logging warning
reminding that the feature is a debugging escape hatch. This is
informational, not an error.
