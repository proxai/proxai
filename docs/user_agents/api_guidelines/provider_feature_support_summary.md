# Provider Feature Support Summary

Quick-skim cheatsheet for which providers support which features.
Snapshot of the `ENDPOINT_CONFIG` declarations in
`src/proxai/connectors/providers/*.py` as of 2026-04-23.

**Source of truth is `px.models`, not this document.** The tables
below aggregate across each provider's endpoints and age with the
code. Always verify with the API before writing code against a
capability:

```python
import proxai as px

# Declared capability (no network call)
px.models.list_models(output_format="json", feature_tags=["thinking"])
px.models.list_models(input_format="image", tool_tags=["web_search"])
px.models.get_model_config(("openai", "gpt-4o"))   # full per-model config

# Verified (runs a health probe; cached)
px.models.list_working_models(output_format="pydantic")
```

If a table cell disagrees with `px.models.*`, trust the API.

---

## Legend

```
✅ SUPPORTED      Declared and exercised by the connector
⚠️  BEST_EFFORT   Library tries; silent degradation possible
❌ NOT_SUPPORTED  Framework rejects or skips the request
—  not applicable (feature doesn't make sense on that axis)
```

Provider-level cells collapse across endpoints: the strongest support
among the provider's endpoints wins. So a provider whose `responses`
endpoint supports `web_search` shows ✅ even if its other endpoints
don't. Use `px.models.get_model_config()` when you need the
per-endpoint, per-model truth.

---

## 1. Provider × endpoint map

```
claude        beta.messages.stream
cohere        chat
databricks    chat.completions.create, beta.chat.completions.parse
deepseek      chat.completions.create
gemini        models.generate_content, models.generate_videos
grok          chat.create
huggingface   chat.completions.create
mistral       chat.complete, chat.parse, beta.conversations.start
openai        responses.create, chat.completions.create,
              beta.chat.completions.parse, images.generate,
              audio.speech.create, videos.create
```

---

## 2. Text parameters

Parameters on `px.generate_text(parameters=...)`.

| provider    | temperature | max_tokens | stop | n   | thinking |
|-------------|-------------|------------|------|-----|----------|
| claude      | ✅          | ✅         | ✅   | ❌  | ✅       |
| cohere      | ✅          | ✅         | ✅   | ❌  | ✅       |
| databricks  | ✅          | ✅         | ✅   | ✅  | ✅       |
| deepseek    | ✅          | ✅         | ✅   | ❌  | ⚠️      |
| gemini      | ✅          | ✅         | ✅   | ❌  | ✅       |
| grok        | ✅          | ✅         | ✅   | ❌  | ✅       |
| huggingface | ✅          | ✅         | ✅   | ❌  | ⚠️      |
| mistral     | ✅          | ✅         | ✅   | ✅  | ⚠️      |
| openai      | ✅          | ✅         | ✅   | ✅  | ✅       |

`n` = multiple candidate responses (`generate_text(parameters={"n": 3})`).
Only databricks / mistral / openai return native multi-choice; the
others reject `n > 1` at the adapter layer.

`thinking` = `low | medium | high` reasoning budget. `⚠️` providers
honor the parameter silently on reasoning-capable models (deepseek-
reasoner, mistral magistral, HF reasoning models) and drop it
elsewhere.

---

## 3. Tools

Tools on `px.generate_text(tools=[...])`.

| provider    | web_search |
|-------------|------------|
| claude      | ✅         |
| cohere      | ❌         |
| databricks  | ❌         |
| deepseek    | ❌         |
| gemini      | ✅         |
| grok        | ✅         |
| huggingface | ❌         |
| mistral     | ✅         |
| openai      | ✅         |

On openai, `web_search` is only supported on the `responses.create`
endpoint — pass the tool and the framework routes you there
automatically. On mistral, it's only on `beta.conversations.start`.

---

## 4. Input formats

Content types accepted inside `messages` / `chat`. `text` is always ✅
and omitted from the table.

| provider    | image | document | audio | video | json | pydantic |
|-------------|-------|----------|-------|-------|------|----------|
| claude      | ✅    | ✅       | ❌    | ❌    | ⚠️  | ⚠️      |
| cohere      | ✅    | ⚠️      | ❌    | ❌    | ⚠️  | ⚠️      |
| databricks  | ✅    | ⚠️      | ❌    | ❌    | ⚠️  | ⚠️      |
| deepseek    | ❌    | ⚠️      | ❌    | ❌    | ⚠️  | ⚠️      |
| gemini      | ✅    | ✅       | ✅    | ✅    | ⚠️  | ⚠️      |
| grok        | ✅    | ⚠️      | ❌    | ❌    | ⚠️  | ⚠️      |
| huggingface | ✅    | ⚠️      | ❌    | ❌    | ⚠️  | ⚠️      |
| mistral     | ✅    | ✅       | ❌    | ❌    | ⚠️  | ⚠️      |
| openai      | ✅    | ✅       | ✅    | ❌    | ⚠️  | ⚠️      |

- `document` ⚠️ = text-based docs (md, csv, txt) are read inline; PDF
  is extracted via pypdf; docx / xlsx are dropped.
- `document` ✅ = provider accepts PDF (and usually more) natively.
- `json` / `pydantic` = sending a dict / BaseModel as an input block.
  Universally BEST_EFFORT — it is serialized to a text block and
  passed through.

For per-endpoint document fidelity (which PDFs, which binaries), read
the `_to_<provider>_part` docstring in the connector file.

---

## 5. Output formats

Accepted values for `response_format=` on `px.generate_text(...)` and
the dedicated media calls (`px.generate_image`, `generate_audio`,
`generate_video`).

| provider    | text | json | pydantic | image | audio | video |
|-------------|------|------|----------|-------|-------|-------|
| claude      | ✅   | ⚠️  | ✅       | ❌    | ❌    | ❌    |
| cohere      | ✅   | ✅   | ✅       | ❌    | ❌    | ❌    |
| databricks  | ✅   | ✅   | ✅       | ❌    | ❌    | ❌    |
| deepseek    | ✅   | ✅   | ⚠️      | ❌    | ❌    | ❌    |
| gemini      | ✅   | ✅   | ⚠️      | ✅    | ✅    | ✅    |
| grok        | ✅   | ✅   | ✅       | ❌    | ❌    | ❌    |
| huggingface | ✅   | ✅   | ✅       | ❌    | ❌    | ❌    |
| mistral     | ✅   | ✅   | ✅       | ❌    | ❌    | ❌    |
| openai      | ✅   | ✅   | ✅       | ✅    | ✅    | ✅    |

`pydantic` ⚠️ = the framework asks for JSON and parses it with
`model_validate_json`; the provider has no native Pydantic / strict
JSON-schema path, so schema violations surface as parse errors rather
than provider-side rejections.

Image / audio / video output is per-endpoint:
- openai: `images.generate`, `audio.speech.create`, `videos.create`.
- gemini: `models.generate_content` for image / audio;
  `models.generate_videos` for video.

`px.models.list_working_models()` refuses `output_format="image" |
"audio" | "video"` — probing would generate real media and is
prohibitive. Verify with a single-model `generate_image()` / etc. call
instead.

---

## 6. How to verify

Always cross-check a cell with the API before writing code against it:

```python
import proxai as px

# Does openai claim to support n > 1?
cfg = px.models.get_model_config(("openai", "gpt-4o"))
print(cfg.features.parameters.n)       # FeatureSupportType.SUPPORTED

# Which providers actually do JSON output today?
px.models.list_providers(output_format="json")

# Which models can take an image and return JSON?
px.models.list_models(input_format="image", output_format="json")

# Working set — add a probe
px.models.list_working_models(
    input_format="image", output_format="json", recommended_only=True)

# Full per-endpoint detail for one model
cfg = px.models.get_model_config(("openai", "gpt-4o"))
print(cfg.features.tools.web_search)           # provider-level aggregate
print(cfg.features.output_format.pydantic)
```

See `px_models_analysis.md` for the full `px.models` surface and
`px_generate_analysis.md` for how the capabilities interact with
request dispatch (feature-mapping strategies, fallback chains).
