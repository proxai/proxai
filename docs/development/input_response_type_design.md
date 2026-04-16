# Design: Splitting CallType into InputType and ResponseType

## 1. Problem Statement

The `CallType` enum (`src/proxai/types.py:28`) serves as the single
classification axis for model capabilities:

```python
class CallType(str, enum.Enum):
  TEXT = "TEXT"
  IMAGE = "IMAGE"
  AUDIO = "AUDIO"
  VIDEO = "VIDEO"
  MULTI_MODAL = "MULTI_MODAL"
  OTHER = "OTHER"
```

This design has five concrete problems:

**P1 — Semantic confusion.** The name "CallType" does not clearly communicate
whether it describes what a model *accepts* or what it *produces*. In
practice it describes the **output modality** of a model (a `CallType.IMAGE`
model generates images), but `MULTI_MODAL` breaks this interpretation — it
means "accepts multimodal input but produces text." The same enum mixes
input and output semantics.

**P2 — Missing JSON and PYDANTIC.** `generate_json` and `generate_pydantic`
both resolve their default model via `CallType.TEXT`
(`src/proxai/client.py:1345,1413`). There is no `CallType.JSON` or
`CallType.PYDANTIC`, so `set_model()` cannot configure independent default
models for structured output vs. free-form text generation.

**P3 — No input modality concept.** The codebase can send multimodal content
blocks in messages via `ContentType` (`src/proxai/chat/message_content.py:55`)
with TEXT, IMAGE, DOCUMENT, AUDIO, VIDEO types. However, there is no
metadata declaring which endpoints accept which input modalities, and no
filtering or validation based on input type.

**P4 — Model discovery is one-dimensional.** `list_models(call_type=...)`
(`src/proxai/connections/available_models.py`) can only filter by a single
axis. A user cannot ask "give me models that accept image input AND produce
text output" — the system conflates these into `MULTI_MODAL`.

**P5 — `set_model` is artificially limited.** The current signature
(`src/proxai/client.py:1101`) only supports `provider_model` and
`generate_text` keywords, both mapping to `CallType.TEXT`. There is no way
to set defaults for `generate_image`, `generate_audio`, `generate_video`,
`generate_json`, or `generate_pydantic` independently.

**P6 — FeatureTagType conflates four unrelated concerns.** The
`FeatureTagType` enum (`src/proxai/types.py:176`) mixes input format tags
(PROMPT, MESSAGES, SYSTEM_PROMPT), parameter tags (TEMPERATURE, MAX_TOKENS,
STOP, N, THINKING), tool tags (WEB_SEARCH), and response format tags
(RESPONSE_TEXT, RESPONSE_JSON, RESPONSE_PYDANTIC, etc.) into a single flat
enum. This forces `list_models(features=[...])` to accept a grab-bag of
unrelated filters in one list, making the API harder to use and understand.
A user writing `features=['response_json', 'web_search', 'temperature']`
is filtering on three completely different dimensions with no type safety
distinguishing them.

## 2. Current Architecture

### 2.1 Three Overlapping Type Systems

| Value | CallType | ResponseFormatType | ContentType |
|---|---|---|---|
| TEXT | YES | YES | YES |
| IMAGE | YES | YES | YES |
| AUDIO | YES | YES | YES |
| VIDEO | YES | YES | YES |
| JSON | **NO** | YES | YES |
| PYDANTIC | **NO** | YES | YES |
| MULTI_MODAL | YES | YES | NO (implicit) |
| DOCUMENT | NO | NO | YES |
| THINKING | NO | NO | YES |
| TOOL | NO | NO | YES |
| OTHER | YES | NO | NO |

`ResponseFormatType` (`types.py:425`) already has the response-oriented
taxonomy that `CallType` lacks (TEXT, IMAGE, AUDIO, VIDEO, JSON, PYDANTIC,
MULTI_MODAL). `ContentType` already captures input modality at the message
content level. `CallType` sits between them, doing neither job well.

### 2.2 Where CallType Is Used Today

**Model metadata** — `ProviderModelMetadataType.call_type` (`types.py:216`)
classifies what a model can do. Stored in model config JSON files
(`model_configs_data/v1.2.0.json`).

**Model filtering** — `model_configs._check_call_type_matches()`
(`model_configs.py:749-777`) filters models by capability. Special case:
`CallType.TEXT` also matches `MULTI_MODAL` models.

**Default model registry** — `client.registered_models: dict[CallType, ProviderModelType]` and `client.registered_model_connectors: dict[CallType, ProviderState]`
(`types.py:846`) store the default model per call type.

**Model cache** — `ModelStatusByCallType = dict[CallType, ModelStatus]`
(`types.py:710`) keys health-check results by CallType.

**Model discovery API** — `list_models(call_type=...)`,
`list_providers(call_type=...)`, etc.
(`available_models.py`) all accept `call_type` as a filter parameter.

**Serialization** — `type_serializer.py` encodes/decodes CallType in model
metadata and `CallTypeMappingType` dicts.

**generate_* routing** — Each generate method uses a specific CallType to
resolve its default model: `generate_text` → TEXT,
`generate_json` → TEXT, `generate_pydantic` → TEXT,
`generate_image` → IMAGE, `generate_audio` → AUDIO,
`generate_video` → VIDEO.

**Not used in execution pipeline.** CallType never appears in
`QueryRecord`, `provider_connector.generate()`, `feature_adapter`,
`result_adapter`, or any executor method. Endpoint selection is driven
entirely by `ResponseFormatType`.

### 2.3 Where Input Modality Lives Today

`ContentType` (`message_content.py:55`) describes content blocks: TEXT,
IMAGE, DOCUMENT, AUDIO, VIDEO, etc. Messages can contain any combination.
However:

- No metadata declares which endpoints accept which content types.
- `FeatureConfigType` (`types.py:163`) has `prompt` and `messages` support
  flags but no per-modality input flags.
- `ResponseFormatConfigType` (`types.py:150`) declares output formats per
  endpoint — there is no parallel `InputFormatConfigType`.
- Feature adapter validates response format compatibility but not input
  format compatibility.

### 2.4 FeatureTagType — One Enum, Four Responsibilities

`FeatureTagType` (`types.py:176`) is a single flat enum used for model
discovery filtering via `list_models(features=[...])`:

```python
class FeatureTagType(str, enum.Enum):
  # Input features
  PROMPT = "prompt"
  MESSAGES = "messages"
  SYSTEM_PROMPT = "system_prompt"
  # Parameter features
  TEMPERATURE = "temperature"
  MAX_TOKENS = "max_tokens"
  STOP = "stop"
  N = "n"
  THINKING = "thinking"
  # Tool features
  WEB_SEARCH = "web_search"
  # Response format features
  RESPONSE_TEXT = "response_text"
  RESPONSE_IMAGE = "response_image"
  RESPONSE_AUDIO = "response_audio"
  RESPONSE_VIDEO = "response_video"
  RESPONSE_JSON = "response_json"
  RESPONSE_PYDANTIC = "response_pydantic"
  RESPONSE_MULTI_MODAL = "response_multi_modal"
```

The `_TAG_TO_FIELD` map in `adapter_utils.py:60-90` resolves each tag to a
lambda that navigates into the nested `FeatureConfigType` structure:

- Input tags → `c.prompt`, `c.messages`, `c.system_prompt` (top-level fields)
- Parameter tags → `c.parameters.temperature`, etc. (nested in `ParameterConfigType`)
- Tool tags → `c.tools.web_search` (nested in `ToolConfigType`)
- Response tags → `c.response_format.text`, etc. (nested in `ResponseFormatConfigType`)

The current `list_models` API accepts all of these in one `features` list:

```python
# Current — flat, untyped, confusing
px.models.list_models(features=['response_json', 'web_search', 'temperature'])
```

This has three problems:

1. **No type safety.** A user can pass any tag in any position. There is no
   way to distinguish "I want models that accept image input" from "I want
   models that produce image output" without knowing the RESPONSE_ prefix
   convention.

2. **Missing input format tags entirely.** There are RESPONSE_TEXT,
   RESPONSE_IMAGE, etc. but no INPUT_TEXT, INPUT_IMAGE, etc. Input format
   filtering does not exist.

3. **Scaling problem.** As new tools, parameters, and formats are added,
   `FeatureTagType` grows without bound and the `_TAG_TO_FIELD` map becomes
   a maintenance burden. The flat structure provides no grouping or
   namespacing.

### 2.5 Current Model Landscape (v1.2.0)

| Model | Current CallType | Actual Input | Actual Output |
|---|---|---|---|
| openai/gpt-4o | TEXT | text, image, document | text, json, pydantic |
| openai/gpt-5-mini | TEXT | text, image, document | text, json, pydantic |
| openai/o3 | TEXT | text, image, document | text, json, pydantic |
| openai/dall-e-3 | IMAGE | text only | image |
| openai/tts-1 | AUDIO | text only | audio |
| openai/sora-2 | VIDEO | text only | video |
| gemini models | TEXT | text, image, doc, audio, video | text, json, pydantic, image |
| claude models | TEXT | text, image, document | text, json, pydantic |
| mistral, cohere, etc. | TEXT | text | text, json, pydantic |

This table reveals the conflation: a single `CallType` cannot express that
GPT-4o accepts image input but produces text output, while DALL-E 3 accepts
text input but produces image output.

## 3. Proposed Design

### 3.1 New Enums

**InputType** — What a model accepts as primary input modality:

```python
class InputType(str, enum.Enum):
  TEXT = "TEXT"
  IMAGE = "IMAGE"
  AUDIO = "AUDIO"
  VIDEO = "VIDEO"
  MULTI_MODAL = "MULTI_MODAL"
```

`MULTI_MODAL` means the model accepts text combined with other modalities
(images, documents, audio, video). This preserves the current semantic of
`CallType.MULTI_MODAL` but places it explicitly in the input dimension.

**ResponseType** — What a model produces as primary output:

```python
class ResponseType(str, enum.Enum):
  TEXT = "TEXT"
  IMAGE = "IMAGE"
  AUDIO = "AUDIO"
  VIDEO = "VIDEO"
  JSON = "JSON"
  PYDANTIC = "PYDANTIC"
  MULTI_MODAL = "MULTI_MODAL"
  OTHER = "OTHER"
```

Adds JSON and PYDANTIC to enable independent default model registration.
`MULTI_MODAL` means the model can produce multiple output formats.

### 3.2 New InputFormatConfigType

Parallel to the existing `ResponseFormatConfigType` (`types.py:150`):

```python
@dataclasses.dataclass
class InputFormatConfigType:
  """Input format configuration for a provider endpoint."""

  text: FeatureSupportType | None = None
  image: FeatureSupportType | None = None
  document: FeatureSupportType | None = None
  audio: FeatureSupportType | None = None
  video: FeatureSupportType | None = None
```

Added to `FeatureConfigType`:

```python
@dataclasses.dataclass
class FeatureConfigType:
  prompt: FeatureSupportType | None = None
  messages: FeatureSupportType | None = None
  system_prompt: FeatureSupportType | None = None
  add_system_to_messages: bool | None = None
  parameters: ParameterConfigType | None = None
  tools: ToolConfigType | None = None
  response_format: ResponseFormatConfigType | None = None
  input_format: InputFormatConfigType | None = None  # NEW
```

### 3.3 Updated Model Metadata

```python
@dataclasses.dataclass
class ProviderModelMetadataType:
  input_type: InputType | None = None     # NEW — replaces call_type
  response_type: ResponseType | None = None  # NEW — replaces call_type
  is_recommended: bool | None = None
  model_size_tags: list[ModelSizeType] | None = None
  tags: list[str] | None = None
```

Each model in `v1.2.0.json` would have both fields:

| Model | input_type | response_type |
|---|---|---|
| openai/gpt-4o | MULTI_MODAL | TEXT |
| openai/dall-e-3 | TEXT | IMAGE |
| openai/tts-1 | TEXT | AUDIO |
| openai/sora-2 | TEXT | VIDEO |
| gemini/gemini-2.5-flash | MULTI_MODAL | TEXT |
| anthropic/claude-4-sonnet | MULTI_MODAL | TEXT |
| mistral/mistral-large | TEXT | TEXT |

### 3.4 Updated set_model()

```python
def set_model(
    provider_model=None,        # sets TEXT (backward compat)
    generate_text=None,         # → ResponseType.TEXT
    generate_json=None,         # → ResponseType.JSON
    generate_pydantic=None,     # → ResponseType.PYDANTIC
    generate_image=None,        # → ResponseType.IMAGE
    generate_audio=None,        # → ResponseType.AUDIO
    generate_video=None,        # → ResponseType.VIDEO
)
```

Multiple kwargs allowed in one call:
`px.set_model(generate_text=A, generate_image=B)`.

### 3.5 Fallback Chain for Default Model Resolution

When `generate_json()` is called without an explicit `provider_model`:

```
ResponseType.JSON → ResponseType.TEXT → default discovery
```

Full chain:

```
generate_text     → TEXT → discovery
generate_json     → JSON → TEXT → discovery
generate_pydantic → PYDANTIC → JSON → TEXT → discovery
generate_image    → IMAGE → discovery
generate_audio    → AUDIO → discovery
generate_video    → VIDEO → discovery
```

JSON and PYDANTIC fall back to TEXT because they use text-capable models.
PYDANTIC falls through JSON first because a model good at structured JSON
output is likely good at Pydantic too.

### 3.6 Updated Model Discovery

```python
# Filter by response type (current behavior, renamed)
px.models.list_models(response_type="text")

# Filter by input type (new capability)
px.models.list_models(input_type="multi_modal")

# Filter by both (new capability)
px.models.list_models(input_type="multi_modal", response_type="text")
```

The matching logic in `model_configs._check_call_type_matches()` becomes:

```python
def _check_response_type_matches(self, response_type, config):
  if response_type == ResponseType.TEXT:
    return config.metadata.response_type in (
        ResponseType.TEXT, ResponseType.MULTI_MODAL)
  if response_type in (ResponseType.JSON, ResponseType.PYDANTIC):
    # JSON/PYDANTIC are registration keys, not metadata values.
    # Text and multi-modal models can produce structured output.
    return config.metadata.response_type in (
        ResponseType.TEXT, ResponseType.MULTI_MODAL)
  return config.metadata.response_type == response_type

def _check_input_type_matches(self, input_type, config):
  if input_type == InputType.TEXT:
    # TEXT input is accepted by all models
    return True
  if input_type == InputType.MULTI_MODAL:
    return config.metadata.input_type == InputType.MULTI_MODAL
  return config.metadata.input_type == input_type
```

### 3.7 Provider Connector ENDPOINT_CONFIG Example

After adding `input_format`, OpenAI's config would look like:

```python
ENDPOINT_CONFIG = {
    'chat.completions.create': FeatureConfigType(
        prompt=SUPPORTED,
        messages=SUPPORTED,
        system_prompt=SUPPORTED,
        input_format=InputFormatConfigType(
            text=SUPPORTED,
            image=SUPPORTED,
            document=SUPPORTED,
            audio=NOT_SUPPORTED,
            video=NOT_SUPPORTED,
        ),
        response_format=ResponseFormatConfigType(
            text=SUPPORTED,
            json=SUPPORTED,
            pydantic=BEST_EFFORT,
        ),
    ),
    'images.generate': FeatureConfigType(
        prompt=SUPPORTED,
        input_format=InputFormatConfigType(
            text=SUPPORTED,
        ),
        response_format=ResponseFormatConfigType(
            image=SUPPORTED,
        ),
    ),
}
```

### 3.8 Split FeatureTagType into Four Separate Tag Enums

Replace the monolithic `FeatureTagType` with four focused enums, each
mapping to a specific dimension of model capability:

**InputFormatTag** — What input formats the model accepts:

```python
class InputFormatTag(str, enum.Enum):
  TEXT = "text"
  IMAGE = "image"
  DOCUMENT = "document"
  AUDIO = "audio"
  VIDEO = "video"
```

Maps to `InputFormatConfigType` fields. Each tag resolves to
`feature_config.input_format.<field>`.

**OutputFormatTag** — What output formats the model produces:

```python
class OutputFormatTag(str, enum.Enum):
  TEXT = "text"
  IMAGE = "image"
  AUDIO = "audio"
  VIDEO = "video"
  JSON = "json"
  PYDANTIC = "pydantic"
  MULTI_MODAL = "multi_modal"
```

Maps to `ResponseFormatConfigType` fields. Each tag resolves to
`feature_config.response_format.<field>`. Replaces the current
`RESPONSE_TEXT`, `RESPONSE_IMAGE`, etc. tags (dropping the `RESPONSE_`
prefix since the parameter name now provides the namespace).

**ToolTag** — What tools the model supports:

```python
class ToolTag(str, enum.Enum):
  WEB_SEARCH = "web_search"
```

Maps to `ToolConfigType` fields. Each tag resolves to
`feature_config.tools.<field>`. As new tools are added (code execution,
function calling, etc.), they go here.

**FeatureTag** — General model features and parameters:

```python
class FeatureTag(str, enum.Enum):
  PROMPT = "prompt"
  MESSAGES = "messages"
  SYSTEM_PROMPT = "system_prompt"
  TEMPERATURE = "temperature"
  MAX_TOKENS = "max_tokens"
  STOP = "stop"
  N = "n"
  THINKING = "thinking"
```

Maps to top-level `FeatureConfigType` fields and nested
`ParameterConfigType` fields. These describe general capabilities and
tunable parameters.

### 3.9 Updated list_models() API

The `features` grab-bag parameter is replaced by four typed parameters:

```python
# Current — flat, untyped
px.models.list_models(
    features=['response_json', 'web_search', 'temperature']
)

# Proposed — separated by concern
px.models.list_models(
    input_format='multi_modal',
    output_format='image',
    feature_tags=['temperature', 'max_tokens'],
    tool_tags=['web_search'],
)
```

Each parameter accepts its corresponding tag enum values or strings
(case-insensitive, following the existing `Param` pattern):

```python
InputFormatTagParam = list[InputFormatTag] | list[str] | InputFormatTag | str
OutputFormatTagParam = list[OutputFormatTag] | list[str] | OutputFormatTag | str
ToolTagParam = list[ToolTag] | list[str] | ToolTag | str
FeatureTagParam = list[FeatureTag] | list[str] | FeatureTag | str
```

Singular values are also accepted for convenience — the user does not need
to wrap a single tag in a list.

The full `list_models` signature becomes:

```python
def list_models(
    model_size: ModelSizeIdentifierType | None = None,
    input_type: InputTypeParam | None = None,
    response_type: ResponseTypeParam = ResponseType.TEXT,
    input_format: InputFormatTagParam | None = None,
    output_format: OutputFormatTagParam | None = None,
    feature_tags: FeatureTagParam | None = None,
    tool_tags: ToolTagParam | None = None,
    recommended_only: bool = True,
) -> list[ProviderModelType]:
```

**Distinction between `input_type`/`response_type` and
`input_format`/`output_format`:**

- `input_type` / `response_type` filter by model **metadata** — the
  primary classification stored in `ProviderModelMetadataType`. These are
  coarse filters ("show me image generation models").
- `input_format` / `output_format` filter by endpoint **capability** —
  the fine-grained support levels in `FeatureConfigType`. These check
  whether at least one endpoint SUPPORTS the specific format ("show me
  models where JSON output is natively SUPPORTED, not just BEST_EFFORT").

Example: `response_type="text"` returns all text models.
`output_format="json"` further filters to only those text models that have
at least one endpoint with `response_format.json = SUPPORTED`.

### 3.10 Updated _TAG_TO_FIELD Maps

The single `_TAG_TO_FIELD` dict in `adapter_utils.py:60-90` splits into
four maps, each self-contained:

```python
_INPUT_FORMAT_TAG_TO_FIELD = {
    InputFormatTag.TEXT: lambda c: (
        c.input_format.text if c.input_format else None),
    InputFormatTag.IMAGE: lambda c: (
        c.input_format.image if c.input_format else None),
    InputFormatTag.DOCUMENT: lambda c: (
        c.input_format.document if c.input_format else None),
    InputFormatTag.AUDIO: lambda c: (
        c.input_format.audio if c.input_format else None),
    InputFormatTag.VIDEO: lambda c: (
        c.input_format.video if c.input_format else None),
}

_OUTPUT_FORMAT_TAG_TO_FIELD = {
    OutputFormatTag.TEXT: lambda c: (
        c.response_format.text if c.response_format else None),
    OutputFormatTag.IMAGE: lambda c: (
        c.response_format.image if c.response_format else None),
    OutputFormatTag.AUDIO: lambda c: (
        c.response_format.audio if c.response_format else None),
    OutputFormatTag.VIDEO: lambda c: (
        c.response_format.video if c.response_format else None),
    OutputFormatTag.JSON: lambda c: (
        c.response_format.json if c.response_format else None),
    OutputFormatTag.PYDANTIC: lambda c: (
        c.response_format.pydantic if c.response_format else None),
    OutputFormatTag.MULTI_MODAL: lambda c: (
        c.response_format.multi_modal if c.response_format else None),
}

_TOOL_TAG_TO_FIELD = {
    ToolTag.WEB_SEARCH: lambda c: (
        c.tools.web_search if c.tools else None),
}

_FEATURE_TAG_TO_FIELD = {
    FeatureTag.PROMPT: lambda c: c.prompt,
    FeatureTag.MESSAGES: lambda c: c.messages,
    FeatureTag.SYSTEM_PROMPT: lambda c: c.system_prompt,
    FeatureTag.TEMPERATURE: lambda c: (
        c.parameters.temperature if c.parameters else None),
    FeatureTag.MAX_TOKENS: lambda c: (
        c.parameters.max_tokens if c.parameters else None),
    FeatureTag.STOP: lambda c: (
        c.parameters.stop if c.parameters else None),
    FeatureTag.N: lambda c: (
        c.parameters.n if c.parameters else None),
    FeatureTag.THINKING: lambda c: (
        c.parameters.thinking if c.parameters else None),
}
```

Each map is used by its own `resolve_*_tag_support` function, keeping
the adapter logic modular and testable.

### 3.11 Updated Adapter Filtering

The `_filter_by_features` method in `available_models.py:362` currently
takes a single `features: list[FeatureTagType]`. It splits into four
filter methods (or one method with four parameters):

```python
def _filter_by_input_format(self, models, input_format_tags):
    ...  # check input_format support on best endpoint

def _filter_by_output_format(self, models, output_format_tags):
    ...  # check response_format support on best endpoint

def _filter_by_feature_tags(self, models, feature_tags):
    ...  # check general feature support on best endpoint

def _filter_by_tool_tags(self, models, tool_tags):
    ...  # check tool support on best endpoint
```

Each filter follows the same pattern as the existing `_filter_by_features`:
iterate models, check support level via the appropriate tag map, keep
models with SUPPORTED or BEST_EFFORT.

### 3.12 Backward Compatibility for FeatureTagType

`FeatureTagType` is currently used in:
- `list_models(features=...)` in `available_models.py`
- `get_feature_tags_support_level()` in `feature_adapter.py`,
  `result_adapter.py`, and `provider_connector.py`
- `_TAG_TO_FIELD` in `adapter_utils.py`
- `type_utils.create_feature_tag_list()` in `type_utils.py`
- Tests: `test_feature_adapter.py`, `test_adapter_utils.py`,
  `test_available_models.py`

Migration approach:
1. Create the four new enums.
2. Keep `FeatureTagType` as a deprecated union/alias that maps old values
   to their new enum counterparts.
3. Keep `features` parameter in `list_models` as deprecated, internally
   routing each tag to the correct new filter.
4. Remove after one major version.

## 4. Impact Analysis

### 4.1 Files Requiring Changes

**Core type definitions:**
- `src/proxai/types.py` — Add InputType, ResponseType enums. Add
  InputFormatConfigType. Replace FeatureTagType with InputFormatTag,
  OutputFormatTag, ToolTag, FeatureTag. Update ProviderModelMetadataType.
  Rename derived types (CallTypeMappingType → ResponseTypeMappingType,
  etc.). Add backward-compat aliases (`CallType = ResponseType`,
  `FeatureTagType` deprecated).

**Model selection and registration:**
- `src/proxai/client.py` — Update `set_model()` signature. Update
  `get_default_provider_model()` to accept `response_type` with fallback.
  Update all `generate_*` methods. Update `registered_models` key type.

**Model configuration and filtering:**
- `src/proxai/connectors/model_configs.py` — Rename `models_by_call_type`
  → `models_by_response_type`. Add `models_by_input_type`. Update filtering
  methods.

**Model discovery:**
- `src/proxai/connections/available_models.py` — Rename all `call_type`
  parameters to `response_type`. Add `input_type` parameter to discovery
  methods. Replace `features` parameter with `input_format`,
  `output_format`, `feature_tags`, `tool_tags`. Split
  `_filter_by_features` into four separate filter methods.

**Adapter utilities:**
- `src/proxai/connectors/adapter_utils.py` — Split `_TAG_TO_FIELD` dict
  into four maps (`_INPUT_FORMAT_TAG_TO_FIELD`,
  `_OUTPUT_FORMAT_TAG_TO_FIELD`, `_TOOL_TAG_TO_FIELD`,
  `_FEATURE_TAG_TO_FIELD`). Add per-category `resolve_*_tag_support`
  functions.

**Caching:**
- `src/proxai/caching/model_cache.py` — Rename cache key type from
  `ModelStatusByCallType` to `ModelStatusByResponseType`.

**Serialization:**
- `src/proxai/serializers/type_serializer.py` — Handle both `call_type`
  and `response_type`/`input_type` JSON keys for backward compatibility.
  Add InputFormatConfigType serialization.

**Model config data:**
- `src/proxai/connectors/model_configs_data/v1.2.0.json` — Replace
  `"call_type"` with `"input_type"` and `"response_type"` in all model
  metadata entries.

**Provider connectors (8 files):**
- `src/proxai/connectors/providers/openai.py` — Add `input_format` to
  ENDPOINT_CONFIG.
- Same for `gemini.py`, `claude.py`, `mistral.py`, `grok.py`,
  `deepseek.py`, `cohere.py`, `huggingface.py`.

**Feature adapter:**
- `src/proxai/connectors/feature_adapter.py` — Add input format validation
  (check that message content types match endpoint's input_format config).

**Utility functions:**
- `src/proxai/type_utils.py` — Rename `check_call_type_param` to
  `check_response_type_param`. Add `check_input_type_param`. Replace
  `create_feature_tag_list` with per-category validation functions
  (`create_input_format_tag_list`, `create_output_format_tag_list`,
  `create_feature_tag_list`, `create_tool_tag_list`).

**Public API:**
- `src/proxai/proxai.py` — Update `set_model()` signature. Update
  `list_models()` and other model discovery forwarding functions.
- `src/proxai/__init__.py` — Export `InputType`, `ResponseType`,
  `InputFormatTag`, `OutputFormatTag`, `ToolTag`, `FeatureTag`.

**Tests (9+ files):**
- `tests/test_client.py`
- `tests/connectors/test_model_configs.py`
- `tests/connections/test_available_models.py`
- `tests/caching/test_model_cache.py`
- `tests/serializers/test_type_serializer.py`
- `tests/logging/test_utils.py`
- `tests/connections/test_proxdash.py`
- `tests/connectors/test_adapter_utils.py` — Update for split tag maps
- `tests/connectors/test_feature_adapter.py` — Update for new tag enums

**Examples:**
- `examples/alias_test.py`, `examples/refactoring_test.py`

**Documentation (4+ files):**
- `docs/development/px_models_analysis.md`
- `docs/development/px_generate_analysis.md`
- `docs/development/px_client_analysis.md`
- `docs/development/provider_connectors.md`

### 4.2 Scope Summary

| Category | File Count |
|---|---|
| Core source files | ~16 |
| Provider connectors | 8 |
| JSON config data | 1-2 |
| Test files | 9+ |
| Example files | 2 |
| Documentation | 5+ |
| **Total** | **~42** |

## 5. Migration Strategy

### Phase 1: Foundation (No Breaking Changes)

1. In `types.py`, create `ResponseType` and `InputType` as new enums.
2. Set `CallType = ResponseType` as a backward-compat alias.
3. Create all new type aliases with backward-compat aliases for old names.
4. Add `InputFormatConfigType` dataclass.
5. In `ProviderModelMetadataType`, add `input_type` and `response_type`
   fields. Keep `call_type` as a deprecated property that proxies to
   `response_type`.
6. Update serializer to accept both old and new JSON key names on decode.
7. All existing code continues to work.

### Phase 2: Migrate Internal Callsites

1. Rename all internal `call_type` parameters and variables to
   `response_type` (and add `input_type` where applicable).
2. Rename internal methods (e.g., `_check_call_type_matches` →
   `_check_response_type_matches`).
3. Rename state fields (`models_by_call_type` → `models_by_response_type`).
4. Keep backward-compat aliases.

### Phase 3: Split FeatureTagType into Four Enums

1. Create `InputFormatTag`, `OutputFormatTag`, `ToolTag`, `FeatureTag`
   enums in `types.py`.
2. Keep `FeatureTagType` as a deprecated alias that maps old values to
   the correct new enum.
3. Split `_TAG_TO_FIELD` in `adapter_utils.py` into four separate maps.
4. Add per-category `resolve_*_tag_support` functions.
5. Update `list_models()` signature: replace `features` with
   `input_format`, `output_format`, `feature_tags`, `tool_tags`.
6. Keep old `features` parameter as deprecated, routing internally.
7. Split `_filter_by_features` in `available_models.py` into four
   filter methods.

### Phase 4: Add InputFormatConfigType to Providers

1. Add `InputFormatConfigType` dataclass to `types.py`.
2. Add `input_format` field to `FeatureConfigType`.
3. Add `input_format` to each provider's `ENDPOINT_CONFIG`.
4. Update feature adapter to validate input content types.
5. Populate `input_type` metadata for all models in JSON config.

### Phase 5: Expand set_model() and generate_* Fallback Logic

1. Add per-generate-function keywords to `set_model()`.
2. Add fallback chain to `get_default_provider_model()`.
3. Update each `generate_*` method to use its own ResponseType key.

### Phase 6: Update JSON Config Data

1. Rename `"call_type"` to `"response_type"` in all JSON entries.
2. Add `"input_type"` to all JSON entries.
3. Keep decoder backward compat (accept old key names).

### Phase 7: Update Tests, Examples, and Docs

### Phase 8: Deprecate and Remove Aliases

1. Add deprecation warnings to `CallType`, `CallTypeMappingType`,
   `FeatureTagType`, and the `features` parameter.
2. After one major version, remove the aliases.

## 6. Potential Problems

### 6.1 Serialized State Migration

The model cache (`model_cache.py`) and `ProxAIClientState` serialize dicts
keyed by `CallType` string values ("TEXT", "IMAGE", etc.). Since the string
values themselves do not change (only the enum name), existing serialized
data deserializes correctly as long as the decoder maps strings to
`ResponseType` values. The `CallType = ResponseType` alias ensures this.

### 6.2 ProxDash Server Compatibility

The ProxDash server sends model configs containing `"call_type"` keys.
The decoder must accept both `"call_type"` and `"response_type"` /
`"input_type"` indefinitely — or until the ProxDash server is updated to
send the new keys. This is a cross-service compatibility concern that
requires coordination.

### 6.3 Model Metadata vs. Registration Keys

A critical distinction: **JSON and PYDANTIC exist as ResponseType
registration keys but NOT as model metadata values.** No model is
inherently a "JSON model" — it is a text model used for JSON generation.
Model metadata uses `response_type: TEXT`; the `ResponseType.JSON` value
is only used in `registered_models[ResponseType.JSON]`. This must be
clearly documented and enforced to prevent confusion.

### 6.4 Fallback Chain Complexity

With `ResponseType.JSON` and `ResponseType.PYDANTIC` as first-class keys,
`get_default_provider_model` needs a fallback chain:
`PYDANTIC → JSON → TEXT → MULTI_MODAL → discovery`. Each step must check
`registered_models` without recursion. Test carefully to avoid infinite
loops or surprising model selection.

### 6.5 MULTI_MODAL Ambiguity

Even after the split, `MULTI_MODAL` appears in both enums:
- `InputType.MULTI_MODAL` — "accepts text + other modalities" (clear)
- `ResponseType.MULTI_MODAL` — "can produce multiple output formats"
  (less common, mostly future-facing)

The input side is well-defined. The response side is speculative — no
current model metadata uses `ResponseType.MULTI_MODAL`. Consider whether
this value is needed in ResponseType at launch, or if it can be added
later.

### 6.6 Two-Dimensional Model Discovery

Adding `input_type` to discovery creates a 2D filter space. The
`models_by_call_type` dict (currently `dict[CallType, list[ProviderModelType]]`)
may need to become two dicts (`models_by_response_type` and
`models_by_input_type`) or a single dict with tuple keys. Two separate
dicts is simpler and avoids combinatorial explosion.

### 6.7 Backward Compatibility of Public API

`CallType` is **not** currently exported in `__init__.py`, which makes
renaming safer. However, `call_type` appears as a parameter name in
user-facing methods like `list_models(call_type=...)`. Renaming to
`response_type` is a breaking change for users who use keyword arguments.
Mitigation: accept both `call_type` and `response_type` kwargs with a
deprecation warning on the old name.

### 6.8 InputFormatConfigType Granularity

The proposed `InputFormatConfigType` has `document` as a field, but
`ResponseFormatConfigType` does not. This asymmetry reflects reality
(models accept document input but don't produce documents), but may
confuse developers expecting a symmetric design. Document the rationale
explicitly.

### 6.9 Feature Adapter Validation Scope

Adding input format validation to `feature_adapter.py` means that sending
an image in a message to a text-only endpoint would now raise an error
instead of silently failing or producing garbage. This is correct behavior
but is technically a breaking change for users who currently send
unsupported content types without error.

### 6.10 FeatureTagType Split — Backward Compatibility

The current `FeatureTagType` is exported and used in user-facing code
(e.g., `list_models(features=[FeatureTagType.RESPONSE_JSON])`). Splitting
into four enums changes the import paths. Mitigation: keep `FeatureTagType`
as a deprecated enum whose values map to the correct new enum. The old
`features` parameter in `list_models` continues to work but routes each
tag to the appropriate new filter internally.

### 6.11 Tag Enum Naming — Potential Confusion

`InputFormatTag` vs `InputType` and `OutputFormatTag` vs `ResponseType`
could confuse developers. The distinction:
- `InputType` / `ResponseType` are **metadata** classifications for model
  discovery (coarse: "this is a text model").
- `InputFormatTag` / `OutputFormatTag` are **capability** filters for
  endpoint features (fine-grained: "this endpoint supports image input
  at SUPPORTED level").

This must be documented clearly, with examples showing when to use each.

### 6.12 Four Filter Parameters vs. One

The current `features` parameter is simple to explain even if semantically
overloaded. Four parameters (`input_format`, `output_format`,
`feature_tags`, `tool_tags`) are more precise but add cognitive load.
However, the separated API reads naturally:

```python
# Self-documenting — each parameter name explains what it filters
px.models.list_models(
    input_format='multi_modal',
    output_format='image',
    tool_tags=['web_search'],
)
```

vs. the current:

```python
# Requires knowing the RESPONSE_ prefix convention
px.models.list_models(
    features=['response_image', 'web_search'],
)
```

The separated version is worth the additional parameters because each
parameter name carries its meaning.

## 7. Alternatives Considered

### A. Just Add JSON and PYDANTIC to CallType

**Pros:** Minimal change, solves the `set_model` problem immediately.
**Cons:** Does not fix naming confusion. No input modality concept.
MULTI_MODAL remains ambiguous. Does not scale as the type system evolves.

### B. Use ResponseFormatType as the Universal Key

**Pros:** Eliminates CallType/ResponseFormatType duplication.
**Cons:** Conflates model capability classification (which model to pick)
with execution-time response parsing configuration (how to handle the
response). These are different layers that should remain separate.

### C. Use FeatureTagType Instead of InputType

**Pros:** FeatureTagType already exists and is extensible.
**Cons:** Feature tags are lists, not single values. A model has multiple
tags. Using them as dict keys for `registered_models` breaks down when
tags are non-exclusive. Tags are for filtering, not keying.

### E. Keep FeatureTagType as a Single Enum, Just Add Input Tags

**Pros:** Minimal structural change. Just add INPUT_TEXT, INPUT_IMAGE, etc.
to the existing enum.
**Cons:** Deepens the conflation problem. The enum grows unbounded. The
`features` parameter remains a grab-bag. No type safety — a user can mix
input and output tags without realizing. The `_TAG_TO_FIELD` map in
`adapter_utils.py` becomes even longer. The `RESPONSE_` prefix convention
(and now `INPUT_` prefix) is a poor substitute for proper namespacing.

### F. Use Separate Param Types but Keep One Enum

**Pros:** Four parameters in `list_models` but tags come from one enum.
Simpler type system.
**Cons:** No compile-time/type-checking guarantee that the right tags go
in the right parameter. A user could pass `FeatureTagType.TEMPERATURE`
into `output_format` — runtime validation catches it, but the API
is misleading.

### D. Defer InputType, Only Rename CallType → ResponseType

**Pros:** Smaller scope, faster to ship.
**Cons:** Misses the opportunity to properly model the input dimension.
The MULTI_MODAL confusion persists. Would need another refactor later when
input validation is needed.

## 8. Summary

| Decision | Choice | Rationale |
|---|---|---|
| Add InputType? | **Yes** | Properly separates input and output modality |
| Rename CallType? | **Yes → ResponseType** | Reflects actual semantics |
| Add InputFormatConfigType? | **Yes** | Parallel to ResponseFormatConfigType |
| Split FeatureTagType? | **Yes → 4 enums** | Each tag type maps to one config dimension |
| Merge with ResponseFormatType? | **No** | Different layers (selection vs. parsing) |
| Handle MULTI_MODAL? | **Keep as-is** | Both enums, documented clearly |
| Backward compat? | **Alias + dual-key decode** | Zero-breakage migration path |

### Complete API Vision

After all changes, the user-facing API looks like:

```python
import proxai as px

# Set default models per generate function
px.set_model(
    generate_text=('openai', 'gpt-4o'),
    generate_json=('anthropic', 'claude-4-sonnet'),
    generate_image=('openai', 'dall-e-3'),
)

# Discover models with precise, typed filtering
models = px.models.list_models(
    input_format='multi_modal',   # accepts image/doc input
    output_format='json',         # can produce JSON output
    feature_tags=['thinking'],    # supports thinking/reasoning
    tool_tags=['web_search'],     # supports web search
)

# Generate with automatic model resolution
text = px.generate_text("Hello")          # uses gpt-4o
data = px.generate_json("List 3 items")   # uses claude-4-sonnet
img = px.generate_image("A sunset")       # uses dall-e-3
```
