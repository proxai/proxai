# ProxAI Integration Test Suite — Design

Status: **DRAFT** — review and adjust before implementation.

## 1. Goals

A human-in-the-loop integration test suite for the public ProxAI surface
(`px.*`, `px.models.*`, `px.files.*`, `px.types.*`, ProxDash). One run gives a
human reviewer high confidence that an end-to-end ProxAI installation works:
provider calls go out, responses come back, files upload, ProxDash records
appear, sensitive content is hidden when configured, etc.

**Three audiences for these tests:**
1. ProxAI maintainers verifying a release before publishing to PyPI.
2. ProxDash backend/UI engineers verifying the SDK still feeds the dashboard.
3. New contributors learning the public API by example.

**Run shape:** one operator, one terminal, ~30–60 min of attention. Each test
prints what it did, prompts the operator on UI checks, and persists state so
re-runs skip already-passed steps.

## 2. Non-goals

- **Not a unit-test replacement.** `tests/` (pytest) covers parser/edge logic
  with no network. Integration tests assume `tests/` already passed.
- **Not a load test.** No QPS, no parallelism beyond what the API exercises.
- **Not provider conformance.** `examples/refactoring_test.py` already runs
  the cross-provider parity matrix; these tests use 2–3 trustworthy provider
  models to exercise SDK code paths, not every provider.
- **Not part of `pytest` collection.** Run via `poetry run python3
  integration_tests/<file>.py`, like the existing `proxai_api_test.py`.

## 3. Source material — what we have

### `integration_tests/proxai_api_test.py` (current, ~1350 lines, ~50 tests)
Single mega-file. Mature pattern: `@integration_block` decorator persists
test state to disk so a half-finished run resumes; `_manual_user_check()`
prompts the operator y/n; `--mode latest|new|<id>`, `--auto-continue`,
`--print-code`, `--env dev|prod` flags. **Keep this pattern**, split the body.

What it covers well:
- px.models.* listing (15+ tests, some redundant)
- px.generate_text variations (11 tests)
- check_health (4 variants)
- Local logging (3 variants)
- query_cache (5 variants)
- ProxDash records / experiments / hide_sensitive_content (4 + 4)
- connect-time options (suppress_provider_errors, feature_mapping_strategy)
- get_current_options

**Major gaps:**
- Zero coverage of `px.files.*` (upload / list / download / remove).
- No image / audio / video generation (`generate_image/audio/video`).
- No multi-modal input (image, audio, video, document, json, pydantic).
- No chat-session usage of `px.generate(messages=...)`.
- No `px.set_model` for non-text models.
- No fallback / endpoint / thinking parameter exercise.
- No ProxDash file integration (proxdash_file_id, dedup).

### `examples/*_test.py` (post-cleanup, ~5 files)
Built during feature development. Mostly assert-driven, a few interactive.
After the recent cleanup:
- `models_api_test.py` — clean px.models.* coverage. **Already canonical.**
- `files_api_test.py` — comprehensive Files API. **Net-new content for
  integration tests.**
- `proxdash_files_api_test.py` — ProxDash + files dedup/dl/delete.
  **Net-new content.**
- `proxdash_test.py` — interactive ProxDash UI sanity, multi-modal in/out.
  **Net-new content.**
- `alias_test.py` — generate_text/json/pydantic/image, set_model.
  **Partial new content (alias funcs were thin in proxai_api_test).**
- `simple_test.py` — playground, no assertions. **Drop.**
- `refactoring_test.py` — cross-provider parity harness. **Stays separate.**

## 4. Proposed file structure

```
integration_tests/
├── INTEGRATION_TESTS_PLAN.md      # this file
├── _utils.py                      # shared utilities (NOT a test file)
├── _assets/                       # cat.pdf, cat.jpeg, cat.md, ...
│   ├── cat.pdf
│   ├── cat.jpeg
│   ├── cat.webp
│   ├── cat.md
│   ├── cat.mp3
│   └── cat.mp4
├── 01_models_test.py              # ~15 tests
├── 02_generate_test.py            # ~25 tests
├── 03_files_test.py               # ~25 tests
├── 04_proxdash_test.py            # ~20 tests (mostly manual UI checks)
└── 05_runtime_test.py             # ~15 tests (connect / cache / logging / errors)
```

5 test files + 1 shared util module + an asset directory. Total ~100 tests
(consolidated from ~50 current + ~110 example = ~160, dropping duplication
and the `simple_test.py` playground).

The numeric prefixes communicate **suggested run order** for a fresh
operator (setup precedes generate; generate precedes files; files precede
proxdash UI checks; runtime config last). They are **not strict
dependencies** — each file's `main()` calls `_utils.ensure_setup_state()`
which lazy-creates / loads the api_key state.

The leading underscore on `_utils.py` and `_assets/` signals "not a test
file, won't be picked up by `--mode <id>`."

`refactoring_test.py` stays in `examples/` as a development tool; not part
of this plan.

## 5. `_utils.py` — shared API

Single source of truth for: paths, CLI parsing, the `@integration_block`
decorator, manual-check helpers, common provider models, asset paths,
and ProxDash setup.

```python
# Paths and run config
def init_run(test_file_label: str) -> RunContext:
  """Parse argv, set up _TEST_PATH/_ROOT_LOGGING_PATH/_ROOT_CACHE_PATH,
  pick test_id (latest|new|N), and return a RunContext.

  Replaces init_test_path() — same flags, but each test file passes its
  own label so logs separate per file.
  """

@dataclass
class RunContext:
  test_id: int
  test_path: str
  root_logging_path: str
  root_cache_path: str
  experiment_path: str
  webview_base_url: str    # http://localhost:3000  or  https://proxai.co
  proxdash_base_url: str   # http://localhost:3001  or  https://proxainest-...
  print_code: bool
  auto_continue: bool

# Setup state — api_key persists across all 5 files
def ensure_setup_state(ctx: RunContext) -> dict:
  """Load ~/proxai_integration_test/test_<id>/_setup.state if present.
  If missing, walk operator through user-creation + api-key generation
  (the existing create_user flow). Returns state dict with at least
  `api_key`. Idempotent across files — only first runner pays the cost."""

# Decorator (kept verbatim from proxai_api_test.py, moved here)
def integration_block(func):
  """State-persisting decorator: skip if <func.__name__>.state exists,
  otherwise run and persist. Honors force_run= and skip= kwargs."""

# Manual UI verification
def manual_check(test_message: str, fail_message: str) -> None:
  """y/n prompt; raise on n. Honors --auto-continue (then auto-y for
  user_check is dangerous — auto-continue should ONLY skip the
  'press enter' beats, not the y/n verifications. Decision needed,
  see §10.)."""

def manual_check_with_url(prompt: str, url: str, expected: str) -> None:
  """Prints url, prompts operator, asserts y. For ProxDash UI tests."""

# Output
def print_separator(status: str, message: str, color: str) -> None:
  """[STARTING|RUNNING|SKIPPED|PASSED|FAILED] colored line."""

# Models registry — one source of truth across all test files
TEXT_MODELS = [
  ('openai',   'gpt-4o'),       # web_search + thinking + json
  ('gemini',   'gemini-3-flash'),
  ('claude',   'sonnet-4.6'),
]
THINKING_MODEL = ('openai', 'o3')
IMAGE_MODEL    = ('openai', 'dall-e-3')           # or gemini-2.5-flash-image
AUDIO_MODEL    = ('openai', 'tts-1')              # or gemini-2.5-flash-tts
VIDEO_MODEL    = ('openai', 'sora-2')             # or veo-3.1-generate
FAILING_MODEL  = ('mock_failing_provider', 'mock_failing_model')

# Asset paths (resolved at import)
ASSET_PDF      = '_assets/cat.pdf'
ASSET_IMAGE    = '_assets/cat.jpeg'
ASSET_WEBP     = '_assets/cat.webp'
ASSET_MD       = '_assets/cat.md'
ASSET_AUDIO    = '_assets/cat.mp3'
ASSET_VIDEO    = '_assets/cat.mp4'
def asset(name: str) -> str: ...
```

## 6. `01_models_test.py` — `px.models.*` and registry

**Pre-condition:** any setup state (api_key not strictly needed — most
tests are offline reads of the bundled JSON registry).

**Theme:** "I just installed ProxAI. Show me what models are available
and which ones actually work right now."

| # | Test | Source | Type | Notes |
|---|------|--------|------|-------|
| 1.1 | `list_models_default`             | new (alias_test) | assert | Default = recommended text. Smoke-check `len > 0`, gpt-4o present, dall-e-3 absent. |
| 1.2 | `list_models_recommended_only_false` | new (alias_test) | assert | Non-recommended models become visible (e.g. mistral-small-latest). |
| 1.3 | `list_models_by_size`             | new (proxai_api_test 1.2 + models_api_test) | assert | small / medium / large / largest filters. |
| 1.4 | `list_models_by_input_format`     | new (models_api_test) | assert | image / audio / video / document on input. |
| 1.5 | `list_models_by_output_format`    | new (models_api_test) | assert | image / audio / video / json / pydantic on output. |
| 1.6 | `list_models_by_tool_tags`        | new (models_api_test) | assert | web_search filter. |
| 1.7 | `list_models_by_feature_tags`     | new (models_api_test) | assert | thinking filter. |
| 1.8 | `list_models_combined_filters`    | new (models_api_test) | assert | image-in + json-out, medium + web_search. |
| 1.9 | `list_providers`                  | new (models_api_test) | assert | All known providers present. |
| 1.10 | `list_provider_models`           | new (models_api_test + proxai_api_test) | assert | per-provider model list, with size filter. |
| 1.11 | `get_model` / `get_model_config` | new (models_api_test + proxai_api_test) | assert | Type, fields, metadata.is_recommended, model_size_tags, features. |
| 1.12 | `get_default_model_list`         | new (models_api_test) | assert | priority list from JSON. |
| 1.13 | `list_working_models_basic`      | proxai_api_test | assert + slow | First call probes — assert it took >1s. |
| 1.14 | `list_working_models_with_filters` | proxai_api_test (consolidated from 4 variants) | assert | model_size, return_all, recommended_only=False all in one block. |
| 1.15 | `list_working_models_clear_cache_verbose` | proxai_api_test | assert + visual | Verbose output to verify pretty-printing. |
| 1.16 | `list_working_providers`         | proxai_api_test | assert | At least openai+gemini+claude. |
| 1.17 | `list_working_provider_models`   | proxai_api_test | assert | + size filter consolidated. |
| 1.18 | `check_health_default`           | proxai_api_test | manual_check | Visual: progress bars, summary. |
| 1.19 | `check_health_no_multiprocessing` | proxai_api_test | manual_check | Visual: sequential output. |
| 1.20 | `check_health_with_timeout`      | proxai_api_test | assert | timeout=1 trims slow probes. |
| 1.21 | `check_health_extensive_return`  | proxai_api_test | assert | working/failed counts ≥ 0. |
| 1.22 | `list_working_methods_refuse_media` | new (models_api_test) | assert | image/audio/video output_format raises ValueError pointing at list_models. |

**Consolidations vs source:**
- proxai_api_test had 5 `list_working_models_*` variants — collapsed to 3.
- proxai_api_test had 4 `check_health_*` — kept all 4 (each tests a real
  flag), but turned 2 into manual_check (visual-only).

## 7. `02_generate_test.py` — generation in all shapes

**Pre-condition:** ProxDash connection (api_key from `_setup.state`).
Tests below run within a `local_proxdash_connection` context block.

**Theme:** "Send things to LLMs and get things back, for every input and
output type ProxAI supports."

### 7.1 Text generation — alias + parameters (10 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 2.1 | `generate_text_basic`            | proxai_api_test | assert (response truthy) |
| 2.2 | `generate_text_with_provider_model` | proxai_api_test | assert |
| 2.3 | `generate_text_with_provider_model_type` | proxai_api_test | assert |
| 2.4 | `generate_text_with_system_prompt` | proxai_api_test | manual (output reflects system) |
| 2.5 | `generate_text_with_message_history` | proxai_api_test | assert |
| 2.6 | `generate_text_with_max_tokens`  | proxai_api_test | assert (truncated) |
| 2.7 | `generate_text_with_temperature` | proxai_api_test | assert |
| 2.8 | `generate_text_with_extensive_return` | proxai_api_test | assert (CallRecord shape) |
| 2.9 | `generate_text_with_thinking`    | new (refactoring_test inspiration) | assert (THINKING content) |
| 2.10 | `set_model_default_text`        | proxai_api_test | manual |

### 7.2 Output formats — JSON / Pydantic / Image / Audio / Video (7 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 2.11 | `generate_json` (response_format json_schema) | proxai_api_test | assert (dict shape) |
| 2.12 | `generate_pydantic` (response_format=pydantic) | proxai_api_test | assert |
| 2.13 | `generate_image` via px.generate_image alias | new (alias_test) | manual (file written, opens) |
| 2.14 | `generate_audio` via px.generate_audio alias | new | manual (audio file plays) |
| 2.15 | `generate_video` via px.generate_video alias | new | manual (video file plays) |
| 2.16 | `set_model_image` / `set_model_audio`        | new (alias_test) | manual |
| 2.17 | `output_format_via_string` ('json'/'image'/'audio'/'video') | new | assert |

### 7.3 Multi-modal input (5 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 2.18 | `input_image` (image + text prompt) | new (refactoring + proxdash_test) | assert (cat in output) |
| 2.19 | `input_document_pdf` | new | assert |
| 2.20 | `input_document_md` | new | assert |
| 2.21 | `input_audio` (gemini) | new | assert |
| 2.22 | `input_video` (gemini) | new | assert |

### 7.4 Tools and connection options (4 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 2.23 | `tools_web_search` | proxai_api_test (basic) + refactoring | manual + assert (citations) |
| 2.24 | `connection_options_fallback` | new (refactoring) | assert |
| 2.25 | `connection_options_endpoint` (openai-only) | new (refactoring) | assert |
| 2.26 | `suppress_provider_errors_in_call` | proxai_api_test | assert (error returned, no raise) |

**Drops vs `proxai_api_test.py`:**
- `generate_text_with_suppress_provider_errors` is moved to runtime_test
  alongside the other `connect`-level error tests; the call-level version
  stays here as 2.26.

## 8. `03_files_test.py` — `px.files.*` + ProxDash file integration

**Pre-condition:** ProxDash connection. This file is **net-new content**;
proxai_api_test.py has zero coverage here.

**Theme:** "Upload files to providers / ProxDash, list them, download
them back, delete them, and exercise the cache when files are involved."

### 8.1 Upload — single provider (8 tests, parameterized)

Per asset type, per provider; each asserts `provider_file_api_status[p].state == ACTIVE`.

| # | Test | Source | Type |
|---|------|--------|------|
| 3.1 | `upload_pdf_all_providers` | files_api_test (×4 collapsed) | assert |
| 3.2 | `upload_image_all_providers` | files_api_test | assert (openai expected to fail) |
| 3.3 | `upload_audio_gemini_only` | files_api_test | assert (3 expected fails) |
| 3.4 | `upload_video_gemini_only` | files_api_test | assert |

Collapsed from 16 individual tests to 4 parameterized tests. The original
verbosity is kept inside the body via for-loops, not 16 top-level functions.

### 8.2 Upload — multi-provider (3 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 3.5 | `upload_multi_sequential` | files_api_test | assert |
| 3.6 | `upload_multi_parallel` | files_api_test | assert (faster than sequential) |
| 3.7 | `upload_multi_mixed_media_fail` | files_api_test | assert (raises) |

### 8.3 Remove (3 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 3.8 | `remove_per_provider`     | files_api_test (×4 collapsed) | assert |
| 3.9 | `remove_all`              | files_api_test | assert |
| 3.10 | `remove_selective`       | files_api_test | assert |

### 8.4 List (3 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 3.11 | `list_per_provider`      | files_api_test (×4 collapsed) | assert |
| 3.12 | `list_all_providers`     | files_api_test | assert |
| 3.13 | `list_with_limit`        | files_api_test | assert |

### 8.5 Download (2 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 3.14 | `download_mistral_success` | files_api_test | assert (bytes match) |
| 3.15 | `download_unsupported_providers` | files_api_test (×3 collapsed) | assert (raises) |

### 8.6 Generate with files — manual + auto upload (4 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 3.16 | `generate_manual_upload_per_media` | files_api_test (×16 collapsed by media type) | assert (cat in output) |
| 3.17 | `generate_auto_upload_pdf_per_provider` | files_api_test (×4 collapsed) | assert |
| 3.18 | `generate_unsupported_media_fails` | files_api_test (×~7 collapsed) | assert |
| 3.19 | `serialization_round_trip` | files_api_test | assert |

### 8.7 Cache + files (2 tests, collapsed from 4)

| # | Test | Source | Type |
|---|------|--------|------|
| 3.20 | `cache_with_file_upload` | files_api_test (×4 collapsed) | assert (call 1 PROVIDER, calls 2/3 CACHE) |
| 3.21 | `cache_with_new_media_object_dedupes` | files_api_test | assert |

### 8.8 ProxDash file integration (5 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 3.22 | `proxdash_only_upload` | proxdash_files_api_test (×4 collapsed) | assert (proxdash_file_id set, no providers) |
| 3.23 | `proxdash_plus_provider_upload` | proxdash_files_api_test | assert (both ids) |
| 3.24 | `proxdash_list_dedup` | proxdash_files_api_test | assert (no duplicate file_id) |
| 3.25 | `proxdash_download` | proxdash_files_api_test | assert (bytes match) |
| 3.26 | `proxdash_delete_clears_all_ids` | proxdash_files_api_test | assert |

### 8.9 Cleanup (1 test)

| # | Test | Source | Type |
|---|------|--------|------|
| 3.27 | `cleanup_all_uploaded` | files_api_test | best-effort, last in file |

**Net effect:** files_api_test.py (~75 tests) + proxdash_files_api_test.py
(~15 tests) → ~27 tests, no loss of intent.

## 9. `04_proxdash_test.py` — ProxDash dashboard human verification

**Pre-condition:** ProxDash UI accessible at `_WEBVIEW_BASE_URL`, operator
logged in, ready to switch focus to a browser tab.

**Theme:** "Records I generate via the SDK appear correctly in the
ProxDash UI — including hidden fields, custom experiment paths, sensitive
content masking, and limited API keys."

This file is **mostly manual_check** — the assertions are about what the
operator sees, not what Python returns.

### 9.1 Connection / stdout / disable (3 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 4.1 | `proxdash_stdout_connection_message` | proxai_api_test | manual |
| 4.2 | `proxdash_disable` | proxai_api_test | manual |
| 4.3 | `proxdash_local_log_file` | proxai_api_test | assert (proxdash.log content) |

### 9.2 Records visibility (2 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 4.4 | `proxdash_logging_record_basic` | proxai_api_test | manual (UI shows record) |
| 4.5 | `proxdash_logging_record_full_options` | proxai_api_test | manual (system, messages, temperature, max_tokens, stop visible) |

### 9.3 Sensitive content (3 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 4.6 | `proxdash_hide_sensitive_prompt` | proxai_api_test | manual + assert |
| 4.7 | `proxdash_hide_sensitive_messages` | proxai_api_test | manual + assert |
| 4.8 | `proxdash_limited_api_key` | proxai_api_test | manual (operator generates limited key) |

### 9.4 Experiment path (1 test)

| # | Test | Source | Type |
|---|------|--------|------|
| 4.9 | `proxdash_experiment_path` | proxai_api_test | manual (folder appears in UI) |

### 9.5 Multi-modal records — content rendering (5 tests)

These come from `proxdash_test.py` and are **interactive demos**: each
generates one call with a specific input/output type and asks the
operator to verify rendering in ProxDash.

| # | Test | Source | Type |
|---|------|--------|------|
| 4.10 | `proxdash_renders_text_input` | proxdash_test | manual |
| 4.11 | `proxdash_renders_json_input` | proxdash_test | manual |
| 4.12 | `proxdash_renders_pydantic_input` | proxdash_test | manual |
| 4.13 | `proxdash_renders_image_input` | proxdash_test | manual |
| 4.14 | `proxdash_renders_audio_input` | proxdash_test | manual |
| 4.15 | `proxdash_renders_video_input` | proxdash_test | manual |
| 4.16 | `proxdash_renders_document_input_md` | proxdash_test | manual |
| 4.17 | `proxdash_renders_image_output` | proxdash_test | manual |
| 4.18 | `proxdash_renders_audio_output` | proxdash_test | manual |
| 4.19 | `proxdash_renders_video_output` | proxdash_test | manual |

### 9.6 Files in ProxDash UI (3 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 4.20 | `proxdash_files_appear_in_ui` | new | manual (uploaded file visible) |
| 4.21 | `proxdash_file_attached_to_call_record` | new | manual |
| 4.22 | `proxdash_files_dedup_visible_in_ui` | new | manual |

## 10. `05_runtime_test.py` — connect, options, caching, logging, errors

**Pre-condition:** api_key present.

**Theme:** "Configure ProxAI itself — connect/reset_state, logging files,
query cache behavior, error handling, get_current_options."

### 10.1 Connection lifecycle (3 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 5.1 | `connect_empty` | proxai_api_test | assert (defaults) |
| 5.2 | `connect_full_options` | proxai_api_test | assert (round-trip via get_current_options) |
| 5.3 | `reset_state_clears_session` | new | assert |

### 10.2 Local logging files (3 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 5.4 | `logging_to_provider_queries_log` | proxai_api_test | assert (last_log_data matches) |
| 5.5 | `logging_with_hide_sensitive_content` | proxai_api_test | assert (prompt/response/system/messages all `<sensitive content hidden>`) |
| 5.6 | `logging_with_stdout` | proxai_api_test | manual_check (record visible in stdout) |

### 10.3 Query cache (5 tests, kept as is — each tests a real flag)

| # | Test | Source | Type |
|---|------|--------|------|
| 5.7 | `query_cache_basic` | proxai_api_test | assert (PROVIDER → CACHE) |
| 5.8 | `query_cache_unique_response_limit` | proxai_api_test | assert (PROVIDER ×3 then CACHE) |
| 5.9 | `query_cache_use_cache_false` | proxai_api_test | assert (skips cache) |
| 5.10 | `query_cache_clear_and_override` | proxai_api_test | assert |
| 5.11 | `query_cache_pydantic_response` | proxai_api_test | assert (cached pydantic equality) |

### 10.4 Error handling and runtime errors (2 tests)

| # | Test | Source | Type |
|---|------|--------|------|
| 5.12 | `connect_suppress_provider_errors` | proxai_api_test | assert (error in record, no raise) |
| 5.13 | `connect_feature_mapping_strategy_strict` | proxai_api_test | assert (raises with helpful message) |

### 10.5 Multiprocessing (1 test)

| # | Test | Source | Type |
|---|------|--------|------|
| 5.14 | `connect_allow_multiprocessing_false` | proxai_api_test | manual (warning printed) |

## 11. Test conventions

### State data
Each file has its own state directory:
`~/proxai_integration_test/test_<id>/<file_label>/<test_name>.state`.

A shared `~/proxai_integration_test/test_<id>/_setup.state` holds the
api_key (and any limited_api_key from 4.8). Any file's `main()` calls
`_utils.ensure_setup_state()` which:
1. If `_setup.state` exists → load and return.
2. Else → walk operator through user creation in ProxDash UI, ask for
   api_key via `input()`, write `_setup.state`, return.

### Test types
- **`assert`** — Python assertions only. Operator presses Enter to advance.
- **`manual_check`** — Code prints expected output / a URL; `manual_check()`
  prompts y/n; n raises.
- **`manual` (visual)** — Operator visually confirms ProxDash UI matches.
  Most of `04_proxdash_test.py` is this.

Use `assert` whenever possible. `manual_check` is for things Python can't
introspect (UI rendering, audio playback, image output looking right).

### CLI flags (unchanged from current)
```
--mode latest|new|<id>      # which test_<id> dir to write to
--print-code                # show inspect.getsource per block
--auto-continue             # skip "Press Enter to continue"
--env dev|prod              # base URLs for ProxDash UI
--file <label>              # NEW: only run tests from one file (01_models, …)
--test <name>               # NEW: only run a specific named test
```

### Provider model selection
Centralized in `_utils.py` (§5). The intent is that each test uses a
**known-good, recommended, documented** model — not the cheapest, not the
most-feature-rich. This keeps test failures actionable: if a test using
`gpt-4o` fails, the failure is in the SDK or in OpenAI's API, not in
"someone bumped the test to the latest preview model and the contract
shifted."

Exceptions documented inline:
- 2.9 (thinking) uses `o3` because gpt-4o doesn't think.
- 2.13–2.15 (image/audio/video output) use the relevant provider models.
- 8.x (files) iterates over `[gemini, claude, openai, mistral]` because the
  point is per-provider behavior.

## 12. Migration: what moves from where

### Drop entirely
- `examples/simple_test.py` — playground, no assertions.

### Keep where it is
- `examples/refactoring_test.py` — cross-provider parity is its own job;
  not part of integration tests.

### Folded into integration tests (then removed from `examples/`)
- `examples/models_api_test.py` → `01_models_test.py`
- `examples/alias_test.py` → fragments into `01_models_test.py` (filters)
  + `02_generate_test.py` (alias funcs, set_model)
- `examples/files_api_test.py` → `03_files_test.py` (collapsed)
- `examples/proxdash_files_api_test.py` → `03_files_test.py` §8.8
- `examples/proxdash_test.py` → `04_proxdash_test.py` §9.5

After migration `examples/` contains only `refactoring_test.py` and its
asset directory; the asset directory moves under
`integration_tests/_assets/` with a one-time copy.

### Replaced
- `integration_tests/proxai_api_test.py` → split across all 5 new files.
  The old file becomes `proxai_api_test.py.deprecated` for one release,
  then is removed.

## 13. Open questions for review

1. **`--auto-continue` and `manual_check`.** Today, `--auto-continue`
   skips the "press Enter" beat between tests. Should it also auto-answer
   `y` to `manual_check`? Probably **no** — that defeats the human-in-the-
   loop purpose. Suggest renaming to `--auto-advance` to make the scope
   clear.

2. **State sharing across files.** Proposal: shared `_setup.state` for
   api_key, per-file directory for everything else. Alternatives:
   (a) one flat state dir (current pattern); (b) re-run setup per file
   (annoying). I prefer the proposal.

3. **Run order.** The numeric prefixes are *suggestions*, not enforced.
   But `04_proxdash_test.py` UI tests assume some records exist, so
   running `02_generate_test.py` first matters. Do we add a dependency
   declaration or just document it?

4. **Cleanup.** `03_files_test.py` §8.9 attempts cleanup but providers may
   accumulate orphaned uploads if a test crashes mid-way. Add a `--cleanup`
   flag that scans `~/proxai_integration_test/` for orphaned media JSON
   and removes them?

5. **Should `01_models_test.py` need an api_key at all?**
   Most of it is offline JSON-registry reads. Only `list_working_*` needs
   real provider keys. Could split into `01a_models_offline_test.py` and
   `01b_models_health_test.py`, but that's two files for one concept.
   Suggest: keep one file, mark working-model tests as needing keys, skip
   them if env vars absent.

6. **Failure modes.** When a manual_check fails (operator says n), the
   exception aborts the run. The state file is *not* written, so re-run
   resumes. Confirm this is the desired behavior (vs. "skip and continue,
   collect failures at the end").

7. **Drift detection.** When a model name changes in `_utils.py`, every
   test in every file picks it up — but provider behavior may shift
   silently (a model becomes more verbose, an output_format changes).
   Should we add a per-provider golden-output check for one canonical
   prompt? Probably out of scope for integration tests; that's
   `refactoring_test.py`'s job.

## 14. Implementation order (after this plan is approved)

1. Create `_utils.py` with `RunContext`, `ensure_setup_state`,
   `integration_block`, `manual_check`, asset paths, model constants.
2. Move `_assets/` from `examples/refactoring_test_assets/` (keep a
   symlink in examples for `refactoring_test.py`).
3. Stand up `01_models_test.py`. Verify against current behavior.
4. Stand up `05_runtime_test.py` (mostly direct moves from
   proxai_api_test.py).
5. Stand up `02_generate_test.py` (mix of moves + new content).
6. Stand up `03_files_test.py` (mostly new content from examples).
7. Stand up `04_proxdash_test.py` (mostly manual_check; do last because
   ProxDash UI must already render the records the earlier files
   produced).
8. Mark old `proxai_api_test.py` as deprecated, remove `examples/*.py`
   except `refactoring_test.py` and `simple_test.py` (or delete the
   latter entirely per §12).
9. Update `CLAUDE.md` with the new run command.

## 15. What "done" looks like

```
$ poetry run python3 integration_tests/01_models_test.py --mode new --auto-continue
... 22 tests, 22 passed, 0 manual prompts (auto-continue)

$ poetry run python3 integration_tests/02_generate_test.py
... 26 tests, 25 assert, 1 manual_check, ALL PASS

$ poetry run python3 integration_tests/03_files_test.py
... 27 tests, 27 assert, ALL PASS

$ poetry run python3 integration_tests/04_proxdash_test.py
... 22 tests, 5 assert, 17 manual_check
[Operator sits in front of ProxDash UI for ~15 minutes, answers y/n]
ALL PASS

$ poetry run python3 integration_tests/05_runtime_test.py
... 14 tests, 14 assert, ALL PASS

Total: ~110 tests across 5 files, ~30 minutes of operator time on a
fresh test_<id> run, ~5 minutes when re-running with state persistence.
```
