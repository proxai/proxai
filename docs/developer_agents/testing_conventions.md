# Testing Conventions

Source of truth: `tests/conftest.py` (shared pytest fixture setup —
the per-session `model_configs_instance`),
`tests/*/test_*.py` (the executable spec for each subsystem),
`src/proxai/connectors/providers/*_mock.py` (mock-client
implementations wired in via `run_type=TEST`),
`src/proxai/connectors/provider_connector.py` (the `api`
property's `run_type`-branched mock/production switch, and the
mock dispatch tables in `files.py` / `file_helpers.py`),
`integration_tests/proxai_api_test.py` (the interactive end-to-end
harness that hits real providers), and `pyproject.toml`
(`[tool.ruff]` and `[tool.yapf]` for the style rules the tests are
held to). If this document disagrees with those files, the files
win — update this document.

This is the definitive reference for how ProxAI is tested: the
two-tier split between unit tests and the integration harness,
the `run_type=TEST` mock pattern every provider connector
follows, the shared fixture that stubs a stable model registry,
the style rules tests are held to, and the coverage expectations
when you land a feature. Read this before adding a new test file,
writing a new mock connector, editing `conftest.py`, extending
the integration harness, or deciding which tier a given
regression belongs in.

See also: `adding_a_new_provider.md` (the executor-side
contract that mocks must duck-type — every method / attribute a
production connector accesses on `self.api` must also exist on
the mock); `files_internals.md` (the `MOCK_*_DISPATCH` tables in
`file_helpers.py` and why mocks return fake file_ids); and
`CLAUDE.md` at the repo root (the **⚠️ Always use `poetry run
python`** rule — non-negotiable for every test command in this
doc).

---

## 1. Test structure (current)

```
proxai/
│
├── tests/                                         # unit tier — pytest, fast, no network
│   ├── conftest.py                                # pytest_configure: model_configs fixture
│   ├── test_proxai.py                             # px.* module facade + user patterns
│   ├── test_client.py                             # ProxAIClient deep coverage
│   ├── test_type_utils.py                         # is_query_record_equal etc.
│   │
│   ├── caching/
│   │   ├── test_query_cache.py
│   │   └── test_model_cache.py
│   ├── chat/
│   │   ├── test_chat_session.py                   # Chat.export, from_dict round-trip
│   │   ├── test_message.py
│   │   └── test_message_content.py
│   ├── connections/
│   │   ├── test_api_key_manager.py
│   │   ├── test_available_models.py
│   │   └── test_proxdash.py                       # uses requests_mock
│   ├── connectors/
│   │   ├── test_provider_connector.py             # __init_subclass__, safe_provider_query, etc.
│   │   ├── test_feature_adapter.py
│   │   ├── test_result_adapter.py
│   │   ├── test_adapter_utils.py
│   │   ├── test_files.py                          # FilesManager dispatch paths
│   │   └── test_model_configs.py
│   ├── experiment/test_experiment.py
│   ├── logging/test_utils.py
│   ├── serializers/
│   │   ├── test_hash_serializer.py
│   │   └── test_type_serializer.py
│   └── state_controllers/test_state_controller.py
│
└── integration_tests/                             # integration tier — interactive, real APIs
    └── proxai_api_test.py                         # @integration_block workflow
```

The test suite maps one-to-one to `src/proxai/` subdirectories.
Adding a new subsystem means creating a matching `tests/<name>/`
directory with at least one `test_*.py` and (usually) no new
top-level entries. Tests at the root (`test_proxai.py`,
`test_client.py`, `test_type_utils.py`) correspond to source at
the root; everything else mirrors a subfolder.

`integration_tests/` is deliberately *not* under `tests/` —
`pytest` collection never picks it up (no `conftest.py`, no
`__init__.py` discovery), and it is run as a standalone script
that writes artifacts under `~/proxai_integration_test/`. See §5.

---

## 2. Two tiers — unit tests and the integration harness

| Dimension | `tests/` (unit) | `integration_tests/proxai_api_test.py` (integration) |
|---|---|---|
| Runner | `poetry run pytest` | `poetry run python3 integration_tests/proxai_api_test.py` |
| Network | None — all providers mocked | Real HTTP — every provider SDK called |
| API keys required | No (env stripped by `monkeypatch`) | Yes — every provider key the test exercises |
| Determinism | Deterministic; runs on every CI | Non-deterministic; manual gate before release |
| Runtime | Seconds to low minutes for full suite | Several minutes; prompts for interactive input |
| Artifacts | None; in-memory | `~/proxai_integration_test/test_<id>/` (state, cache, logs) |
| When to add | Every PR that changes behavior | When wire-format changes need provider validation |
| Failure semantics | Any failure blocks merge | Flaky provider = investigate, not a block |

The split is not negotiable: if your test needs a real API key to
run, it belongs in `integration_tests/`, not in `tests/`. Unit
tests must pass offline on every developer's laptop without
secrets.

---

## 3. The `run_type=TEST` mock pattern

Every provider connector has two sibling files under
`src/proxai/connectors/providers/`:

```
providers/
├── openai.py               # real connector
├── openai_mock.py          # structural duck-typed mock client
├── claude.py
├── claude_mock.py
├── gemini.py
├── gemini_mock.py
├── mistral.py
├── mistral_mock.py
└── ...                     # one _mock.py per real connector
```

The switch between them lives in `ProviderConnector.api`
(`provider_connector.py:144-151`):

```python
@property
def api(self) -> Any:
    if not getattr(self, '_api', None):
        if self.run_type == types.RunType.PRODUCTION:
            self._api = self.init_model()
        else:
            self._api = self.init_mock_model()
    return self._api
```

`run_type=TEST` is the only non-production value — there is no
"staging" or "dev" mode. Tests construct connectors with
`ProviderConnectorParams(run_type=types.RunType.TEST, ...)` and
every SDK call then resolves to the mock client's method.

### 3.1 The mock contract

Each mock class is a **structural duck type** of the real SDK
client. `gemini_mock.py` looks like this:

```python
class _MockResponse:
    text: str
    def __init__(self):
        self.text = 'mock response'

class _MockModel:
    def generate_content(self, *args, **kwargs) -> _MockResponse:
        return _MockResponse()

class GeminiMock:
    """Mock Gemini API client for testing."""
    models: _MockModel
    def __init__(self):
        self.models = _MockModel()
```

Three rules follow from this style:

- **The mock's public shape mirrors the SDK's.** Every attribute
  the real connector reads via `self.api.<path>` (for Gemini:
  `self.api.models.generate_content(...)`) must exist on the
  mock. If the connector adds a new SDK call, the mock must grow
  the matching method.
- **The return shape mirrors what the connector parses.** Mocks
  return classes with the attributes / fields the connector
  reads (e.g., `response.choices[0].message.content` for OpenAI,
  `response.text` for Gemini). Anything the connector doesn't
  read can be omitted. Use bare Python classes with typed
  attributes — not dataclasses, not `types.SimpleNamespace`.
- **Mocks never raise on `*args, **kwargs`.** A real SDK may
  reject an unknown keyword; the mock must silently accept every
  call signature the connector could produce. This avoids
  coupling the mock to the current executor body — adding a new
  kwarg in the connector never breaks the mock.

Mocks return a single, deterministic response shape. They are
**not** meant to simulate error responses, rate limits, or
provider-specific edge cases — tests that need those behaviors
construct custom connectors (see §4.3 `_CaptureTestConnector`) or
patch `_safe_provider_query` directly.

### 3.2 The file-operations mock dispatch

The `FilesManager` subsystem has its own TEST-mode switch via
the mock dispatch tables in
`src/proxai/connectors/file_helpers.py`:

```python
MOCK_UPLOAD_DISPATCH   = {p: mock_upload   for p in _MOCK_PROVIDERS}
MOCK_REMOVE_DISPATCH   = {p: mock_remove   for p in _MOCK_PROVIDERS}
MOCK_LIST_DISPATCH     = {p: mock_list     for p in _MOCK_PROVIDERS}
MOCK_DOWNLOAD_DISPATCH = {p: mock_download for p in _MOCK_PROVIDERS}
```

The mock upload returns a fake `file_id` (`mock-file-<hex>`) so
the test flow exercises the same code path as production — the
downstream executor still reads `provider_file_api_ids[provider]`
and builds a `file_id` content block. See `files_internals.md`
§2.1 for details. The rule: mock dispatches exist so tests can
exercise the `file_id` code branch without a real upload; if a
test needs to verify provider wire format, it goes in the
integration harness.

---

## 4. Unit test conventions

### 4.1 Style rules

Tests are held to most of ruff's rules, with per-folder
relaxations in `pyproject.toml`:

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "D101",   # Missing docstring in public class
    "D103",   # Missing docstring in public function
]
```

Class and function docstrings are *not* required in tests —
class names (`TestGetCachedResult`) and function names
(`test_cache_hit_returns_result_with_refreshed_timestamps`) are
the documentation. Module-level docstrings are still required
(D100 is not relaxed), and should briefly describe the test
file's focus and the test-class organization.

Indentation, column width, and import ordering follow the
project-wide rules: 2-space indent, 80-col lines, isort with
`proxai` as first-party.

### 4.2 Shared fixtures via `conftest.py`

`tests/conftest.py` runs once per pytest session and:

1. **Strips `PROXDASH_API_KEY` from the environment** before any
   test imports happen (`pytest_configure` at
   `conftest.py:68-72`). Stored and restored in
   `pytest_unconfigure` so CI jobs that had the key set don't
   lose it after tests run.
2. **Builds a shared `model_configs_instance`** and attaches it
   to `pytest` itself so tests can access it via
   `pytest.model_configs_instance`. The registry is built from
   the curated `example_proxdash_model_configs.json` (a stable
   reference config) *plus* three programmatically-registered
   mock provider/model configs: `mock_provider/mock_model`,
   `mock_failing_provider/mock_failing_model`,
   `mock_slow_provider/mock_slow_model`.

The rationale (from the `conftest.py` docstring) is that the
bundled `v1.3.x.json` registry is a moving-target production
snapshot — its churn would silently drift tests that hard-code
model names. `example_proxdash_model_configs.json` is
explicitly stable for test use. Mock providers are test
scaffolding, not production models, so they are registered
programmatically rather than polluting the reference JSON.

### 4.3 Per-file fixtures

Common patterns that repeat across test files:

```python
@pytest.fixture(autouse=True)
def setup_test(monkeypatch, requests_mock):
    """Clean env keys and stub the ProxDash key-verification endpoint."""
    monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
    for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
        for api_key in api_key_list:
            monkeypatch.setenv(api_key, 'test_api_key')
    requests_mock.get(
        'https://proxainest-production.up.railway.app/ingestion/verify-key',
        text='{"success": true, "data": {"permission": "ALL"}}',
        status_code=200,
    )
    yield
```

Patterns visible above:

- **`autouse=True`**: the fixture runs before every test in the
  file with no explicit mention at the test site. Used for setup
  that *every* test needs (env, HTTP stubs, singleton reset).
- **`monkeypatch`**: scoped env-var manipulation. Every env var
  set with `monkeypatch.setenv` is reverted after the test — do
  NOT use `os.environ[...]=...` directly, it leaks.
- **`requests_mock`**: from `requests-mock` package. Stubs
  `requests.*` calls at the URL level, so any test that
  accidentally triggers a real HTTP call fails loudly ("no mock
  address matches"). This is the project's primary guard against
  tests accidentally hitting the network.
- **`tmp_path`**: pytest-builtin that provides a fresh temp
  directory per test. Used throughout cache and file tests
  (`cache_options=types.CacheOptions(cache_path=str(tmp_path))`).
  Cleaned up automatically.
- **`yield`**: runs teardown code after the test. When the
  fixture needs to restore state (e.g.,
  `px.reset_state()` before and after), use the
  `setup → yield → teardown` pattern.

### 4.4 Singleton reset for `px.*` tests

Tests that touch the module-level `px.*` façade must reset the
default client:

```python
@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Strip provider keys, set fast mocks, reset the px singleton each test."""
    monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
    for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
        for api_key in api_key_list:
            monkeypatch.delenv(api_key, raising=False)
    for k, v in _MOCK_KEYS.items():
        monkeypatch.setenv(k, v)
    px.reset_state()
    yield
    px.reset_state()
```

The module-level `_DEFAULT_CLIENT` persists across tests by
default. Without the reset, a `px.connect(...)` in test A would
color test B's default client and cause cross-test order
dependence. The pre-yield and post-yield `reset_state()` calls
ensure a clean slate in both directions.

### 4.5 Test class organization

Tests are grouped into `TestName` classes by responsibility,
with a module docstring enumerating the groups. Example from
`tests/connectors/test_provider_connector.py`:

```python
"""Tests for ProviderConnector.

The request-pipeline base class. Every provider subclass inherits this;
generate() is the one public entry point. Tests are organized by
responsibility:

  - Subclass contract validation (__init_subclass__)
  - Token map validation
  - Helpers: JSON extraction, token count, cost estimation
  - Safe execution wrapper
  - ExecutorResult + keep_raw_provider_response contract
  - Endpoint selection
  - Cache integration
  - Auto-upload media
  - generate() orchestration (the main public entry point)
  - Feature-tag rollup introspection
"""
```

Each bullet maps to a `TestXYZ` class in the file. Running one
group in isolation is cheap:

```bash
poetry run pytest tests/connectors/test_provider_connector.py::TestGetCachedResult -x
poetry run pytest tests/connectors/test_provider_connector.py::TestGenerate -x
```

Methods inside the class follow the pattern
`test_<behavior>_<condition>` — e.g.,
`test_cache_hit_returns_result_with_refreshed_timestamps`,
`test_skip_cache_disables_write`. Avoid generic names like
`test_basic` or `test_it_works`.

### 4.6 Synthetic connectors for contract tests

For testing base-class behaviors (like
`ProviderConnector.__init_subclass__` or the
`ExecutorResult` contract) without tying a test to a specific
provider, define a minimal synthetic connector inline in the
test file:

```python
class _CaptureTestConnector(provider_connector.ProviderConnector):
    PROVIDER_NAME = 'capture_test_provider'
    PROVIDER_API_KEYS = ['CAPTURE_TEST_KEY']
    ENDPOINT_PRIORITY = ['generate.text']
    ENDPOINT_CONFIG = {
        'generate.text': types.FeatureConfigType(
            prompt=types.FeatureSupportType.SUPPORTED,
        ),
    }
    ENDPOINT_EXECUTORS = {'generate.text': '_generate_text_executor'}

    def init_model(self):        return None
    def init_mock_model(self):   return None
    def _generate_text_executor(self, query_record):
        ...
```

Two conventions: the class name starts with `_` (not importable
from outside the test module), and `init_model` /
`init_mock_model` return `None` because the connector body never
reads `self.api` — the executor uses a closure directly. See
`tests/connectors/test_provider_connector.py:160-221` for the
full pattern including the paired `_FailingCaptureTestConnector`.

---

## 5. The integration harness

`integration_tests/proxai_api_test.py` is a single-file,
interactive harness that exercises every public `px.*` surface
against real provider APIs. It is run manually, typically before
a release or after a non-trivial feature change that touches
wire format.

### 5.1 Running the harness

```bash
# Resume the latest test (re-run only blocks that haven't been checkpointed)
poetry run python3 integration_tests/proxai_api_test.py

# Start a new numbered test run
poetry run python3 integration_tests/proxai_api_test.py --mode new --print-code

# Resume a specific test run
poetry run python3 integration_tests/proxai_api_test.py --mode 42

# Run against production ProxDash (default is local dev at :3001)
poetry run python3 integration_tests/proxai_api_test.py --env prod --print-code

# Non-interactive — don't wait for user "press enter"
poetry run python3 integration_tests/proxai_api_test.py --auto-continue
```

The `--mode` flag controls state: `latest` resumes the most
recent run from `~/proxai_integration_test/test_<id>/`
(checkpointed blocks are skipped); `new` allocates a fresh
directory. `--print-code` dumps each block's source before
running it, which makes the harness double as live
documentation.

### 5.2 The `@integration_block` decorator

Every test step is wrapped in `@integration_block`
(`proxai_api_test.py:109-135`):

```python
@integration_block
def local_proxdash_connection(state_data):
    px.reset_state()
    px.connect(
        experiment_path=_EXPERIMENT_PATH,
        logging_options=px.types.LoggingOptions(
            logging_path=_ROOT_LOGGING_PATH,
        ),
        cache_options=px.types.CacheOptions(
            clear_model_cache_on_connect=False,
            clear_query_cache_on_connect=True,
            cache_path=_ROOT_CACHE_PATH,
        ),
        ...
    )
    return state_data
```

The decorator:

- Checkpoints `state_data` to `{test_path}/{func.__name__}.state`
  on success. Subsequent runs skip the block unless
  `force_run=True`.
- Prompts the operator to "press Enter to continue" after each
  block (unless `--auto-continue` is set) — the operator visually
  verifies the output against the block's comments.
- Optionally prints the block's source via
  `inspect.getsource(func)` before execution (`--print-code`), so
  the output doubles as a readable trace of "what the library
  did and how."

### 5.3 When to add an integration block

Add a new `@integration_block` when one of these is true:

- You added a new `px.*` public method that doesn't have
  coverage in the harness.
- You changed a wire format (new provider, new endpoint key,
  new request parameter that reaches the SDK).
- You added a ProxDash interaction (upload, confirm, list)
  that the harness doesn't exercise.

Do **not** add integration blocks for pure-internal refactors
or coverage gaps in the unit tier — that's what `tests/` is for.
The integration harness is load-bearing manual gate; keep it
focused on behavior that only a real API can validate.

### 5.4 The `main()` entry point

`proxai_api_test.py:main()` is a flat list of `state_data =
block_name(state_data=state_data)` calls — no loops, no
conditionals. Every block appears there explicitly, in the
order operators are expected to validate them. Skipped blocks
pass `skip=True`; force-reruns pass `force_run=True`. Adding a
new block means adding its name to `main()` at the right
position.

---

## 6. Running tests

### 6.1 Full suite

```bash
poetry run pytest                              # every unit test
poetry run pytest -q                           # quiet; no per-test dots
poetry run pytest --tb=short                   # short tracebacks
poetry run pytest -x                           # stop on first failure
poetry run pytest -k "cache and not pydantic"  # keyword filter
```

The full suite is pure-Python and hits no network — it should
complete in seconds to low minutes.

### 6.2 Targeted

```bash
# One file
poetry run pytest tests/connectors/test_feature_adapter.py

# One class
poetry run pytest tests/connectors/test_feature_adapter.py::TestAdaptInputFormat

# One test
poetry run pytest tests/connectors/test_feature_adapter.py::TestAdaptInputFormat::test_drops_on_best_effort -x
```

The `-x` flag is especially useful here — for a targeted run
you almost always want to stop on the first failure.

### 6.3 Style checks

```bash
poetry run ruff check src tests                # lint
poetry run yapf -i -r src tests                # format in place
```

Tests are held to the same style rules as `src/` (see §4.1 for
the relaxations). CI runs both on every PR.

### 6.4 Never skip `poetry run`

Per `CLAUDE.md` at the repo root:

> **NEVER invoke `python` / `python3` / `pytest` / `ruff` / `yapf`
> directly.** Every Python command in this repo MUST be prefixed
> with `poetry run`.

Bare `python` resolves to the system interpreter, which does
NOT have the project dependencies installed — imports will fail
or, worse, silently pick up stale globally-installed versions of
`openai`, `anthropic`, etc. and produce misleading results. This
rule applies to every command in this doc.

---

## 7. Coverage expectations

When you land a feature, the test tier depends on what you
changed:

| Change | Unit tier (`tests/`) | Integration tier |
|---|---|---|
| New public `px.*` method / arg | Required — `test_proxai.py` and/or the relevant `test_<subsystem>.py`. | Add an `@integration_block` before release. |
| New provider connector | Required — per-feature tests in `tests/connectors/` plus a mock client. | Add at least one block exercising the new provider in `generate_text*`. |
| New endpoint on existing provider | Required — `test_provider_connector.py::TestFindCompatibleEndpoint` pattern. | Optional — add if the endpoint has novel wire format. |
| New `FeatureConfigType` field | Required — `test_feature_adapter.py` + `test_adapter_utils.py` for merge / min semantics. | Optional. |
| New `MessageContent` field | Required — `test_message_content.py` + `test_hash_serializer.py` (hash identity) + `test_type_utils.py` (equality mirror). See §7.1. | Optional. |
| Cache algorithm change | Required — `test_query_cache.py`. | Optional (caches are deterministic). |
| Pure internal refactor | Existing tests must still pass. No new tests required if behavior is unchanged. | No. |

### 7.1 The hash / equality parallel rule

Any change to `QueryRecord` / `MessageContent` fields touches
two places that must stay in lockstep:

- `src/proxai/serializers/hash_serializer.py` — `get_query_record_hash`
- `src/proxai/type_utils.py` — `is_query_record_equal` and
  `_normalize_chat_for_comparison`

Both are tested:

- `tests/serializers/test_hash_serializer.py` verifies the hash
  includes / excludes each field as intended.
- `tests/test_type_utils.py` verifies the equality check
  mirrors the hash's exclusions.

Whenever you add a field to one, add a test to both. A hash that
includes a field the equality check excludes produces silent
`CACHE_NOT_MATCHED` on every call. See `cache_internals.md` §5
for the full contract.

### 7.2 Mock completeness

When you add an SDK call to a production connector (new
attribute / method accessed on `self.api`), the paired
`*_mock.py` must grow a matching mock so `run_type=TEST` still
constructs a working connector. Tests that run against the
connector will fail at `AttributeError: 'OpenAIMock' object has
no attribute 'new_method'` — fix the mock, don't patch the test.

### 7.3 Structural-contract tests

Certain invariants are enforced at import time:

- `ProviderConnector.__init_subclass__` validates every
  subclass has the five required class attributes and that their
  keys match up. Covered by
  `test_provider_connector.py::TestInitSubclassContract`.
- `FilesManager` capability tables
  (`UPLOAD_SUPPORTED_MEDIA_TYPES` vs.
  `REFERENCE_SUPPORTED_MEDIA_TYPES`) must not declare an upload
  type that can't be referenced in generate — otherwise
  `is_upload_supported` would uselessly upload and drop the
  file_id. Covered by `test_files.py`.

When you add a new invariant of this shape, add a structural
test rather than relying on runtime checks.

---

## 8. Inputs the test conventions do not cover

A few things look like they should be here but live elsewhere:

- **CI configuration** is outside the repo's current set of
  Layer A docs. If GitHub Actions / other CI adds project-specific
  test gates, they should be documented in a dedicated ops file.
- **Hypothesis / property-based testing** is not currently used.
  Tests rely on concrete scenarios; there is no property-test
  framework in the dependency set.
- **Benchmarks** — no performance suite exists. Timing
  assertions in the integration harness (`assert end_time -
  start_time < 1`) are the only explicit performance checks, and
  they guard against regressions in `list_models()` cache hits,
  not general throughput.
- **Snapshot testing** — no snapshot / golden-file framework.
  Round-trip tests (`to_dict` / `from_dict`, `encode_cache_record`
  / `decode_cache_record`) substitute for snapshots where the
  serialized shape matters.

---

## 9. Where to read next

- `adding_a_new_provider.md` §11 "The mock model" — the
  executor-side contract mocks must satisfy, and the minimum
  duck-typed shape for a new provider's `*_mock.py`.
- `files_internals.md` §2.1 "Run-type dispatch" — why mock file
  helpers return fake `file_id`s instead of empty strings.
- `CLAUDE.md` — the project-wide `poetry run` rule and the
  one-line summaries of the canonical test commands.
- `pyproject.toml` `[tool.ruff]` / `[tool.yapf]` blocks — the
  style rules tests are held to and their per-folder
  relaxations.
- Any specific `tests/<folder>/test_*.py` file — when you need
  a template for how a new test file should look, copy the
  closest sibling and replace the body.
