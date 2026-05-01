# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

ProxAI is a lightweight abstraction layer that exposes a single Python API
(`import proxai as px`) for many foundational LLM providers (OpenAI, Anthropic
Codex, Google Gemini, Mistral, Cohere, DeepSeek, Grok/xAI, HuggingFace,
Databricks, …). The library is published to PyPI as `proxai`.

Python `>=3.10,<4.0`. Managed with Poetry.

## ⚠️ Always use `poetry run python`

**NEVER invoke `python` / `python3` / `pytest` / `ruff` / `yapf` directly.**
Every Python command in this repo MUST be prefixed with `poetry run` (e.g.
`poetry run python3 ...`, `poetry run pytest ...`, `poetry run ruff ...`).
Bare `python` resolves to the system interpreter, which does NOT have the
project dependencies installed, so imports will fail or — worse — silently
pick up stale/globally-installed versions of `openai`, `anthropic`, etc. and
produce misleading results. This rule is non-negotiable: if you catch
yourself typing `python3 something.py`, stop and prepend `poetry run`.

## Common commands

```bash
# Install / refresh dependencies
poetry install

# Cross-provider parity harness (real providers, all 9 providers in a matrix)
poetry run python3 examples/refactoring_test.py --provider openai

# Full unit test suite (fast; pure-Python, no network)
poetry run pytest

# Run a single test file / test
poetry run pytest tests/connectors/test_feature_adapter.py
poetry run pytest tests/connectors/test_feature_adapter.py::TestName::test_case -x

# Lint / format
poetry run ruff check src tests
poetry run yapf -i -r src tests   # in-place, google-based, 2-space indent, 80 col

# End-to-end integration tests (hits real providers + ProxDash UI; needs API
# keys). Five files, run independently. See integration_tests/README.md.
poetry run python3 integration_tests/01_models_test.py
poetry run python3 integration_tests/02_generate_test.py
poetry run python3 integration_tests/03_files_test.py
poetry run python3 integration_tests/04_proxdash_test.py
poetry run python3 integration_tests/05_runtime_test.py
# Run one named block: --test <block_name>. Skip "Press Enter" beats with
# --auto-continue (manual_check y/n prompts still need input). Switch URLs
# with --env prod.
```

`tests/` is unit tests (no network). `integration_tests/` is an interactive
human-in-the-loop suite that calls real providers and writes artifacts under
`~/proxai_integration_test/`; it is not part of `pytest` collection. The five
files are runnable independently; `_utils.py` holds shared infrastructure
(setup, decorator, manual_check, model constants, asset paths).
`examples/refactoring_test.py` is a separate cross-provider parity matrix
(stays in `examples/`); the old monolithic `integration_tests/proxai_api_test.py`
is preserved as `proxai_api_test.py.deprecated` for historical reference.

Style: ruff + yapf (google base, 2-space indent, 80 col line length). Ruff
enforces google-style docstrings (`D` rules), pyupgrade, bugbear, and isort
with `proxai` as first-party. Per-file ignores relax docstring rules for
`tests/`, `integration_tests/`, `examples/`, and `docs/` — don't re-tighten
them without reason.

## Architecture — the big picture

The request pipeline is the most important thing to internalize. Everything
hangs off `ProviderConnector.generate()` in
`src/proxai/connectors/provider_connector.py`:

```
client.generate_text(prompt, parameters, response_format)
  └─ QueryRecord built from the request
     └─ FeatureAdapter.adapt_query_record()   # drops/injects per support level
        └─ ENDPOINT_PRIORITY picks an endpoint
           └─ _<endpoint>_executor(query_record)         # per-provider code
              └─ functools.partial(SDK call)
              └─ self._safe_provider_query(partial)      # retries, error mapping
              └─ result_record.content = [MessageContent, ...]
           └─ ResultAdapter.adapt_result_record()        # transforms per response_format
           └─ usage / cost / cache / timestamp computed
  └─ CallRecord returned to user
```

Dependency layering (bottom → top) — respect this when adding imports;
circular deps will break lazy imports in `proxai.py`:

```
Layer 0: types, stat_types, type_utils, state_controllers, serializers
Layer 1: caching/query_cache, caching/model_cache, logging/utils
Layer 2: connectors/provider_connector, connections/proxdash, connectors/model_configs
Layer 3: connectors/model_registry ← connectors/providers/*
Layer 4: connections/available_models
Layer 5: proxai.py  (public API)
```

Key files to read before deep work:

- `src/proxai/types.py` — all dataclasses, `StateContainer` base, the
  `FeatureConfigType` / `ParameterConfigType` / `ResponseFormatConfigType`
  hierarchy, `QueryRecord` / `ResultRecord` / `CallRecord`.
- `src/proxai/connectors/provider_connector.py` — `ProviderConnector`,
  `generate()`, `_safe_provider_query`, and the `__init_subclass__` contract
  validator.
- `src/proxai/connectors/feature_adapter.py` and `result_adapter.py` — the
  transform steps on either side of the executor.
- `src/proxai/connectors/model_registry.py` — `_MODEL_CONNECTOR_MAP` (add new
  providers here) plus `model_configs.PROVIDER_KEY_MAP`.
- `src/proxai/client.py` — `ProxAIClient`, the single object the public
  `proxai.py` module forwards to.
- `src/proxai/proxai.py` — the module-level functional façade
  (`connect`, `generate_text`, `set_model`, `models.list_models`, …).

### Provider connectors

Each provider under `src/proxai/connectors/providers/` ships a real connector
(`openai.py`, `Codex.py`, `gemini.py`, …) and a mock sibling
(`*_mock.py`, e.g. `openai_mock.py`) used when `run_type == TEST`. Adding or
modifying a connector is governed by a formal contract — **read
`docs/development/provider_connectors.md` first**; it's the source of truth
for the five required class attributes (`PROVIDER_NAME`, `PROVIDER_API_KEYS`,
`ENDPOINT_PRIORITY`, `ENDPOINT_CONFIG`, `ENDPOINT_EXECUTORS`), endpoint-key
naming, the system-prompt patterns, and the SUPPORTED / BEST_EFFORT /
NOT_SUPPORTED taxonomy. `__init_subclass__` enforces most of this at import
time, so broken connectors fail the whole test run.

Finalized reference connectors (use these as templates, not older ones):
`openai.py`, `gemini.py`, `Codex.py`, `mistral.py`, `grok.py`, `deepseek.py`,
`cohere.py`, `huggingface.py`.

Static model/price metadata lives in
`src/proxai/connectors/model_configs_data/*.json` (packaged via
`pyproject.toml`'s `include` list — don't forget to add new JSON files there).

### StateControlled system

A handful of core objects (`ProxAIClient`, `ModelConnector`, caches,
`ProxDashConnection`, `ModelConfigs`, `AvailableModels`) inherit from
`StateControlled` in `src/proxai/state_controllers/state_controller.py`.
Their fields are stored in a single serializable `StateContainer` dataclass
so they can be passed across processes / threads and cached. Writes flow
through a `handle_changes` callback that validates and propagates state down
the object tree. If you're editing one of these classes (or adding a new
one), read `docs/development/state_controller.md` — the corner cases around
nested state, getter functions, and the handle_changes contract are easy to
get wrong and silently break multiprocessing.

### Chat / sessions

`src/proxai/chat/chat_session.py` wraps the connector pipeline with a
stateful message history. Re-exported as `proxai.Chat`.

## Conventions worth knowing

- 2-space indent everywhere (Python, JSON, Markdown code blocks where
  possible). yapf + ruff enforce 80-col line length.
- Google-style docstrings; `D100`/`D104`/`D105`/`D107`/`D102` are
  intentionally ignored — don't "fix" them by adding stub docstrings.
- `types.py` is the foundation layer and imports nothing from the rest of
  `proxai`. Keep it that way.
- Providers must only import `types`, their own `*_mock` file, and
  `provider_connector` (plus `model_configs` if needed). See
  `docs/development/dependency_graph.md`.
- Never declare a feature `SUPPORTED` on an endpoint that will silently drop
  the field — prefer `NOT_SUPPORTED` so the framework surfaces a clear error,
  or `BEST_EFFORT` if degrading is safe. (See `provider_connectors.md` §7.)
