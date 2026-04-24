# Sanity Check Before Merge

Source of truth: the CI workflow at `.github/workflows/ci.yml`
(what actually blocks merge), the test suite under `tests/` (what
proves behavior), the `[tool.ruff]` / `[tool.yapf]` / `[tool.poetry]`
sections of `pyproject.toml` (the style and packaging rules), and
`CLAUDE.md` at the repo root (the project-wide rules CI does not
mechanically enforce). If this document disagrees with those
files, the files win — update this document.

This is the definitive pre-merge audit for changes to ProxAI.
Every item below is either enforced by CI (and listed so you know
what's being checked) or a consistency invariant CI cannot see
that past regressions have shown to matter. Work through the
relevant sections before you mark a change ready for review —
skipping a conditional section you didn't need is fine; skipping
one you did need is the most common way behavior drift ships to
users.

See also: `testing_conventions.md` (what a "good test" looks
like on this project, the unit vs. integration tier split, and
the fixture patterns that this audit calls "write a test for
it"); `dependency_graph.md` (the layering rule behind §4.1 in
this doc); `CLAUDE.md` (the shorter project-wide contract this
audit references repeatedly).

---

## 1. Audit flow (current)

```
Pre-merge audit
│
│   # Gate 1 — CI-enforced (will block merge on a PR)
├── §2  Lint + format
│     ├── ruff check src/ tests/
│     └── yapf --diff --recursive src/ tests/
├── §3  Unit tests on Python 3.10 / 3.11 / 3.12
│     └── poetry run pytest tests/ -v
│
│   # Gate 2 — conventions CI does not enforce mechanically
├── §4  Dependency layering (types → caches → connectors → client)
├── §5  `poetry run` everywhere (no bare python)
├── §6  CLAUDE.md contract spot-checks
│
│   # Gate 3 — conditional on what you changed
├── §7  If you changed public behavior         → docs + tests
├── §8  If you added a provider connector       → mock + config + registry
├── §9  If you added a MessageContent / QueryRecord field
│                                                → hash + equality mirror
├── §10 If you edited a StateControlled class   → state propagation
├── §11 If you changed wire format               → integration harness
├── §12 If you touched the cache                 → freshness + LRU
│
│   # Gate 4 — documentation currency
├── §13 Layer A docs reflect behavior changes
├── §14 documentation_outline.md in sync
│
│   # Gate 5 — release-adjacent (usually skip on feature PRs)
└── §15 Packaging / deps / version bumps
```

Not every gate applies on every PR. The `Gate 3` sections have
trigger conditions at the top — if the condition is false, skip
the section. The sections themselves flow roughly top-down; a
change that hits §7 also usually hits §13, so reading in order
saves double-work.

---

## 2. Lint + format (CI enforced)

```bash
poetry run ruff check src/ tests/
poetry run yapf --diff --recursive src/ tests/
```

Both must be clean. CI runs them as the first gate (`ci.yml`
`lint` job); a failure there blocks the `test` job.

- Ruff enforces pycodestyle, pyflakes, isort, google-style
  docstrings (`D` rules), pyupgrade, bugbear, and the
  `proxai`-first-party isort order.
- yapf enforces the google base style with 2-space indent,
  80-column line length, and the project-specific
  `dedent_closing_brackets=true` / `coalesce_brackets=true`
  rules.
- Per-file relaxations live in `[tool.ruff.lint.per-file-ignores]`.
  `tests/`, `integration_tests/`, `examples/`, and `docs/` have
  `D101` / `D103` / similar relaxed — do not re-tighten them
  without reason.

If ruff complains about a legitimate-looking rule, check the
per-file ignores before disabling the rule inline. Inline
`# noqa: <code>` is reserved for genuine one-off exceptions, not
a first-line fix.

---

## 3. Unit tests (CI enforced)

```bash
poetry run pytest                          # the one-shot command
poetry run pytest -q                       # when you want less output
poetry run pytest -x --tb=short            # when the first failure is enough
```

CI runs `poetry run pytest tests/ -v` on Python 3.10, 3.11, and
3.12 — failures on any of the three block merge. The full suite
is pure-Python and hits no network, so "passes on my laptop"
should match "passes in CI" modulo Python-version differences.

Two practical notes:

- **Test targeted first, full suite last.** When iterating, use
  `poetry run pytest tests/<file>.py::<Class>::<test> -x` to
  close the loop fast; only run the whole suite before pushing.
- **Do not mark CI failures as flaky.** Every test in `tests/` is
  deterministic (no real network, no sleep-based synchronization,
  no wall-clock dependencies beyond freshness checks that use
  `datetime.datetime.now()` at insertion time). If a test is
  flaky, it's a bug — investigate rather than re-running CI.

See `testing_conventions.md` for the full running guide, class-
grouping conventions, and the shared fixture contract.

---

## 4. Dependency layering (cannot be CI-enforced directly)

ProxAI's layering contract
(`dependency_graph.md`, bottom → top):

```
Layer 0: types, stat_types, type_utils, state_controllers, serializers
Layer 1: caching/query_cache, caching/model_cache, logging/utils
Layer 2: connectors/provider_connector, connections/proxdash, connectors/model_configs
Layer 3: connectors/model_registry ← connectors/providers/*
Layer 4: connections/available_models
Layer 5: proxai.py  (public API)
```

Check whenever you add a new `import proxai.<something>` line:

- Lower layers must not import from higher layers. A `types.py`
  edit that imports `provider_connector` is immediately wrong.
- Providers under `connectors/providers/` must only import
  `types`, their own `*_mock` sibling, `provider_connector`, and
  (when needed) `model_configs`.
- Circular imports break lazy-loading in `proxai.py` and show up
  as cryptic `ImportError: cannot import name 'X' from partially
  initialized module` at import time — not test runtime.

If you need a symbol from a higher layer, the usual fix is to
relocate the symbol to a lower layer or inject it at construction
time.

---

## 5. `poetry run` everywhere

From `CLAUDE.md`:

> **NEVER invoke `python` / `python3` / `pytest` / `ruff` /
> `yapf` directly.** Every Python command in this repo MUST be
> prefixed with `poetry run`.

Self-check before committing:

- Any new script you wrote that ships in the repo (examples,
  tools, integration blocks) uses `poetry run python3 ...` in
  its docstring command lines.
- Any new shell snippet in a doc uses `poetry run`.
- Any CI step you added runs under poetry (`poetry run <cmd>`).

Bare `python` silently picks up the system interpreter, which
does NOT have the project dependencies — imports will fail at
best and silently resolve to stale global versions of `openai`,
`anthropic`, etc. at worst.

---

## 6. `CLAUDE.md` contract spot-checks

A handful of project rules from `CLAUDE.md` that CI cannot catch
but reviewers will. Skim each before pushing:

| Rule | What to check |
|---|---|
| 2-space indent everywhere (Python, JSON, Markdown code blocks) | yapf handles Python; manually verify JSON and Markdown |
| 80-column line length | Check long-looking prose in new code / docstrings |
| Google-style docstrings | New public functions / classes have them; the ignored `D100` / `D104` / `D105` / `D107` / `D102` rules are NOT fixed with stub docstrings |
| `types.py` is the foundation layer | No imports from the rest of `proxai` into `types.py` |
| Provider `BEST_EFFORT` only where degrading is safe | Review every new `FeatureSupportType.BEST_EFFORT` — `SUPPORTED` on an endpoint that silently drops fields is the specific anti-pattern |
| Never declare a feature `SUPPORTED` that will silently drop | If the executor body doesn't read `query_record.<feature>`, the config must not claim `SUPPORTED` |

If you find yourself adding a `# noqa`, a stub docstring to
satisfy a relaxed rule, or a `SUPPORTED` level without a
matching executor branch — stop and reconsider. Those three are
the most common regressions a reviewer will flag.

---

## 7. Public behavior changes → update tests AND docs

**Trigger:** you added, removed, or changed a public surface —
any `px.*`, `px.Client`, `px.Chat`, `px.files`, `px.models`
method, field, or argument; any `CallRecord` / `QueryRecord` /
`ResultRecord` field; any `CacheOptions` / `ConnectionOptions` /
`ProviderCallOptions` / `DebugOptions` flag.

Checklist:

- [ ] Unit test covers the new behavior on the happy path.
  `test_proxai.py` (module facade) and/or
  `test_client.py` (ProxAIClient) are the usual homes; deeper
  tests live in the subsystem's folder (`test_query_cache.py`,
  `test_feature_adapter.py`, etc.).
- [ ] Unit test covers the failure mode. Every `raise` you added
  has a test that exercises it. Every `if ... return None` /
  `CacheLookFailReason` / `ResultStatusType.FAILED` branch has a
  test.
- [ ] `docs/user_agents/api_guidelines/<file>.md` reflects the
  new surface — the relevant file (`px_client_api.md`,
  `px_generate_api.md`, `px_models_api.md`, `px_files_api.md`,
  `px_chat_api.md`, `call_record.md`, `cache_behaviors.md`,
  `raw_provider_response.md`,
  `provider_feature_support_summary.md`) has the new
  parameter / field / default / error row in its tree diagram
  and the relevant numbered section.
- [ ] Default values are stated explicitly
  ("Default `None`.", "Default: empty list.") — the
  `px_generate_api.md` style, not "defaults to the usual value".
- [ ] Errors table updated if you added a raise path with a
  caller-facing exception.

See `testing_conventions.md` §4 for the test style; see the
specific `user_agents/api_guidelines/*.md` file for the surface
you changed to match its structure.

---

## 8. New provider connector → five edits

**Trigger:** you added a new file under
`src/proxai/connectors/providers/`.

Checklist (every item is enforced at import time via
`__init_subclass__` — skipping one breaks every test run):

- [ ] Connector class has all five required attributes:
  `PROVIDER_NAME`, `PROVIDER_API_KEYS`, `ENDPOINT_PRIORITY`,
  `ENDPOINT_CONFIG`, `ENDPOINT_EXECUTORS`.
- [ ] Keys across `ENDPOINT_PRIORITY` / `ENDPOINT_CONFIG` /
  `ENDPOINT_EXECUTORS` match exactly. The validator compares
  them on subclass registration.
- [ ] `init_model()` returns the real SDK client;
  `init_mock_model()` returns the paired `*_mock.py` class.
- [ ] Sibling `*_mock.py` file exists with the duck-typed shape
  the executor reads on `self.api.<path>` — every method the
  executor calls is mocked, and mocks accept `*args, **kwargs`.
- [ ] Three registration edits are in:
  - `src/proxai/connectors/model_registry.py::_MODEL_CONNECTOR_MAP`
    — add the connector.
  - `src/proxai/connectors/model_configs.py::PROVIDER_KEY_MAP`
    — add the env-var names.
  - `src/proxai/connectors/model_configs_data/<provider>.json`
    — add at least one model with pricing + features +
    metadata.
- [ ] `pyproject.toml` `[tool.poetry]` `include` list covers the
  new JSON file (`"src/proxai/connectors/model_configs_data/*.json"`
  already matches, but double-check if you introduced a new
  glob).
- [ ] The support levels on each endpoint are honest — every
  `SUPPORTED` feature is actually read by the executor, and
  every `BEST_EFFORT` has a documented degradation.

See `adding_a_new_provider.md` for the full walkthrough
(class-attribute contract in §3, registration in §12, pitfalls in
§16); `testing_conventions.md` §3.1 for the mock contract.

---

## 9. New `MessageContent` / `QueryRecord` field → hash + equality mirror

**Trigger:** you added, removed, or changed semantics of a field
on `QueryRecord`, `MessageContent`, `ParameterType`,
`OutputFormat`, `ConnectionOptions`, or any nested type the
cache hashes.

This is the #1 source of silent cache regressions.
`hash_serializer.get_query_record_hash` and
`type_utils.is_query_record_equal` MUST agree on whether a field
is part of cache identity. Checklist:

- [ ] Decide: is the new field part of cache identity?
- [ ] **If yes** — in:
  - `src/proxai/serializers/hash_serializer.py` —
    `get_query_record_hash` feeds the field into the hasher.
  - `src/proxai/type_utils.py::is_query_record_equal` — the
    field is not normalized away in `_normalize_*` helpers.
  - `tests/serializers/test_hash_serializer.py` — a test
    asserts two records with different values for this field
    hash differently.
  - `tests/test_type_utils.py` — a test asserts the equality
    check distinguishes them.
- [ ] **If no** — in:
  - `hash_serializer.py` — the field is explicitly skipped
    (not read at all, or popped in `_content_hash_dict`).
  - `type_utils.py` — the field is cleared in the
    `_normalize_*` helper so equality mirrors.
  - Both places are documented in the module docstring's
    "Currently excluded from cache identity" list.
  - Tests assert cache identity is preserved across the
    field's variation.

The failure mode when one side drifts: hash matches but equality
doesn't (or vice versa), and every cache lookup returns
`CACHE_NOT_MATCHED`. The cache "works" but every call is a miss.
See `cache_internals.md` §5 for the full contract.

---

## 10. StateControlled class edits → propagation check

**Trigger:** you edited a class that inherits from
`state_controller.StateControlled` — `ProxAIClient`,
`ProviderConnector`, `QueryCacheManager`, `ModelCacheManager`,
`ProxDashConnection`, `ModelConfigs`, `AvailableModels`,
`ApiKeyManager`, `FilesManager`, or a new subclass.

Checklist:

- [ ] New public fields are stored via `set_property_value` /
  `set_state_controlled_property_value` (for nested
  StateControlled) and exposed via `@property` getters /
  setters, not as plain attributes.
- [ ] The corresponding `*State` dataclass in `types.py` carries
  the new field (if it must survive serialization).
- [ ] `handle_changes` validates any cross-field invariants the
  new field introduces. Do not put validation in the setter —
  it breaks deserialization, which calls
  `set_property_value_without_triggering_getters`.
- [ ] For nested StateControlled values, a
  `<property_name>_deserializer` method exists and round-trips
  the nested state container back into a live instance.
- [ ] `tests/state_controllers/test_state_controller.py` or the
  class's own test file exercises the new field through a
  round-trip (`get_state()` → `load_state()`).

See `state_controller.md` for the full contract — corner cases
around nested state, getter functions, and the handle_changes
ordering are easy to get wrong and silently break
multiprocessing.

---

## 11. Wire-format changes → integration harness

**Trigger:** you changed anything the provider SDK sees on the
wire — a new request body field, a new endpoint call, a new
response-parsing path, a provider File API upload flow, a new
capability table entry under `file_helpers.py`.

Unit tests don't validate wire format (mocks accept anything).
The integration harness is the only gate that proves real
providers accept the new shape.

Checklist:

- [ ] Add an `@integration_block` to
  `integration_tests/proxai_api_test.py` exercising the new
  path. See `testing_conventions.md` §5.3 for the trigger
  criteria and the decorator contract.
- [ ] Register the block's name in `main()` at the right
  sequence position.
- [ ] Run the harness end-to-end against at least one real
  provider:
  ```bash
  poetry run python3 integration_tests/proxai_api_test.py --mode new --print-code
  ```
- [ ] If the change affects ProxDash upload / confirm / list
  flows, run against a real ProxDash backend
  (`--env prod` or a local dev ProxDash on `localhost:3001`).

The integration harness is *not* part of CI — you run it
yourself before declaring a wire-touching change ready to merge.
A flaky result is a signal to investigate, not a reason to skip.

---

## 12. Cache-touching changes → freshness + LRU

**Trigger:** you edited `src/proxai/caching/query_cache.py`,
`src/proxai/caching/model_cache.py`, or any read / write path in
`provider_connector.py::_get_cached_result` /
`_update_cache`.

Checklist:

- [ ] Every read path that decodes a `CacheRecord` from disk
  calls `_check_cache_record_is_up_to_date` before acting
  on it. Freshness invariant in
  `cache_internals.md` §4.
- [ ] Every mutation to `_light_cache_records` goes through
  `_update_cache_record` (updates the shard heap, write
  tombstone, etc. atomically). Hand-rolled equivalents drift.
- [ ] Status-machine checks: `look()` returns
  `CacheLookResult(cache_look_fail_reason=CACHE_UNAVAILABLE)`
  on non-`WORKING` status; `cache()` logs a warning and
  no-ops. No `ValueError` on misconfiguration.
- [ ] Hash + equality pact upheld (§9 above) — any new field
  the cache touches must appear in both.
- [ ] Tests added to `tests/caching/test_query_cache.py` /
  `test_model_cache.py` for the new behavior. If the change
  affects the LRU eviction path, assert on the heap size
  after a push.

---

## 13. Layer A docs reflect behavior changes

**Trigger:** any of §7 / §8 / §9 / §10 / §11 / §12 fired.

Public-facing changes land in `docs/user_agents/`; internal
contract changes land in `docs/developer_agents/`. Checklist:

- [ ] `user_agents/api_guidelines/<file>.md` — the relevant API
  reference (`px_client_api.md`, `px_generate_api.md`, etc.)
  has the new surface in its tree + numbered section.
- [ ] `user_agents/api_guidelines/call_record.md` — if any
  `CallRecord` / `QueryRecord` / `ResultRecord` field changed.
- [ ] `user_agents/api_guidelines/cache_behaviors.md` — if any
  `CacheOptions` / `ConnectionOptions` cache flag changed, or
  you added a new `CacheLookFailReason`.
- [ ] `user_agents/api_guidelines/provider_feature_support_summary.md`
  — if provider feature support changed.
- [ ] `user_agents/troubleshooting.md` — if you added a new
  error path users are likely to hit.
- [ ] `developer_agents/<relevant_doc>.md` — the internals doc
  for the subsystem you edited. `feature_adapters_logic.md`,
  `chat_export_logic.md`, `cache_internals.md`,
  `files_internals.md`, `state_controller.md`,
  `adding_a_new_provider.md`, `testing_conventions.md`.

A `developer_agents/` doc pointing at `X` is a claim that `X`
exists. Rename, move, or delete `X` without updating the doc
and you ship a broken reference.

Per-doc audit: every `See §X` cross-link resolves, every
`src/proxai/<file>.py:<lineno>` citation still points at the
relevant code (line numbers drift; keep them close to the
truth or drop them and name the function instead).

---

## 14. `documentation_outline.md` in sync

**Trigger:** you added, renamed, or substantively changed a
Layer A doc.

Checklist:

- [ ] **§1 "Complete repo and docs layout"** — the new doc
  appears in the tree. `*` markers reflect which files are
  drafted (have real content) vs. placeholders.
- [ ] **§2.3 / §2.4 / §2.7** — the scope cell for the file you
  wrote matches what you wrote. One-line concrete description,
  no hedging.
- [ ] **§7.1 Completed** — a bullet for the new / revised doc
  with date (absolute — `2026-04-23`, not "Thursday"), a
  one-paragraph summary of what the doc covers, and the
  cross-reference updates applied elsewhere.
- [ ] **§7.2 Staging files** — if you consumed content from
  `docs/development/<file>.md`, mark the row as drained /
  partial. Delete the staging file if fully consumed.
- [ ] **§8 Design decisions** — if anything you learned while
  writing contradicts a decision recorded here, stop and flag
  to the maintainer. Do not silently change a design decision
  via a doc write.

Skipping the outline update is the single most common way
future agents form confident-but-false beliefs about the docs
tree. The `px-create-doc` skill enforces this in its §8, but
the same discipline applies for any doc write.

---

## 15. Packaging / deps / version bumps

**Trigger:** you changed `pyproject.toml`, `poetry.lock`, or
anything that affects the published wheel.

Checklist:

- [ ] `pyproject.toml` `[tool.poetry] include` covers every
  non-`.py` resource the library ships — JSON model configs,
  bundled skills, any new templated text. If you added a JSON
  file under `src/proxai/connectors/model_configs_data/`,
  verify it's inside the glob.
- [ ] New runtime dependency has a documented reason, sits
  under `[tool.poetry.dependencies]`, pins an acceptable
  version range.
- [ ] Dev / test-only dependencies go under
  `[tool.poetry.group.dev.dependencies]`, not the main list.
- [ ] `poetry.lock` regenerated and committed alongside the
  `pyproject.toml` change (`poetry lock --no-update` for a
  minimal resolve, `poetry update <package>` when upgrading).
- [ ] If you bumped the ProxAI version, the bump is in the
  `[tool.poetry]` `version` field and the CHANGELOG is
  updated with the diff summary.

Never commit a `pyproject.toml` change without the matching
`poetry.lock` update — the next `poetry install` run will hang
or resolve something surprising.

---

## 16. Final self-review

A short mental pass, immediately before pushing:

- Run `git status` — are there untracked files you forgot to
  add? Stray `~/proxai_integration_test/` artifacts that should
  not be in the commit?
- Run `git diff --stat` — are the files touched the files you
  expected? A refactor that unexpectedly edited a provider
  connector is a signal to split the commit.
- `git log --oneline main..HEAD` — does each commit message
  describe a coherent unit of change? Squash noise commits.
- Read your own PR description once more — does it match what
  the diff does? If the description says "refactor X" but the
  diff also adds a new endpoint, the description is wrong or
  the commit should be split.

A clean commit, a focused diff, and a PR description that
matches the diff are the final sanity check CI cannot do for
you.

---

## 17. Where to read next

- `testing_conventions.md` — the full story behind §3, §7
  (writing a test), §11 (integration harness structure), and
  the `poetry run pytest` command menu.
- `dependency_graph.md` — the full layering chart §4 alludes
  to, plus the import rules for each layer.
- `adding_a_new_provider.md` — the full new-provider walkthrough
  behind §8 (class-attribute contract, registration, executor
  anatomy, pitfalls).
- `state_controller.md` — the handle_changes / deserializer
  contract behind §10.
- `cache_internals.md` — the freshness invariant and hash /
  equality pact behind §9 and §12.
- `CLAUDE.md` — the project-wide rules this audit cross-links
  repeatedly.
