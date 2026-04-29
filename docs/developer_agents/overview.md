# Developer Agents — Overview

Source of truth: `CLAUDE.md` at the repo root (the project-wide
contract — style rules, `poetry run` non-negotiable, dependency
layering, architectural big picture),
`src/proxai/connectors/provider_connector.py` (the request pipeline
every call flows through), `src/proxai/client.py` (`ProxAIClient` —
the single object the public `proxai.py` façade forwards to), and
`src/proxai/types.py` (every dataclass, enum, and `StateContainer`
the rest of the tree depends on). If this document disagrees with
those files, the files win — update this document.

This is the landing page for agents (and humans) modifying ProxAI
itself. It shows the request pipeline every call flows through, the
import-order layering that prevents circular dependencies, and a
decision tree from "what are you changing" to "which doc owns the
invariants you are about to touch." Read this before opening a PR —
the decision tree in §2 will point you at the doc that lists the
contracts CI does not mechanically enforce, and §4 lists the pre-
merge audit that is mandatory for every change, not optional.

See also: [`sanity_check_before_merge.md`](./sanity_check_before_merge.md)
(the pre-merge audit you must run),
[`dependency_graph.md`](./dependency_graph.md) (import layering with
the full per-file table),
[`adding_a_new_provider.md`](./adding_a_new_provider.md) (the most
common "first big change" a new contributor makes),
[`../../CLAUDE.md`](../../CLAUDE.md) (the short repo-root contract
this doc defers to rather than duplicating), and
[`../user_agents/overview.md`](../user_agents/overview.md) for the
caller-side view when you need to understand what behaviour a user
is relying on.

---

## 1. Architecture at a glance

Two diagrams you should internalize before touching `src/proxai/`:
the **request pipeline** (what happens on one `.generate()` call)
and the **dependency layering** (which files may import which).

### 1.1 Request pipeline (current)

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

Every provider call lives inside `ProviderConnector.generate` in
`src/proxai/connectors/provider_connector.py`. The adapters on
either side of the executor are what lets a single caller surface
work across wildly different provider SDKs — see
[`feature_adapters_logic.md`](./feature_adapters_logic.md) §1 for
the per-stage contract and
[`adding_a_new_provider.md`](./adding_a_new_provider.md) §6 for the
executor-author's view of what `query_record` carries on entry.

### 1.2 Dependency layering (current)

Bottom → top. Respect this order when adding imports — circular
dependencies break the lazy imports in `src/proxai/proxai.py` and
fail the whole test run on import.

```
Layer 5: proxai.py                                        (public façade)
         │
Layer 4: connections/available_models
         │
Layer 3: connectors/model_registry ← connectors/providers/*
         │
Layer 2: connectors/provider_connector,
         connections/proxdash,
         connectors/model_configs
         │
Layer 1: caching/query_cache, caching/model_cache, logging/utils
         │
Layer 0: types, stat_types, type_utils,
         state_controllers, serializers
```

Key invariant: **`types.py` imports nothing else from `proxai`.**
It is the foundation layer. Providers (Layer 3 leaves) only import
`types`, their own `*_mock` sibling, and
`connectors/provider_connector` (plus `model_configs` if they need
to read static metadata). For the full file-by-file import table
see [`dependency_graph.md`](./dependency_graph.md).

### 1.3 Subsystems to know about

Pipeline-adjacent systems worth knowing exist, each with its own
doc:

```
ProviderConnector                                     # the hot path
├── FeatureAdapter / ResultAdapter                    → feature_adapters_logic.md
├── _get_cached_result / _update_cache                → cache_internals.md
├── _auto_upload_media                                → files_internals.md §4
└── ProxDashConnection                                → (logging / upload)

ProxAIClient                                          # the user-facing object
├── QueryCacheManager      (StateControlled)          → cache_internals.md
├── ModelCacheManager      (StateControlled)          → cache_internals.md §11
├── FilesManager           (StateControlled)          → files_internals.md
├── ProxDashConnection     (StateControlled)          → (see proxdash.py)
├── ModelConfigs           (StateControlled)          → adding_a_new_provider.md §12
└── AvailableModels        (StateControlled)          → px.models surface

Chat                                                  # input normalization
└── Chat.export(...)                                  → chat_export_logic.md
```

`StateControlled` is the framework every cache, proxdash, files, and
model-config object inherits from — it lets them be serialized,
cached, and passed across processes. Rules and corner cases live in
[`state_controller.md`](./state_controller.md).

---

## 2. Decision tree

Find your change on the left, open the doc on the right. Every doc
listed below has substantive content; placeholders are noted at the
bottom of §5.

```
I'm about to …                                         → Read
│
│   # Providers
├── add a new provider connector                       →  adding_a_new_provider.md
├── edit an existing connector's endpoint / executor   →  adding_a_new_provider.md §6
├── change FeatureConfigType or a support-level rule   →  feature_adapters_logic.md §2
│
│   # Request pipeline
├── change what FeatureAdapter drops or injects        →  feature_adapters_logic.md §2
├── change what ResultAdapter transforms               →  feature_adapters_logic.md §3
├── edit ENDPOINT_PRIORITY selection                   →  feature_adapters_logic.md §1
├── change an `_<endpoint>_executor` contract          →  adding_a_new_provider.md §6
│
│   # Chat / messages
├── add a new ContentType                              →  chat_export_logic.md §3
├── add or change a Chat.export flag                   →  chat_export_logic.md §3
├── alter the deep-copy invariant in Chat              →  chat_export_logic.md §3.4
│
│   # Caching
├── edit query cache (read / write / LRU / shards)     →  cache_internals.md §3-§10
├── add a field to QueryRecord / MessageContent        →  cache_internals.md §5
│   │                                                     (hash + equality mirror rule)
├── edit ModelCacheManager or the default model cache  →  cache_internals.md §11
│
│   # Files and multi-modal
├── add a new provider File API dispatch               →  files_internals.md §1.1
├── change the auto-upload hook                        →  files_internals.md §4
├── extend UPLOAD / REFERENCE media-type allow-lists   →  files_internals.md §1.2
│
│   # State propagation
├── add or edit a StateControlled class                →  state_controller.md
├── change how nested state propagates                 →  state_controller.md
├── add a new StateContainer dataclass                 →  state_controller.md
│
│   # Dependency / import layering
├── add a new file under src/proxai/                   →  dependency_graph.md
├── unsure where an import belongs                     →  dependency_graph.md  (Dependency Layers)
│
│   # Tests
├── add a unit test                                    →  testing_conventions.md §4
├── write a mock for a new provider                    →  testing_conventions.md §3
├── extend integration_tests/proxai_api_test.py        →  testing_conventions.md §5
├── use conftest fixtures (model_configs_instance)     →  testing_conventions.md §4.2
│
│   # Pre-merge
└── I'm ready to open / land a PR                      →  sanity_check_before_merge.md
                                                           (see §4 of this doc — mandatory)
```

A change that doesn't fit any row above either touches the public
caller surface (cross over to
[`../user_agents/overview.md`](../user_agents/overview.md)) or is a
housekeeping edit (packaging, CI, workflow files) covered by
[`sanity_check_before_merge.md`](./sanity_check_before_merge.md) §15.

---

## 3. Global rules

The repo-root [`CLAUDE.md`](../../CLAUDE.md) is the authoritative
project contract. The rules below are the non-negotiables every
`developer_agents/` doc assumes — they are restated here so you
don't have to chase them down mid-review, but `CLAUDE.md` wins if
they drift.

### 3.1 Always use `poetry run`

**Never invoke `python` / `python3` / `pytest` / `ruff` / `yapf`
directly.** Every Python command in this repo is prefixed with
`poetry run` (`poetry run python3 …`, `poetry run pytest …`,
`poetry run ruff check …`). Bare interpreters resolve to the
system Python, which does not have the project's dependencies
installed — imports fail, or worse, silently pick up stale
globally-installed `openai` / `anthropic` / `google-genai` and
produce misleading results. This rule is non-negotiable; see
[`CLAUDE.md`](../../CLAUDE.md) and
[`sanity_check_before_merge.md`](./sanity_check_before_merge.md) §5.

### 3.2 Style is 2-space indent, 80-col, google docstrings

Enforced by `ruff` + `yapf` on every CI run. Ruff additionally
checks google-style docstrings (`D` rules), `pyupgrade`, `bugbear`,
and `isort` with `proxai` as first-party. Per-file ignores relax
docstring rules for `tests/`, `integration_tests/`, `examples/`,
and `docs/` — don't re-tighten them without a reason. Specific `D`
codes the repo intentionally ignores (`D100`, `D104`, `D105`,
`D107`, `D102`) should not be "fixed" by adding stub docstrings.

Run before you push:

```bash
poetry run ruff check src tests
poetry run yapf -i -r src tests
```

### 3.3 Respect the dependency layering

Imports go **upward only** (Layer 0 → Layer 5). The most common
violation: a provider (Layer 3 leaf) reaching back into
`caching/` or `connections/available_models`. If you find yourself
writing an import that fights the tree in §1.2, ask whether the
right move is to push a helper *down* into `type_utils` or `types`
rather than pull a high-level dependency into a low-level file. The
file-by-file import table in
[`dependency_graph.md`](./dependency_graph.md) is the enforcement
record.

### 3.4 `types.py` is the foundation — keep it that way

`src/proxai/types.py` imports nothing from the rest of `proxai`.
New dataclasses / enums / `StateContainer` subclasses live here. If
a new type needs a helper function that imports something else,
the helper lives in `type_utils.py` (Layer 0), not in `types.py`.

### 3.5 Provider imports are restricted

Files under `src/proxai/connectors/providers/` may only import:
- `proxai.types`
- their own `*_mock.py` sibling
- `proxai.connectors.provider_connector`
- `proxai.connectors.model_configs` (if static metadata is needed)

No caching, no proxdash, no `available_models`, no cross-provider
imports. See [`dependency_graph.md`](./dependency_graph.md) and
[`adding_a_new_provider.md`](./adding_a_new_provider.md) §3 for
what the `__init_subclass__` contract validator enforces at import
time.

### 3.6 Hash and equality are a paired contract

Any field you add to `QueryRecord`, `MessageContent`, or any
type that participates in cache identity must be reflected in **both**
`src/proxai/serializers/hash_serializer.py` (the cache key) **and**
`src/proxai/type_utils.py::is_query_record_equal` (the collision
check). They are a single contract split across two files; one
without the other lets a cache hit return the wrong record. The
invariant is explained in
[`cache_internals.md`](./cache_internals.md) §5 and enforced during
review by
[`sanity_check_before_merge.md`](./sanity_check_before_merge.md) §9.

### 3.7 Feature support rules out "silently drop"

A feature declared `SUPPORTED` on an endpoint that actually ignores
the field silently ships a regression — the framework cannot raise
for the caller. Prefer `NOT_SUPPORTED` (the adapter surfaces a
clear error) or `BEST_EFFORT` (degradation is explicit and the
support level travels onto `CallRecord.connection`). See
[`adding_a_new_provider.md`](./adding_a_new_provider.md) §5 for the
SUPPORTED / BEST_EFFORT / NOT_SUPPORTED taxonomy and
[`feature_adapters_logic.md`](./feature_adapters_logic.md) §2 for
what each level does at pipeline time.

---

## 4. Before you merge — mandatory

[`sanity_check_before_merge.md`](./sanity_check_before_merge.md) is
the pre-merge audit. It is not optional. CI catches lint / yapf /
`pytest` (Gate 1), but most regressions the repo has shipped came
from the conventions CI does not mechanically enforce (Gate 2:
layering, `poetry run`, `CLAUDE.md` spot-checks) or from changes
that needed a paired edit in a second file (Gate 3: the new-
provider 5-attribute contract, the hash / equality mirror, the
StateControlled propagation contract, the integration-harness wire
check, the cache freshness invariant).

Work through **every section in `sanity_check_before_merge.md` that
applies to your change** before you mark a PR ready for review. The
checklist calls out which sections are conditional on what you
edited, so skipping an unrelated one is fine — skipping a relevant
one is the most common way drift ships to users.

Gate 4 (docs currency) is what forces this overview, the
`api_guidelines/*.md` reference docs, and
[`../documentation_outline.md`](../documentation_outline.md) to stay
truthful. If you change behaviour callers can observe, update the
relevant Layer A doc in the same PR.

---

## 5. Further reading

Per-topic internals docs (listed in the order the outline's §2.7
scope table has them):

| Doc | What it owns |
|---|---|
| [`dependency_graph.md`](./dependency_graph.md) | Import layering and the file-by-file import table. |
| [`state_controller.md`](./state_controller.md) | StateControlled base, StateContainer, handle_changes, propagation corner cases. |
| [`adding_a_new_provider.md`](./adding_a_new_provider.md) | Self-contained end-to-end guide for a new connector — the five-attribute contract, executor anatomy, the 3-gate flow. |
| [`feature_adapters_logic.md`](./feature_adapters_logic.md) | FeatureAdapter + ResultAdapter pipeline — what gets dropped / injected per support level. |
| [`chat_export_logic.md`](./chat_export_logic.md) | `Chat` construction / normalization and the 10-step `Chat.export` pipeline. |
| [`cache_internals.md`](./cache_internals.md) | Query cache (shards / heap / freshness invariant), model cache, hash + equality pact. |
| [`files_internals.md`](./files_internals.md) | FilesManager lifecycle, four dispatch tables, capability split, auto-upload hook. |
| [`testing_conventions.md`](./testing_conventions.md) | Unit tier vs. integration harness, mock contract, fixtures, coverage expectations. |
| [`sanity_check_before_merge.md`](./sanity_check_before_merge.md) | Pre-merge audit (mandatory). |

Two docs in this folder are still **placeholders** (design decision
§8 in [`../documentation_outline.md`](../documentation_outline.md)
for the shipping plan): `feature_roadmap.md` (planned work with
status / date) and `skills_authoring.md` (rules for authoring
bundled Claude skills under `src/proxai/agent_skills/`). Scope
definitions live in outline §2.7; both will land with substantive
content before the Layer B skills ship.

When you need the **caller surface** — what a user sees when they
call the library you just edited — cross over to
[`../user_agents/overview.md`](../user_agents/overview.md) and its
`api_guidelines/` siblings. The pair of overviews plus the full
[`../../CLAUDE.md`](../../CLAUDE.md) is the complete orientation
for most contributor tasks.
