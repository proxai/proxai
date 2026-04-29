# ProxAI Documentation — Structure and Plan

This document is the structural plan for ProxAI's documentation. It
enumerates every file in the docs tree, states the scope of each,
tracks migration status, and records the design decisions behind
the layout.

ProxAI's documentation is organized in three layers, each solving a
distinct distribution problem. The rationale and ecosystem context
for this split live in [`skill_analyses.md`](./skill_analyses.md);
this document is the execution plan.

- **Layer A — Repo docs.** `docs/user_agents/` and
  `docs/developer_agents/`, read by AI agents working inside the
  ProxAI repository.
- **Layer B — Bundled skills.** `src/proxai/agent_skills/`,
  shipped inside the `proxai` PyPI wheel and installed into the
  user's agent-discoverable paths via `proxai skills install`.
  Reachable by AI agents working inside a client's repository
  that has `proxai` installed.
- **Layer C — `llms.txt`.** `docs/llms.txt` and
  `docs/llms-full.txt`, mirrored to `proxai.co` at deploy time.
  A retrieval fallback for agents without skills installed.

All three ship from the same repository and the same release.

---

## 1. Complete repo and docs layout

```
proxai/
├── src/proxai/
│   ├── ...                              (library code)
│   ├── agent_skills/                    (Layer B — shipped in wheel)
│   │   ├── proxai-migrate/
│   │   │   ├── SKILL.md
│   │   │   └── patterns/
│   │   │       ├── openai.md
│   │   │       ├── anthropic.md
│   │   │       ├── gemini.md
│   │   │       └── ...
│   │   ├── proxai-setup/SKILL.md
│   │   ├── proxai-debug/SKILL.md
│   │   └── proxai-best-practices/SKILL.md
│   └── cli.py                           (`proxai` CLI entrypoint,
│                                         includes `skills install`)
│
├── docs/
│   ├── README.md  *                     (top-level router)
│   ├── llms.txt                         (Layer C — curated index)
│   ├── llms-full.txt                    (Layer C — CI-generated)
│   ├── documentation_outline.md  *      (this file)
│   ├── skill_analyses.md  *             (analysis + rationale)
│   │
│   ├── user_agents/                     (Layer A — for library users)
│   │   ├── overview.md  *
│   │   ├── api_guidelines/
│   │   │   ├── px_client_api.md  *
│   │   │   ├── px_generate_api.md  *
│   │   │   ├── px_models_api.md  *
│   │   │   ├── px_models_model_config_api.md  *
│   │   │   ├── px_files_api.md  *
│   │   │   ├── px_chat_api.md  *
│   │   │   ├── call_record.md  *
│   │   │   ├── raw_provider_response.md  *
│   │   │   ├── provider_feature_support_summary.md  *
│   │   │   └── cache_behaviors.md  *
│   │   ├── recipes/
│   │   │   ├── proxdash_onboarding.md
│   │   │   ├── refactoring_existing_codebase.md
│   │   │   ├── best_practices.md
│   │   │   └── production_best_practices.md
│   │   └── troubleshooting.md  *
│   │
│   └── developer_agents/                (Layer A — for contributors)
│       ├── overview.md  *
│       ├── dependency_graph.md  *
│       ├── state_controller.md  *
│       ├── adding_a_new_provider.md  *
│       ├── feature_adapters_logic.md  *
│       ├── chat_export_logic.md  *
│       ├── cache_internals.md  *
│       ├── files_internals.md  *
│       ├── model_registry_type.md  *
│       ├── testing_conventions.md  *
│       ├── sanity_check_before_merge.md  *
│       ├── feature_roadmap.md
│       └── skills_authoring.md
│
└── README.md                            (root — PyPI / GitHub landing)
```

`*` marks files that have been drafted with substantive content;
unmarked entries are still placeholders pending a write (see §7.3).

Counts:
- **Layer A (repo docs):** 22 markdown files (9 top-level +
  ancillary), of which 11 migrated content from the old
  `docs/development/` tree and 11 are new writes.
- **Layer B (bundled skills):** 4 `SKILL.md` files plus per-skill
  pattern files. Sourced from Layer A recipes.
- **Layer C (`llms.txt`):** 2 files; `llms-full.txt` generated in
  CI from `docs/user_agents/`.

---

## 2. Layer A — Repo docs

Purpose: reference material read by AI agents (or humans) working
inside the ProxAI repository.

### 2.1 `docs/README.md`

A top-of-`docs/` router page. Directs visitors based on task
("using ProxAI → `user_agents/overview.md`; modifying ProxAI →
`developer_agents/overview.md`") and carries a flat sorted
inventory of every doc in the tree with a one-sentence scope per
entry, so an agent that landed at `docs/` can reach any target
without opening an overview first. No deep content — every entry
links out to the reference doc that owns the surface.

### 2.2 `user_agents/overview.md`

Mandatory first read for user-agents. Contains:
- One paragraph describing what ProxAI is.
- Decision tree: "task is calling a model → `px_generate_api.md`;
  task is picking a model → `px_models_api.md`; …"
- Global rules: `import proxai as px`, `px.models.*` as the source
  of truth for capabilities, integer nano-USD cost convention,
  error surfaces.
- Pointer to `troubleshooting.md` for errors and to the bundled
  Skills for task-oriented workflows.

### 2.3 `user_agents/api_guidelines/` (10 files)

| File | Scope |
|---|---|
| `px_client_api.md` | `px.connect()` / `ProxAIClient` construction, option trees (ConnectionOptions, ProviderCallOptions, ModelProbeOptions, DebugOptions). |
| `px_generate_api.md` | `generate_text` / `generate_image` / `generate_audio` / `generate_video` / `generate_json` / `generate_pydantic`. Parameters, response_format, tools, fallback chains. |
| `px_models_api.md` | `px.models.*` discovery surface — list / working / provider methods, filters, health checks. |
| `px_models_model_config_api.md` | `px.models.model_config.*` registry-mutation surface — register / unregister models, override the default fallback priority list, load / export the registry as JSON. Sibling of `px_models_api.md` (read-only discovery). |
| `px_files_api.md` | `px.files.*` upload / list / delete surface; semantics of local paths vs. uploaded file IDs. |
| `px_chat_api.md` | `px.Chat` session API, Message / MessageContent shape. |
| `call_record.md` | Return shape: fields on CallRecord, QueryRecord, ResultRecord, MessageContent. |
| `raw_provider_response.md` | `DebugOptions.keep_raw_provider_response` — the client-level debug flag that attaches the provider SDK's raw response to `CallRecord.debug.raw_provider_response`. |
| `provider_feature_support_summary.md` | Quick-skim tables for SUPPORTED / BEST_EFFORT / NOT_SUPPORTED per provider (already written). |
| `cache_behaviors.md` | User-facing cache behavior: implications + simple examples. Detailed outline in §6.1. |

### 2.4 `user_agents/recipes/` (4 files)

| File | Scope |
|---|---|
| `proxdash_onboarding.md` | Getting a ProxDash key, registering provider API keys, connecting. First-time-setup flow. Source for the `proxai-setup` skill. |
| `refactoring_existing_codebase.md` | Step-by-step migration of an existing OpenAI / Anthropic / Gemini / etc. codebase onto ProxAI. Links to `proxdash_onboarding.md`. Source for the `proxai-migrate` skill. |
| `best_practices.md` | General rules: load files first then generate, prefer `px.generate()` over aliases in source code, add a ProxDash key. Colab subsection for key management. |
| `production_best_practices.md` | Additive to `best_practices.md` — always use `fallback_models`, avoid the query cache in production, multi-provider resilience. Source for the `proxai-best-practices` skill. |

### 2.5 `user_agents/troubleshooting.md`

Error-driven lookup table. Format: common error symptom → likely
cause → concrete fix. Covers: probe failures, feature-mapping
rejections, multi-choice errors, bad fallback chains, auto-upload
surprises, stale cache hits, missing API keys. Source for the
`proxai-debug` skill.

### 2.6 `developer_agents/overview.md`

Mandatory first read for developer-agents. Contains:
- One-paragraph architecture summary (pipeline stages).
- Decision tree: "adding a provider → `adding_a_new_provider.md`;
  changing state propagation → `state_controller.md`; …"
- Global rules: 2-space indent, 80-col, `poetry run`,
  google-style docstrings, dependency layering.
- Explicit deference to `CLAUDE.md` at repo root as the
  authoritative project contract.
- Pointer to `sanity_check_before_merge.md` as the mandatory
  pre-merge audit.

### 2.7 `developer_agents/` (12 files)

| File | Scope |
|---|---|
| `dependency_graph.md` | Layering rules (types → caches → connectors → client → proxai), import contracts. |
| `state_controller.md` | StateControlled base, StateContainer, handle_changes contract, propagation rules (absorbs `state_propagation_analysis.md`). |
| `adding_a_new_provider.md` | Self-contained end-to-end guide for writing a new connector — both the procedural recipe and the normative contract. Covers: (0) throwaway SDK-probing scripts and the observation→`FeatureSupportType` table; (1-3) the 3-gate flow — code (create `providers/<name>.py` + mock, register in `_MODEL_CONNECTOR_MAP` + `PROVIDER_KEY_MAP`, add to `model_configs_data/v1.3.x.json`, add pyproject dep), optional File API dispatch integration, tests + integration-harness block; plus inline reference for `FeatureConfigType` fields (top-level, `ParameterConfigType`, `ToolConfigType`, `InputFormatConfigType`, `OutputFormatConfigType`, system-prompt Patterns 1/2), SUPPORTED / BEST_EFFORT / NOT_SUPPORTED semantics, executor anatomy (`ExecutorResult`, `_safe_provider_query`, what `query_record` carries after the adapter), per-provider parameter cross-reference, JSON / PYDANTIC Pattern A/B, multimodal `_to_<provider>_part` conversion, thinking / web-search citations, mock contract, common pitfalls. |
| `feature_adapters_logic.md` | FeatureAdapter + ResultAdapter mechanics; what gets dropped or injected per support level. |
| `chat_export_logic.md` | `Chat` construction / normalization and the `Chat.export` pipeline: deep-copy invariant, the 10-step processing order, the ten private helpers, how `_adapt_chat` drives `export` via six flags, `to_dict` / `from_dict` vs. `export` serialization. |
| `cache_internals.md` | `QueryCacheManager` / `ShardManager` / `HeapManager` three-layer architecture, on-disk sharded JSONL layout, the freshness invariant (in-memory index as source of truth), hash + equality identity pact (`hash_serializer` ↔ `type_utils.is_query_record_equal`), LRU eviction, backlog flush / shard merge, restart recovery, and `ModelCacheManager` (separate single-file design). |
| `files_internals.md` | `FilesManager` lifecycle and state fields, the four dispatch tables (`UPLOAD` / `REMOVE` / `LIST` / `DOWNLOAD`) plus mocks, capability-table split (`UPLOAD_SUPPORTED_MEDIA_TYPES` vs. `REFERENCE_SUPPORTED_MEDIA_TYPES`), parallelism rules, ProxDash dual-persistence story, the `_auto_upload_media` hook in `ProviderConnector.generate`, and the per-provider `_to_*_part` consumers. |
| `model_registry_type.md` | The `ModelRegistry` dataclass in `types.py` — its three fields (`metadata`, `default_model_priority_list`, `provider_model_configs`), supporting nested types, the load lifecycle (BUILT_IN bundled JSON → optional PROXDASH override via `proxdash.get_model_registry` → user-supplied JSON), the validation invariants enforced by `ModelConfigs.reload_from_registry` (`_validate_provider_model_configs`, `_validate_default_model_priority_list`, `_validate_min_proxai_version`), the JSON serializer round-trip (`encode_model_registry` / `decode_model_registry`), the `ModelConfigsState` / StateControlled integration, and the consumer matrix (`ModelConfigs`, `ProxDashConnection`, `AvailableModels`, `ProviderConnector`). |
| `testing_conventions.md` | Unit tier (`tests/`) vs. interactive integration harness (`integration_tests/proxai_api_test.py`), the `run_type=TEST` mock switch in `ProviderConnector.api`, structural-duck-typed `*_mock.py` contract, `MOCK_*_DISPATCH` for files, the `conftest.py` shared `model_configs_instance` fixture, common per-file fixtures (monkeypatch / requests_mock / tmp_path), style / class-grouping rules, `@integration_block` decorator and checkpointing, and coverage expectations (including the hash / equality parallel rule). |
| `sanity_check_before_merge.md` | Pre-merge audit organized by gates: CI-enforced lint / yapf / pytest (§2-3), conventions CI doesn't catch (§4 layering, §5 `poetry run`, §6 CLAUDE.md spot-checks), conditional checklists triggered by the kind of change (§7 public behavior, §8 new provider, §9 hash/equality mirror for new fields, §10 StateControlled edits, §11 wire-format → integration harness, §12 cache changes), docs currency (§13-14 Layer A + outline), and packaging / version bumps (§15). |
| `feature_roadmap.md` | Planned work with `Status: planned / in-progress / blocked / shipped` and a date. Tool usage, router, more providers, i2i / t+i2i / i2v. |
| `skills_authoring.md` | Rules for authoring files under `src/proxai/agent_skills/`: 5k-token body cap, `description` phrasing, progressive-disclosure pattern, versioning / CI checks for code snippets. |

---

## 3. Layer B — Bundled skills

Purpose: task-oriented playbooks loaded by a client's AI agent. Each
skill is a directory inside `src/proxai/agent_skills/` and ships
with the `proxai` PyPI wheel. The `proxai skills install` CLI
copies them into the user's agent-discoverable paths
(`~/.claude/skills/`, `~/.codex/skills/`, …) on demand.

### 3.1 Minimum skill set

| Skill | Auto-invoke trigger (description) | Body content | Source in Layer A |
|---|---|---|---|
| `proxai-migrate` | "Migrate an existing OpenAI, Anthropic, Gemini, or other LLM client codebase to ProxAI. Use when user mentions switching providers, multi-provider support, or adding fallback logic." | Step-by-step migration playbook. Points to bundled `patterns/<provider>.md` via progressive disclosure. | `recipes/refactoring_existing_codebase.md` |
| `proxai-setup` | "Set up ProxAI — install, configure API keys, onboard ProxDash. Use on first-time ProxAI integration." | Install, env vars, `px.connect()`, ProxDash key setup. | `recipes/proxdash_onboarding.md` |
| `proxai-debug` | "Debug a ProxAI call — cache hits, fallback chains, feature-mapping errors, auto-upload surprises." | Decision tree mapping symptoms → causes → fixes. | `troubleshooting.md` |
| `proxai-best-practices` | "Production best practices for ProxAI — fallback chains, cache hygiene, multi-provider resilience." | Condensed production-ready rules. | `recipes/production_best_practices.md` |

### 3.2 Directory layout under `src/proxai/agent_skills/`

```
src/proxai/agent_skills/
├── proxai-migrate/
│   ├── SKILL.md
│   └── patterns/
│       ├── openai.md
│       ├── anthropic.md
│       ├── gemini.md
│       ├── mistral.md
│       ├── cohere.md
│       ├── grok.md
│       ├── deepseek.md
│       ├── huggingface.md
│       └── databricks.md
├── proxai-setup/SKILL.md
├── proxai-debug/SKILL.md
└── proxai-best-practices/SKILL.md
```

Each `SKILL.md` body stays under 5k tokens. Long reference material
(per-provider migration patterns, extended examples) lives in
supporting files loaded on demand via progressive disclosure.

Packaging: add one entry to `pyproject.toml`:

```toml
[tool.poetry]
include = [
  "src/proxai/connectors/model_configs_data/*.json",
  "src/proxai/agent_skills/**/*",      # Layer B
]
```

### 3.3 CLI commands (design; implementation tracked separately)

The `proxai` CLI entrypoint exposes the skills lifecycle:

```bash
proxai skills install            # first-time install; auto-detect tools
proxai skills install --update   # refresh to match current wheel
proxai skills install --force    # overwrite local modifications
proxai skills install --diff     # preview without writing
proxai skills install --prune    # remove deprecated skills
proxai skills status             # wheel version vs installed per skill
proxai skills uninstall          # remove proxai-* skills everywhere
```

Full upgrade-lifecycle behavior (version stamp, runtime staleness
warning, modification detection) is specified in
[`skill_analyses.md`](./skill_analyses.md) §5 Layer B.

### 3.4 `developer_agents/skills_authoring.md`

The authoring guide for bundled skills. Lives in Layer A because it
is a developer-facing contributor doc, not a runtime artifact.
Scope:
- `SKILL.md` frontmatter schema (`name`, `description`,
  `license`, `metadata`, `allowed-tools`) with worked examples.
- Body constraints: under 5k tokens, task-oriented imperative voice,
  concrete commands, progressive-disclosure references to
  supporting files.
- `description` phrasing rules ("Use when user mentions X") for
  reliable auto-invocation.
- Versioning stamps and CI checks that verify code snippets still
  import current ProxAI API.
- Review checklist before merging a skill change.

---

## 4. Layer C — `llms.txt`

Purpose: retrieval fallback for agents without the bundled skills
installed, and a search / browse entry point for humans. Canonical
files live in the repo; the docs site mirrors them at deploy time.

### 4.1 `docs/llms.txt`

Short, hand-written curated index. Structure (Stripe-pattern):

```
# ProxAI

> One-paragraph description.

## Instructions for agents
- Fetch *.md variants of any page link below.
- For task workflows, install the bundled Skills: `pip install proxai && proxai skills install`.

## Core
- [Overview](https://proxai.co/docs/overview.md): what ProxAI is.
- [Install](https://proxai.co/docs/install.md): setup and API keys.
- [Feature support](https://proxai.co/docs/provider_feature_support_summary.md): provider capability matrix.

## API guidelines
- [px.generate](...)
- [px.models](...)
- ...

## Recipes
- [Refactoring to ProxAI](...)
- ...
```

### 4.2 `docs/llms-full.txt`

CI-generated. A concatenated dump of all Markdown files under
`docs/user_agents/` with simple separators, so an agent can ingest
the user-facing documentation in a single fetch. Script lives in the
repo (`scripts/build_llms_full.py` or similar); runs in the docs
deploy pipeline.

Developer-agent content (`docs/developer_agents/`) is excluded — it
is not user-facing.

---

## 5. Root `README.md` (PyPI / GitHub landing page)

The root `README.md` is a separate deliverable from the
documentation restructure. It sits at the top of the repo and
renders on PyPI and GitHub — the first surface both humans and
agents encounter.

Full requirements are specified in
[`skill_analyses.md`](./skill_analyses.md) §7. Summary:

- Hero with a 5-line before/after migration snippet showing
  multi-provider fallback.
- Install block featuring `pip install proxai && proxai skills
  install`, plus the upgrade flow (`pip install -U proxai && proxai
  skills install`).
- Short capability sections: multi-provider fallback, ProxDash
  observability, the 9 providers, feature callouts.
- "For AI agents" subsection linking to `docs/llms.txt`,
  `docs/user_agents/overview.md`, and
  `src/proxai/agent_skills/`.
- Metadata badges: PyPI version, Python support, license,
  `llms.txt` link, CI status.
- Target: under 300 lines; links out to `docs/user_agents/`
  rather than duplicating content.

Sequencing: this rewrite should happen **after**
`docs/user_agents/` is largely populated, so links point to real
content rather than placeholders.

---

## 6. Per-document outlines

Where a document's scope is non-obvious, its detailed outline lives
here. More will be added as each document is written.

### 6.1 `cache_behaviors.md`

User-facing cache behavior. Implications and simple examples.
Implementation details belong in `developer_agents/cache_internals.md`.

```
1. What ProxAI caches
   - query_cache: (model, prompt, params) → response
   - model_cache: health-check results per (provider, model)

2. When the cache helps
   - Iterating on prompt engineering (free retries)
   - Running the same eval harness repeatedly
   - Example: skip_cache=False + override_cache_value=False

3. When the cache hurts
   - Production traffic (users expect a fresh answer)
   - Non-deterministic evaluations (variance matters)
   - Example: cache_options=None or skip_cache=True

4. Cache options reference
   - cache_path: where the files live
   - skip_cache: read bypass (still writes)
   - override_cache_value: write-always (overwrites hits)
   - clear_model_cache: one-shot invalidation

5. Implications
   - Sharing cache across processes / machines → safe, file-locked
   - Changing parameters (temperature, n) → different cache key, miss
   - Fallback models → only the successful model's response is cached
   - Auto-uploaded files → cache key includes content hash, not path
   - Health-probe cache TTL → stale probes can hide a regressed model

6. Common problems
   - "I changed the prompt and still got the old response" → exact
     string match; whitespace matters
   - "model_cache says X works but it doesn't" → clear_model_cache=True
   - "Cache hit on a different model" → cache key includes
     provider_model; verify the model passed
```

Target: 150-250 lines, matching the length of
`provider_feature_support_summary.md`.

---

## 7. Migration status

Layer A restructure is structurally complete — directories created,
existing content moved, empty placeholders in place for new docs.
Content rewrites are pending per doc.

### 7.1 Completed

The `*` markers in §1 "Complete repo and docs layout" are the
authoritative list of completed docs. This section captures the
structural bootstrap and a dated drafting log only — scope lives
in §2.3 / §2.6 / §2.7, not here.

**Structural bootstrap**

- `docs/user_agents/`, `docs/developer_agents/`,
  `user_agents/api_guidelines/`, `user_agents/recipes/`
  directories created.
- 11 files moved via `git mv` from `docs/development/` into Layer
  A homes (history preserved as renames).
- 20 empty placeholders created for new docs, each with a one-line
  scope pointer back to this outline.
- `docs/README.md`, `docs/llms.txt`, `docs/llms-full.txt`
  placeholders created.

**Drafts**

- `user_agents/api_guidelines/cache_behaviors.md` — 2026-04-23
- `user_agents/api_guidelines/raw_provider_response.md` — 2026-04-23
- `user_agents/troubleshooting.md` — 2026-04-23
- `developer_agents/feature_adapters_logic.md` — 2026-04-23
- `developer_agents/adding_a_new_provider.md` — 2026-04-23
- `developer_agents/sanity_check_before_merge.md` — 2026-04-23
- `developer_agents/testing_conventions.md` — 2026-04-23
- `developer_agents/cache_internals.md` — 2026-04-23 (drains
  `docs/development/query_cache.md` — deleted; see §7.2)
- `developer_agents/files_internals.md` — 2026-04-23
- `developer_agents/chat_export_logic.md` — 2026-04-23
- `user_agents/overview.md` — 2026-04-24
- `developer_agents/overview.md` — 2026-04-24
- `docs/README.md` — 2026-04-24 (router + flat doc inventory;
  `/px-create-doc` §7.1 keeps the inventory in sync)
- `user_agents/api_guidelines/px_models_model_config_api.md` — 2026-04-24
- `developer_agents/model_registry_type.md` — 2026-04-26

### 7.2 Pending — staging files in `docs/development/`

Three files remain in `docs/development/` as content sources for
merges / splits. They will be consumed and deleted as the new docs
are written:

| Staging file | Consumes into |
|---|---|
| ~~`query_cache.md`~~ | Fully drained and deleted 2026-04-23. User-facing content → `user_agents/api_guidelines/cache_behaviors.md`; internals → `developer_agents/cache_internals.md`. Stale `retry_if_error_cached` / `PROVIDER_ERROR_CACHED` descriptions in the staging file were intentionally NOT carried forward — the current source has those removed. |
| `multimodal_large_file_analysis.md` | Split: internals half (FilesManager + auto-upload) consumed into `developer_agents/files_internals.md` (drafted 2026-04-23). User-facing halves already folded into `user_agents/api_guidelines/px_files_api.md` and `px_generate_api.md`. Per-provider size-limit / delivery-method reference material (§3) and SDK code examples (§5) in the staging file are NOT yet migrated — keep the file until either that reference material is folded into Layer A or the maintainer confirms it can be dropped. |
| `state_propagation_analysis.md` | Merged into `developer_agents/state_controller.md` |

`docs/development/` will be removed once all three staging files
have been consumed.

### 7.3 Pending — writes

Each empty placeholder needs its content written. The files that
map directly from existing moved content (e.g., `px_client_api.md`,
`call_record.md`) are in good shape; they need only editorial passes
to align with the restructure's tone and cross-links.

The genuinely new docs (10+) need to be written from scratch —
outline-first, then draft, then review — the same way
`provider_feature_support_summary.md` was built.

### 7.4 Pending — Layer B and Layer C

- `src/proxai/agent_skills/` directory does not yet exist. Skill
  content is sourced from the Layer A recipes, so skills are
  authored after recipes are written.
- `docs/llms.txt` and `docs/llms-full.txt` are placeholders;
  `llms-full.txt` requires the CI script after `user_agents/` is
  largely written.
- `proxai` CLI / `proxai skills install` command is not yet
  implemented. Deferred until the documentation content is complete.

---

## 8. Design decisions

Resolved items from the planning discussion:

1. **Two-folder split, no separate development docs.** One tree for
   library users (`user_agents/`), one for library contributors
   (`developer_agents/`). Replaces the previous
   `docs/development/` monolith.

2. **API reference uses `_api.md` suffix.** Former `*_analysis.md`
   files renamed during migration.

3. **Single-package skill distribution.** Skills ship inside the
   `proxai` wheel under `src/proxai/agent_skills/`; no separate
   `proxai-skills` package. Rationale in
   [`skill_analyses.md`](./skill_analyses.md) §5 Layer B.

4. **`llms.txt` in the repo.** `docs/llms.txt` and
   `docs/llms-full.txt` are canonical; `proxai.co` mirrors them at
   deploy time.

5. **`skills_authoring.md` lives under `developer_agents/`.**
   The audience is contributors editing skill source files, not
   library users.

6. **Colab best practices fold into `best_practices.md`.** Single
   subsection rather than a standalone file, since the content is
   a short list of environment-specific notes.

7. **`raw_provider_response.md` in `user_agents/`.** The doc covers
   the client-level `keep_raw_provider_response` debug flag — a
   caller-facing escape hatch for reading the provider SDK's raw
   response object off a returned `CallRecord`. Users who reach for
   this flag are still in user-facing territory.

8. **`docs/README.md` (not `index.md`).** GitHub renders it
   automatically at the `docs/` path.

9. **Overview files written last.** `user_agents/overview.md` and
   `developer_agents/overview.md` are written after their sibling
   docs exist so their decision trees can link to real files.

10. **Staging files stay in `docs/development/` until consumed.**
    No content loss during the split/merge operations; the folder
    is removed entirely once its three remaining files are
    rewritten into their Layer A targets.

---

## 9. Remaining open questions

Items that still need a decision before implementation:

1. **Per-provider patterns in `proxai-migrate`.** Ship one pattern
   file per provider (9+ files) or a single consolidated
   `patterns.md`? Leaning toward one file per provider for
   progressive-disclosure benefit, but the exact list to ship on
   v1 is open.

2. **`proxai skills install` default behavior.** Auto-detect all
   installed agent tools and install to each, or require
   `--tool=<name>` / `--all` explicitly? Depends on observed client
   usage patterns.

3. **CI rot guard for skills.** Minimum version: a check that
   validates code snippets in SKILL.md bodies against the current
   ProxAI API. Exact implementation (regex extraction vs.
   Markdown-aware parser vs. executing a generated Python script)
   open.

4. **MCP server.** Not in v1. Revisit if clients start asking for
   live provider-health querying through an agent interface rather
   than static docs.
