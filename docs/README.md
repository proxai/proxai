# ProxAI Documentation

Reference docs for ProxAI — the single-API Python library for
calling OpenAI, Anthropic, Google, Mistral, Cohere, DeepSeek,
Grok/xAI, HuggingFace, and Databricks through one `import proxai as
px` surface. If you're new here, start below based on your task.

## For AI agents (and humans) using ProxAI

Start at [`user_agents/overview.md`](./user_agents/overview.md) —
the complete `px.*` surface, a decision tree from task to doc, and
the library-wide conventions (import alias, nano-USD cost unit, the
two calling styles, error surfaces). From there, every deep topic
is one link away: calling a model, picking a model, attaching
files, building a Chat, reading a `CallRecord`, cache behaviour,
debugging raw provider responses, and a symptom-first error lookup
([`user_agents/troubleshooting.md`](./user_agents/troubleshooting.md)).

## For AI agents (and humans) modifying ProxAI

Start at [`developer_agents/overview.md`](./developer_agents/overview.md)
— the request-pipeline diagram, the import layering that prevents
circular dependencies, and a decision tree from "what are you
changing" to "which doc owns the invariants you're about to touch."
Before you open a PR, walk through
[`developer_agents/sanity_check_before_merge.md`](./developer_agents/sanity_check_before_merge.md)
— it's mandatory, not optional. The repo root's
[`CLAUDE.md`](../CLAUDE.md) is the authoritative project contract
this tree defers to.

## Complete doc inventory

If the two overviews didn't land you on the right page, scan this
flat list. Paths are sorted within each group; `(placeholder)` marks
files with only a scope pointer back to
[`documentation_outline.md`](./documentation_outline.md).

**`user_agents/`** — reference material for calling ProxAI.

- [`user_agents/overview.md`](./user_agents/overview.md) — Landing page and decision tree for library users.
- [`user_agents/troubleshooting.md`](./user_agents/troubleshooting.md) — Symptom-first error lookup with concrete fixes.
- [`user_agents/api_guidelines/call_record.md`](./user_agents/api_guidelines/call_record.md) — Every field on the `CallRecord` / `QueryRecord` / `ResultRecord` / `MessageContent` returned from a call.
- [`user_agents/api_guidelines/cache_behaviors.md`](./user_agents/api_guidelines/cache_behaviors.md) — Query-cache and model-cache behaviour, bypass flags, replay semantics.
- [`user_agents/api_guidelines/provider_feature_support_summary.md`](./user_agents/api_guidelines/provider_feature_support_summary.md) — Cheat-sheet of `SUPPORTED` / `BEST_EFFORT` / `NOT_SUPPORTED` per provider.
- [`user_agents/api_guidelines/px_chat_api.md`](./user_agents/api_guidelines/px_chat_api.md) — `px.Chat`, `Message`, `MessageContent` — building multi-turn conversations.
- [`user_agents/api_guidelines/px_client_api.md`](./user_agents/api_guidelines/px_client_api.md) — `px.connect()` / `px.Client()`, option trees, lifecycle, defaults.
- [`user_agents/api_guidelines/px_files_api.md`](./user_agents/api_guidelines/px_files_api.md) — `px.files.*` upload / list / remove / download plus the auto-upload hook inside `generate()`.
- [`user_agents/api_guidelines/px_generate_api.md`](./user_agents/api_guidelines/px_generate_api.md) — The seven `generate_*` functions — parameters, output formats, fallback chains.
- [`user_agents/api_guidelines/px_models_api.md`](./user_agents/api_guidelines/px_models_api.md) — `px.models.*` discovery — filters, health probes, configured vs. working.
- [`user_agents/api_guidelines/raw_provider_response.md`](./user_agents/api_guidelines/raw_provider_response.md) — The `DebugOptions.keep_raw_provider_response` local-debug hatch.
- [`user_agents/recipes/best_practices.md`](./user_agents/recipes/best_practices.md) — General do/don't rules for library use _(placeholder)_.
- [`user_agents/recipes/production_best_practices.md`](./user_agents/recipes/production_best_practices.md) — Production resilience additions — fallback chains, cache hygiene _(placeholder)_.
- [`user_agents/recipes/proxdash_onboarding.md`](./user_agents/recipes/proxdash_onboarding.md) — First-time ProxDash setup and API-key registration _(placeholder)_.
- [`user_agents/recipes/refactoring_existing_codebase.md`](./user_agents/recipes/refactoring_existing_codebase.md) — Step-by-step migration of an OpenAI / Anthropic / Gemini / ... codebase onto ProxAI _(placeholder)_.

**`developer_agents/`** — reference material for modifying ProxAI itself.

- [`developer_agents/overview.md`](./developer_agents/overview.md) — Landing page and decision tree for contributors.
- [`developer_agents/adding_a_new_provider.md`](./developer_agents/adding_a_new_provider.md) — End-to-end guide plus normative contract for a new provider connector.
- [`developer_agents/cache_internals.md`](./developer_agents/cache_internals.md) — Query cache (shards, heap, freshness invariant) and `ModelCacheManager` internals.
- [`developer_agents/chat_export_logic.md`](./developer_agents/chat_export_logic.md) — `Chat` construction / normalization and the 10-step `Chat.export` pipeline.
- [`developer_agents/dependency_graph.md`](./developer_agents/dependency_graph.md) — Import layering (Layer 0 → Layer 5) and file-by-file import table.
- [`developer_agents/feature_adapters_logic.md`](./developer_agents/feature_adapters_logic.md) — `FeatureAdapter` + `ResultAdapter` mechanics — what each support level drops or injects.
- [`developer_agents/feature_roadmap.md`](./developer_agents/feature_roadmap.md) — Planned work with status / date — tool usage, router, more providers _(placeholder)_.
- [`developer_agents/files_internals.md`](./developer_agents/files_internals.md) — `FilesManager` lifecycle, four dispatch tables, capability split, auto-upload hook.
- [`developer_agents/sanity_check_before_merge.md`](./developer_agents/sanity_check_before_merge.md) — Mandatory pre-merge audit organised by CI / conventions / conditional gates.
- [`developer_agents/skills_authoring.md`](./developer_agents/skills_authoring.md) — Rules for authoring bundled Claude skills under `src/proxai/agent_skills/` _(placeholder)_.
- [`developer_agents/state_controller.md`](./developer_agents/state_controller.md) — `StateControlled` base, `handle_changes` contract, propagation rules for nested state.
- [`developer_agents/testing_conventions.md`](./developer_agents/testing_conventions.md) — Unit tier vs. integration harness, mock contract, fixtures, coverage expectations.

**Top-level `docs/`** — meta and retrieval-fallback files.

- [`documentation_outline.md`](./documentation_outline.md) — Canonical doc inventory, per-file scope, migration status, and design decisions.
- [`llms-full.txt`](./llms-full.txt) — CI-generated flat dump of `user_agents/` docs for single-fetch ingestion _(placeholder)_.
- [`llms.txt`](./llms.txt) — Curated index for agents that can fetch one file but cannot clone the repo _(placeholder)_.
- [`skill_analyses.md`](./skill_analyses.md) — Rationale for the three-layer documentation strategy (Layer A / B / C).

---

## For agents without these docs checked out

Two retrieval fallbacks ship alongside the repo (both are
placeholders today — see
[`documentation_outline.md`](./documentation_outline.md) §4 and §7
for the shipping plan):

- **Bundled skills** — `proxai-setup`, `proxai-migrate`,
  `proxai-debug`, `proxai-best-practices`. Task-oriented playbooks
  that will ship inside the `proxai` PyPI wheel at
  `src/proxai/agent_skills/` and install via
  `proxai skills install`.
- **[`llms.txt`](./llms.txt) / [`llms-full.txt`](./llms-full.txt)**
  — a curated index plus a CI-generated flat dump of
  `user_agents/`, for agents that can fetch a single file but
  cannot clone a repo.

## For maintainers

- [`documentation_outline.md`](./documentation_outline.md) — canonical
  inventory of every doc in the tree, per-file scope, migration
  status, and design decisions. Every new or substantively revised
  doc updates this file in the same PR.
- [`skill_analyses.md`](./skill_analyses.md) — analysis and
  rationale behind the three-layer documentation strategy (Layer A
  repo docs, Layer B bundled skills, Layer C `llms.txt`).

For the library's public landing page (what PyPI and GitHub show),
see the [root `README.md`](../README.md).
