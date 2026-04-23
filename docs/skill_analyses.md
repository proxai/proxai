# Skills — Current State and Applicability to ProxAI Docs

This document surveys the state of AI-agent "Skills" as of early 2026
and explains how they fit alongside ProxAI's documentation strategy.

Short version: Skills complement reference docs, they don't replace
them. The place where Skills earn their keep is a distribution
problem — reaching an AI agent that's running inside someone else's
codebase (for example, a client refactoring an `openai`- or
`google-genai`-based project onto ProxAI). A repo-checked-in doc
can't reach that agent unless the user clones the repo; a Skill
bundled with the installed library can.

ProxAI therefore adopts a three-layer documentation strategy:

1. **Repo docs** (`docs/user_agents/`, `docs/developer_agents/`) —
   the reference layer, read by agents working inside ProxAI.
2. **Bundled Skills** (`src/proxai/agent_skills/`) — installed
   alongside the library and reachable by a client's agent.
3. **`llms.txt`** mirrored from the repo to the docs site — a
   retrieval fallback for agents without the skills installed.

All three ship from one package, one repository, one release.

---

## 1. What a Skill actually is

A Skill is a **directory** with a required `SKILL.md` file (YAML
frontmatter + markdown body) and optional supporting files (scripts,
reference docs, assets). The runtime loads it via **progressive
disclosure**:

```
Level 1  — frontmatter only (~100 tokens, always in context)
           name + description. Agent uses description to decide
           relevance. Load-bearing field.

Level 2  — SKILL.md body (<5000 tokens, loaded on trigger)
           Task instructions, decision trees, inline examples.

Level 3  — referenced files (unlimited, on-demand)
           reference docs, code snippets, scripts that run in a
           sandbox. Zero context cost if not accessed.
```

**SKILL.md frontmatter schema** (open spec, agentskills.io):

```yaml
---
name: proxai-migrate                     # 1-64 chars, lowercase+hyphens
description: Migrate an OpenAI / ...     # 1-1024 chars, BOTH what & when
license: MIT                             # optional
compatibility: Python 3.10+              # optional
allowed-tools: Bash(poetry *) Read       # optional, experimental
metadata: { version: 1.0 }               # optional, free-form
---

## Body starts here. Markdown. No schema.
```

**Activation model:**
- **Auto-invoke**: agent reads `description` of all installed skills
  at startup, pulls in SKILL.md when user request matches.
- **Explicit**: user types `/proxai-migrate` (Claude Code, Codex,
  Cursor all wire a slash menu).
- Claude Code adds non-standard extensions: `paths: "src/**"` (glob
  auto-activation), `disable-model-invocation` (user-only),
  `user-invocable: false` (Claude-only), `context: fork` (run in
  subagent).

---

## 2. Ecosystem state (early 2026)

Two open standards are converging, at **different altitudes**:

| Standard | Altitude | Analogy | Who signed on |
|---|---|---|---|
| **AGENTS.md** (Aug 2025, Linux Foundation Agentic AI Foundation) | Project-level "README for agents" — conventions, commands, style | `README.md` for humans | OpenAI, Sourcegraph (Amp), Google (Jules), Cursor, Factory; 20k+ repos |
| **Agent Skills** (Dec 2025, open spec at agentskills.io) | Task-level procedural knowledge | Recipes / playbooks | Anthropic (Claude Code, Claude.ai, API), OpenAI Codex, Cursor, GitHub Copilot, VS Code, Windsurf, Gemini CLI, Roo Code, OpenHands, Goose — 25+ platforms in ~12 weeks |

A **third layer** — `llms.txt` (Jeremy Howard, 2024) — is for making
published docs sites machine-readable. Adopted by Anthropic, Vercel,
Cloudflare, **Stripe**, Clerk, Pydantic, Prisma, Cursor. Stripe's
`llms.txt` has an `# Instructions` section telling agents to fetch
`*.md` variants of every URL — a pattern worth borrowing.

### 2.1 Per-tool quick matrix

| Tool | Native convention | Skills support | AGENTS.md | llms.txt reader |
|---|---|---|---|---|
| Claude Code | `CLAUDE.md`, `.claude/skills/` | ✅ native | ✅ | via WebFetch |
| Codex CLI | `AGENTS.md` | ✅ (Dec 2025) | ✅ | via fetch |
| Cursor | `.cursor/rules/*.mdc` | ✅ (Dec 2025) | ✅ | via fetch |
| GitHub Copilot | `.github/copilot-instructions.md` | ✅ | partial | — |
| Windsurf | `.windsurf/rules/` | ✅ | partial | — |
| Roo Code | Custom modes + skills | ✅ native | partial | — |
| Continue.dev | `config.yaml` rules/prompts | ❌ not yet | partial | — |
| Aider | `CONVENTIONS.md` | ❌ | ✅ (2025) | — |
| Sourcegraph Cody | `.vscode/cody.json` | ❌ | ? | — |

Pattern: **YAML frontmatter + markdown body** is the convergent
format. The `description` field is always the load-bearing one — it
is what the agent reads to decide whether a skill is relevant to the
current request.

### 2.2 How major libraries ship agent-docs today

| Library | Pattern |
|---|---|
| **Stripe** | llms.txt + per-URL `.md` variants + [Agent Toolkit Skills package](https://github.com/stripe/ai) + MCP server. Three-layer strategy. |
| **Pydantic** | `ai.pydantic.dev/llms-full.txt` — whole docs as one file |
| **Vercel AI SDK** | llms.txt + `.md` docs + skills ([vercel-labs/skills](https://github.com/vercel-labs/skills)) |
| **LangChain** | llms.txt |
| **Supabase, Clerk** | llms.txt |
| **Django** | open thread, nothing shipped |
| **FastAPI** | nothing formal |

Emerging practice: a PyPI/npm package that carries Skills. `pip
install proxai` makes the skills reachable by the user's Claude
Code / Cursor / Codex agent. There is active ecosystem work on this
pattern (`pydantic-ai-skills` on PyPI,
[neovateai/agent-skill-npm-boilerplate](https://github.com/neovateai/agent-skill-npm-boilerplate),
[vercel-labs/skills](https://github.com/vercel-labs/skills)).

---

## 3. The consensus patterns

**What the ecosystem has converged on:**

1. **Progressive disclosure beats monolithic docs.** A lean entry
   file that points to detailed references loads only what's needed.
   CLAUDE.md files bloating past 2k lines hurt performance.
2. **`description` is load-bearing.** Agents decide what to read
   based on it. Be specific — "Migrate an OpenAI client to ProxAI"
   beats "ProxAI helper."
3. **Recipe > API reference for agents.** Task-oriented imperative
   prose ("to migrate a handler, do X → Y → Z") beats surface
   dumps. Render's benchmark found LLM-generated context files
   *decreased* task success by roughly 3% and added 20%+ inference
   cost — quality and intent matter more than completeness.
4. **YAML frontmatter with `description`** is the universal entry
   point across Skills, Cursor MDC, and Roo Code.
5. **Repo-local vs. user-level distribution are different
   products.** Repo-local is for the agent working *inside the
   library's own repo*. User-level (skills, global rules) is for
   the agent working *inside someone else's repo that has the
   library installed*.

**Where skills fall short:**

- **Fragmentation tax.** Libraries realistically need AGENTS.md
  *and* SKILL.md *and* llms.txt — each addresses a different layer.
- **Maintenance burden.** Skills are prompts — they drift from
  source docs faster than auto-generated reference material.
- **Cross-surface installation is manual.** A skill in Claude Code
  does not auto-appear in Claude.ai or the API. Each surface has
  its own install flow (filesystem / zip upload / API call).
- **Not a silver bullet for reference material.** API surfaces and
  type hierarchies are still better served by structured reference
  docs than by skill bodies.

---

## 4. Skills complement reference docs, not replace them

The repo-docs layer (`docs/user_agents/` + `docs/developer_agents/`)
is the **reference layer** — what an agent reads while working
inside the ProxAI repository itself, whether to use the library or
to extend it. Skills do not replace this. Libraries that ship
Skills (Stripe, Pydantic, Vercel) all keep a structured reference
corpus alongside them.

Where repo docs alone fall short is distribution. When a client
agent is refactoring a codebase from `openai` or `google-genai` onto
`proxai`, that agent is running in the **client's** repository —
not ProxAI's. It will never see
`docs/user_agents/recipes/refactoring_existing_codebase.md` unless:

- the client has explicitly cloned ProxAI, or
- the recipe has been published as a Skill the client's agent can
  reach, or
- the client's agent fetches `llms.txt` from the docs site.

The content of the recipes folder is right. Without a distribution
layer, it is just in the wrong place for the client-refactoring
use case. Skills (and `llms.txt`) supply that missing distribution.

---

## 5. ProxAI's three-layer approach

All three layers ship from the same repository and the same PyPI
release. Each solves a distinct problem.

### Layer A — Repo docs

Purpose: an agent working **inside** ProxAI's repository
(developer-agents extending the library, or a user-agent inspecting
the library's source).

Location: `docs/user_agents/` (public-facing API/usage docs) and
`docs/developer_agents/` (library-internals docs for contributors).
Structure and content are defined in
[`outline_proposal.md`](./outline_proposal.md).

### Layer B — Skills bundled into the main `proxai` wheel

Purpose: an agent working **inside a client's repository**,
migrating code onto ProxAI or debugging a ProxAI-powered
integration. This is the refactoring scenario.

Shape: SKILL.md directories ship as data files inside the existing
`proxai` PyPI wheel (same mechanic as
`src/proxai/connectors/model_configs_data/*.json`). A small
`proxai skills install` CLI copies them into each installed agent
tool's discoverable path. One package, one release cadence, no
drift between library and skill versions.

```
src/proxai/
├── agent_skills/                      # shipped with the wheel
│   ├── proxai-migrate/
│   │   ├── SKILL.md
│   │   └── patterns/openai.md, anthropic.md, gemini.md, ...
│   ├── proxai-setup/SKILL.md
│   ├── proxai-debug/SKILL.md
│   └── proxai-best-practices/SKILL.md
└── cli.py                             # adds `proxai` CLI entrypoint
```

`pyproject.toml` — add one line to the existing include list:

```toml
[tool.poetry]
include = [
  "src/proxai/connectors/model_configs_data/*.json",
  "src/proxai/agent_skills/**/*",      # new
]
```

User flow:

```bash
pip install proxai
proxai skills install             # one-shot; detects installed tools
# copies to ~/.claude/skills/, ~/.codex/skills/ as applicable
```

At runtime the CLI uses `importlib.resources.files("proxai.agent_skills")`
to locate bundled skills, then writes copies (or dev symlinks) to
each agent's expected path. Version bumps to `proxai` carry skill
updates automatically — there is no separate package to track.

**Gotcha to document clearly:** wheel-bundled skills are not
auto-discovered by agents. They have to live at the agent's known
path (`~/.claude/skills/`, `.claude/skills/`, etc.). The
`proxai skills install` step is mandatory, one-time, per
environment. The README and the `proxai-setup` skill both need to
call this out.

#### Upgrade lifecycle — what happens when `proxai` is updated

Scenario: a user runs `pip install proxai` (v1.0), runs
`proxai skills install`, uses skills for weeks. Later, they run
`pip install -U proxai` (now v1.1).

**By default, nothing happens to the installed skills** — pip has no
reliable post-install hook across environments, and silent
overwrites of files living under the user's home directory are
undesirable. Without explicit handling, the agent would keep reading
v1.0 skills while calling v1.1 code. Subtly wrong.

The CLI handles this in four mechanisms:

1. **Version stamp on install.** At `proxai skills install` time,
   the CLI writes `proxai_version: "1.0.0"` into each installed
   skill's frontmatter `metadata` block (or a sibling
   `.proxai-version` file — a lightweight choice to make at
   implementation time). The stamp matches `proxai.__version__`
   from the installed wheel.

2. **Runtime staleness warning.** The first `import proxai` in a
   process compares the stamp on installed skills to the current
   wheel version. On mismatch:

   ```
   ProxAI skills are v1.0.0; library is v1.1.0.
   Run `proxai skills install` to update.
   ```

   Cached per-process so it fires at most once. Suppressible via
   `PROXAI_SKILLS_WARN=0`. Users hit this within a week of an
   upgrade; their agents pick it up on the next Python run.

3. **Idempotent, version-aware install command.**
   `proxai skills install` is safe to re-run:

   - Already present, same version → "Already up to date."
   - Present, older version, **unmodified** → overwrite, report
     diff summary.
   - Present, older version, **modified by user** (mtime or
     checksum diverges from what v1.0 shipped) → prompt per skill:
     "foo/SKILL.md was modified locally. Overwrite / skip / diff?"
   - `--force` skips prompts, `--diff` previews without writing.
   - `--prune` removes installed skills that no longer exist in
     the new wheel (a deprecated-skill case).

4. **Status and uninstall commands.**
   - `proxai skills status` — shows wheel version vs. installed
     version per skill. Agents and humans can self-diagnose.
   - `proxai skills uninstall` — removes every `proxai-*` skill
     from known agent paths. Needed because `pip uninstall proxai`
     does not touch files outside `site-packages/`.

Documented upgrade workflow (appears in the README install block
and in the `proxai-setup` skill):

```bash
pip install -U proxai
proxai skills install            # refresh skills to match new lib
```

User customizations survive upgrades under two conditions:
- The customized skill is named distinctly
  (`proxai-migrate-myorg/`) — the CLI never touches names it did
  not ship.
- Or the user answers "skip" on the overwrite prompt when the CLI
  detects local modifications; the skill is left alone and
  `proxai skills status` flags it as "customized, v1.0.0 (wheel is
  v1.1.0)."

Minimum skill set (mirrors the `recipes/` folder under
`docs/user_agents/`):

| Skill | Auto-invoke trigger (description) | Body content |
|---|---|---|
| `proxai-migrate` | "Migrate an existing OpenAI, Anthropic, Gemini, or other LLM client codebase to ProxAI. Use when user mentions switching providers, multi-provider support, or adding fallback logic." | Step-by-step migration playbook. Points to bundled `patterns/openai.md`, `patterns/anthropic.md` etc. via progressive disclosure. |
| `proxai-setup` | "Set up ProxAI — install, configure API keys, onboard ProxDash. Use on first-time ProxAI integration." | Install, env vars, `px.connect()`, ProxDash key setup. |
| `proxai-debug` | "Debug a ProxAI call — cache hits, fallback chains, feature-mapping errors, auto-upload surprises." | Decision tree mapping symptoms → causes → fixes. Sourced from the `troubleshooting.md` recipe. |
| `proxai-best-practices` | "Production best practices for ProxAI — fallback chains, cache hygiene, multi-provider resilience." | Condensed from the `production_best_practices.md` recipe. |

Key design rules:
- Each skill's body stays under 5k tokens. Long content
  (per-provider migration patterns, code examples) goes in bundled
  files loaded on demand.
- The `description` field names specific symptoms / triggers so
  auto-invocation works. "Use when user mentions X" is the phrasing
  that gets picked up reliably.
- Skills are versioned alongside the library — skill v1.2.0 matches
  `proxai` v1.2.0. Drift risk is real, so a CI check that verifies
  skill code snippets still import current ProxAI API is worth the
  effort.

### Layer C — `llms.txt` in the repo, mirrored to the docs site

Purpose: a retrieval fallback for agents that do not have the skill
installed, and a search/browse entry point for humans.

Shape — **source files live in `docs/`, the site mirrors them**:

```
docs/
├── llms.txt              # curated index, hand-written
└── llms-full.txt         # generated from user_agents/*.md at CI time
```

- **`llms.txt`** — short hand-written index. Stripe-style, with an
  `# Instructions` section telling agents to fetch `*.md` variants
  of each page and pointing at the bundled Skills.
- **`llms-full.txt`** — concatenated dump of `user_agents/` as one
  file. A 20-line CI script produces it from the user-agents folder
  so it never drifts from source.
- **`.md` twins** — every page served at `proxai.co/<page>` should
  also be reachable at `proxai.co/<page>.md`. Mintlify and most
  doc hosts auto-generate these.

The docs site (`proxai.co`) pulls `docs/llms.txt` and
`docs/llms-full.txt` from this repo at deploy time — a GitHub
Action, a Netlify sync, or a build-time copy, whichever fits the
site stack. Single source of truth; the repo wins; the site mirrors.

Minimum content: everything in `user_agents/` (Layer A), published
as standalone docs. Developer-agents content stays repo-only — it
is not user-facing.

Even if the docs site is offline or changes hosts, `llms.txt` and
`llms-full.txt` are still retrievable via `raw.githubusercontent.com`
— a fallback URL like
`https://raw.githubusercontent.com/proxai/proxai/main/docs/llms-full.txt`
remains valid.

### What is explicitly not built

- **AGENTS.md is not the primary vehicle.** AGENTS.md's job is to
  tell an agent how to work *inside the ProxAI repo itself*
  (`poetry run`, style rules, test commands). ProxAI already has
  `CLAUDE.md` serving that role. A short AGENTS.md that points to
  `CLAUDE.md` is fine but not load-bearing.
- **No separate MCP server for now.** Stripe maintains one because
  their API is live and stateful; ProxAI is mostly a thin
  abstraction over other providers' APIs, where static docs plus
  skills cover the use cases.
- **No per-tool variants published** (`.cursorrules`,
  `.github/copilot-instructions.md`, etc.). The Skills spec is now
  the lowest common denominator; the symlink-to-AGENTS.md pattern
  the community has settled on handles tool-specific fallbacks.

---

## 6. Resulting repo and docs layout

Adding Skills does not reshuffle the docs content. The content
inventory planned in
[`outline_proposal.md`](./outline_proposal.md) stays intact. A few
details shift:

1. **Recipes serve double duty.**
   `refactoring_existing_codebase.md`, `proxdash_onboarding.md`,
   `production_best_practices.md`, and `troubleshooting.md` are
   the source material for Layer B skills. They are written as
   recipes — imperative, step-by-step, runnable — so they extract
   cleanly into SKILL.md bodies. Non-recipe reference content
   (the API guidelines) stays in Layer A only.

2. **Skill sources live inside the package tree.** They ship with
   the wheel, so they sit under `src/proxai/agent_skills/` rather
   than under `docs/`. The authoring guide for them
   (`developer_agents/skills_authoring.md`, see point 4) lives in
   the repo docs.

3. **`llms.txt` lives in `docs/`, mirrors to the site.**
   Hand-written index plus CI-generated full dump. No separate
   repo-side copy and no site-owned duplicate.

4. **Add `developer_agents/skills_authoring.md`** — the rules for
   writing skills (5k-token cap, `description` phrasing guidance,
   progressive-disclosure pattern, versioning / CI check for code
   snippets). Audience: developer-agents editing the library's
   bundled data files, not library users.

5. **Updated repo layout after restructure:**

   ```
   proxai/
   ├── src/proxai/
   │   ├── ...                        (library code)
   │   ├── agent_skills/              (Layer B — shipped in wheel)
   │   │   ├── proxai-migrate/
   │   │   │   ├── SKILL.md
   │   │   │   └── patterns/...
   │   │   ├── proxai-setup/SKILL.md
   │   │   ├── proxai-debug/SKILL.md
   │   │   └── proxai-best-practices/SKILL.md
   │   └── cli.py                     (proxai CLI, includes
   │                                   `proxai skills install`)
   │
   └── docs/
       ├── user_agents/               (Layer A — repo docs)
       ├── developer_agents/          (Layer A — repo docs)
       │   └── skills_authoring.md    (new)
       ├── llms.txt                   (Layer C — curated index)
       ├── llms-full.txt              (Layer C — CI-generated)
       ├── outline_proposal.md
       └── skill_analyses.md
   ```

6. **`developer_agents/` content is unchanged.** Those docs are
   repo-local by design; the audience is agents editing ProxAI
   itself, not clients using it.

---

## 7. README.md requirements

The root `README.md` is the first surface that humans and agents
encounter — on PyPI, on GitHub, in search results. Aligning it with
the three-layer strategy is a separate deliverable from the docs
restructure, but an important one: without it, agents landing on the
README miss both the Skills entry point and the capability pitch.

**Two audiences served in the first screen:**
- A human developer deciding whether to use ProxAI.
- An AI agent reading the README to learn how to call the library
  (either inside ProxAI's repo or after a client has pip-installed
  it).

### 7.1 What the README needs

1. **Hero: show the value in 5 lines.** A before/after snippet
   contrasting native provider SDK boilerplate against
   `px.generate_text(...)` with multi-provider fallback. Agents
   learn what the library does from this alone.

2. **`proxai skills install` appears in the install block.** Two
   commands now, not one:

   ```bash
   pip install proxai
   proxai skills install            # one-time; loads ProxAI skills
                                    # into your agent (Claude Code,
                                    # Codex, Cursor, …)
   ```

   One sentence explaining what the second command does and why it
   matters (so client-side agents can auto-invoke `proxai-migrate`,
   `proxai-debug`, etc.). Link to the skills list under
   `src/proxai/agent_skills/`.

   **The upgrade flow is documented alongside it** — re-run after
   library updates:

   ```bash
   pip install -U proxai
   proxai skills install            # refresh skills to match lib
   ```

   Point at §5 upgrade lifecycle for staleness behavior (warning on
   mismatch, customization safety, `--force` / `--diff` / `--prune`
   flags).

3. **Show the power, don't hide it.** Explicit short sections:
   - Multi-provider fallback chains (3-line example)
   - ProxDash observability (one-liner + screenshot if possible)
   - The 9 providers supported, as a grid or badge row
   - Feature callouts: multi-modal I/O, thinking budgets, web
     search, caches, structured output (Pydantic / JSON)

4. **Agent-ready structure.** Mirror the Stripe / Pydantic pattern:
   - One-paragraph "what ProxAI is"
   - Five-line usage example
   - Install block (including `proxai skills install`)
   - "For AI agents" subsection with direct links to:
     - `docs/llms.txt` — retrieval index
     - `docs/user_agents/overview.md` — repo docs entry
     - `src/proxai/agent_skills/` — bundled skill sources
   - Feature support matrix link
     (`docs/user_agents/api_guidelines/provider_feature_support_summary.md`)

5. **Structured metadata at the top.** Badges for:
   - PyPI version + Python support matrix
   - License
   - llms.txt link (so agents ingesting the README spot the docs
     retrieval path immediately)
   - CI status / test coverage

6. **Short, skim-friendly.** Target under 300 lines. Deep content
   lives in `docs/user_agents/`. The README is a front door, not a
   reference manual.

7. **Trim anything that duplicates `docs/user_agents/`.** After the
   restructure, the full API reference lives there and on
   `proxai.co`; the README links to it rather than reproducing it.

### 7.2 Why this matters for agents specifically

Client-side agents fall into two modes:
- **Skills installed** → `proxai-migrate` / `proxai-setup` fire
  automatically on relevant prompts. The README is a secondary
  reference.
- **Skills not yet installed** → the agent reads the README as its
  primary source of truth. The README needs enough to get the user
  to run `proxai skills install` and to recognize the library's
  capabilities before hitting a dead end.

Both modes need the README to be explicit about the skills install
step. Burying it two screens down is equivalent to not having it.

### 7.3 Scope note

The README rewrite is a separate deliverable from the docs
restructure and from skill authoring. It should be sequenced
**after** `docs/user_agents/` is largely written, so the README
links to real URLs instead of placeholders.

---

## 8. Design decisions and open questions

**Decided:**

- **Single-package distribution.** Skills ship inside the existing
  `proxai` wheel under `src/proxai/agent_skills/`. The `proxai
  skills install` CLI places them into the user's agent-discoverable
  paths. One version, one release, no decoupling.
- **`llms.txt` lives in the repo.** `docs/llms.txt` (curated) and
  `docs/llms-full.txt` (CI-generated) are canonical; `proxai.co`
  pulls them at deploy time.

**Still open:**

1. **MCP server later?** Skippable for now. Worth revisiting if
   clients start asking for live provider-health querying through
   an agent interface rather than static docs.

2. **Versioning / rot guard.** Skills drift faster than reference
   docs because they are prompts. Minimum safety net: a CI check
   that parses code snippets in each SKILL.md body and validates
   they still import the current ProxAI API (failing the build on
   broken `import proxai as px; px.generate_text(...)` examples).
   Without this, skills rot and agents serve buggy playbooks.

3. **Multi-agent install paths.** `proxai skills install` needs to
   know which tools to target. MVP: auto-detect Claude Code
   (`~/.claude/`). Later: add flags for `--tool=codex`,
   `--tool=cursor`, etc., or `--all` for every installed tool.
   Concrete shape to be decided at implementation time based on
   observed client usage.

---

## 9. Summary

- **Skills are not a replacement for repo docs.** They are an
  additional layer aimed at a different distribution problem.
- **For the refactoring scenario, Layer B is not optional.**
  Without it, a client-side agent has no way to reach the migration
  playbook.
- **The docs restructure stays intact.** Skills ship inside the
  existing `proxai` wheel under `src/proxai/agent_skills/`; clients
  run `proxai skills install` once; there is no separate package
  to manage.
- **`llms.txt` lives in the repo** and the docs site mirrors it.
  Cheap insurance for agents without Skills installed — well under
  a day of work to set up.

**One-line summary:** repo docs for the agent inside ProxAI;
bundled Skills for the agent inside a client's codebase; `llms.txt`
as the fallback when neither is installed — all three shipped from
one package, one repo, one release.

---

## Sources

- [Agent Skills open spec](https://agentskills.io) — Dec 2025
- [Anthropic Agent Skills overview](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview)
- [Claude Code skills docs](https://code.claude.com/docs/en/skills.md)
- [anthropics/skills on GitHub](https://github.com/anthropics/skills)
- [AGENTS.md standard](https://agents.md/) — Aug 2025
- [OpenAI Agentic AI Foundation (Linux Foundation stewardship)](https://openai.com/index/agentic-ai-foundation/)
- [OpenAI Codex Skills](https://developers.openai.com/codex/skills) — Dec 2025
- [Cursor rules](https://cursor.com/docs/context/rules)
- [Simon Willison — Agent Skills writeup (Dec 19 2025)](https://simonwillison.net/2025/Dec/19/agent-skills/)
- [Stripe building with LLMs](https://docs.stripe.com/building-with-llms)
- [stripe/ai Skills package](https://github.com/stripe/ai)
- [vercel-labs/skills](https://github.com/vercel-labs/skills)
- [Pydantic AI llms-full.txt](https://ai.pydantic.dev/llms-full.txt)
- [HumanLayer — Writing a good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [alexop.dev — Progressive disclosure, stop bloating CLAUDE.md](https://alexop.dev/posts/stop-bloating-your-claude-md-progressive-disclosure-ai-coding-tools/)
- [Render — AI coding agents benchmark (context quality > quantity)](https://render.com/blog/ai-coding-agents-benchmark)
