---
name: px-create-doc
description: Create a new ProxAI Layer A documentation file at a given path inside docs/user_agents/ or docs/developer_agents/. Use when populating an empty placeholder with real content or scaffolding a new doc from scratch, following ProxAI's documentation conventions. Example invocation -- /px-create-doc user_agents/api_guidelines/raw_provider_response.md.
disable-model-invocation: true
---

# Create a ProxAI documentation file

A maintainer has asked to create or populate a Layer A documentation
file. The target path was provided as an argument. Follow this
workflow end to end.

## 1. Resolve the target path

Expected argument: a path relative to `docs/`, e.g.
`user_agents/api_guidelines/raw_provider_response.md` or
`developer_agents/cache_internals.md`.

- If the argument is missing a subfolder (e.g.
  `user_agents/raw_provider_response.md`), try to resolve it by
  searching `docs/documentation_outline.md` for the file name and
  placing the doc in the matching folder. Confirm with the
  maintainer before writing if the resolution was ambiguous.
- If the argument does not start with `user_agents/` or
  `developer_agents/`, stop and ask. Layer A docs live in those two
  folders only.
- If the file already exists and is NOT an empty placeholder
  (anything longer than ~5 lines counts as real content), stop and
  confirm before overwriting.

## 2. Internalize the audience split

ProxAI's Layer A documentation has two distinct audiences. Getting
this wrong is the single most common drafting error.

**`docs/user_agents/`** — reference material for AI agents (and
humans) *using* the ProxAI library in their own code. Focus on API
shape, call patterns, behavioral contracts, error modes, worked
examples. Prioritize what a caller needs to predict library
behavior.

Do **not** include in a `user_agents/` doc:
- Conversion / transformation pipelines (e.g., how a dict becomes a
  provider-specific payload, how JSON becomes Pydantic, mime-type
  sniffing).
- Internal state machines, thread pools, async mechanics — unless
  the caller directly observes them.
- Private method signatures, `_` -prefixed helpers, import graphs.
- "We used to do X, now we do Y" internal evolution stories.

These belong in `developer_agents/`. The test: if removing the
paragraph would not change what a caller predicts about the
library's behavior, delete it from `user_agents/`.

**`docs/developer_agents/`** — reference material for AI agents (and
humans) *modifying* ProxAI itself. Covers internals, invariants,
cross-module contracts, data flow, subsystem mechanics. Prioritize
consistency rules that must hold across changes. A reader of this
doc is about to edit source code.

The target path tells which audience applies. Any instinct to
include implementation internals in a `user_agents/` doc, or to
skim over internals in a `developer_agents/` doc, is a signal to
re-read this section.

## 3. Locate the scope in documentation_outline.md

Open `docs/documentation_outline.md`. Find the entry for the target
path. The scope cell in that file is authoritative — the new
document covers what is described there and does not expand beyond
it without flagging the divergence.

If the target path is not listed in `documentation_outline.md`, stop
and ask the maintainer whether the outline needs an update or the
path is wrong. Do not silently add docs outside the outline.

## 4. Read source code and style references

Before writing a single line, read in this order:

1. **The source code the doc describes.** This is the source of
   truth the doc will cite. A doc about `CallRecord` reads the
   CallRecord dataclass in `src/proxai/types.py` and every field on
   it. A doc about `px.models.*` reads the ModelConnector class in
   `src/proxai/client.py`. Current source always wins over memory
   or prior drafts.

2. **At least two sibling docs as style references.** Strongly
   prefer finished docs in the same folder. Canonical references:

   - `docs/user_agents/api_guidelines/px_client_api.md`
   - `docs/user_agents/api_guidelines/px_generate_api.md`
   - `docs/user_agents/api_guidelines/px_models_api.md`

   These establish the intro block, the tree diagram style, section
   numbering, code-sample conventions, and the cross-link pattern.

   If the target is the first doc in a new folder (no finished
   siblings exist), still read the three canonical references
   above — they set the style baseline for the whole project.
   Then pause and confirm with the maintainer whether this doc
   should establish a new sub-style for its folder before drafting.

3. **Any existing placeholder at the target path.** If it has only
   the standard H1 + scope pointer, treat it as empty. If there is
   any substantive prose, confirm before overwriting.

## 5. Ask clarifying questions if needed

Stop and ask the maintainer BEFORE writing when:
- The scope in `documentation_outline.md` is ambiguous.
- The source code and the outline disagree about the surface.
- The target path does not match the audience split (for example,
  a deep subsystem internal landing under `user_agents/`).
- No clear sibling exists to match the style against.

One clarification round is cheaper than a full rewrite.

## 6. Draft the document

Every Layer A doc starts with a consistent opening and includes a
tree diagram near the top.

### 6.1 Intro block with source-of-truth statement and "See also"

The first paragraph below the H1 names the source of truth and
makes it explicit that the source wins over the doc. The
verbatim closing phrase "the files win — update this document" is
the project convention; keep it.

Pattern (adapt the file list and topic; keep the structure):

```markdown
# <Doc Title>

Source of truth: `src/proxai/<file>.py` (role), `src/proxai/<other>.py`
(role), and `src/proxai/types.py` (the dataclasses). If this
document disagrees with those files, the files win — update this
document.

This is the definitive reference for <topic>. Read this before
<use case>.

See also: `<sibling_doc>.md` (role), `<other_sibling>.md` (role),
`<related_doc>.md` (role).
```

If the doc covers a surface governed by a runtime API rather than
a specific source file (e.g., feature-support data queried via
`px.models.*`), the source-of-truth line should say so explicitly
— for example: "Source of truth is `px.models` at runtime, not
this document."

The "See also" block in the intro lists sibling docs the reader
might confuse with this one, each with a one-phrase role. It is
distinct from the end-of-doc "See also" (§6.4), which covers
related topics a reader may want next.

### 6.2 A tree diagram near the top

A tree showing the API surface, type hierarchy, or call path makes
a doc skimmable. Place it in §1, titled `## 1. <Topic> structure
(current)`. The `(current)` suffix is a project convention
signalling that the tree is a snapshot of the present surface, not
an aspirational or historical one. Use a fenced code block with
Unicode box characters.

Concrete example from `px_models_api.md` — copy this style:

```
px.models                                            # same on client.models
│
│   # Configured models (read from registry, never hit the network)
├── .list_models(...)                → list[ProviderModelType]
├── .list_providers(...)             → list[str]
├── .list_provider_models(...)       → list[ProviderModelType]
├── .get_model(...)                  → ProviderModelType
├── .get_model_config(...)           → ProviderModelConfig
├── .get_default_model_list()        → list[ProviderModelType]
│
│   # Working models (run health probes; cached)
├── .list_working_models(...)        → list[ProviderModelType] | ModelStatus
├── .list_working_providers(...)     → list[str]
├── .list_working_provider_models(...) → list[ProviderModelType] | ModelStatus
├── .get_working_model(...)          → ProviderModelType
└── .check_health(...)               → ModelStatus
```

What this example does well:
- Uses Unicode box-drawing characters (`├──`, `└──`, `│`) for
  clean ASCII tree visuals.
- Aligns return-type annotations in a column so the eye can scan.
- Uses inline `# Comment` blocks to group related entries without
  breaking the tree.
- Shows blank-line `│` separators between logical groups.

For deeper patterns study at least two of these finished docs
before drafting:

- `px_client_api.md` — ProxAIClient construction tree with nested
  option types (ConnectionOptions, ProviderCallOptions,
  ModelProbeOptions, DebugOptions).
- `px_generate_api.md` — generate signature tree with parameters
  grouped by function.
- `px_models_api.md` — `px.models` method tree with return types
  (example above comes from here).

Copy the style (box characters, indentation, inline type
annotations, grouping comments) from the closest example. Trees
that no reader can skim fast have failed their job.

### 6.3 Numbered sections with concrete examples

Each major topic gets a numbered H2 section. Short intro, then at
least one runnable code snippet illustrating the topic. Avoid
abstract theory without a worked example — a doc that never shows
the caller's code has failed.

Minimal pattern to match:

    ## 3. Fallback chains

    When a call fails (provider error, rate limit, timeout), ProxAI
    can transparently retry against a different model or provider.

    ```python
    import proxai as px

    px.connect(fallback_models=[
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku"),
    ])

    response = px.generate_text("Summarize this article...")
    ```

    The first entry is primary; subsequent entries are tried in
    order. Default `None` (no fallback). See §5.3 for edge cases.

The shape: numbered H2 → one-paragraph problem statement → code
block → one or two sentences of detail → a `See §X` pointer when
helpful.

**State defaults and behavioral edges explicitly.** When describing
an optional parameter, add a phrase like "Default `None`." or
"Default: empty list." When describing a method that returns the
same instance vs. a new one, say so ("mutates in place; returns
the same instance"). When an error path has non-obvious routing,
call it out ("raises `ValueError` synchronously — not routed
through `suppress_provider_errors`"). These short clarifications
catch the defensive-programming anti-patterns that would otherwise
surface as bug reports.

### 6.4 "See also" at the end

Close with cross-references to related topics a reader may want
next. This is distinct from the intro "See also" (§6.1), which
disambiguates sibling docs covering adjacent concerns.

- Other files in the same folder covering adjacent topics.
- Files in the opposite folder when a reader might want to cross
  over (for example, a `user_agents/` doc may point into
  `developer_agents/` for internals a curious reader might want).

### 6.5 Errors table (for API-surface docs)

Each API-surface doc (`px_client_api.md`, `px_generate_api.md`,
etc.) closes functional sections with a short Errors table listing
every exception a caller can hit at that surface. Pattern:

| Trigger | Error |
|---|---|
| Calling `px.generate_text()` without `px.connect()` | `RuntimeError: no active ProxAIClient` |
| `fallback_models` contains a tuple that is not `(provider, model)` | `ValueError` |

Keep triggers concrete (actual cause of the error, not abstract
descriptions) and error messages short-but-greppable. Open the
table with a one-sentence preamble explaining which exception
class is used and whether these are synchronous vs. routed through
provider-error handling — this pre-empts defensive-programming
mistakes.

Type-reference docs (`call_record.md`) do not need an Errors table
— they describe return shapes, not call sites.

### 6.6 Deprecated or removed surfaces

When a doc describes a surface where a field / parameter / method
used to exist and has been removed, note it in a short paragraph
inside the affected section using the project idiom:

> "The previous `X.y: ZType` field and the `ZType` dataclass have
> been removed entirely. The deserializer silently ignores a
> legacy `y` key in older cached records so historical data still
> loads without error."

This phrasing signals (a) the current surface no longer has the
field, (b) backward-compat behavior for persisted data, (c) why
old callers don't crash. Prefer one inline paragraph at the
relevant section over a separate "Deprecations" section.

## 7. Cross-reference audit

After drafting, scan for other docs that should now link to this
new one, and check that every cross-link in the new doc resolves.

### 7.1 Inbound links

Find docs that should link to the new one. Update them. Typical
candidates:

- `docs/user_agents/overview.md` decision tree.
- `docs/developer_agents/overview.md` decision tree (when the new
  doc is in `developer_agents/`).
- Sibling `*_api.md` files in the same folder whose "See also"
  section covers the same adjacent surface.
- `docs/README.md` "Complete doc inventory" — add one bulleted
  line for the new doc in the correct group (alphabetical within
  `user_agents/` / `developer_agents/` / top-level), and drop
  any `(placeholder)` flag if you just populated an empty file.

Do not leave dangling references. If another doc discusses the
surface you just documented, add a link from there.

### 7.2 Outbound link verification

Every cross-link in the new doc must point at a file that actually
exists. Known drift to guard against: finished API docs in the repo
still reference neighbors by their pre-rename name (e.g.,
`px_client_analysis.md`) rather than the current filename
(`px_client_api.md`). Do not propagate this bug into new work.

For each cross-link:
- Run a shell check: `ls docs/<path>` or grep for the filename in
  the repo tree. If it does not exist, fix the link before
  committing.
- Filenames take precedence over memory or nearby references in
  older docs.

(Updates to `documentation_outline.md` itself are handled
separately in §8 — do not skip that step.)

## 8. Keep documentation_outline.md in sync

The outline (`docs/documentation_outline.md`) is the canonical
inventory of ProxAI's documentation. Every new or substantively
revised doc requires an outline update. Never batch this for
later — when the outline and reality diverge, future agents form
confident-but-false beliefs from the outline.

Apply the minimum edits below after drafting.

### 8.1 Check §1 "Complete repo and docs layout"

The top-level tree in §1 of the outline must include the new file
at its correct path. If absent, add it in the appropriate position
(alphabetical inside a folder, or grouped with related files where
the existing tree groups topically). Preserve indentation and the
Unicode box-drawing style.

### 8.2 Update the scope table for the relevant subsection

Each folder has a table listing its files and one-line scopes:

- §2.3 `user_agents/api_guidelines/`
- §2.4 `user_agents/recipes/`
- §2.7 `developer_agents/`
- §3.1 bundled skills (if the doc sources a Layer B skill)

Find the row for the new file:

- If the row exists and the scope cell still accurately describes
  what was written → no change.
- If the scope cell is narrower, broader, or off-topic relative to
  the doc that actually shipped → update the cell to match
  reality. One line, concrete, no hedging.
- If the row is missing → add it alphabetically or grouped with
  topical neighbors, matching the style of adjacent rows.

### 8.3 Update migration status in §7

The outline's §7 Migration status tracks what has been written vs.
placeholder. After populating a placeholder, reflect that:

- Move the entry from §7.3 "Pending writes" to the "Completed"
  view (or add to §7.1 if that section enumerates individual
  files).
- If a staging file in §7.2 was consumed during the drafting
  process (e.g., `docs/development/query_cache.md` contributed to
  a new `cache_behaviors.md`), update §7.2 to reflect the
  consumption, and delete the staging file if fully drained.

### 8.4 Check §8 Design decisions for contradictions

If something learned while writing the new doc contradicts a
decision recorded in §8 (Design decisions), stop and flag it to
the maintainer. A design-decision change is its own conversation,
not a silent side effect of a doc write.

Example contradiction worth flagging: the outline's §8 states
"Skills ship inside the `proxai` wheel under
`src/proxai/agent_skills/`" but reading the current source reveals
skills are being loaded from a different path. Flag it — either
the code drifted from the decision or the decision needs to be
updated. Either way, one doc write is not the right venue to
settle it.

### 8.5 Verify outgoing section numbers still resolve

If outline edits shifted any numbered section, confirm that
incoming links from placeholder docs and from `skill_analyses.md`
still point at the right sections. Update any that broke.

## 9. Length discipline

Target lengths (adjust as content demands, don't pad):

- API guideline docs: 400-800 lines.
- Recipes: 200-500 lines.
- Troubleshooting: 200-400 lines.
- Feature summaries / cheat-sheets: 150-250 lines.

If the content naturally exceeds ~1000 lines, pause and ask whether
the doc should be split into subtopics rather than shipped as one
unwieldy file.

## 10. Hand off

After the draft is saved, report back with:

- One-paragraph summary of what was written.
- The cross-references that were added or updated in other files.
- The `documentation_outline.md` edits applied in §8 (rows added,
  scope cells updated, migration status changes, staging files
  consumed or deleted).
- Any questions that remained unanswered (if the maintainer
  deferred them) and what default assumptions were made.
- Any design-decision contradictions flagged per §8.4 that the
  maintainer should resolve separately.

Do not mark the doc as "done" — the maintainer reviews and decides.
