# Strategy Analysis — `keep_raw_provider_response`

Status: **unimplemented design proposal**. This document walks through
the option space for exposing the raw provider SDK object
(`openai.types.chat.ChatCompletion`, `anthropic.types.Message`,
`google.genai.types.GenerateContentResponse`, …) on a `CallRecord` so
that end users can reach native provider fields without waiting for
ProxAI to model them first. Nothing here is merged — it is a decision
aid.

Related reading:

- `docs/development/call_record_analysis.md` — the shape of
  `CallRecord` that any new field has to fit into.
- `docs/development/px_client_analysis.md` — the user-facing
  reference for `px.Client(...)` / `px.connect(...)` options. §2 is
  where any new client-level option gets documented, §3 is where
  per-call `ConnectionOptions` fields live, §4 is the error catalog
  this proposal adds to, and §5 is the implicit-behaviours section
  the one-time warning belongs in.

---

## 1. Problem statement

ProxAI's abstraction is lossy by design: it folds many providers into
a common `CallRecord` shape. Users repeatedly run into fields that
ProxAI hasn't modelled yet — `logprobs`, `system_fingerprint`,
`prompt_filter_results`, Anthropic cache read/write tokens, reasoning
details, detailed finish reasons, safety ratings, and so on. Today
their only options are to fork ProxAI or drop back to the raw SDK.

A natural escape hatch is to expose the raw SDK response object that
the connector executor already has in scope. Implementation cost is
roughly one line per executor. The hard part is not the write — it is
deciding what guarantees ProxAI is willing to make about the field,
what happens when the cache takes over, and how strongly to discourage
users from relying on it for portable code.

The rest of this document enumerates the design axes and the option
space on each, with pros and cons. Section 9 assembles a recommended
combination that matches the user's stated constraints:

- Client-level opt-in flag.
- **Error** when cache is enabled *and* the flag is on.
- **Warn** every time the flag is used even without a cache, flagging
  the feature as debug-only / unstable / "contact us for a feature
  request".
- Hand back the live SDK object — not a dict form.

---

## 2. Design axes

Any implementation has to answer nine independent questions. The
sections that follow walk through each axis and its options. You can
mix and match — they are mostly orthogonal, though some combinations
are called out as incompatible.

1. **Where does the flag live?** Client-level, per-call, both,
   environment variable.
2. **Where does the captured data live?** `ResultRecord` field,
   `CallRecord` field, separate `Debug` sidecar, weak reference.
3. **What is the default?** Off, on, on-in-dev, on-in-TEST only.
4. **What happens when the query cache is enabled?** Error, warn,
   silent `None`, refuse to enable cache.
5. **How loudly do we warn when the flag is on?** Never, once per
   process, every call, configurable, log-only.
6. **Which attempts get a captured object in a fallback chain?** Only
   the returned record, all attempts, first attempt only.
7. **What is the mock-mode contract?** Mock object, `None`,
   passthrough.
8. **How does serialization behave?** Skipped in all serializers,
   serialized best-effort, serialized with round-trip guarantees.
9. **What is the field named and typed?** `raw_provider_response`,
   `_raw_provider_response`, `native_response`, `provider_sdk_object`,
   `debug.raw_response`. Type as `Any` vs. `object` vs. a Protocol.

---

## 3. Axis 1 — where does the flag live?

### 3.1 Option A: Client-level only

Add a `keep_raw_provider_response: bool = False` kwarg to
`px.Client(...)` and `px.connect(...)`, alongside the other
construction-time options documented in
`px_client_analysis.md §2`. The flag is read inside the connector
executor when deciding whether to capture the raw response.

**Pros.**
- Matches how every other client-wide default works
  (`suppress_provider_errors`, `feature_mapping_strategy`, etc.);
  zero surprise for users reading `px_client_analysis.md §2`.
- Single place to validate-and-error when the cache is enabled (§6).
- A user who needs it sets it once on their dev `Client` and never
  thinks about it again.
- Easy to put behind a prominent warning on enable (§7).

**Cons.**
- No per-call opt-out for the rare case of "I only want the raw object
  for one debug call but keep my production client clean."
- Users who share a `Client` between production and debug code paths
  have to weigh risk on a client boundary rather than a call boundary.

### 3.2 Option B: Per-call only

Add `keep_raw_provider_response: bool | None = None` to
`types.ConnectionOptions`. Lives alongside the other per-call fields
documented in `px_client_analysis.md §3`.

**Pros.**
- Precise scoping — one call at a time. No "did someone else turn this
  on on the shared client?" ambiguity.
- Matches `skip_cache` / `override_cache_value` as another debug-y
  per-call switch.

**Cons.**
- Every debug session becomes "thread the kwarg through every call
  site" — users will instead copy/paste `connection_options=...` and
  sometimes miss places.
- Harder to put a loud one-time warning in front of it (you'd be
  warning on every call by default).
- Doesn't solve the "I want this to be on during my notebook session
  but off in CI" use case cleanly.

### 3.3 Option C: Both (client default + per-call override)

Client-level `keep_raw_provider_response: bool = False` sets the
default; `ConnectionOptions.keep_raw_provider_response: bool | None =
None` overrides it for a single call (`None` = inherit).

**Pros.**
- Full flexibility — users get precise scoping when they need it and
  a simple global toggle when they don't.
- Mirrors how `suppress_provider_errors` already splits between client
  default and per-call override (see `px_client_analysis.md §2.6` and
  §3.2).

**Cons.**
- Two surfaces to document, test, and validate. The cache-interaction
  check (§6) has to run on whichever value actually ends up effective
  per call, not on either in isolation.
- Easier for users to leave the flag on by accident — "I turned it on
  once per call" is less visible than "I turned it on at connect()".

### 3.4 Option D: Environment variable

`PROXAI_KEEP_RAW_PROVIDER_RESPONSE=1`.

**Pros.**
- Zero code changes at call sites; ideal for ad-hoc debugging an
  already-deployed script.
- Mirrors how debug flags are typically wired in other ecosystems.

**Cons.**
- Invisible in the code — reproducibility suffers. Tests pass locally
  and fail in CI (or vice versa) because of an env var nobody
  documented.
- ProxAI has no existing pattern of reading env vars at call time;
  introducing one just for this feature is disproportionate.
- Cannot be audited via `client.get_current_options()`.

### 3.5 Recommendation on Axis 1

**Option A (client-level only).** The user explicitly said "Having on
client make sense", and client-level keeps validation honest (§6) and
warnings easy to scope (§7). Start with A; if per-call override turns
out to be necessary, upgrade to C later as a strict addition (adding a
per-call override to a client-level default is backward compatible;
the reverse is not).

---

## 4. Axis 2 — where does the captured data live?

### 4.1 Option A: `ResultRecord.raw_provider_response`

Put the field directly on `ResultRecord` next to `content`, `choices`,
`error` (see `call_record_analysis.md §1`).

**Pros.**
- Parallel to `content` / `output_text` — users read
  `call_record.result.raw_provider_response` the same way they read
  `call_record.result.output_text`. Shortest navigation path.
- Makes sense at the conceptual level: the raw response *is* part of
  the result.

**Cons.**
- Pollutes the canonical result shape. Every diff that touches
  `ResultRecord` now has to reason about whether `raw_provider_response`
  applies.
- Hard to communicate "this field is not part of the stable contract"
  when it sits shoulder-to-shoulder with stable ones.
- Forces `call_record_analysis.md` and the serializers to carry a
  permanent caveat for a field that is supposed to be debug-only.

### 4.2 Option B: `CallRecord.raw_provider_response`

Put the field on the top-level `CallRecord` (sibling of `query`,
`result`, `connection`).

**Pros.**
- Clearly signals "this is record-level metadata, not part of the
  result shape" — the field sits next to `connection` (which also
  carries flow metadata).
- `ResultRecord` stays clean — users eyeing `result.*` will not even
  see it.
- Single, obvious site for serializers/exporters to skip.

**Cons.**
- Slightly awkward ergonomics: `call_record.raw_provider_response`
  reads fine, but naming like "result" is what users expect to contain
  the raw response.

### 4.3 Option C: `CallRecord.debug: DebugInfo | None`

Introduce a small sidecar dataclass:

```python
@dataclass
class DebugInfo:
    raw_provider_response: Any | None = None
    # future: provider_request_payload, retry_counters, ...
```

and hang it off `CallRecord.debug`.

**Pros.**
- **Strongest stability signal.** The word `debug` in the accessor
  path (`call_record.debug.raw_provider_response`) tells every reader
  "not part of the stable contract."
- Future-proof: other debug-only fields (the outbound request
  payload, SDK version tags, retry counters, timing breakdown) can
  join without sprawling `CallRecord` further.
- Serializer contract becomes cleanly "skip the `debug` subtree
  entirely" rather than "skip this one field inside
  `ResultRecord`."

**Cons.**
- Adds a new dataclass type that users have to learn.
- Slightly longer access path.
- If only one debug field ever exists, the sidecar is over-engineered.

### 4.4 Option D: Side-channel (weak reference, context-local dict)

Keep the raw object in a process-local
`dict[CallRecord, Any]` or a `weakref.WeakKeyDictionary` that the
connector populates and expose a `px.last_raw_response(call_record)`
accessor.

**Pros.**
- Completely off the `CallRecord` — zero impact on serialization,
  pickling, multiprocessing, and downstream tooling.
- Feels like the strongest "this is debug, don't rely on it" signal.

**Cons.**
- Multiprocessing is broken by definition — the raw object doesn't
  survive the IPC boundary, and the side-channel dict isn't shared
  anyway.
- Subtle lifetime semantics (weakref collection races, etc.).
- Accessor API is harder to discover and harder to document. Users
  will miss it.
- Breaks the "look at the `CallRecord` and see everything about the
  call" mental model that ProxAI currently has.

### 4.5 Recommendation on Axis 2

**Option C (`CallRecord.debug.raw_provider_response`).** It scores
highest on the "tell users this is unstable" axis, which is the whole
point of the feature. If you are worried about the overhead of a new
sidecar type, Option B (`CallRecord.raw_provider_response`) is a good
compromise — still off `ResultRecord`, still skippable by serializers
as a single top-level field. Do **not** ship Option A; it makes
`ResultRecord` harder to reason about long-term.

---

## 5. Axis 3 — what is the default?

### 5.1 Option A: Off by default (opt-in)

Users must explicitly pass `keep_raw_provider_response=True` to get the
field. Default-off.

**Pros.**
- Zero impact on users who don't want it. Production code stays the
  same. No surprise pickling failures, no inflated memory usage in
  hot loops, no cache interaction footguns for anyone who didn't
  opt in.
- Naturally enforces the "debug/escape-hatch only" framing.
- Pairs cleanly with a loud warning on enable (§7).

**Cons.**
- Users first trying to debug a tricky provider response have to
  discover the flag exists before they can use it.

### 5.2 Option B: On by default

Always capture the raw object unless explicitly disabled.

**Pros.**
- Maximum convenience — no "wait, I have to reconnect" friction.

**Cons.**
- Cache interaction (§6) becomes a silent footgun for everyone.
- Pickling in `allow_multiprocessing=True` now fails for everyone
  unless every SDK object happens to pickle cleanly.
- Memory blow-up on high-throughput callers who are batching
  `CallRecord`s for later analytics.
- Hard to later ship a warning — every existing user already has it
  on and will see the warning spam.

### 5.3 Option C: On in `RunType.TEST`, off in production

Auto-enable when the client is in `TEST` mode (mocks) and disable
otherwise.

**Pros.**
- Test suites can always inspect the mock object without opt-in.
- No production cost.

**Cons.**
- Surprising: the same code behaves differently under `TEST` and
  `PRODUCTION` purely around a debug-ergonomics feature.
- Mock objects are not the same shape as real SDK objects, so the
  tests that touch them end up locked to mock internals.

### 5.4 Recommendation on Axis 3

**Option A (off by default).** This is the only sane answer given the
cache-interaction story (§6) and the user's intent to keep the feature
positioned as debug-only. Defaulting to on would require suppressing
every warning we want to add, and would silently break multiprocessing
for the average user.

---

## 6. Axis 4 — cache interaction

This is the single biggest footgun in the whole design. The raw SDK
object is unambiguously a live object: pydantic models for OpenAI and
Anthropic, plain Python objects for Google, etc. The query cache only
persists the pieces of `ResultRecord` that ProxAI knows how to
serialize. A cached reply can reconstruct `content`, `output_text`,
`usage`, etc., but cannot reconstruct the SDK object — the original is
long gone.

### 6.1 Option A: Hard error when both enabled

Raise `ValueError` at client construction time if both `cache_options`
and `keep_raw_provider_response=True` are set. This becomes a new row
in the error catalog in `px_client_analysis.md §4`.

**Pros.**
- No footgun — the impossible configuration is rejected at the point
  it is assembled, not discovered later. Users get a clear, early,
  fixable error message.
- Matches how the client already treats mutually exclusive options
  (see `px_client_analysis.md §4` for the existing catalog:
  `fallback_models` + `suppress_provider_errors`,
  `fallback_models` + `endpoint`, `cache_options` missing
  `cache_path`, etc.).
- Documentation becomes trivial: "`keep_raw_provider_response` and
  `cache_options` are mutually exclusive."

**Cons.**
- Users who want both convenience *and* occasional raw-object access
  must pick one. They can construct a second `Client` for the debug
  path, but that's a workflow change.
- Slightly blunt: the flag works fine on a cache *miss*; the error is
  really about cache *hits*. But distinguishing per-call is more
  complicated than it is worth.

### 6.2 Option B: Warn at setup, silently return `None` on cache hit

Allow the combination but log a warning once at setup and set
`raw_provider_response = None` on cache hits. Provider-path calls get
the object; cache-path calls get `None`.

**Pros.**
- Maximum flexibility: the flag is useful on the provider path and
  harmless on the cache path.

**Cons.**
- The exact "asymmetry footgun" from the earlier analysis: user code
  that does `assert call_record.debug.raw_provider_response is not None`
  works in tests and breaks in production as soon as cache warms up.
- Users learn about the gotcha only after a bug report.
- Warning is easy to miss (it fires once at setup, long before the
  cache hit that actually breaks things).

### 6.3 Option C: Warn on every cache hit

Same as B, but emit a warning every time a cache hit returns `None`
for `raw_provider_response`.

**Pros.**
- Much harder to ignore than B — the warning fires exactly at the
  point where user code would fail.

**Cons.**
- Warning spam on hot paths that legitimately mix cache hits and
  provider calls.
- Adds conditional logic in the cache-hit branch of the request flow.
- Does not change the underlying fact that the field silently flips
  between two shapes (None vs. live object).

### 6.4 Option D: Disable cache for calls with the flag on

Per-call override: when `keep_raw_provider_response=True` is effective
for a call, force `skip_cache=True` internally. Client-level flag
quietly overrides cache for every call.

**Pros.**
- No more silent `None`. Every call that asks for the raw object
  actually gets one.

**Cons.**
- Extremely surprising: setting a debug flag silently disables an
  important performance feature the user explicitly configured. Most
  users will not expect the flag to nuke their cache.
- If the flag is client-wide, this effectively disables the cache for
  the whole client. The user would have been better off not
  configuring the cache at all.
- Hides the real incompatibility rather than surfacing it.

### 6.5 Recommendation on Axis 4

**Option A (hard error).** Matches the user's explicit preference and
matches how the client already handles mutually exclusive options.
It is the only choice that prevents the asymmetry footgun entirely
rather than papering over it. Users who need the rare "both"
combination can construct two separate clients (a cached production
client + an uncached debug client), which is also the honest shape of
what they are doing anyway.

**Implementation constraint.** The mutual-exclusion check must run at
client construction so that **no** construction path can produce the
forbidden combination — both the direct kwarg form
(`px.Client(cache_options=..., keep_raw_provider_response=True)`) and
any internal state-restoration path must reject it. This matches the
eager-raise pattern for other client-level conflicts in
`px_client_analysis.md §4`.

---

## 7. Axis 5 — warning policy when the flag is on

Independent of cache interaction. The user wants a clear signal that
this field is for debug only and may disappear / change shape.

### 7.1 Option A: Never warn

Relies entirely on documentation.

**Pros.**
- Clean runtime output.

**Cons.**
- Users miss the "unstable, debug-only" framing; the feature becomes
  de facto stable.

### 7.2 Option B: Warn once per process on first enable

`warnings.warn(...)` via the standard `warnings` module, filtered to
`default`, emitted the first time the flag is set to `True` on any
client in the process.

**Pros.**
- Visible, but not spammy.
- Standard Python idiom; users know how to silence it if they truly
  need to.
- Fires at `connect()`-time, which is the right moment to include
  a "contact us for a feature request" pointer.

**Cons.**
- A user who mishears the warning once and ignores it will never see
  it again for the rest of the process.
- Doesn't distinguish between "I enabled this on purpose" and "a
  library I depend on enabled this on my behalf."

### 7.3 Option C: Warn on every call with the flag effective

Emit a warning inside the connector every time the captured field is
actually populated.

**Pros.**
- Impossible to miss.
- Forces users to treat production usage as abnormal.

**Cons.**
- Noisy at scale. A batch job with 10k calls produces 10k warnings.
- Users will install blanket filters — the *next* warning they need
  to see gets hidden too.
- Adds per-call overhead to the hot path.

### 7.4 Option D: Log at connect time, not via warnings module

Route through ProxAI's existing logging utilities (the same layer
`logging_options` feeds into — see `px_client_analysis.md §2.3`) at
the moment the flag is enabled, instead of raising a Python warning.

**Pros.**
- Respects the user's logging setup. Users who already route ProxAI
  logs to stdout / a file see it immediately.
- Does not pollute stderr in environments where `warnings` are
  captured aggressively (Jupyter, pytest's `-W error`).

**Cons.**
- Easier to miss entirely if the user hasn't wired any logging.
- Python warnings are the more universally recognised signal.

### 7.5 Option E: Both — one warning + one log

Warn once via `warnings.warn(...)` (standard Python signal) **and**
emit a log line once through the ProxAI logging utilities at the same
moment, so the user's own log stream also records it. Subsequent
`generate()` calls are silent.

**Pros.**
- Maximum visibility at enable time; no runtime overhead afterwards.
- Matches what libraries like `pandas` and `transformers` do for
  experimental features.
- Gives a natural place to put the "contact the ProxAI team" ask.

**Cons.**
- Two warnings to maintain and keep phrased consistently.
- Slightly more code.

### 7.6 Recommendation on Axis 5

**Option E (warn once + log once at enable time).** The user
explicitly asked for a warning message even when cache is off, with
wording like "this is debugging intention only, please contact the
ProxAI team for a feature request, unstable in the long term." E is
the only option that makes that message both conspicuous *and*
debuggable by the user's own logging plumbing, without producing
per-call spam.

Sketch of the message:

> `keep_raw_provider_response=True` is a debugging-only escape hatch.
> The raw provider response is not part of ProxAI's stable contract,
> is not serialized to the query cache or ProxDash, and may break at
> any provider SDK upgrade. If you need a specific provider field
> surfaced as a first-class `CallRecord` attribute, please reach out
> to the ProxAI team so we can model it properly instead of having
> you depend on this hatch long-term.

The message should also mention that the flag is mutually exclusive
with `cache_options` — users who trip the §6.1 error should already
know why, but reinforcing it in the warning text at enable time makes
the constraint obvious even when no cache is configured yet.

---

## 8. Remaining axes (brief)

These are smaller decisions. A single recommended option is enough.

### 8.1 Axis 6 — fallback chain semantics

Only the **returned** `CallRecord` carries the raw object — the one
the caller actually receives from `client.generate()`. Attempts that
failed and were internally logged (see
`call_record_analysis.md §2.7` for the fallback semantics) have their
own `debug.raw_provider_response = None`: either there was no raw
object (the call failed) or it belongs to a previous attempt the
caller never sees. This matches the existing "one record out" contract
of the client and keeps the implementation to a single per-attempt
capture point.

### 8.2 Axis 7 — mock mode

In `RunType.TEST`, the connector talks to a mock SDK client instead of
a real provider SDK. Whatever that mock returns *is* the raw response
from the field's point of view. Do not special-case mock mode — if the
flag is on, the captured object is the mock. This keeps the contract
simple and teachable: "the field holds whatever the executor got back
from the provider call." Document it as a known caveat: tests written
against `raw_provider_response` in `TEST` mode couple to the mock
shape.

### 8.3 Axis 8 — serialization

- **Query cache:** skip the field. Cached replies reconstruct
  everything *except* the raw object, which would become `None` on
  deserialize. Note that the Axis 4 error prevents users from ever
  observing this branch in practice.
- **ProxDash upload:** skip the field. Telemetry is not the place for
  provider-native objects.
- **Client option persistence:** this is about the *flag* itself, not
  the captured field. `keep_raw_provider_response: bool` is stored
  alongside every other client option and appears in the
  `client.get_current_options()` snapshot just like
  `suppress_provider_errors` does (see
  `px_client_analysis.md §5.9`). The captured *object* is never
  persisted anywhere — it lives only inside
  `CallRecord.debug` during the hot request path.
- **Pickling / multiprocessing:** if the user calls
  `copy.deepcopy(call_record)` or passes a record across a
  `multiprocessing` boundary, the raw field may fail to pickle. This
  is unavoidable given the design — document it, and note that
  ProxAI's own multiprocessing (the health-check worker pool gated
  by `allow_multiprocessing`, see `px_client_analysis.md §2.7`) does
  not pass `CallRecord`s across processes at all, so the core library
  is unaffected.

### 8.4 Axis 9 — naming and typing

- **Name:** `raw_provider_response`. Clearer than `native_response`;
  less jargon than `provider_sdk_object`. Staying literal keeps the
  field honestly labelled.
- **Location:** `CallRecord.debug.raw_provider_response` (per §4.5
  recommendation). If Option B is chosen instead,
  `CallRecord.raw_provider_response` is the alternative.
- **Type annotation:** `Any | None`. Use `Any` deliberately, not
  `object`, so that `pyright` / `mypy` treat downstream attribute
  access as unchecked. The whole point of the field is that its
  shape is owned by a third-party SDK.
- **Docstring:** must state (1) debug-only, (2) unstable across SDK
  upgrades, (3) mutually exclusive with `cache_options` at the client
  level, (4) not pickle-safe, (5) how to request a properly modelled
  alternative.

---

## 9. Recommended combination

Pulling all the axis recommendations together:

| Axis | Decision |
|------|----------|
| 1. Flag location | **Client-level only** — a `keep_raw_provider_response` kwarg on `px.Client(...)` / `px.connect(...)`, alongside the other construction-time options in `px_client_analysis.md §2`. Upgradable to client+per-call later if needed. |
| 2. Field location | **`CallRecord.debug.raw_provider_response`** via a new `DebugInfo` sidecar dataclass. Acceptable fallback: `CallRecord.raw_provider_response`. |
| 3. Default | **Off**. |
| 4. Cache interaction | **Hard `ValueError`** at client construction when `cache_options` and `keep_raw_provider_response=True` are both set. Added to the `px_client_analysis.md §4` error catalog. |
| 5. Warning policy | **Once at enable time, via both `warnings.warn` and the ProxAI logging utilities** (Option E). Message includes the "debug-only, unstable, file a feature request" framing. |
| 6. Fallback semantics | **Only the returned record carries the object**; intermediate failed records have `debug.raw_provider_response = None`. |
| 7. Mock mode | **Passthrough** — whatever the mock SDK returns is the captured value. Documented caveat. |
| 8. Serialization | Query cache, ProxDash, and client-option serializers **skip** the field. Pickling caveat documented. |
| 9. Naming & typing | **`raw_provider_response: Any | None`** on `DebugInfo`. Docstring states the five caveats above. |

### 9.1 Concrete API shape

```python
import proxai as px

# Enabling the flag
client = px.Client(
    keep_raw_provider_response=True,
    # cache_options=...  # ← raises ValueError if both are set (§6)
)
# At this moment:
#   1. warnings.warn(<debug-only message>, UserWarning, stacklevel=2)
#   2. ProxAI logging utility emits the same message once through the
#      client's logging_options pipeline.

rec = client.generate(prompt="hi")

# Canonical, stable access — unchanged:
print(rec.result.output_text)
print(rec.result.content)

# Debug escape hatch — new, documented as unstable:
raw = rec.debug.raw_provider_response  # type: Any
# For openai chat.completions.create this is a ChatCompletion;
# for anthropic.messages.create it is a Message; for google.genai
# it is a GenerateContentResponse; etc.
print(raw.system_fingerprint)  # provider-specific — not portable
```

Attempting the forbidden combination:

```python
client = px.Client(
    keep_raw_provider_response=True,
    cache_options=px.CacheOptions(cache_path="/tmp/cache"),
)
# ValueError: keep_raw_provider_response=True is incompatible with
# cache_options (cache lookups cannot recover provider SDK objects).
# To use both, construct two clients: one cached for production and
# one with keep_raw_provider_response=True for debugging.
```

### 9.2 Implementation checklist

Minimum set of changes, scoped to user-observable impact where
possible:

1. **`types.py`** — Add a `DebugInfo` dataclass with
   `raw_provider_response: Any | None = None`. Add
   `CallRecord.debug: DebugInfo | None = None`. Nothing else on
   `CallRecord` changes.
2. **Client construction** — Add
   `keep_raw_provider_response: bool = False` to both `px.Client(...)`
   and `px.connect(...)`. Enforce the mutual exclusion with
   `cache_options` at construction time: raise `ValueError` if both
   are truthy, along every path that builds a client. On first
   enable, emit the debug-only message (§7.6) once via
   `warnings.warn(...)` and once via the ProxAI logging utility so it
   flows through the user's `logging_options`.
3. **Connector capture point** — In the request pipeline, after the
   executor returns a successful response and before the result
   adapter runs, attach the raw response object to
   `CallRecord.debug.raw_provider_response` if the flag is on. Leave
   `debug = None` on any cache-hit path (moot because of §6.1, but
   defensive).
4. **Serialization** — Skip `CallRecord.debug` in the query cache
   serializer and the ProxDash upload payload. Add a regression test
   that round-tripping a `CallRecord` through each serializer zeroes
   the field.
5. **`get_current_options()`** — Include `keep_raw_provider_response`
   in the `RunOptions` snapshot so reproducibility manifests
   (`px_client_analysis.md §5.9`) reflect the setting.
6. **Documentation.**
   - `call_record_analysis.md`: add `CallRecord.debug` to §1 with a
     one-line note pointing at this proposal.
   - `px_client_analysis.md`: add a `§2.9 keep_raw_provider_response`
     block in the client-level options reference; add one row to
     the §4 error catalog for the new `ValueError`; add one bullet to
     §5 (implicit behaviours) describing the one-time warning + log
     at enable time.

---

## 10. Things to reconsider later

- **Per-call override (Axis 1, Option C).** If users report that the
  client-level flag is too coarse, adding
  `ConnectionOptions.keep_raw_provider_response` is a strictly
  additive change. Defer until real user demand shows up.
- **Optional dict form (Axis 9 alt).** The user rejected a dict
  variant in favour of the raw object. If pickling / cache
  interaction turns out to be painful in practice, reintroducing a
  `raw_provider_response_dict` that *does* round-trip is a possible
  follow-up — but it should be a separate field, never a replacement
  for the live object.
- **Namespace.** If more debug fields accumulate on `DebugInfo`
  (outbound request payloads, retry timing, connector-level caches,
  ...), it might warrant a dedicated subsystem alongside the existing
  caching / logging ones. Defer until there are at least three
  concrete fields.
- **Typing.** If the `Any` annotation causes downstream type-checker
  complaints, a `typing.TypeAlias` (`RawProviderResponse = Any`) gives
  a named handle without narrowing. Purely cosmetic.
