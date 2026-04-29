# Cache Internals

Source of truth: `src/proxai/caching/query_cache.py`
(`QueryCacheManager`, `ShardManager`, `HeapManager` — query response
storage), `src/proxai/caching/model_cache.py` (`ModelCacheManager`
— model-status persistence),
`src/proxai/serializers/hash_serializer.py` (the query-record hash
function that defines cache identity),
`src/proxai/type_utils.py` (`is_query_record_equal` and
`_normalize_chat_for_comparison` — the paired equality check that
guards against hash collisions), and `src/proxai/types.py` (the
dataclasses: `CacheRecord`, `LightCacheRecord`, `CacheLookResult`,
`CacheLookFailReason`, `CacheOptions`, `QueryCacheManagerState`,
`ModelCacheManagerState`). If this document disagrees with those
files, the files win — update this document.

This is the definitive reference for how ProxAI's two caches are
implemented — on-disk layout, the freshness invariant that keeps
deleted buckets from coming back to life, the hash-and-equality
contract for cache identity, LRU eviction, shard / backlog
dynamics, restart recovery, and the model-cache single-file
design. Read this before changing any cache algorithm, adding a
field to `QueryRecord` (without also updating hash + equality),
editing `ShardManager`, tweaking eviction, or changing the model-
cache file format.

See also: `../user_agents/api_guidelines/cache_behaviors.md`
(caller-level view — what each flag does to read/write paths and
the matrix of `CacheOptions` / `ConnectionOptions` combinations);
`state_controller.md` (`QueryCacheManager` and `ModelCacheManager`
are both `StateControlled`, so their fields follow the
state-container nesting rules); `adding_a_new_provider.md`
(the executor-side consumer — cache hooks run before and after
the executor but executors themselves are cache-agnostic).

---

## 1. Cache subsystem structure (current)

```
ProxAIClient                                         # src/proxai/client.py
│
├── _query_cache_manager:  QueryCacheManager    ← state-controlled
├── _model_cache_manager:  ModelCacheManager    ← state-controlled
└── _default_model_cache_manager: ModelCacheManager (platformdirs fallback)

ProviderConnector                                    # pipes the client's managers through
│
├── _get_cached_result(query_record, conn_opts)     # before executor
│     └── query_cache_manager.look(query_record)
├── _update_cache(call_record, conn_opts)            # after executor
│     └── query_cache_manager.cache(...)
└── query_cache_manager: QueryCacheManager          # shared reference

QueryCacheManager (query_cache.py:455)                # public surface
│
├── .status              → QueryCacheManagerStatus
├── .look(qr, update=True, unique_response_limit=None) → CacheLookResult
├── .cache(qr, rr, unique_response_limit=None, override_cache_value=False)
└── .clear_cache()                                   # wipe entire query_cache dir
      │
      ├─ ShardManager (query_cache.py:133)           # storage layer
      │    │
      │    ├── _light_cache_records: dict[hash, LightCacheRecord]
      │    │   │   # in-memory index; single source of truth for "what's live"
      │    │   └── persisted to light_cache_records_{shard_count:05}.json
      │    │
      │    ├── _loaded_cache_records: dict[hash, CacheRecord]
      │    │   │   # lazy cache of full records; populated by _load_shard
      │    │
      │    ├── _map_shard_to_cache: dict[shard_id, set[hash]]
      │    ├── _shard_active_count: dict[shard_id, int]
      │    └── _shard_heap: HeapManager (smallest-first)
      │
      ├─ HeapManager (query_cache.py:75)            # priority queue w/ lazy deletion
      │    │
      │    ├── _heap: heapq of (value, key)
      │    ├── _active_values: dict[key, value]
      │    └── with_size=True mode tracks _total_size for LRU gating
      │
      └─ _record_heap: HeapManager(with_size=True)  # LRU eviction per-bucket

ModelCacheManager (model_cache.py:22)                 # separate manager, simpler model
│
├── .status              → ModelCacheManagerStatus
├── .get(output_format_type)         → ModelStatus   # expire-on-read per model
├── .update(model_status_updates, output_format_type)  # incremental merge
├── .save(model_status, output_format_type)          # replace wholesale
└── .clear_cache()
      │
      └── _model_status_by_output_format_type        # in-memory dict
            │   # persisted as one JSON file: {cache_path}/available_models.json

Query cache on-disk layout: {cache_path}/query_cache/
  ├── light_cache_records_{shard_count:05}.json          ← index (JSONL, append-only)
  ├── light_cache_records_{shard_count:05}.json_backup   ← pre-compact snapshot
  ├── shard_{i:05}-of-{shard_count:05}.jsonl             ← committed shards
  └── shard_backlog_{shard_count:05}.jsonl               ← write buffer

Model cache on-disk layout:
  {cache_path}/available_models.json                     ← single JSON dict per manager
```

The query cache and model cache are independent managers with
different shapes — the query cache is append-heavy, sharded, and
LRU-bounded; the model cache is a small in-memory dict flushed
wholesale to one JSON file. They share only a config dataclass
(`CacheOptions`) and the `state_controller.StateControlled` base.

### 1.1 Three-layer query cache (public / storage / priority queue)

| Layer | Class | Responsibility |
|---|---|---|
| Public API | `QueryCacheManager` (`query_cache.py:458`) | `look()` / `cache()` / `clear_cache()`, LRU heap, status machine, state propagation |
| Storage | `ShardManager` (`query_cache.py:135`) | Sharded JSONL files, in-memory indices, backlog flush + shard merge |
| Priority queue | `HeapManager` (`query_cache.py:77`) | Lazy-deletion heap, used twice: (a) shard-fill ordering (by active count), (b) LRU eviction (by last_access_time, `with_size=True`) |

The storage layer never reads `CacheOptions` or `CacheLookResult`
— it speaks purely in `CacheRecord` / `LightCacheRecord`. The
public layer never reads a shard file directly — it goes through
`ShardManager`. Respect this split when adding functionality;
mixing layers is the shortest path to a hash-equality drift bug
(§5).

### 1.2 Status machine

Both managers have a `status` property whose value drives every
operation:

```
QueryCacheManagerStatus
│
├── INITIALIZING                   # transient; __init__ sets it first
├── CACHE_OPTIONS_NOT_FOUND        # cache_options is None
├── CACHE_PATH_NOT_FOUND           # cache_options.cache_path is None
├── CACHE_PATH_NOT_WRITABLE        # makedirs or os.access failed
└── WORKING                        # the only state in which look / cache do I/O

ModelCacheManagerStatus adds:
└── DISABLED                       # cache_options.disable_model_cache is True
```

`QueryCacheManager.look()` returns
`CacheLookResult(cache_look_fail_reason=CACHE_UNAVAILABLE)` in any
non-`WORKING` state; `cache()` logs a warning and returns a no-op.
The cache is an *optimization*, and its unavailability must never
crash the calling path. If you find cache code raising
`ValueError` on a non-`WORKING` status outside of
`clear_cache()`, that's a bug.

---

## 2. Core data types

Defined in `src/proxai/types.py`.

| Type | Lines | Purpose |
|---|---|---|
| `CacheRecord` | `types.py:702-710` | Full bucket stored on disk: `query`, `results: list[ResultRecord]`, `shard_id`, `last_access_time`, `call_count`. One per query hash. |
| `LightCacheRecord` | `types.py:713-721` | Index entry held in memory for every bucket: `query_hash`, `results_count`, `shard_id`, `last_access_time`, `call_count`. |
| `CacheLookResult` | `types.py:724-729` | Return value of `look()`. Either `result` (a `ResultRecord`) or `cache_look_fail_reason` set. |
| `CacheLookFailReason` | `types.py:659-665` | `CACHE_NOT_FOUND`, `CACHE_NOT_MATCHED`, `UNIQUE_RESPONSE_LIMIT_NOT_REACHED`, `CACHE_UNAVAILABLE`. |
| `CacheOptions` | `types.py:349-385` | User-facing config: `cache_path`, `unique_response_limit=1`, `clear_query_cache_on_connect=False`, `disable_model_cache=False`, `clear_model_cache_on_connect=False`, `model_cache_duration=None`. |
| `QueryCacheManagerState` | `types.py:811+` | State container: `status`, `cache_options`, `shard_count`, `response_per_file`, `cache_response_size`. |
| `ModelCacheManagerState` | `types.py` | State container: `status`, `cache_options`, `model_status_by_output_format_type`. |

`LightCacheRecord` is the compact form that lives in the in-memory
`_light_cache_records` dict for every bucket the cache knows
about; the full `CacheRecord` is loaded lazily per shard on
demand. A `CacheRecord`'s `shard_id` can be an `int` (committed
shard 0..shard_count-1) or the literal string `'backlog'`.

`CacheLookFailReason` has four values, each with a distinct
meaning the caller can act on. `CACHE_UNAVAILABLE` (from the
status machine) is indistinguishable from `CACHE_NOT_FOUND` from
the provider's perspective — both route to a provider call — but
they differ in observability: `CACHE_UNAVAILABLE` surfaces on
`CallRecord.connection.cache_look_fail_reason` so callers can
detect misconfiguration. The previous `PROVIDER_ERROR_CACHED` fail
reason and the `retry_if_error_cached` `CacheOptions` field have
been removed entirely; the deserializer silently ignores a legacy
`retry_if_error_cached` key in older cached records so historical
data still loads without error.

---

## 3. On-disk layout

### 3.1 Query cache

Under `{cache_options.cache_path}/query_cache/`:

```
query_cache/
├── light_cache_records_00800.json           ← index, append-only during session
├── light_cache_records_00800.json_backup    ← pre-compact snapshot on startup
├── shard_00000-of-00800.jsonl               ← committed shards (1 CacheRecord / line)
├── shard_00001-of-00800.jsonl
├── ...
├── shard_00799-of-00800.jsonl
└── shard_backlog_00800.jsonl                ← write buffer; flushed when full
```

File-naming conventions are enforced by `ShardManager` properties
(`query_cache.py:164-192`):

- All integer shard ids are zero-padded to 5 digits
  (`f'{i:05}'`). The cap at 99999 is enforced in
  `ShardManager.__init__` (`query_cache.py:151`).
- The shard-count suffix is embedded in every filename. Changing
  `shard_count` on the next start produces a fresh set of files
  and orphans the old ones — they are not read again. Clean up
  manually if disk usage matters.
- Files are JSONL (one JSON per line), encoded via
  `type_serializer.encode_cache_record` and
  `encode_light_cache_record`. Corrupt / unparseable lines are
  skipped individually on read.
- Writes to the light index and backlog are pure appends. Only
  `_move_backlog_to_shard` (§4.5) rewrites a shard file, and that
  is an atomic `tmp → rename`.

### 3.2 Model cache

Under `{cache_options.cache_path}/available_models.json`:

```json
{
  "TEXT":  {"working_models": [...], "failed_models": [...], ...},
  "JSON":  {"working_models": [...], ...},
  "IMAGE": {...}
}
```

Keyed by `OutputFormatType` value string. Each entry is the
encoded form of a `ModelStatus` (the five sets plus
`provider_queries` dict). `ModelCacheManager` loads the file in
full on construction (`_load_from_cache_path` at
`model_cache.py:133-160`) and writes it in full after every
`update` / `save` / per-model expiry (`_save_to_cache_path` at
`model_cache.py:124-131`). No sharding, no light-vs-full split —
the file is small and the surface is simple.

The default model cache (used when no `cache_options.cache_path`
is set) writes to a `platformdirs.user_cache_dir` location, or to
a per-process `tempfile.TemporaryDirectory()` if `platformdirs`
fails. That default path has a 4-hour TTL via
`cache_options.model_cache_duration`; per-process temp dirs live
for the lifetime of the client and die with the process.

---

## 4. The freshness invariant (query cache)

This is the single load-bearing idea that keeps the query cache
correct without requiring an fsck or compaction step.

> **The in-memory `_light_cache_records` dict is the only
> authority on what is live.** Every decode from a shard file is
> validated against it.

`_check_cache_record_is_up_to_date`
(`query_cache.py:329-341`) returns `True` iff the decoded
`CacheRecord`'s light form equals the in-memory
`LightCacheRecord` for that hash (ignoring `call_count`). Any row
whose `last_access_time`, `results_count`, or `shard_id` no
longer matches the index is *skipped on read*. Every read path
calls it:

```python
# _load_shard() (query_cache.py:343-364)
for line in shard_file:
    cache_record = type_serializer.decode_cache_record(...)
    if not _check_cache_record_is_up_to_date(cache_record):
        continue                              # ← skip the stale row
    ...
```

### 4.1 Consequences

- **Deletion never rewrites a shard file.** Removing the entry
  from `_light_cache_records` and appending a `{hash: {}}`
  tombstone to the light index is enough. Stale bytes in shards
  become invisible automatically.
- **Updates work the same way.** The new row is appended to the
  backlog; old rows in any shard become invisible because the
  in-memory light entry for that hash advanced (new
  `last_access_time` or `results_count`).
- **Garbage accumulates on disk until
  `_move_backlog_to_shard`.** That function rewrites the target
  shard from live state (`_loaded_cache_records`), so every
  orphan in that shard is dropped during the rewrite. Other
  shards keep their orphans until their own merge.

### 4.2 What this means for editors

- Any new read path that decodes a `CacheRecord` from disk MUST
  call `_check_cache_record_is_up_to_date` before acting. Without
  it, deleted buckets come back to life on the next lookup.
- Any new write path that mutates
  `_light_cache_records[hash]` MUST go through
  `_update_cache_record` (`query_cache.py:198-249`). That method
  updates the shard-active-count map, the shard heap, and the
  tombstone / append, all atomically — hand-rolled equivalents
  drift.
- The light index's `_backup` copy is not a fallback for data
  loss — it's a restart-safety measure for a corrupted primary.
  Do not rely on it as a transaction log.

---

## 5. Hash identity: the hash + equality pact

The query cache determines "is this the same query we saw
before?" with two functions that must stay in lockstep:

1. `hash_serializer.get_query_record_hash(qr)` — computes the
   16-char hex prefix of `sha256` over a canonical string
   representation of the query. Produces the bucket key.
2. `type_utils.is_query_record_equal(a, b)` — does a
   field-by-field structural compare after normalizing out
   anything excluded from identity. Runs on cache hit to guard
   against hash collisions.

If one function excludes a field and the other does not, cache
lookups silently fail — the hash matches but the equality check
rejects the hit, producing `CACHE_NOT_MATCHED` on every call. The
`hash_serializer.py` module docstring calls this out explicitly;
reread it before touching either function.

### 5.1 What's in the hash (fields that define identity)

`get_query_record_hash` (`hash_serializer.py:167-194`) feeds the
following into the hasher, in order:

| Field | Helper |
|---|---|
| `query_record.prompt` | bare string |
| `query_record.system_prompt` | bare string |
| `query_record.chat` (messages + system + content blocks) | `_hash_chat` → `_content_hash_dict` per block |
| `provider_model.provider` / `.model` / `.provider_model_identifier` | bare strings |
| `parameters.temperature` / `.max_tokens` / `.stop` / `.n` / `.thinking` | `_hash_parameters` |
| `output_format.type` + pydantic class identity | `_hash_output_format` (name + JSON schema, never the live class) |
| `tools[i]` | `_hash_tools` |
| `connection_options.endpoint` (only this field) | `_hash_connection_options` |

Two details that matter:

- **`MessageContent.path` folds in file mtime_ns + size.**
  `_hash_chat` calls `os.stat` on the path and appends
  `{mtime_ns}:{size}` to the hash key. In-place edits invalidate
  the cache; replacing a file with an identical-stat copy does
  not. The `MessageContent` docstring at
  `message_content.py:248-253` documents this for callers.
- **Pydantic identity is the name + JSON schema, not the live
  class.** `_hash_output_format` resolves
  `output_format.pydantic_class` to `__name__` +
  `model_json_schema()` (or uses the stored
  `pydantic_class_name` / `pydantic_class_json_schema` if the
  live class is absent). Two schema-identical classes with
  different Python identities hit the same bucket.

### 5.2 What's excluded from identity

Documented at the top of `hash_serializer.py`:

| Field | Why excluded |
|---|---|
| `MessageContent.provider_file_api_ids` (if local content also present) | Transport metadata — uploading to a new provider must not cache-miss |
| `MessageContent.provider_file_api_status` | Transport metadata |
| `MessageContent.filename` | Informational label only |
| `MessageContent.proxdash_file_id` / `proxdash_file_status` | ProxDash tracking, not content identity |
| `ConnectionOptions` fields other than `endpoint` | `skip_cache`, `override_cache_value` are per-call flags that would poison future lookups |

For remote-only content (no `path` / `data` / `source` — only a
`provider_file_api_ids` dict), identity falls back to *just the
current provider's file_id*, extracted by `_content_hash_dict`
(`hash_serializer.py:37-71`). This ensures that a message
referencing Gemini's file_id for the same uploaded file produces
the same bucket as the previous call that uploaded it.

### 5.3 The equality normalization mirror

`is_query_record_equal` in `type_utils.py:202-266` does the same
exclusions as the hash, just in reverse (it normalizes both sides
to empty values before the `==` check):

- `_normalize_output_format` strips `pydantic_class` and
  substitutes `pydantic_class_name` + `model_json_schema()` so
  two classes with the same name + schema compare equal.
- The equality check zeroes `connection_options` down to just
  `endpoint`. Without this, a stored record written with
  `override_cache_value=True` would never match a later lookup
  without the flag.
- `_normalize_chat_for_comparison` deep-copies the chat and
  clears `provider_file_api_ids`, `provider_file_api_status`,
  `proxdash_file_id`, `proxdash_file_status`, `filename` on every
  `MessageContent`.

If you add a field to `QueryRecord`, you must:

1. Decide whether it's part of cache identity.
2. If yes: add it to `get_query_record_hash` and make sure
   `__eq__` on `QueryRecord` (dataclass default) will compare it
   field-by-field.
3. If no: add it to both `_normalize_*` helpers so equality
   mirrors the hash's exclusion.

Skipping step 3 is the classic silent-miss bug: hash matches,
equality fails, every call reports `CACHE_NOT_MATCHED`.

---

## 6. Request lifecycle (query cache hooks)

`ProviderConnector.generate()` (`provider_connector.py:825+`)
calls the cache twice per request: once to read before the
executor, once to write after.

```
generate(...)
│
├── _auto_upload_media(query_record)        # see files_internals.md, populates ids
├── _prepare_execution(...)                 # feature adapter
│
├── cached_result = _get_cached_result(query_record, connection_options)
│     │   # provider_connector.py:659-690
│     │
│     ├── if skip_cache or override_cache_value or no query_cache_manager → None
│     ├── result = query_cache_manager.look(query_record)
│     ├── if result.result is not None:
│     │     # Rewrite timestamp so the cached result reports as
│     │     # "just happened now" to the caller; cache_response_time
│     │     # records the actual lookup latency.
│     │     result.timestamp.end_utc_date = now
│     │     result.timestamp.start_utc_date = end - response_time
│     │     result.timestamp.cache_response_time = lookup_elapsed
│     │     return result
│     └── else: return cache_look_fail_reason
│
├── if cached_result is a ResultRecord:
│     connection_metadata.result_source = ResultSource.CACHE
│     return CallRecord(...)               # short-circuit; no provider call
│
├── (executor runs) ...
│
└── _update_cache(call_record, connection_options)
      │   # provider_connector.py:692-708
      │
      ├── if skip_cache or no query_cache_manager → return
      └── query_cache_manager.cache(
              query_record=call_record.query,
              result_record=call_record.result,
              override_cache_value=connection_options.override_cache_value,
          )
```

Three behaviors worth internalizing:

- **`override_cache_value` bypasses `_get_cached_result` entirely
  and flows to `_update_cache`.** The real provider is called;
  the returned result then wipes the existing bucket and starts
  fresh with a single-entry bucket (`call_count=0`). See §7.2.
- **`skip_cache` disables both sides.** No read, no write. The
  cache never learns about the call.
- **Cached results re-stamp their timestamp.** The caller sees
  "this happened just now" but
  `timestamp.cache_response_time` records the real lookup
  latency. If you need the original call's latency, read
  `timestamp.response_time` (unchanged).

---

## 7. Query cache algorithms

### 7.1 `look()` — the read path

`QueryCacheManager.look()` (`query_cache.py:636-682`):

```
1. If self.status != WORKING       → CACHE_UNAVAILABLE
2. Validate query_record type      → ValueError on wrong type (a real bug)
3. cache_record = ShardManager.get_cache_record(query_record)
4. If cache_record is None         → CACHE_NOT_FOUND
5. is_query_record_equal(cached.query, query_record)?
   If not                          → CACHE_NOT_MATCHED
6. If len(results) < unique_response_limit
                                   → UNIQUE_RESPONSE_LIMIT_NOT_REACHED
7. Pick slot:
      result = results[call_count % len(results)]
8. If update:
      cache_record.last_access_time = now
      cache_record.call_count += 1
      ShardManager.save_record(cache_record)   # delete-then-add
      _push_record_heap(cache_record)
9. Return CacheLookResult(result=result)
```

Steps worth flagging:

- **Step 3** is the only function that touches shard files on
  the read path. `ShardManager.get_cache_record` looks up the
  light record, validates the `shard_id`, then calls
  `_load_shard` which iterates lines, applies the freshness
  check (§4), and populates `_loaded_cache_records`. Cache hits
  after the first call for a hash are served from the in-memory
  dict.
- **Step 5** catches hash collisions. `_check_cache_record_is_up_to_date`
  only validates *light-record* identity; step 5 does the full
  field-by-field compare via `is_query_record_equal`. See §5.3.
- **Step 6** is why `unique_response_limit > 1` forces extra
  provider calls — a partial bucket is reported as a miss so
  the caller hits the provider and contributes a new response.
- **Step 7** is round-robin via modulo, *not* random. Reads
  cycle through the bucket deterministically. `call_count`
  resets to 0 on restart (§10), so the first call after a
  process boot always picks `results[0]`.
- **Step 8's `update=False` mode** exists so look-ahead or
  diagnostic tools can peek at the cache without disturbing LRU
  / call_count state. The public caller (`_get_cached_result`)
  always passes the default `update=True`.

### 7.2 `cache()` — the write path

`QueryCacheManager.cache()` (`query_cache.py:684-738`):

```
1. If self.status != WORKING  → log warning, no-op
2. If override_cache_value    → cache_record = None (force branch 3a)
   Else                       → cache_record = get_cache_record(query_record)
3a. If cache_record is None   → create new CacheRecord(
                                  query=query_record,
                                  results=[result_record],
                                  call_count=0,
                                  last_access_time=now,
                                )
                                save_record, push heap, return
3b. If len(results) < limit   → append(result_record),
                                last_access_time = now,
                                save_record, push heap, return
3c. Else (bucket full, no override) → return (no-op)
```

The `override_cache_value` branch is the only way to *shrink* a
bucket. It routes through 3a, and `save_record` internally calls
`delete_record` first (§8.1), so any prior bucket for this hash
is wiped. With `unique_response_limit > 1` this is a one-call
refresh that then requires the caller to make `limit - 1` more
provider calls to refill the bucket.

The previous "replace errored slot" branch and its paired
`retry_if_error_cached` config field have been removed entirely.
A call that returns an error still writes an error record into
the bucket via the normal 3a/3b branches; it is not distinguished
from a good result on subsequent reads.

### 7.3 `ShardManager.save_record` — delete-then-add

`ShardManager.save_record()` (`query_cache.py:427-443`):

```
1. delete_record(hash)                       # removes from in-memory index +
                                             # writes {hash: {}} tombstone
2. backlog_size  = _shard_active_count['backlog']
   record_size   = _get_cache_size(cache_record)       # results_count + 1
   lowest_value, lowest_key = _shard_heap.top()
3. If backlog_size + record_size >
        response_per_file - lowest_value:
      _move_backlog_to_shard(shard_id=lowest_key)      # flush least-full shard
      _add_to_backlog(cache_record)
      _save_light_cache_records()                      # compact index
   Else:
      _add_to_backlog(cache_record)                    # hot path
```

The delete-then-add pattern is *why* `override` works for free:
every write passes through delete. From the storage layer's
perspective there is no "update in place" — updates are
represented as delete + add.

`_record_size_map` and `_shard_active_count` use
`_get_cache_size(record) = results_count + 1` so an empty bucket
still counts as size 1. This prevents pathological cases where a
thousand empty buckets would appear to take no space.

---

## 8. Shard dynamics

### 8.1 Tombstones and the index

`delete_record` (`ShardManager.delete_record`, `query_cache.py:414-425`)
is shallow: it removes the hash from `_light_cache_records`, updates
`_shard_active_count` and `_map_shard_to_cache`, writes a `{hash:
{}}` tombstone line to the light index file, and does nothing to
the shard file. The tombstone is how restarts know the record is
gone — the load routine (`_load_light_cache_records`,
`query_cache.py:269-327`) skips `{}`-valued entries explicitly.

### 8.2 Backlog flush — `_move_backlog_to_shard`

`_move_backlog_to_shard` (`query_cache.py:366-389`) is the only
path that rewrites a committed shard file. It:

1. Calls `_load_shard(shard_id)` for the target — populates
   `_loaded_cache_records` with *live* records in that shard
   (stale ones drop out per §4).
2. Calls `_load_shard('backlog')` — populates backlog records.
3. Writes the union of both (from `_loaded_cache_records`, so no
   stale bytes) to `{shard_path}_backup`.
4. Atomically renames `_backup → shard_path`.
5. Removes the backlog file.

Step 3 is where GC happens for old shard bytes. Orphaned rows
that failed the freshness check in step 1 never make it to the
rewrite.

### 8.3 Which shard gets flushed

`_shard_heap` is a `HeapManager` (without size tracking) keyed by
`shard_id` with value = `_shard_active_count[shard_id]`.
`heap.top()` returns the *smallest*-value entry, i.e., the
*least-full* committed shard. That's the flush target when the
backlog needs to drain.

The flush condition (`save_record`, `query_cache.py:435-441`) is
triggered when adding one more record to the backlog would push
the chosen shard past `response_per_file`:

```
backlog_size + record_size > response_per_file - lowest_shard_value
```

Defaults on `QueryCacheManagerState` are `shard_count=800`,
`response_per_file=200`, `cache_response_size=40000`
(`types.py:811+`). With those defaults the cache holds up to
`~800 * 200 = 160000` bucket-slots on disk but evicts down to
`cache_response_size=40000` slots by LRU before that wall is
reached in practice.

### 8.4 `HeapManager` — lazy deletion

`HeapManager` (`query_cache.py:77-132`) is a thin priority queue
over `heapq` with two properties:

- Pushing a key that already exists overwrites the old value and
  subtracts the old `record_size` before adding the new one.
- Old `(value, key)` tuples still sit in the underlying `heapq`
  until `pop()` / `top()` encounters them; they're discarded
  lazily via the `_active_values[key] == value` check.

Two instances exist on `QueryCacheManager` / `ShardManager`:

- `ShardManager._shard_heap` — `with_size=False`; tracks
  `shard_active_count` per shard for flush-target selection.
- `QueryCacheManager._record_heap` — `with_size=True`; tracks
  every bucket's `last_access_time.timestamp()` with
  `_get_cache_size(record)` as size. Used for LRU eviction.

---

## 9. LRU eviction

Every successful `look()` and every `cache()` write calls
`_push_record_heap` (`query_cache.py:623-634`), which pushes the
bucket's `(query_hash, last_access_time.timestamp(),
record_size)` into `_record_heap` and then pops until the total
size is under `cache_response_size`:

```python
def _push_record_heap(self, cache_record):
    self._record_heap.push(
        key=cache_record.query.hash_value,   # always computed
        value=last_access_time.timestamp(),  # eviction priority
        record_size=_get_cache_size(record), # results_count + 1
    )
    while len(self._record_heap) > self.cache_response_size:
        _, hash_value = self._record_heap.pop()
        self._shard_manager.delete_record(hash_value)
```

Three properties worth knowing:

- **Eviction uses `last_access_time`, not `call_count`.** A query
  touched once recently wins over one hit frequently but long
  ago. Intentional: the cache optimizes for "will this query
  repeat soon?" not "has this query been valuable historically?"
- **Eviction is write-path-only.** A read that returns
  `CACHE_UNAVAILABLE` / `CACHE_NOT_FOUND` does not trigger
  eviction. Eviction requires a push, and only successful reads
  and writes push.
- **`cache_response_size` counts results, not buckets.** A bucket
  with `unique_response_limit=3` counts as size 4
  (`results_count + 1`) against the budget. A cache with
  `cache_response_size=40000` holds up to ~10000 fully-filled
  buckets at `limit=3`.

---

## 10. Restart recovery

On `ShardManager.__init__` (`query_cache.py:150-162`):

```python
self._load_light_cache_records()
```

`_load_light_cache_records` (`query_cache.py:269-327`):

1. Reset all in-memory state
   (`_shard_active_count = {}`, `_shard_heap = HeapManager()`,
   `_loaded_cache_records = {}`, `_light_cache_records = {}`,
   `_map_shard_to_cache = defaultdict(set)`).
2. Read the light index file line by line, parsing each line as
   JSON. Corrupt lines are skipped silently (`try/except`).
   Later lines for the same hash win (later appends override
   earlier appends).
3. For each non-tombstone (`record != {}`) entry:
   - Decode via `type_serializer.decode_light_cache_record`.
     Corrupt records skipped.
   - Sanity check: `query_record_hash` matches
     `light_cache_record.query_hash`, `shard_id` in range.
   - **Reset `call_count` to 0** — round-robin state does not
     persist across processes.
   - Call `_update_cache_record(write_to_file=False)` to
     populate `_light_cache_records`, `_shard_active_count`, the
     shard heap, and `_map_shard_to_cache`.
4. If the primary file fails to open, fall back to
   `light_cache_records_*.json_backup`.
5. Call `_save_light_cache_records()` — rewrites the primary
   file from the clean in-memory state, creating a fresh backup
   along the way.

Key properties:

- **Shards are not loaded at startup.** Only the light index.
  Individual shards are lazily loaded on the first
  `get_cache_record` for a hash in that shard.
- **Tombstones win over earlier appends.** A record deleted
  during a session is represented by `{hash: {}}` after the
  earlier `{hash: <record>}` line; step 2's "later wins" loop
  sees the tombstone last and drops the earlier insert.
- **`call_count` reset is deliberate.** Persisting it across
  restarts would make multi-process sharing (two workers sharing
  one cache_path) subtly unfair — whichever process last wrote
  would dictate the modulo start point. Resetting gives each
  fresh process a clean round-robin.
- **No fsck step.** The freshness invariant (§4) means a
  restarted process rejects stale shard rows naturally; there is
  nothing to repair.

---

## 11. `ModelCacheManager` — separate, simpler

`ModelCacheManager` (`model_cache.py:22-261`) caches model-probe
results, not query responses. Its data shape is a
`ModelStatusByOutputFormatType = dict[OutputFormatType,
ModelStatus]` — one `ModelStatus` per output-format type, each
with five sets (`unprocessed_models`, `working_models`,
`failed_models`, `filtered_models`) and a `provider_queries` dict
mapping model to the probe's `ProviderQuery` result.

### 11.1 Three operations

| Method | Effect | Persistence |
|---|---|---|
| `get(output_format_type)` (`model_cache.py:189-212`) | Returns a `ModelStatus`. If `cache_options.model_cache_duration` is set, walks `provider_queries` and evicts models whose probe is older than the duration. Writes the file if any eviction happened. | Writes on expiry |
| `update(updates, output_format_type)` (`model_cache.py:214-252`) | Incremental merge. Every model named in `updates` must be in at least one of the four sets, else `ValueError`. Then the model is cleaned from all existing sets and re-added to the one(s) present in `updates`. | Writes unconditionally |
| `save(model_status, output_format_type)` (`model_cache.py:254-261`) | Replace the entry wholesale with a deep-copied `ModelStatus`. | Writes unconditionally |

### 11.2 Per-model expiry semantics

`model_cache_duration` (on `CacheOptions`) is a *per-model*
eviction threshold applied on read, not a file-level TTL. When
`get()` is called:

- Iterate `model_status.provider_queries`.
- For each, compute
  `now - provider_query.result.timestamp.end_utc_date` in seconds.
- If it exceeds `model_cache_duration`, call
  `_clean_model_from_tested_models` — removes the model from
  `working_models` / `failed_models` and clears its
  `provider_queries` entry. Crucially, it does NOT touch
  `unprocessed_models` or `filtered_models`; those are static
  classifications, not probe outcomes.

The expiry pass can rewrite the file even on a read path. This
is intentional: the next probe pass sees the evicted model as
"never tested" and re-probes it.

### 11.3 Why two managers (not one combined)

`QueryCacheManager` is tuned for many small records with
individual identity (one bucket per query). `ModelCacheManager`
is tuned for few large records with wholesale replacement (one
per `OutputFormatType`). Their access patterns, eviction
strategies, and persistence granularity are incompatible — one
file per output format, no sharding, no LRU, no per-entry
equality check.

Two-manager split also lets callers disable one independently
(`disable_model_cache=True` keeps the query cache on; setting
`cache_path=None` disables both).

---

## 12. Inputs the cache subsystem does not own

Several concerns that look like cache concerns aren't — keep the
layering clean:

- **`CacheOptions` validation** happens implicitly in
  `QueryCacheManager.init_status` (`query_cache.py:507-544`) via
  the status machine. Adding custom validation should happen in
  that method, not in `look` / `cache`.
- **`ConnectionOptions.skip_cache` / `override_cache_value` gating**
  lives in `ProviderConnector._get_cached_result` /
  `_update_cache`. The cache manager accepts
  `override_cache_value` but does not inspect `skip_cache` — the
  caller already short-circuits before calling `look` or
  `cache`.
- **Pydantic instance reconstruction from cache.** When a cached
  result is returned with a `PYDANTIC_INSTANCE` content block,
  `ProviderConnector._reconstruct_pydantic_from_cache`
  (`provider_connector.py:710+`) walks the blocks and calls
  `model_validate` on the stored JSON. The cache only stores
  serializable JSON (`PydanticContent.instance_json_value`), not
  live instances.
- **File-content hashing.** `MessageContent.path` / `data` are
  hashed in `hash_serializer._hash_chat` by folding `os.stat`
  into the hash (§5.1). `FilesManager` does not interact with
  the cache — if an auto-upload happens, it mutates the
  `MessageContent` after the hash has already been computed; the
  excluded fields (§5.2) make this safe.
- **ProxDash logging.** Cache hits log through
  `logging_utils.log_message` but never flow through
  `ProxDashConnection`. ProxDash writes only on successful
  provider calls; a cached response does not re-log.

---

## 13. Where to read next

- `../user_agents/api_guidelines/cache_behaviors.md` — the
  caller view: what each `CacheOptions` / `ConnectionOptions`
  combination does to reads and writes, the `ResultSource` /
  `cache_look_fail_reason` fields that surface on a returned
  `CallRecord`, and the common surprises.
- `state_controller.md` — both cache managers inherit from
  `StateControlled`. Fields on
  `QueryCacheManagerState` / `ModelCacheManagerState` follow the
  handle_changes / deserializer contract documented there.
- `adding_a_new_provider.md` §5.2 — where the `BEST_EFFORT`
  behaviors note that `output_format.type` is intentionally
  left in place so the executor can enable native JSON mode
  even when the framework already injected prompt guidance.
  Cache identity follows the resolved `output_format.type`, not
  the caller's original intent, so a cached BEST_EFFORT response
  for `type=JSON` stays valid on replay.
- `feature_adapters_logic.md` §2.1-2.2 — adapters run *before*
  the cache hook on the write path is irrelevant, but on the
  read path the cache is consulted *after* feature-adapter
  `_prepare_execution`. That ordering means a query whose
  adapter dropped a BEST_EFFORT parameter hits the same cache
  bucket as a query that never set the parameter — identity is
  the resolved (adapted) query, not the caller's input.
- `tests/caching/test_query_cache.py` /
  `tests/caching/test_model_cache.py` — the executable spec.
  When the source is ambiguous (or when you think you've found
  an invariant violation), find the corresponding test.
