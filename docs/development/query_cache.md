# Query Cache — How It Works

The query cache stores provider responses keyed by a hash of the
`QueryRecord`, so identical requests can be served from disk instead of
re-hitting the provider. It is sharded, LRU-bounded, crash-tolerant, and
designed to be safe under restart without a fsck step.

**Source of truth**: `src/proxai/caching/query_cache.py`. All line
references in this doc point there unless noted. User-facing options
live in `CacheOptions` (`types.py:297`) and `ConnectionOptions`
(`types.py:500`).

## Table of Contents

1. [Mental Model](#1-mental-model)
2. [Core Data Types](#2-core-data-types)
3. [On-Disk Layout](#3-on-disk-layout)
4. [The Freshness Invariant](#4-the-freshness-invariant)
5. [Request Lifecycle](#5-request-lifecycle)
6. [Algorithms](#6-algorithms)
7. [User-Facing Options](#7-user-facing-options)
8. [LRU Eviction](#8-lru-eviction)
9. [Backlog Flush and Shard Merge](#9-backlog-flush-and-shard-merge)
10. [Restart Recovery](#10-restart-recovery)
11. [Pitfalls](#11-pitfalls)

---

## 1. Mental Model

Every query hashes to a **bucket**. A bucket holds one or more
`ResultRecord`s (up to `unique_response_limit`). Reads round-robin
through the bucket via `call_count % len(results)`. Writes fill the
bucket until it is full; once full, the bucket is frozen except for
error-slot retries and explicit override.

```
QueryRecord ──hash──▶ bucket  ┌─── ResultRecord #0
                              ├─── ResultRecord #1
                              └─── ResultRecord #2   (call_count cycles)
```

Buckets live in sharded `.jsonl` files on disk. A single index file
(`light_cache_records_*.json`) tells the cache which shard currently
owns each bucket. That index is the **single source of truth**; stale
bytes in shard files are invisible (see §4).

The cache has three layers:

| Layer | Class | Responsibility |
|---|---|---|
| Public API | `QueryCacheManager` (line 455) | `look()` / `cache()`, LRU heap, status, state wiring |
| Storage | `ShardManager` (line 133) | Sharded files, in-memory indices, delete/save/merge |
| Priority queue | `HeapManager` (line 75) | Lazy-deletion heap for LRU and shard-fill ordering |

---

## 2. Core Data Types

Defined in `types.py:626-662`.

| Type | Purpose |
|---|---|
| `CacheRecord` | Full bucket: `query`, `results: list[ResultRecord]`, `shard_id`, `last_access_time`, `call_count`. What lives in shard files. |
| `LightCacheRecord` | Index entry: `query_hash`, `results_count`, `shard_id`, `last_access_time`, `call_count`. What lives in the light index file. |
| `CacheLookResult` | Return value of `look()`. Either `result` or `cache_look_fail_reason`. |
| `CacheLookFailReason` | `CACHE_NOT_FOUND`, `CACHE_NOT_MATCHED`, `UNIQUE_RESPONSE_LIMIT_NOT_REACHED`, `PROVIDER_ERROR_CACHED`. |

The `LightCacheRecord` is the compact form kept in memory for **every**
bucket; the full `CacheRecord` is loaded lazily per shard.

---

## 3. On-Disk Layout

Under `{cache_path}/query_cache/`:

```
query_cache/
├── light_cache_records_00100.json     ← index (JSONL, append-only writes)
├── light_cache_records_00100.json_backup
├── shard_00000-of-00100.jsonl         ← committed shards (one CacheRecord per line)
├── shard_00001-of-00100.jsonl
├── ...
├── shard_00099-of-00100.jsonl
└── shard_backlog_00100.jsonl          ← backlog, flushed into a committed shard when full
```

- File name suffix `_00100` is the shard count. Change `shard_count`
  and you get a fresh set of files; the old ones are ignored.
- All files are JSONL (one record per line), encoded via
  `type_serializer.encode_cache_record` /
  `encode_light_cache_record`.
- Writes are **append-only**. Nothing rewrites shard files mid-session
  except `_move_backlog_to_shard` (§9), which does an atomic
  `tmp → rename`.
- The light index is also append-only during a session. On startup
  it's compacted via `_save_light_cache_records()` (line 249).

---

## 4. The Freshness Invariant

This is the load-bearing idea. **The in-memory
`_light_cache_records` dict is the only authority on what is live.**
Every read path that decodes a `CacheRecord` from a shard file calls:

```python
_check_cache_record_is_up_to_date(cache_record)   # line 327
```

which returns True iff the decoded record's light form equals the
in-memory index entry for that hash (ignoring `call_count`). A row
whose `last_access_time`, `results_count`, or `shard_id` no longer
matches the index is **skipped on read** as if it weren't there.

Consequences:

- Deletion does not need to rewrite shard files. Removing the entry
  from the index + writing a `{hash: {}}` tombstone to the light
  index is enough. Stale bytes in shards become invisible.
- Updates work the same way: new row appended, old rows filtered out
  on the next read because the in-memory light entry advanced.
- Garbage accumulates until `_move_backlog_to_shard` (§9) rewrites
  the target shard from live data.

If you modify any read path, **keep the freshness check intact**.
Without it, deleted buckets reappear.

---

## 5. Request Lifecycle

From the caller's perspective, the provider connector calls two
methods: `look()` before the provider call, and `cache()` after. The
cache manager handles everything else.

```
ProviderConnector._get_cached_result(query_record, connection_options)
  │
  ├─ if skip_cache or override_cache_value or no manager → return None
  │
  └─ query_cache_manager.look(query_record)   ← line 609
       ├─ ShardManager.get_cache_record(query_record)   ← line 397
       │    ├─ light index lookup
       │    ├─ _load_shard(shard_id)                    ← line 341
       │    └─ return _loaded_cache_records[hash]
       ├─ freshness / equality re-check
       ├─ unique_response_limit gate
       ├─ error-slot retry gate (retry_if_error_cached)
       ├─ bump call_count, save_record, push heap
       └─ return CacheLookResult(result=...)

ProviderConnector._update_cache(call_record, connection_options)
  │
  ├─ if skip_cache or no manager → return
  │
  └─ query_cache_manager.cache(query, result, override_cache_value)   ← line 660
       ├─ if override_cache_value → cache_record = None   (full wipe)
       ├─ if bucket empty → create, save, push
       ├─ if bucket < limit → append, save, push
       └─ if bucket full + error slot → replace slot, save, push
```

Everything after the executor in `provider_connector.py::generate()`
(usage / cost / timestamp) runs regardless of whether the result came
from cache.

---

## 6. Algorithms

### 6.1 Lookup — `QueryCacheManager.look()` (line 609)

```
1. Load bucket        → ShardManager.get_cache_record(query_record)
2. If None            → return CACHE_NOT_FOUND
3. Structural compare → type_utils.is_query_record_equal(cached.query, query)
                        If not equal → return CACHE_NOT_MATCHED
4. Size gate          → if len(results) < unique_response_limit
                        → return UNIQUE_RESPONSE_LIMIT_NOT_REACHED
5. Pick slot          → result = results[call_count % len(results)]
6. Error-retry gate   → if result.error and retry_if_error_cached
                        and call_count < len(results)
                        → bump call_count + save + return PROVIDER_ERROR_CACHED
7. Return result      → bump call_count + save + push heap + return result
```

**Step 3** catches hash collisions. `_check_cache_record_is_up_to_date`
only validates light-record equality; Step 3 does a full field-by-field
compare of the stored `QueryRecord` against the current one.

**Step 4** is the reason `unique_response_limit > 1` forces extra
provider calls until the bucket fills: partial buckets are treated as a
miss so callers hit the provider and contribute a new response.

**Step 6** lets an errored slot be retried once per cycle when
`retry_if_error_cached=True`; if the retry succeeds it overwrites the
errored slot inside `cache()` (line 691-701).

### 6.2 Write — `QueryCacheManager.cache()` (line 660)

```
1. Load existing bucket
2. If override_cache_value  → bucket = None  (force branch 3a)
3a. If bucket is None       → create new CacheRecord with [result],
                              call_count=0, save, push, return
3b. If len(results) < limit → append(result), save, push, return
3c. If len(results) == limit and retry_if_error_cached and
    result.error is None    → replace first errored slot with result,
                              save, push, return
3d. Otherwise               → no-op (bucket is already full of good responses)
```

The `override_cache_value` branch is the only way to **shrink** a
bucket. It routes through 3a, which builds a single-result bucket from
scratch; `save_record` internally deletes any prior entry before
writing, and the freshness invariant (§4) purges stale rows on read.

### 6.3 Save — `ShardManager.save_record()` (line 425)

```
1. delete_record(hash)                       # wipe in-memory + tombstone
2. If backlog + new_size > room in lowest shard:
     _move_backlog_to_shard(lowest_shard)    # flush, atomic rename
     _add_to_backlog(record)
     _save_light_cache_records()             # compact index
   Else:
     _add_to_backlog(record)                 # hot path: append only
```

The delete-then-add pattern is why override works for free: every write
already passes through delete. You can think of `cache()` override as
"treat this as a brand-new bucket" — the storage layer does the rest.

---

## 7. User-Facing Options

### 7.1 `CacheOptions` (client-level, `types.py:297`)

| Option | Meaning |
|---|---|
| `cache_path` | Enables the cache. `None` disables it entirely. |
| `unique_response_limit` | Target bucket size. Reads return `UNIQUE_RESPONSE_LIMIT_NOT_REACHED` until the bucket holds this many responses. Default `1`. |
| `retry_if_error_cached` | If an errored response is in the bucket, retry one at a time until it's replaced by a good one. |
| `clear_query_cache_on_connect` | Wipe the entire cache dir on `connect()`. |

### 7.2 `ConnectionOptions` (per-call, `types.py:500`)

| Option | Effect on read | Effect on write |
|---|---|---|
| `skip_cache` | Skip `look()` entirely | Skip `cache()` entirely (no read, no write) |
| `override_cache_value` | Skip `look()`, always hit provider | Wipe existing bucket for this hash, store fresh result as single-entry bucket (`call_count=0`) |

### 7.3 Behaviour Matrix

For a query whose existing bucket has `results=[R1, R2, R3]`,
`call_count=5`, `unique_response_limit=3`:

| Flags | Next call result | Bucket after |
|---|---|---|
| `(none)` | `R3` (call_count becomes 6) | `[R1, R2, R3]`, cc=6 |
| `skip_cache=True` | provider call, no cache touch | `[R1, R2, R3]`, cc=5 |
| `override_cache_value=True` | provider call, result `R_new` | `[R_new]`, cc=0 |
| `(none)` with `R3.error` and `retry_if_error_cached=True` | provider call, `R_new` replaces `R3` | `[R1, R2, R_new]`, cc=bumped |

---

## 8. LRU Eviction

`QueryCacheManager` holds a size-aware `HeapManager` keyed by
`last_access_time`:

```
_record_heap = HeapManager(with_size=True)
  ├── key      = query_hash
  ├── value    = last_access_time.timestamp()   ← eviction priority (oldest first)
  └── size     = results_count + 1              ← counted toward cache_response_size
```

Every successful `look()` and `cache()` calls `_push_record_heap`
(line 596). When the heap's total size exceeds `cache_response_size`,
`_record_heap.pop()` returns the oldest key and the corresponding
bucket is deleted from `ShardManager`.

`cache_response_size`, along with `shard_count` and `response_per_file`,
is a **cache-manager tuning knob** stored on `QueryCacheManagerState`
(`types.py:748`), not on `CacheOptions`. Defaults: `shard_count=800`,
`response_per_file=200`, `cache_response_size=40000`. They're passed
via `QueryCacheManagerParams` at construction — callers touching them
is rare.

Two details that matter:

- `HeapManager.push` is **idempotent on key**. Pushing an existing
  key subtracts the old size before adding the new one, and old heap
  tuples become garbage that lazy-deletion discards on the next
  `pop()`/`top()`.
- Eviction uses `last_access_time`, not `call_count`. A rarely-hit
  query that was just touched wins over a hot query that went quiet.

---

## 9. Backlog Flush and Shard Merge

All new writes land in `shard_backlog_*.jsonl`. The backlog is
flushed into a committed shard the moment adding the next record would
push the **fullest possible committed shard** past
`response_per_file`:

```python
# save_record(), line 432-441
lowest_shard_value, lowest_shard_key = self._shard_heap.top()
if backlog_size + record_size > response_per_file - lowest_shard_value:
    self._move_backlog_to_shard(shard_id=lowest_shard_key)
    self._add_to_backlog(cache_record)
    self._save_light_cache_records()
else:
    self._add_to_backlog(cache_record)
```

`_shard_heap` is ordered smallest-first, so the **least-full** shard
is picked for the merge. `_move_backlog_to_shard`:

1. Reloads both the target shard and the backlog from disk.
2. Rewrites target shard + backlog contents into
   `{shard}.jsonl_backup`.
3. Atomically renames `_backup → final`.
4. Removes the backlog file.

Because step 2 serialises from `_loaded_cache_records` (live state),
stale rows that failed the freshness check in step 1 are dropped
during the rewrite. **The merge is the only GC path for old shard
bytes.**

---

## 10. Restart Recovery

On construction, `ShardManager.__init__` calls
`_load_light_cache_records()` (line 267):

```
1. Reset in-memory state (indices, heaps, loaded records)
2. Read light_cache_records_*.json line-by-line (JSONL, later entries
   win because data[hash] = record in the loop)
3. For each non-tombstone record:
     - decode LightCacheRecord
     - skip if shard_id is out of bounds
     - reset call_count=0
     - insert via _update_cache_record(write_to_file=False)
4. Compact the index file via _save_light_cache_records()
```

Key properties:

- Tombstones (`{hash: {}}`) are skipped at line 303.
- Corrupt lines are skipped individually — the cache never fails the
  whole load on a single bad record.
- If the primary file is unreadable, `_load_light_cache_records` falls
  back to `_backup`.
- Shards themselves are not loaded at startup. They are lazily
  loaded on the first `get_cache_record` for a hash in that shard.
- `call_count` is reset to 0 on restart by design — the round-robin
  state does not persist across processes.

---

## 11. Pitfalls

**Do not bypass the freshness check.** Any code that decodes a
`CacheRecord` from disk must call `_check_cache_record_is_up_to_date`
before acting on it, or deleted buckets will come back to life.

**Do not mutate a returned `ResultRecord` in place.** `look()` returns
the cached record directly (not a copy); mutations will be written
back on the next save. Copy if you need to transform.

**`unique_response_limit > 1` changes cost.** Each fresh query
contributes one provider call until the bucket is full. With
`unique_response_limit=3`, a new query pays 3× before it starts
serving from cache. Document this to callers.

**Override starts a new bucket from size 1.** With
`unique_response_limit > 1`, the next few calls after an override will
refill the bucket from the provider, not serve from cache. This is
intentional (see §6.2) but surprises users who expect a single-call
refresh.

**`skip_cache` and `override_cache_value` are not the same.**
`skip_cache` makes the call invisible to the cache (no read, no write).
`override_cache_value` still writes; it just wipes first. Choose based
on whether you want the new result to affect future lookups.

**Changing `shard_count` orphans old files.** Shard file names embed
the count. If you bump it, old shards become dead files on disk and a
new set is written. Clean them up manually if disk usage matters.

**Hash collisions are caught at lookup, not at write.** `look()` step
3 does a full `is_query_record_equal` check. If you add new fields to
`QueryRecord`, make sure the equality helper and
`hash_serializer.get_query_record_hash` both see them, or you'll get
silent false hits.

---

*Last updated: 2026-04-15*
