# Findings — first end-to-end run

Notes from the first full automated pass of all 5 integration test files
(2026-04-27, against ProxDash dev / localhost). Captures bugs fixed in
the test harness, drift between local and ProxDash-served registries,
and SDK issues worth filing.

## Total

107 test blocks + 3 setup blocks across 5 files, all passing.

| File | Blocks |
|---|---|
| `01_models_test.py` | 22 / 22 |
| `02_generate_test.py` | 25 / 25 |
| `03_files_test.py` | 26 / 26 |
| `04_proxdash_test.py` | 21 / 21 |
| `05_runtime_test.py` | 13 / 13 (1 dropped — see below) |

## SDK API drift — caller-side fixes

Old patterns from `proxai_api_test.py.deprecated` no longer worked:

1. **Mock provider env-var names** — per
   `src/proxai/connectors/model_configs.py:PROVIDER_KEY_MAP`:
   - `MOCK_PROVIDER_API_KEY` (with `_API_KEY` suffix)
   - `MOCK_FAILING_PROVIDER`, `MOCK_SLOW_PROVIDER` (no suffix).
   Set at the top of `01_models_test.py` and `05_runtime_test.py`.

2. **`provider_queries` values** are now `CallRecord` (was
   `LoggingRecord`). Field rename: `lq.response_record.error` →
   `call_record.result.error`.

3. **`check_health()` signature simplified** — only accepts `verbose`.
   The retired kwargs moved to connect-time options:
   - `allow_multiprocessing=...` → `model_probe_options=ModelProbeOptions(allow_multiprocessing=...)`
   - `model_test_timeout=...` → `model_probe_options=ModelProbeOptions(timeout=...)`
   - `extensive_return=True` is now the default; `check_health()` returns
     `ModelStatus` directly.

4. **`system_prompt` + `messages` are mutually exclusive.** SDK error
   recommends `role='system'` but `MessageRoleType` accepts only
   `USER`/`ASSISTANT`. Workaround in tests: drop the system field, or
   put the instruction in the first user turn.

5. **Local `provider_queries.log` no longer exists.** Logging now writes
   only `merged.log` + `proxdash.log` with INFO-level connection
   messages — provider call records go straight to ProxDash via API
   instead of a local file. Effects:
   - `logging_to_provider_queries_log` rewritten as
     `logging_to_merged_log` checking the connection messages.
   - `logging_with_hide_sensitive_content` dropped (the field still
     exists on `LoggingOptions` but doesn't have a local log to mask;
     the corresponding ProxDash-side test is in `04_proxdash_test`).

6. **`generate_auto_upload_pdf_per_provider` test bug** (mine): created
   a `media` object then called a helper that built its own —
   `media.provider_file_api_ids` stayed `None`. Fixed by reusing the
   same `media` object in the messages payload.

## ProxDash registry differs from local `model_configs_data/v1.3.0.json`

ProxDash serves its own copy of v1.3.0. Some entries are server-side
overridden to all-`NOT_SUPPORTED`, making them unreachable through the
SDK.

| Model | Local | ProxDash |
|---|---|---|
| `gemini/gemini-2.5-flash-tts` | `out.audio=SUPPORTED` | every feature `NOT_SUPPORTED` |
| `gemini/veo-3.1-generate` | `out.video=SUPPORTED` | every feature `NOT_SUPPORTED` |
| `openai/dall-e-3` | `in.text=SUPPORTED, out.image=SUPPORTED` | `in.text=NOT_SUPPORTED` |
| `openai/tts-1` | `in.text=SUPPORTED, out.audio=SUPPORTED` | `in.text=NOT_SUPPORTED` |
| `openai/sora-2` | `in.text=SUPPORTED, out.video=SUPPORTED` | every input format `NOT_SUPPORTED` |

Effect: `openai/dall-e-3`, `openai/tts-1`, `openai/sora-2` cannot be
called with a text prompt through ProxDash because the feature adapter
rejects the prompt as `input.text`. **Test-harness changes:**
- `IMAGE_MODEL` constant in `_utils.py` switched to
  `('gemini', 'gemini-2.5-flash-image')`.
- `AUDIO_MODEL` switched to `('gemini', 'lyria-3-clip')` — the only
  audio model in the registry with `in.text=SUPPORTED`.
- `VIDEO_MODEL` left as `('openai', 'sora-2')`. Both video tests
  (`generate_video_call`, `proxdash_renders_video_output`) now skip
  gracefully when they hit the "No compatible endpoint" error — there
  is no video model in the registry with a usable input path.
- `01_models_test.list_models_by_output_format` no longer asserts on
  `gemini-2.5-flash-tts` or `veo-3.1-generate` (they don't show up in
  the audio/video filter lists under the ProxDash registry).

## Operator-input replacements

`proxdash_limited_api_key` requires generating a UI-side limited key.
Added `PROXAI_LIMITED_API_KEY` env-var fallback so an automated run can
substitute the regular key. **The substituted regular key does not
actually mask content** — manual_check assertions become trivially y on
auto-y runs. A real operator should re-run this block with no env var
to actually verify the limit.

## SDK-side issues worth filing

These are not test-harness problems; the tests are working around them.

1. **`role='system'` rejected even though SDK error recommends it.**
   `client.py:1796` raises with the message `'Please use "system"
   message in messages to set the system prompt'` but
   `MessageRoleType` accepts only `USER`/`ASSISTANT`.

2. **`openai/responses.create` mishandles assistant turns.** ✅ FIXED
   in `src/proxai/connectors/providers/openai.py`. The connector was
   role-blind: `_to_responses_part` and `_build_responses_input`
   unconditionally emitted `'input_text'` for text content regardless
   of role. OpenAI's Responses API requires `'output_text'` (or
   `'refusal'`) for assistant role and rejected the request.
   Fix: `_to_responses_part(part_dict, role='user')` now branches on
   role; `_build_responses_input` passes the role and skips system-role
   messages (those are routed to `instructions=` by the executor,
   which now also extracts system from a role='system' message as a
   defensive fallback for `add_system_to_messages=True` exports). All
   1322 unit tests still pass; the original failing case
   (multi-turn `gpt-4o`) now succeeds.

3. **ProxDash registry blocks all media-generation prompts** for
   `dall-e-3` / `tts-1` / `sora-2`. Either the registry should mark
   these models' `input.text` as `SUPPORTED`, or the feature adapter
   needs a special path for "prompt-only generation" that doesn't
   require `input_format.text`.

4. **`gemini-2.5-flash-tts` and `veo-3.1-generate`** appear in the
   ProxDash registry but every feature is `NOT_SUPPORTED`. They are
   effectively unreachable. Either fill in the features or remove from
   the listing.

## Open items needing a real operator

The auto-y pass let everything through every `manual_check`. For
genuine UI verification an operator should re-run with no `yes y` pipe
and confirm:

- ProxDash UI rendering of every multi-modal record (image preview,
  audio playback, video playback, JSON / Pydantic / markdown blocks).
- That sensitive-content masking actually masks (regular vs
  `hide_sensitive_content=True`, plus a real limited key — not the env
  var fallback).
- That generated image / audio / video files actually look / sound
  right (current tests check non-empty bytes only).

Re-running with a clean session is `--mode new`. Cached blocks skip
fast — only the manual_check ones actually need attention on a re-run
of the same `--mode <id>`.
