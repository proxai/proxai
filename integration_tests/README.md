# ProxAI Integration Tests

Human-in-the-loop integration tests for the public ProxAI API. Five
files, each independently runnable.

```
01_models_test.py    px.models.* + check_health
02_generate_test.py  px.generate / generate_text / json / pydantic / image / audio / video
03_files_test.py     px.files.* + ProxDash file integration
04_proxdash_test.py  ProxDash UI verification (mostly manual)
05_runtime_test.py   connect / cache / logging / errors
```

See [USAGE.md](./USAGE.md) for shell command examples and
[INTEGRATION_TESTS_PLAN.md](./INTEGRATION_TESTS_PLAN.md) for design
rationale.

## Pre-requisites

- Provider API keys in env (`OPENAI_API_KEY`, `GEMINI_API_KEY`, ...).
- ProxDash backend running at `http://localhost:3001` (dev) or use
  `--env prod`.
- ProxDash UI at `http://localhost:3000` (dev).

## State persistence

Each session lives under `~/proxai_integration_test/test_<id>/`. The
api_key from first-time setup persists in `_setup.state`. Per-block
state goes to `<test_dir>/<file_label>/<block>.state` — blocks with an
existing `.state` file skip on subsequent runs. Force a re-run by
deleting the relevant `.state` file or by passing `--mode new`.

## Test types

- **assert** — Python assertions only. Failure raises in Python.
- **manual_check** — code prints something, then prompts y/n. Operator
  must press y or n. n raises and aborts.

`--auto-continue` skips the inter-block "Press Enter" pause but does
**not** auto-answer manual_check prompts.
