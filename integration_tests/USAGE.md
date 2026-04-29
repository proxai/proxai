# Usage — running the integration tests

```bash
# Run a whole file
poetry run python3 integration_tests/01_models_test.py

# Run one named block from a file
poetry run python3 integration_tests/01_models_test.py --test list_models_default

# Skip "Press Enter" pauses between blocks
# (manual_check y/n prompts still require input)
poetry run python3 integration_tests/05_runtime_test.py --auto-continue

# Print each block's source as it runs
poetry run python3 integration_tests/02_generate_test.py --print-code

# Resume an interrupted session (default)
poetry run python3 integration_tests/03_files_test.py --mode latest

# Start a fresh session
poetry run python3 integration_tests/03_files_test.py --mode new

# Re-run a specific previous session by id
poetry run python3 integration_tests/03_files_test.py --mode 3

# Use production URLs (proxai.co + proxainest-production)
poetry run python3 integration_tests/04_proxdash_test.py --env prod

# Combine flags
poetry run python3 integration_tests/02_generate_test.py \
    --mode new --auto-continue --print-code --test generate_text_basic
```
