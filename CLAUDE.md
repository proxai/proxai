# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
```bash
# Install dependencies
poetry install

# Run a simple test
poetry run python examples/simple_test.py
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_proxai.py

# Run tests with verbose output
poetry run pytest -v

# Run tests with coverage
poetry run pytest --cov=src/proxai
```

### Building
```bash
# Build the package
poetry build
```

### Documentation
```bash
# Build documentation
cd docs
make html
```

## Code Architecture

ProxAI is a Python library that provides a unified interface for working with multiple AI providers. It simplifies AI integration by creating a common API across different models and providers.

### Key Components

1. **Core Interface** (`src/proxai/proxai.py`): Contains the main API functions like `generate_text()`, `connect()`, `check_health()`, etc.

2. **Model Connectors** (`src/proxai/connectors/`): Provides integration with different AI providers:
   - Each provider has its own connector implementation in `connectors/providers/`
   - `model_connector.py` defines the base connector interface
   - `model_registry.py` handles registration and retrieval of connectors

3. **Caching** (`src/proxai/caching/`): Implements caching for query responses and model availability:
   - `query_cache.py` for caching responses
   - `model_cache.py` for caching model availability information

4. **Types & Serialization** (`src/proxai/types.py`, `src/proxai/serializers/`):
   - Defines data structures used throughout the library
   - Provides serialization/deserialization for caching and logging

5. **Logging & Stats** (`src/proxai/logging/`, `src/proxai/stat_types.py`):
   - Tracks usage statistics and handles logging

6. **ProxDash Integration** (`src/proxai/connections/proxdash.py`):
   - Optional integration with ProxDash monitoring platform

### Main Workflows

1. **Connection Initialization**: User calls `px.connect()` to set up caching, logging, and connection options.

2. **Model Selection**: User selects a model using `px.set_model()` or lets the library choose a default.

3. **Text Generation**: User calls `px.generate_text()` with a prompt, which:
   - Checks cache for existing responses
   - Routes to the appropriate model connector
   - Handles response processing and error handling
   - Updates statistics

4. **Model Discovery**: `px.models.list_models()` or `px.check_health()` discover available and working models.

### Configuration Options

The library supports various configuration options through:
- `CacheOptions`: Controls caching behavior and locations
- `LoggingOptions`: Controls logging behavior and locations
- `ProxDashOptions`: Controls integration with ProxDash monitoring