# ProxAI Dependency Graph

A simple reference showing which files import which modules within `src/proxai/`.

---

## Visual Overview

```
                                ┌─────────────┐
                                │   types.py  │  ← Foundation (no internal deps)
                                └──────┬──────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│   stat_types.py  │        │   type_utils.py  │        │ state_controller │
└──────────────────┘        └──────────────────┘        └──────────────────┘
           │                                                       │
           ▼                                                       │
┌──────────────────┐                                               │
│ type_serializer  │◄──────────────────────────────────────────────┘
└──────────────────┘
           │
           ▼
┌──────────────────┐        ┌──────────────────┐
│ hash_serializer  │        │ logging/utils.py │
└──────────────────┘        └──────────────────┘
           │                           │
           └───────────┬───────────────┘
                       ▼
              ┌────────────────┐
              │  query_cache   │
              │  model_cache   │
              └────────────────┘
                       │
                       ▼
              ┌────────────────┐        ┌────────────────┐
              │ model_configs  │        │   proxdash     │
              └────────────────┘        └────────────────┘
                       │                        │
                       ▼                        │
              ┌────────────────┐                │
              │model_connector │◄───────────────┘
              └────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ model_registry │◄── providers/*
              └────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │available_models│
              └────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   proxai.py    │  ← Main entry point
              └────────────────┘
```

---

## File-by-File Dependencies

### Core Foundation

| File | Imports |
|------|---------|
| `types.py` | *(none - base types)* |
| `stat_types.py` | `types` |
| `type_utils.py` | `types` |
| `experiment/experiment.py` | *(none - standalone utils)* |

### State Management

| File | Imports |
|------|---------|
| `state_controllers/state_controller.py` | `types` |

### Serializers

| File | Imports |
|------|---------|
| `serializers/type_serializer.py` | `types`, `stat_types` |
| `serializers/hash_serializer.py` | `types` |

### Logging

| File | Imports |
|------|---------|
| `logging/utils.py` | `types`, `type_serializer` |

### Caching

| File | Imports |
|------|---------|
| `caching/model_cache.py` | `types`, `type_serializer`, `state_controller` |
| `caching/query_cache.py` | `types`, `type_serializer`, `hash_serializer`, `state_controller` |

### Connections

| File | Imports |
|------|---------|
| `connections/proxdash.py` | `types`, `experiment`, `logging/utils`, `state_controller` |
| `connections/available_models.py` | `types`, `model_cache`, `proxdash`, `model_registry`, `model_connector`, `model_configs`, `logging/utils`, `state_controller`, `type_utils` |

### Connectors

| File | Imports |
|------|---------|
| `connectors/model_configs.py` | `types`, `type_serializer`, `state_controller` |
| `connectors/model_connector.py` | `types`, `logging/utils`, `query_cache`, `type_utils`, `stat_types`, `hash_serializer`, `proxdash`, `model_configs`, `state_controller` |
| `connectors/model_registry.py` | `model_connector`, `types`, `model_configs`, *all providers* |

### Providers (all follow same pattern)

| File | Imports |
|------|---------|
| `connectors/providers/openai.py` | `types`, `openai_mock`, `model_connector` |
| `connectors/providers/claude.py` | `types`, `claude_mock`, `model_connector` |
| `connectors/providers/gemini.py` | `types`, `gemini_mock`, `model_connector`, `model_configs` |
| `connectors/providers/cohere_api.py` | `types`, `cohere_api_mock`, `model_connector` |
| `connectors/providers/databricks.py` | `types`, `databricks_mock`, `model_connector` |
| `connectors/providers/mistral.py` | `types`, `mistral_mock`, `model_connector` |
| `connectors/providers/huggingface.py` | `types`, `huggingface_mock`, `model_connector` |
| `connectors/providers/deepseek.py` | `types`, `openai_mock`, `model_connector` |
| `connectors/providers/grok.py` | `types`, `openai_mock`, `model_connector` |
| `connectors/providers/mock_provider.py` | `types`, `model_connector` |

### Main Entry Point

| File | Imports |
|------|---------|
| `proxai.py` | `types`, `type_utils`, `model_connector`, `model_registry`, `query_cache`, `model_cache`, `type_serializer`, `stat_types`, `available_models`, `proxdash`, `experiment`, `model_configs`, `logging/utils` |

---

## Dependency Layers (Bottom to Top)

```
Layer 5: proxai.py (main API)
         │
Layer 4: available_models
         │
Layer 3: model_registry ← providers/*
         │
Layer 2: model_connector, proxdash, model_configs
         │
Layer 1: query_cache, model_cache, logging/utils
         │
Layer 0: types, stat_types, type_utils, state_controller, serializers
```

---

## Key Observations

1. **`types.py`** is the foundation - imported by almost everything
2. **`state_controller.py`** is used by all StateControlled classes (caches, proxdash, model_connector, model_configs, available_models)
3. **`model_connector.py`** is the heaviest single file - it coordinates caching, logging, and proxdash
4. **Providers** are leaf nodes - they only depend on `model_connector` and `types`
5. **`proxai.py`** is the root - it imports nearly everything to provide the unified API

---

*Last updated: 2025-12-01*
