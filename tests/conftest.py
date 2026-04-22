import os
from importlib import resources

import pytest

import proxai.connectors.model_configs as model_configs
import proxai.types as types

# Store original value globally
_ORIGINAL_API_KEY = None

# TODO: these test files were written against pre-multi-modal-refactor types
# (types.Response, types.QueryResponseRecord, types.ModelConfigsSchemaType,
# types.EndpointFeatureInfoType, types.ProviderModelConfigType, etc. — none
# of which exist anymore). They need from-scratch rewrites against the new
# ResultRecord / ModelRegistry surface. Skipping here so the rest of the
# suite stays collectable. Delete entries as they are rewritten.
collect_ignore = [
    'connections/test_available_models.py',
    'connectors/test_model_configs.py',
    'connectors/test_provider_connector.py',
    'logging/test_utils.py',
    'test_type_utils.py',
]


def _build_test_model_configs_instance() -> model_configs.ModelConfigs:
  """Build the shared pytest fixture registry.

  Tests load the curated `example_proxdash_model_configs.json` (reference
  config whose contents are documented and stable — see
  `src/proxai/connectors/model_configs_data/example_proxdash_model_configs.md`)
  rather than the moving-target bundled `v1.3.x.json`. The bundled registry is
  explicitly a work-in-progress production snapshot; its churn would silently
  drift test assertions that hard-code model names.

  The example config deliberately contains only two realistic providers
  (`openai`, `gemini`). The three `mock_*` providers that many tests rely on
  for stubbing behaviour are test scaffolding, not production models, so they
  are registered here programmatically instead of polluting the reference
  JSON.
  """
  instance = model_configs.ModelConfigs()
  data = resources.files(
      'proxai.connectors.model_configs_data'
  ).joinpath('example_proxdash_model_configs.json').read_text()
  instance.load_model_registry_from_json_string(data)

  S = types.FeatureSupportType.SUPPORTED
  NS = types.FeatureSupportType.NOT_SUPPORTED
  for provider, model in [
      ('mock_provider', 'mock_model'),
      ('mock_failing_provider', 'mock_failing_model'),
      ('mock_slow_provider', 'mock_slow_model'),
  ]:
    instance.register_provider_model_config(
        types.ProviderModelConfig(
            provider_model=types.ProviderModelType(
                provider=provider, model=model,
                provider_model_identifier=model),
            pricing=types.ProviderModelPricingType(
                input_token_cost=0.0, output_token_cost=0.0),
            metadata=types.ProviderModelMetadataType(
                is_recommended=False,
                model_size_tags=[types.ModelSizeType.SMALL]),
            features=types.FeatureConfigType(
                prompt=S, messages=S, system_prompt=S,
                parameters=types.ParameterConfigType(
                    temperature=S, max_tokens=S, stop=NS, n=NS, thinking=NS),
                tools=types.ToolConfigType(web_search=NS),
                input_format=types.InputFormatConfigType(
                    text=S, image=NS, document=NS, audio=NS, video=NS,
                    json=NS, pydantic=NS),
                output_format=types.OutputFormatConfigType(
                    text=S, json=S, pydantic=S, image=NS, audio=NS,
                    video=NS, multi_modal=NS),
            ),
        )
    )
  return instance


def pytest_configure(config):
  """Configure pytest before any imports happen."""
  global _ORIGINAL_API_KEY
  _ORIGINAL_API_KEY = os.environ.pop('PROXDASH_API_KEY', None)
  pytest.model_configs_instance = _build_test_model_configs_instance()


def pytest_unconfigure(config):
  """Restore environment after all tests complete."""
  if _ORIGINAL_API_KEY is not None:
    os.environ['PROXDASH_API_KEY'] = _ORIGINAL_API_KEY
