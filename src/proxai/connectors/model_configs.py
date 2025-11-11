from types import MappingProxyType
import datetime
from typing import Dict, Tuple, List
import proxai.types as types
import proxai.type_utils as type_utils
import proxai.connectors.provider_model_configs as provider_model_configs
import proxai.connectors.model_pricing_configs as model_pricing_configs
import proxai.connectors.model_features_configs as model_features_configs

PROVIDER_KEY_MAP: Dict[str, Tuple[str]] = MappingProxyType({
    'claude': tuple(['ANTHROPIC_API_KEY']),
    'cohere': tuple(['CO_API_KEY']),
    'databricks': tuple(['DATABRICKS_TOKEN', 'DATABRICKS_HOST']),
    'deepseek': tuple(['DEEPSEEK_API_KEY']),
    'gemini': tuple(['GEMINI_API_KEY']),
    'grok': tuple(['XAI_API_KEY']),
    'huggingface': tuple(['HUGGINGFACE_API_KEY']),
    'mistral': tuple(['MISTRAL_API_KEY']),
    'openai': tuple(['OPENAI_API_KEY']),

    'mock_provider': tuple(['MOCK_PROVIDER_API_KEY']),
    'mock_failing_provider': tuple(['MOCK_FAILING_PROVIDER']),
    'mock_slow_provider': tuple(['MOCK_SLOW_PROVIDER']),
})

FEATURED_MODELS: Tuple[types.ProviderModelIdentifierType] = (
    ('mock_provider', 'mock_model'),

    ('mock_failing_provider', 'mock_failing_model'),

    ('mock_slow_provider', 'mock_slow_model'),

    ('openai', 'gpt-4.1'),
    ('openai', 'gpt-4.1-mini'),
    ('openai', 'gpt-4.1-nano'),
    ('openai', 'gpt-4.5-preview'),
    ('openai', 'gpt-4o'),
    ('openai', 'gpt-4o-mini'),
    ('openai', 'o1'),
    ('openai', 'o1-pro'),
    ('openai', 'o3'),
    ('openai', 'o4-mini'),
    ('openai', 'o3-mini'),
    ('openai', 'o1-mini'),
    ('openai', 'gpt-4o-mini-search-preview'),
    ('openai', 'gpt-4o-search-preview'),
    ('openai', 'chatgpt-4o-latest'),
    ('openai', 'gpt-4-turbo'),
    ('openai', 'gpt-4'),
    ('openai', 'gpt-3.5-turbo'),

    ('claude', 'sonnet-4'),
    ('claude', 'opus-4'),
    ('claude', 'sonnet-3.7'),
    ('claude', 'haiku-3.5'),
    ('claude', 'sonnet-3.5'),
    ('claude', 'sonnet-3.5-old'),
    ('claude', 'opus-3'),
    ('claude', 'sonnet-3'),
    ('claude', 'haiku-3'),

    ('gemini', 'gemini-2.5-pro'),
    ('gemini', 'gemini-2.5-flash'),
    ('gemini', 'gemini-2.5-flash-lite-preview-06-17'),
    ('gemini', 'gemini-2.0-flash'),
    ('gemini', 'gemini-2.0-flash-lite'),
    ('gemini', 'gemini-1.5-flash'),
    ('gemini', 'gemini-1.5-flash-8b'),
    ('gemini', 'gemini-1.5-pro'),

    ('cohere', 'command-a'),
    ('cohere', 'command-r7b'),
    ('cohere', 'command-r-plus'),
    ('cohere', 'command-r-08-2024'),
    ('cohere', 'command-r'),
    ('cohere', 'command'),
    ('cohere', 'command-nightly'),
    ('cohere', 'command-light'),
    ('cohere', 'command-light-nightly'),

    ('databricks', 'llama-4-maverick'),
    ('databricks', 'claude-3-7-sonnet'),
    ('databricks', 'meta-llama-3-1-8b-it'),
    ('databricks', 'meta-llama-3-3-70b-it'),
    ('databricks', 'dbrx-it'),
    ('databricks', 'mixtral-8x7b-it'),

    ('mistral', 'codestral'),
    ('mistral', 'ministral-3b'),
    ('mistral', 'ministral-8b'),
    ('mistral', 'mistral-large'),
    ('mistral', 'mistral-medium'),
    ('mistral', 'mistral-saba'),
    ('mistral', 'mistral-small'),
    ('mistral', 'open-mistral-7b'),
    ('mistral', 'open-mistral-nemo'),
    ('mistral', 'open-mixtral-8x22b'),
    ('mistral', 'open-mixtral-8x7b'),
    ('mistral', 'pixtral-12b'),
    ('mistral', 'pixtral-large'),

    ('huggingface', 'gemma-2-2b-it'),
    ('huggingface', 'meta-llama-3.1-8b-it'),
    ('huggingface', 'phi-4'),
    ('huggingface', 'qwen3-32b'),
    ('huggingface', 'deepseek-r1'),
    ('huggingface', 'deepseek-v3'),

    ('deepseek', 'deepseek-v3'),
    ('deepseek', 'deepseek-r1'),

    ('grok', 'grok-3-beta'),
    ('grok', 'grok-3-fast-beta'),
    ('grok', 'grok-3-mini-beta'),
    ('grok', 'grok-3-mini-fast-beta'),
)

MODELS_BY_CALL_TYPE: Dict[CallType, Tuple[ProviderModelIdentifierType]] = MappingProxyType({
    types.CallType.GENERATE_TEXT: (
        ('mock_provider', 'mock_model'),

        ('mock_failing_provider', 'mock_failing_model'),

        ('mock_slow_provider', 'mock_slow_model'),

        ('openai', 'gpt-4.1'),
        ('openai', 'gpt-4.1-mini'),
        ('openai', 'gpt-4.1-nano'),
        ('openai', 'gpt-4.5-preview'),
        ('openai', 'gpt-4o'),
        ('openai', 'gpt-4o-mini'),
        ('openai', 'o1'),
        ('openai', 'o1-pro'),
        ('openai', 'o3'),
        ('openai', 'o4-mini'),
        ('openai', 'o3-mini'),
        ('openai', 'o1-mini'),
        ('openai', 'gpt-4o-mini-search-preview'),
        ('openai', 'gpt-4o-search-preview'),
        ('openai', 'chatgpt-4o-latest'),
        ('openai', 'gpt-4-turbo'),
        ('openai', 'gpt-4'),
        ('openai', 'gpt-3.5-turbo'),

        ('claude', 'sonnet-4'),
        ('claude', 'opus-4'),
        ('claude', 'sonnet-3.7'),
        ('claude', 'haiku-3.5'),
        ('claude', 'sonnet-3.5'),
        ('claude', 'sonnet-3.5-old'),
        ('claude', 'opus-3'),
        ('claude', 'sonnet-3'),
        ('claude', 'haiku-3'),

        ('gemini', 'gemini-2.5-pro'),
        ('gemini', 'gemini-2.5-flash'),
        ('gemini', 'gemini-2.5-flash-lite-preview-06-17'),
        ('gemini', 'gemini-2.0-flash'),
        ('gemini', 'gemini-2.0-flash-lite'),
        ('gemini', 'gemini-1.5-flash'),
        ('gemini', 'gemini-1.5-flash-8b'),
        ('gemini', 'gemini-1.5-pro'),

        ('cohere', 'command-a'),
        ('cohere', 'command-r7b'),
        ('cohere', 'command-r-plus'),
        ('cohere', 'command-r-08-2024'),
        ('cohere', 'command-r'),
        ('cohere', 'command'),
        ('cohere', 'command-nightly'),
        ('cohere', 'command-light'),
        ('cohere', 'command-light-nightly'),

        ('databricks', 'llama-4-maverick'),
        ('databricks', 'claude-3-7-sonnet'),
        ('databricks', 'meta-llama-3-1-8b-it'),
        ('databricks', 'meta-llama-3-3-70b-it'),
        ('databricks', 'meta-llama-3-1-405b-it'),
        ('databricks', 'dbrx-it'),
        ('databricks', 'mixtral-8x7b-it'),

        ('mistral', 'codestral'),
        ('mistral', 'ministral-3b'),
        ('mistral', 'ministral-8b'),
        ('mistral', 'mistral-large'),
        ('mistral', 'mistral-medium'),
        ('mistral', 'mistral-saba'),
        ('mistral', 'mistral-small'),
        ('mistral', 'open-mistral-7b'),
        ('mistral', 'open-mistral-nemo'),
        ('mistral', 'open-mixtral-8x22b'),
        ('mistral', 'open-mixtral-8x7b'),
        ('mistral', 'pixtral-12b'),
        ('mistral', 'pixtral-large'),

        ('huggingface', 'gemma-2-2b-it'),
        ('huggingface', 'meta-llama-3.1-8b-it'),
        ('huggingface', 'phi-4'),
        ('huggingface', 'qwen3-32b'),
        ('huggingface', 'deepseek-r1'),
        ('huggingface', 'deepseek-v3'),

        ('deepseek', 'deepseek-v3'),
        ('deepseek', 'deepseek-r1'),

        ('grok', 'grok-3-beta'),
        ('grok', 'grok-3-fast-beta'),
        ('grok', 'grok-3-mini-beta'),
        ('grok', 'grok-3-mini-fast-beta'),
    ),
})

MODELS_BY_SIZE: Dict[ModelSizeType, Tuple[ProviderModelIdentifierType]] = MappingProxyType({
    'small': (
        ('claude', 'haiku-3'),
        ('cohere', 'command-light'),
        ('cohere', 'command-r'),
        ('cohere', 'command-r7b'),
        ('deepseek', 'deepseek-v3'),
        ('gemini', 'gemini-2.0-flash'),
        ('grok', 'grok-3-mini-beta'),
        ('mistral', 'codestral'),
        ('mistral', 'mistral-small'),
        ('mistral', 'pixtral-12b'),
        ('openai', 'gpt-4.1-nano'),
        ('openai', 'gpt-4o-mini'),
        ('openai', 'gpt-4o-mini-search-preview'),
    ),
    'medium': (
        ('claude', 'haiku-3.5'),
        ('cohere', 'command'),
        ('cohere', 'command-nightly'),
        ('deepseek', 'deepseek-r1'),
        ('gemini', 'gemini-1.5-pro'),
        ('gemini', 'gemini-2.5-flash'),
        ('grok', 'grok-3-mini-fast-beta'),
        ('huggingface', 'gemma-2-2b-it'),
        ('huggingface', 'meta-llama-3.1-8b-it'),
        ('mistral', 'mistral-large'),
        ('mistral', 'open-mixtral-8x22b'),
        ('mistral', 'pixtral-large'),
        ('openai', 'gpt-3.5-turbo'),
        ('openai', 'gpt-4.1-mini'),
        ('openai', 'o1-mini'),
        ('openai', 'o4-mini'),
        ('openai', 'o3-mini'),
    ),
    'large': (
        ('claude', 'opus-4'),
        ('claude', 'sonnet-4'),
        ('cohere', 'command-a'),
        ('cohere', 'command-r-plus'),
        ('databricks', 'claude-3-7-sonnet'),
        ('databricks', 'meta-llama-3-1-8b-it'),
        ('databricks', 'meta-llama-3-3-70b-it'),
        ('databricks', 'dbrx-it'),
        ('databricks', 'mixtral-8x7b-it'),
        ('gemini', 'gemini-2.5-pro'),
        ('grok', 'grok-3-beta'),
        ('grok', 'grok-3-fast-beta'),
        ('huggingface', 'deepseek-r1'),
        ('huggingface', 'deepseek-v3'),
        ('huggingface', 'phi-4'),
        ('huggingface', 'qwen3-32b'),
        ('openai', 'gpt-4.1'),
        ('openai', 'gpt-4.5-preview'),
        ('openai', 'gpt-4o'),
        ('openai', 'o1'),
        ('openai', 'o1-pro'),
        ('openai', 'o3'),
        ('openai', 'gpt-4o-search-preview'),
        ('openai', 'chatgpt-4o-latest'),
        ('openai', 'gpt-4-turbo'),
        ('openai', 'gpt-4'),
    ),
    'largest': (
        ('claude', 'opus-4'),
        ('cohere', 'command-a'),
        ('databricks', 'dbrx-it'),
        ('databricks', 'meta-llama-3-3-70b-it'),
        ('deepseek', 'deepseek-r1'),
        ('gemini', 'gemini-2.5-pro'),
        ('grok', 'grok-3-beta'),
        ('huggingface', 'phi-4'),
        ('huggingface', 'qwen3-32b'),
        ('mistral', 'mistral-large'),
        ('openai', 'gpt-4.1'),
    ),
})

DEFAULT_MODEL_PRIORITY_LIST: Tuple[ProviderModelIdentifierType] = (
    ('openai', 'gpt-4o-mini'),
    ('gemini', 'gemini-2.0-flash'),
    ('claude', 'haiku-3.5'),
    ('grok', 'grok-3-mini-fast-beta'),
    ('cohere', 'command-r'),
    ('mistral', 'mistral-small'),
    ('deepseek', 'deepseek-v3'),
    ('huggingface', 'gemma-2-2b-it'),
    ('databricks', 'llama-4-maverick'),
)


def get_provider_model_configs() -> Tuple[ProviderModelConfigType]:
  result_config = []
  for provider in provider_model_configs.PROVIDER_MODELS:
    for model in provider_model_configs.PROVIDER_MODELS[provider]:
      is_featured = (provider, model) in FEATURED_MODELS

      model_size = None
      for model_size_type in types.ModelSizeType:
        if (provider, model) in MODELS_BY_SIZE[model_size_type]:
          model_size = model_size_type
          break

      is_default_candidate = False
      if (provider, model) in DEFAULT_MODEL_PRIORITY_LIST:
        is_default_candidate = True
        default_candidate_priority = DEFAULT_MODEL_PRIORITY_LIST.index(
            (provider, model))

      call_type = None
      for call_type in types.CallType:
        if (provider, model) in MODELS_BY_CALL_TYPE[call_type]:
          call_type = call_type
          break

      result_config.append(types.ProviderModelConfigType(
          provider_model=provider_model_configs.PROVIDER_MODELS[
              provider][model],
          pricing=model_pricing_configs.PROVIDER_MODEL_PRICING[
              provider][model],
          features=model_features_configs.PROVIDER_MODEL_FEATURES[
              provider][model],
          metadata=types.ProviderModelMetadataType(
            call_type=call_type,
            is_featured=is_featured,
            model_size=model_size,
            is_default_candidate=is_default_candidate,
            default_candidate_priority=default_candidate_priority,
          ),
      ))
  return tuple(result_config)


ALL_MODELS_CONFIG: types.AllModelsConfigType = types.AllModelsConfigType(
    version='1.0.0',
    released_at=datetime.datetime(2025, 11, 11),
    config_origin=types.ConfigOriginType.BUILT_IN,
    release_notes='First release of the model configs',

    provider_model_configs=get_provider_model_configs(),

    featured_models=FEATURED_MODELS,
    models_by_call_type=MODELS_BY_CALL_TYPE,
    models_by_size=MODELS_BY_SIZE,
    default_model_priority_list=DEFAULT_MODEL_PRIORITY_LIST,
)


def get_provider_model_config(
    model_identifier: types.ProviderModelIdentifierType
) -> types.ProviderModelType:
  if type_utils.is_provider_model_tuple(model_identifier):
    return ALL_MODELS_CONFIG.provider_model_configs[
      model_identifier[0]][model_identifier[1]].provider_model
  else:
    return model_identifier
