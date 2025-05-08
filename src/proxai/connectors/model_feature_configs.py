from types import MappingProxyType
from typing import Dict
import proxai.types as types

GENERATE_TEXT_FEATURES: Dict[
    str, Dict[str, types.ProviderModelFeatureType]] = MappingProxyType({
  'mistral': MappingProxyType({
    'ministral-3b': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'ministral-8b': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'open-mistral-7b': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'open-mistral-nemo': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'open-mixtral-8x7b': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'open-mixtral-8x22b': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'mistral-small': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'mistral-large': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'pixtral-large': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'codestral': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'pixtral-12b': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
    'mistral-saba': types.ProviderModelFeatureType(
      not_supported_features=['stop'],
    ),
  }),

  'openai': MappingProxyType({
    'gpt-4o-search-preview': types.ProviderModelFeatureType(
      not_supported_features=['temperature', 'stop'],
    ),
    'gpt-4o-mini-search-preview': types.ProviderModelFeatureType(
      not_supported_features=['temperature', 'stop'],
    ),
    'o1-mini': types.ProviderModelFeatureType(
      not_supported_features=['system', 'max_tokens', 'temperature', 'stop'],
    ),
    'o1': types.ProviderModelFeatureType(
      not_supported_features=['temperature'],
    ),
    'o3-mini': types.ProviderModelFeatureType(
      not_supported_features=['max_tokens', 'temperature'],
    ),
    'o4-mini': types.ProviderModelFeatureType(
      not_supported_features=['max_tokens', 'temperature', 'stop'],
    ),
  }),
})
