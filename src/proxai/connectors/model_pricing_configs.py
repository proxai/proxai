import math
from types import MappingProxyType
from typing import Dict
import proxai.types as types
import proxai.connectors.model_configs as model_configs

GENERATE_TEXT_PRICING: Dict[
    str, Dict[str, types.ProviderModelPricingType]] = MappingProxyType({
  # Mock provider
  'mock_provider': MappingProxyType({
    'mock_model': types.ProviderModelPricingType(
      per_response_token_cost=1.0,
      per_query_token_cost=2.0,
    ),
  }),

  # Mock failing provider
  'mock_failing_provider': MappingProxyType({
    'mock_failing_model': types.ProviderModelPricingType(
      per_response_token_cost=3.0,
      per_query_token_cost=4.0,
    ),
  }),

  # OpenAI
  'openai': MappingProxyType({
    'gpt-4': types.ProviderModelPricingType(
      per_response_token_cost=60.0,
      per_query_token_cost=30.0,
    ),
    'gpt-4-turbo-preview': types.ProviderModelPricingType(
      per_response_token_cost=30.0,
      per_query_token_cost=10.0,
    ),
    'gpt-3.5-turbo': types.ProviderModelPricingType(
      per_response_token_cost=1.5,
      per_query_token_cost=0.5,
    ),
    'babbage': types.ProviderModelPricingType(
      per_response_token_cost=0.4,
      per_query_token_cost=0.4,
    ),
    'davinci': types.ProviderModelPricingType(
      per_response_token_cost=2.0,
      per_query_token_cost=2.0,
    ),
    'gpt-3.5-turbo-instruct': types.ProviderModelPricingType(
      per_response_token_cost=2.0,
      per_query_token_cost=1.5,
    ),
  }),

  # Claude
  'claude': MappingProxyType({
    'claude-3-opus': types.ProviderModelPricingType(
      per_response_token_cost=15.0,
      per_query_token_cost=75.0,
    ),
    'claude-3-sonnet': types.ProviderModelPricingType(
      per_response_token_cost=3.0,
      per_query_token_cost=15.0,
    ),
    'claude-3-haiku': types.ProviderModelPricingType(
      per_response_token_cost=0.25,
      per_query_token_cost=1.25,
    ),
  }),

  # Gemini
  'gemini': MappingProxyType({
    'gemini-1.5-pro': types.ProviderModelPricingType(
      per_response_token_cost=5.00,
      per_query_token_cost=1.25,
    ),
    'gemini-1.5-flash-8b': types.ProviderModelPricingType(
      per_response_token_cost=0.15,
      per_query_token_cost=0.0375,
    ),
    'gemini-1.5-flash': types.ProviderModelPricingType(
      per_response_token_cost=0.075,
      per_query_token_cost=0.30,
    ),
    'gemini-2.0-flash-lite': types.ProviderModelPricingType(
      per_response_token_cost=0.075,
      per_query_token_cost=0.30,
    ),
    'gemini-2.0-flash': types.ProviderModelPricingType(
      per_response_token_cost=0.1,
      per_query_token_cost=0.4,
    ),
    'gemini-2.5-pro-preview-03-25': types.ProviderModelPricingType(
      per_response_token_cost=1.25,
      per_query_token_cost=10.00,
    ),
  }),
  # Cohere
  'cohere': MappingProxyType({
    'command-light': types.ProviderModelPricingType(
      per_response_token_cost=0.5,
      per_query_token_cost=1.5,
    ),
    'command-light-nightly': types.ProviderModelPricingType(
      per_response_token_cost=0.5,
      per_query_token_cost=1.5,
    ),
    'command': types.ProviderModelPricingType(
      per_response_token_cost=0.5,
      per_query_token_cost=1.5,
    ),
    'command-nightly': types.ProviderModelPricingType(
      per_response_token_cost=0.5,
      per_query_token_cost=1.5,
    ),
    'command-r': types.ProviderModelPricingType(
      per_response_token_cost=0.5,
      per_query_token_cost=1.5,
    ),
    'command-r-plus': types.ProviderModelPricingType(
      per_response_token_cost=3.0,
      per_query_token_cost=15.0,
    ),
  }),

  # Databricks
  'databricks': MappingProxyType({
    'dbrx-instruct': types.ProviderModelPricingType(
      per_response_token_cost=32.14,
      per_query_token_cost=96.42,
    ),
    'mixtral-8x7b-instruct': types.ProviderModelPricingType(
      per_response_token_cost=21.42,
      per_query_token_cost=21.42,
    ),
    'llama-2-70b-chat': types.ProviderModelPricingType(
      per_response_token_cost=28.57,
      per_query_token_cost=28.57,
    ),
    'llama-3-70b-instruct': types.ProviderModelPricingType(
      per_response_token_cost=14.28,
      per_query_token_cost=42.85,
    ),
    'bge-large-en': types.ProviderModelPricingType(
      per_response_token_cost=1.42,
      per_query_token_cost=1.42,
    ),
    'mpt-30b-instruct': types.ProviderModelPricingType(
      per_response_token_cost=14.28,
      per_query_token_cost=14.28,
    ),
    'mpt-7b-instruct': types.ProviderModelPricingType(
      per_response_token_cost=7.14,
      per_query_token_cost=7.14,
    ),
  }),

  # Mistral
  'mistral': MappingProxyType({
    'open-mistral-7b': types.ProviderModelPricingType(
      per_response_token_cost=0.25,
      per_query_token_cost=0.25,
    ),
    'open-mixtral-8x7b': types.ProviderModelPricingType(
      per_response_token_cost=0.7,
      per_query_token_cost=0.7,
    ),
    'open-mixtral-8x22b': types.ProviderModelPricingType(
      per_response_token_cost=2.0,
      per_query_token_cost=6.0,
    ),
    'mistral-small-latest': types.ProviderModelPricingType(
      per_response_token_cost=2.0,
      per_query_token_cost=6.0,
    ),
    'mistral-medium-latest': types.ProviderModelPricingType(
      per_response_token_cost=2.7,
      per_query_token_cost=8.1,
    ),
    'mistral-large-latest': types.ProviderModelPricingType(
      per_response_token_cost=8.0,
      per_query_token_cost=24.0,
    ),
  }),

  # Hugging Face
  'hugging_face': MappingProxyType({
    'google-gemma-7b-it': types.ProviderModelPricingType(
      per_response_token_cost=0.0,
      per_query_token_cost=0.0,
    ),
    'mistral-mixtral-8x7b-instruct': types.ProviderModelPricingType(
      per_response_token_cost=0.0,
      per_query_token_cost=0.0,
    ),
    'mistral-mistral-7b-instruct': types.ProviderModelPricingType(
      per_response_token_cost=0.0,
      per_query_token_cost=0.0,
    ),
    'nous-hermes-2-mixtral-8x7b': types.ProviderModelPricingType(
      per_response_token_cost=0.0,
      per_query_token_cost=0.0,
    ),
    'openchat-3.5': types.ProviderModelPricingType(
      per_response_token_cost=0.0,
      per_query_token_cost=0.0,
    ),
  }),
})


def get_provider_model_cost(
    provider_model_identifier: types.ProviderModelIdentifierType,
    query_token_count: int,
    response_token_count: int,
) -> int:
  provider_model = model_configs.get_provider_model_config(
      provider_model_identifier)
  pricing = GENERATE_TEXT_PRICING[provider_model.provider][provider_model.model]
  return math.floor(query_token_count * pricing.per_query_token_cost +
                    response_token_count * pricing.per_response_token_cost)
