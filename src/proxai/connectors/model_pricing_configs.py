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
    'gpt-4.1': types.ProviderModelPricingType(
      per_response_token_cost=2.00,
      per_query_token_cost=8.00,
    ),
    'gpt-4.1-mini': types.ProviderModelPricingType(
      per_response_token_cost=0.40,
      per_query_token_cost=1.60,
    ),
    'gpt-4.1-nano': types.ProviderModelPricingType(
      per_response_token_cost=0.10,
      per_query_token_cost=0.40,
    ),
    'gpt-4.5-preview': types.ProviderModelPricingType(
      per_response_token_cost=75.00,
      per_query_token_cost=150.00,
    ),
    'gpt-4o': types.ProviderModelPricingType(
      per_response_token_cost=2.50,
      per_query_token_cost=10.00,
    ),
    'gpt-4o-audio-preview': types.ProviderModelPricingType(
      per_response_token_cost=2.50,
      per_query_token_cost=10.00,
    ),
    'gpt-4o-realtime-preview': types.ProviderModelPricingType(
      per_response_token_cost=5.00,
      per_query_token_cost=20.00,
    ),
    'gpt-4o-mini': types.ProviderModelPricingType(
      per_response_token_cost=0.15,
      per_query_token_cost=0.60,
    ),
    'gpt-4o-mini-audio-preview': types.ProviderModelPricingType(
      per_response_token_cost=0.15,
      per_query_token_cost=0.60,
    ),
    'gpt-4o-mini-realtime-preview': types.ProviderModelPricingType(
      per_response_token_cost=0.60,
      per_query_token_cost=2.40,
    ),
    'o1': types.ProviderModelPricingType(
      per_response_token_cost=15.00,
      per_query_token_cost=60.00,
    ),
    'o1-pro': types.ProviderModelPricingType(
      per_response_token_cost=150.00,
      per_query_token_cost=600.00,
    ),
    'o3': types.ProviderModelPricingType(
      per_response_token_cost=10.00,
      per_query_token_cost=40.00,
    ),
    'o4-mini': types.ProviderModelPricingType(
      per_response_token_cost=1.10,
      per_query_token_cost=4.40,
    ),
    'o3-mini': types.ProviderModelPricingType(
      per_response_token_cost=1.10,
      per_query_token_cost=4.40,
    ),
    'o1-mini': types.ProviderModelPricingType(
      per_response_token_cost=1.10,
      per_query_token_cost=4.40,
    ),
    'gpt-4o-mini-search-preview': types.ProviderModelPricingType(
      per_response_token_cost=0.15,
      per_query_token_cost=0.60,
    ),
    'gpt-4o-search-preview': types.ProviderModelPricingType(
      per_response_token_cost=2.50,
      per_query_token_cost=10.00,
    ),
    'computer-use-preview': types.ProviderModelPricingType(
      per_response_token_cost=3.00,
      per_query_token_cost=12.00,
    ),
    'chatgpt-4o-latest': types.ProviderModelPricingType(
      per_response_token_cost=5.00,
      per_query_token_cost=15.00,
    ),
    'gpt-4-turbo': types.ProviderModelPricingType(
      per_response_token_cost=10.00,
      per_query_token_cost=30.00,
    ),
    'gpt-4': types.ProviderModelPricingType(
      per_response_token_cost=30.00,
      per_query_token_cost=60.00,
    ),
    'gpt-4-32k': types.ProviderModelPricingType(
      per_response_token_cost=60.00,
      per_query_token_cost=120.00,
    ),
    'gpt-3.5-turbo': types.ProviderModelPricingType(
      per_response_token_cost=0.50,
      per_query_token_cost=1.50,
    ),
  }),

  # Claude
  'claude': MappingProxyType({
    'sonnet': types.ProviderModelPricingType(
      per_response_token_cost=3.0,
      per_query_token_cost=15.0,
    ),
    'haiku': types.ProviderModelPricingType(
      per_response_token_cost=0.8,
      per_query_token_cost=4.0,
    ),
    '3.5-sonnet-v2': types.ProviderModelPricingType(
      per_response_token_cost=3.0,
      per_query_token_cost=15.0,
    ),
    '3.5-sonnet': types.ProviderModelPricingType(
      per_response_token_cost=3.0,
      per_query_token_cost=15.0,
    ),
    'opus': types.ProviderModelPricingType(
      per_response_token_cost=15.0,
      per_query_token_cost=75.0,
    ),
    '3-sonnet': types.ProviderModelPricingType(
      per_response_token_cost=3.00,
      per_query_token_cost=15.00,
    ),
    '3-haiku': types.ProviderModelPricingType(
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
    'command-a': types.ProviderModelPricingType(
      per_response_token_cost=2.5,
      per_query_token_cost=10.0,
    ),
    'command-r7b': types.ProviderModelPricingType(
      per_response_token_cost=0.0375,
      per_query_token_cost=0.15,
    ),
    'command-r-plus': types.ProviderModelPricingType(
      per_response_token_cost=2.5,
      per_query_token_cost=10.0,
    ),
    'command-r-08-2024': types.ProviderModelPricingType(
      per_response_token_cost=0.15,
      per_query_token_cost=0.6,
    ),
    'command-r': types.ProviderModelPricingType(
      per_response_token_cost=0.15,
      per_query_token_cost=0.6,
    ),
    'command': types.ProviderModelPricingType(
      per_response_token_cost=0.5,
      per_query_token_cost=1.5,
    ),
    'command-nightly': types.ProviderModelPricingType(
      per_response_token_cost=0.5,
      per_query_token_cost=1.5,
    ),
    'command-light': types.ProviderModelPricingType(
      per_response_token_cost=0.0375,
      per_query_token_cost=0.15,
    ),
    'command-light-nightly': types.ProviderModelPricingType(
      per_response_token_cost=0.0375,
      per_query_token_cost=0.15,
    ),
  }),

  # Databricks
  'databricks': MappingProxyType({
    'llama-4-maverick': types.ProviderModelPricingType(
      per_response_token_cost=2.5,
      per_query_token_cost=12.5,
    ),
    'claude-3-7-sonnet': types.ProviderModelPricingType(
      per_response_token_cost=2.5,
      per_query_token_cost=12.5,
    ),
    'meta-llama-3-1-8b-it': types.ProviderModelPricingType(
      per_response_token_cost=2.5,
      per_query_token_cost=12.5,
    ),
    'meta-llama-3-3-70b-it': types.ProviderModelPricingType(
      per_response_token_cost=2.5,
      per_query_token_cost=12.5,
    ),
    # TODO: This is extremely slow model. Until better filtering, it is not
    # included in the list.
    # 'meta-llama-3-1-405b-it': types.ProviderModelPricingType(
    #   per_response_token_cost=2.5,
    #   per_query_token_cost=12.5,
    # ),
    'dbrx-it': types.ProviderModelPricingType(
      per_response_token_cost=2.5,
      per_query_token_cost=12.5,
    ),
    'mixtral-8x7b-it': types.ProviderModelPricingType(
      per_response_token_cost=2.5,
      per_query_token_cost=12.5,
    ),
  }),

  # Mistral
  'mistral': MappingProxyType({
    'ministral-3b': types.ProviderModelPricingType(
      per_response_token_cost=0.04,
      per_query_token_cost=0.04,
    ),
    'ministral-8b': types.ProviderModelPricingType(
      per_response_token_cost=0.1,
      per_query_token_cost=0.1,
    ),
    'open-mistral-7b': types.ProviderModelPricingType(
      per_response_token_cost=0.25,
      per_query_token_cost=0.25,
    ),
    'open-mistral-nemo': types.ProviderModelPricingType(
      per_response_token_cost=0.15,
      per_query_token_cost=0.15,
    ),
    'open-mixtral-8x7b': types.ProviderModelPricingType(
      per_response_token_cost=0.7,
      per_query_token_cost=0.7,
    ),
    'open-mixtral-8x22b': types.ProviderModelPricingType(
      per_response_token_cost=2,
      per_query_token_cost=6,
    ),
    'mistral-small': types.ProviderModelPricingType(
      per_response_token_cost=0.1,
      per_query_token_cost=0.3,
    ),
    'mistral-large': types.ProviderModelPricingType(
      per_response_token_cost=2,
      per_query_token_cost=6,
    ),
    'pixtral-large': types.ProviderModelPricingType(
      per_response_token_cost=2,
      per_query_token_cost=6,
    ),
    'codestral': types.ProviderModelPricingType(
      per_response_token_cost=0.3,
      per_query_token_cost=0.9,
    ),
    'pixtral-12b': types.ProviderModelPricingType(
      per_response_token_cost=0.15,
      per_query_token_cost=0.15,
    ),
    'mistral-saba': types.ProviderModelPricingType(
      per_response_token_cost=0.2,
      per_query_token_cost=0.6,
    ),
  }),

  # Hugging Face
  'huggingface': MappingProxyType({
    'gemma-2-2b-it': types.ProviderModelPricingType(
      per_response_token_cost=1.48,
      per_query_token_cost=3.7,
    ),
    'meta-llama-3.1-8b-it': types.ProviderModelPricingType(
      per_response_token_cost=1.94,
      per_query_token_cost=2.22,
    ),
    'phi-4': types.ProviderModelPricingType(
      per_response_token_cost=33.33,
      per_query_token_cost=37.0,
    ),
    'qwen2.5-coder-32b-it': types.ProviderModelPricingType(
      per_response_token_cost=28.57,
      per_query_token_cost=31.74,
    ),
    'deepseek-r1': types.ProviderModelPricingType(
      per_response_token_cost=50.0,
      per_query_token_cost=55.56,
    ),
    'deepseek-v3': types.ProviderModelPricingType(
      per_response_token_cost=25.0,
      per_query_token_cost=27.78,
    ),
  }),

  # DeepSeek
  'deepseek': MappingProxyType({
    'deepseek-v3': types.ProviderModelPricingType(
      per_response_token_cost=0.27,
      per_query_token_cost=1.10,
    ),
    'deepseek-r1': types.ProviderModelPricingType(
      per_response_token_cost=0.55,
      per_query_token_cost=2.19,
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
