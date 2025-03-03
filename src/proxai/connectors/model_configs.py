from types import MappingProxyType
from typing import Dict, Tuple
import proxai.types as types
import proxai.type_utils as type_utils

PROVIDER_KEY_MAP: Dict[str, Tuple[str]] = MappingProxyType({
    'openai': tuple(['OPENAI_API_KEY']),
    'claude': tuple(['ANTHROPIC_API_KEY']),
    'gemini': tuple(['GOOGLE_API_KEY']),
    'cohere': tuple(['CO_API_KEY']),
    'databricks': tuple(['DATABRICKS_TOKEN', 'DATABRICKS_HOST']),
    'mistral': tuple(['MISTRAL_API_KEY']),
    'hugging_face': tuple(['HUGGINGFACE_API_KEY']),
    'mock_provider': tuple(['MOCK_PROVIDER_API_KEY']),
    'mock_failing_provider': tuple(['MOCK_FAILING_PROVIDER']),
})


ALL_MODELS: Dict[str, Dict[str, types.ProviderModelType]] = MappingProxyType({
  # Mock provider
  'mock_provider': MappingProxyType({
    'mock_model': types.ProviderModelType(
      provider='mock_provider',
      model='mock_model',
      provider_model_identifier='mock_model'
    ),
  }),

  # Mock failing provider
  'mock_failing_provider': MappingProxyType({
    'mock_failing_model': types.ProviderModelType(
      provider='mock_failing_provider',
      model='mock_failing_model',
      provider_model_identifier='mock_failing_model'
    ),
  }),

  # OpenAI models.
  # Models provided by OpenAI:
  # https://platform.openai.com/docs/guides/text-generation
  'openai': MappingProxyType({
    # Newer models (2023–)
    'gpt-4': types.ProviderModelType(
      provider='openai',
      model='gpt-4',
      provider_model_identifier='gpt-4'
    ),
    'gpt-4-turbo-preview': types.ProviderModelType(
      provider='openai',
      model='gpt-4-turbo-preview',
      provider_model_identifier='gpt-4-turbo-preview'
    ),
    'gpt-3.5-turbo': types.ProviderModelType(
      provider='openai',
      model='gpt-3.5-turbo',
      provider_model_identifier='gpt-3.5-turbo'
    ),

    # Updated legacy models (2023)
    'babbage': types.ProviderModelType(
      provider='openai',
      model='babbage',
      provider_model_identifier='babbage-002'
    ),
    'davinci': types.ProviderModelType(
      provider='openai',
      model='davinci',
      provider_model_identifier='davinci-002'
    ),
    'gpt-3.5-turbo-instruct': types.ProviderModelType(
      provider='openai',
      model='gpt-3.5-turbo-instruct',
      provider_model_identifier='gpt-3.5-turbo-instruct'
    ),
  }),

  # Claude models.
  # Models provided by Claude:
  # https://claude.ai/docs/models
  'claude': MappingProxyType({
    'claude-3-opus': types.ProviderModelType(
      provider='claude',
      model='claude-3-opus',
      provider_model_identifier='claude-3-opus-20240229'
    ),
    'claude-3-sonnet': types.ProviderModelType(
      provider='claude',
      model='claude-3-sonnet',
      provider_model_identifier='claude-3-sonnet-20240229'
    ),
    'claude-3-haiku': types.ProviderModelType(
      provider='claude',
      model='claude-3-haiku',
      provider_model_identifier='claude-3-haiku-20240307'
    ),
  }),

  # Gemini models.
  # Models provided by Gemini:
  # https://ai.google.dev/models/gemini
  'gemini': MappingProxyType({
    'gemini-1.0-pro': types.ProviderModelType(
      provider='gemini',
      model='gemini-1.0-pro',
      provider_model_identifier='gemini-1.0-pro'
    ),
    'gemini-1.0-pro-001': types.ProviderModelType(
      provider='gemini',
      model='gemini-1.0-pro-001',
      provider_model_identifier='gemini-1.0-pro-001'
    ),
    'gemini-1.0-pro-latest': types.ProviderModelType(
      provider='gemini',
      model='gemini-1.0-pro-latest',
      provider_model_identifier='gemini-1.0-pro-latest'
    ),
    'gemini-1.0-pro-vision-latest': types.ProviderModelType(
      provider='gemini',
      model='gemini-1.0-pro-vision-latest',
      provider_model_identifier='gemini-1.0-pro-vision-latest'
    ),
    'gemini-1.5-pro-latest': types.ProviderModelType(
      provider='gemini',
      model='gemini-1.5-pro-latest',
      provider_model_identifier='gemini-1.5-pro-latest'
    ),
    'gemini-pro': types.ProviderModelType(
      provider='gemini',
      model='gemini-pro',
      provider_model_identifier='gemini-pro'
    ),
    'gemini-pro-vision': types.ProviderModelType(
      provider='gemini',
      model='gemini-pro-vision',
      provider_model_identifier='gemini-pro-vision'
    ),
  }),

  # Cohere models.
  # Models provided by Cohere:
  # https://docs.cohere.com/docs/models
  'cohere': MappingProxyType({
    'command-light': types.ProviderModelType(
      provider='cohere',
      model='command-light',
      provider_model_identifier='command-light'
    ),
    'command-light-nightly': types.ProviderModelType(
      provider='cohere',
      model='command-light-nightly',
      provider_model_identifier='command-light-nightly'
    ),
    'command': types.ProviderModelType(
      provider='cohere',
      model='command',
      provider_model_identifier='command'
    ),
    'command-nightly': types.ProviderModelType(
      provider='cohere',
      model='command-nightly',
      provider_model_identifier='command-nightly'
    ),
    'command-r': types.ProviderModelType(
      provider='cohere',
      model='command-r',
      provider_model_identifier='command-r'
    ),
    'command-r-plus': types.ProviderModelType(
      provider='cohere',
      model='command-r-plus',
      provider_model_identifier='command-r-plus'
    ),
  }),

  # Databricks models.
  # Models provided by Databricks:
  # https://docs.databricks.com/en/machine-learning/foundation-models/index.html#provisioned-throughput-foundation-model-apis
  'databricks': MappingProxyType({
    'dbrx-instruct': types.ProviderModelType(
      provider='databricks',
      model='dbrx-instruct',
      provider_model_identifier='databricks-dbrx-instruct'
    ),
    'mixtral-8x7b-instruct': types.ProviderModelType(
      provider='databricks',
      model='mixtral-8x7b-instruct',
      provider_model_identifier='databricks-mixtral-8x7b-instruct'
    ),
    'llama-2-70b-chat': types.ProviderModelType(
      provider='databricks',
      model='llama-2-70b-chat',
      provider_model_identifier='databricks-llama-2-70b-chat'
    ),
    'llama-3-70b-instruct': types.ProviderModelType(
      provider='databricks',
      model='llama-3-70b-instruct',
      provider_model_identifier='databricks-meta-llama-3-70b-instruct'
    ),
    'bge-large-en': types.ProviderModelType(
      provider='databricks',
      model='bge-large-en',
      provider_model_identifier='databricks-bge-large-en'
    ),
    'mpt-30b-instruct': types.ProviderModelType(
      provider='databricks',
      model='mpt-30b-instruct',
      provider_model_identifier='databricks-mpt-30b-instruct'
    ),
    'mpt-7b-instruct': types.ProviderModelType(
      provider='databricks',
      model='mpt-7b-instruct',
      provider_model_identifier='databricks-mpt-7b-instruct'
    ),
  }),

  # Mistral models.
  # Models provided by Mistral:
  # https://docs.mistral.ai/platform/endpoints/
  'mistral': MappingProxyType({
    'open-mistral-7b': types.ProviderModelType(
      provider='mistral',
      model='open-mistral-7b',
      provider_model_identifier='open-mistral-7b'
    ),
    'open-mixtral-8x7b': types.ProviderModelType(
      provider='mistral',
      model='open-mixtral-8x7b',
      provider_model_identifier='open-mixtral-8x7b'
    ),
    'open-mixtral-8x22b': types.ProviderModelType(
      provider='mistral',
      model='open-mixtral-8x22b',
      provider_model_identifier='open-mixtral-8x22b'
    ),
    'mistral-small-latest': types.ProviderModelType(
      provider='mistral',
      model='mistral-small-latest',
      provider_model_identifier='mistral-small-latest'
    ),
    'mistral-medium-latest': types.ProviderModelType(
      provider='mistral',
      model='mistral-medium-latest',
      provider_model_identifier='mistral-medium-latest'
    ),
    'mistral-large-latest': types.ProviderModelType(
      provider='mistral',
      model='mistral-large-latest',
      provider_model_identifier='mistral-large-latest'
    ),
  }),

  # Hugging Face models.
  # Models provided by Hugging Face on HuggingFaceChat:
  # https://huggingface.co/chat/models
  'hugging_face': MappingProxyType({
    'google-gemma-7b-it': types.ProviderModelType(
      provider='hugging_face',
      model='google-gemma-7b-it',
      provider_model_identifier='google/gemma-7b-it'
    ),
    'mistral-mixtral-8x7b-instruct': types.ProviderModelType(
      provider='hugging_face',
      model='mistral-mixtral-8x7b-instruct',
      provider_model_identifier='mistralai/Mixtral-8x7B-Instruct-v0.1'
    ),
    'mistral-mistral-7b-instruct': types.ProviderModelType(
      provider='hugging_face',
      model='mistral-mistral-7b-instruct',
      provider_model_identifier='mistralai/Mistral-7B-Instruct-v0.2'
    ),
    'nous-hermes-2-mixtral-8x7b': types.ProviderModelType(
      provider='hugging_face',
      model='nous-hermes-2-mixtral-8x7b',
      provider_model_identifier='NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO'
    ),
    'openchat-3.5': types.ProviderModelType(
      provider='hugging_face',
      model='openchat-3.5',
      provider_model_identifier='openchat/openchat-3.5-0106'
    ),
  }),
})


GENERATE_TEXT_MODELS: Dict[
    str, Dict[str, types.ProviderModelType]] = MappingProxyType({
  'mock_provider': MappingProxyType({
    'mock_model': ALL_MODELS['mock_provider']['mock_model'],
  }),

  'mock_failing_provider': MappingProxyType({
    'mock_failing_model': ALL_MODELS[
        'mock_failing_provider']['mock_failing_model'],
  }),

  'openai': MappingProxyType({
    'gpt-4': ALL_MODELS['openai']['gpt-4'],
    'gpt-4-turbo-preview': ALL_MODELS['openai']['gpt-4-turbo-preview'],
    'gpt-3.5-turbo': ALL_MODELS['openai']['gpt-3.5-turbo'],
  }),

  'claude': MappingProxyType({
    'claude-3-opus': ALL_MODELS['claude']['claude-3-opus'],
    'claude-3-sonnet': ALL_MODELS['claude']['claude-3-sonnet'],
    'claude-3-haiku': ALL_MODELS['claude']['claude-3-haiku'],
  }),

  'gemini': MappingProxyType({
    'gemini-1.0-pro': ALL_MODELS['gemini']['gemini-1.0-pro'],
    'gemini-1.0-pro-001': ALL_MODELS['gemini']['gemini-1.0-pro-001'],
    'gemini-1.0-pro-latest': ALL_MODELS['gemini']['gemini-1.0-pro-latest'],
    'gemini-1.5-pro-latest': ALL_MODELS['gemini']['gemini-1.5-pro-latest'],
    'gemini-pro': ALL_MODELS['gemini']['gemini-pro'],
  }),

  'cohere': MappingProxyType({
    'command-light': ALL_MODELS['cohere']['command-light'],
    'command-light-nightly': ALL_MODELS['cohere']['command-light-nightly'],
    'command': ALL_MODELS['cohere']['command'],
    'command-nightly': ALL_MODELS['cohere']['command-nightly'],
    'command-r': ALL_MODELS['cohere']['command-r'],
    'command-r-plus': ALL_MODELS['cohere']['command-r-plus'],
  }),

  'databricks': MappingProxyType({
    'dbrx-instruct': ALL_MODELS['databricks']['dbrx-instruct'],
    'mixtral-8x7b-instruct': ALL_MODELS['databricks']['mixtral-8x7b-instruct'],
    'llama-2-70b-chat': ALL_MODELS['databricks']['llama-2-70b-chat'],
    'llama-3-70b-instruct': ALL_MODELS['databricks']['llama-3-70b-instruct'],
    'bge-large-en': ALL_MODELS['databricks']['bge-large-en'],
    'mpt-30b-instruct': ALL_MODELS['databricks']['mpt-30b-instruct'],
    'mpt-7b-instruct': ALL_MODELS['databricks']['mpt-7b-instruct'],
  }),

  'mistral': MappingProxyType({
    'open-mistral-7b': ALL_MODELS['mistral']['open-mistral-7b'],
    'open-mixtral-8x7b': ALL_MODELS['mistral']['open-mixtral-8x7b'],
    'mistral-small-latest': ALL_MODELS['mistral']['mistral-small-latest'],
    'mistral-medium-latest': ALL_MODELS['mistral']['mistral-medium-latest'],
    'mistral-large-latest': ALL_MODELS['mistral']['mistral-large-latest'],
  }),

  'hugging_face': MappingProxyType({
    'google-gemma-7b-it': ALL_MODELS['hugging_face']['google-gemma-7b-it'],
    'mistral-mixtral-8x7b-instruct': ALL_MODELS[
        'hugging_face']['mistral-mixtral-8x7b-instruct'],
    'mistral-mistral-7b-instruct': ALL_MODELS[
        'hugging_face']['mistral-mistral-7b-instruct'],
    'nous-hermes-2-mixtral-8x7b': ALL_MODELS[
        'hugging_face']['nous-hermes-2-mixtral-8x7b'],
    'openchat-3.5': ALL_MODELS['hugging_face']['openchat-3.5'],
  }),
})


LARGEST_GENERATE_TEXT_MODELS: Dict[
    str, types.ProviderModelType] = MappingProxyType({
  'openai': MappingProxyType({
    'gpt-4-turbo-preview': ALL_MODELS['openai']['gpt-4-turbo-preview'],
  }),
  'claude': MappingProxyType({
    'claude-3-opus': ALL_MODELS['claude']['claude-3-opus'],
  }),
  'gemini': MappingProxyType({
    'gemini-1.5-pro-latest': ALL_MODELS['gemini']['gemini-1.5-pro-latest'],
  }),
  'cohere': MappingProxyType({
    'command-r-plus': ALL_MODELS['cohere']['command-r-plus'],
  }),
  'databricks': MappingProxyType({
    'llama-3-70b-instruct': ALL_MODELS['databricks']['llama-3-70b-instruct'],
    'dbrx-instruct': ALL_MODELS['databricks']['dbrx-instruct'],
  }),
  'mistral': MappingProxyType({
    'mistral-large-latest': ALL_MODELS['mistral']['mistral-large-latest'],
  }),
})

def get_provider_model_config(
    model_identifier: types.ProviderModelIdentifierType
) -> types.ProviderModelType:
  if type_utils.is_provider_model_tuple(model_identifier):
    return ALL_MODELS[model_identifier[0]][model_identifier[1]]
  else:
    return model_identifier
