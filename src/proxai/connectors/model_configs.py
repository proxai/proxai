from types import MappingProxyType
from typing import Dict, Tuple
import proxai.types as types
import proxai.type_utils as type_utils

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

  # Mock slow provider
  'mock_slow_provider': MappingProxyType({
    'mock_slow_model': types.ProviderModelType(
      provider='mock_slow_provider',
      model='mock_slow_model',
      provider_model_identifier='mock_slow_model'
    ),
  }),

  # OpenAI models.
  # Models provided by OpenAI:
  # https://platform.openai.com/docs/guides/text-generation
  'openai': MappingProxyType({
    'gpt-4.1': types.ProviderModelType(
      provider='openai',
      model='gpt-4.1',
      provider_model_identifier='gpt-4.1-2025-04-14'
    ),
    'gpt-4.1-mini': types.ProviderModelType(
      provider='openai',
      model='gpt-4.1-mini',
      provider_model_identifier='gpt-4.1-mini-2025-04-14'
    ),
    'gpt-4.1-nano': types.ProviderModelType(
      provider='openai',
      model='gpt-4.1-nano',
      provider_model_identifier='gpt-4.1-nano-2025-04-14'
    ),
    'gpt-4.5-preview': types.ProviderModelType(
      provider='openai',
      model='gpt-4.5-preview',
      provider_model_identifier='gpt-4.5-preview-2025-02-27'
    ),
    'gpt-4o': types.ProviderModelType(
      provider='openai',
      model='gpt-4o',
      provider_model_identifier='gpt-4o-2024-08-06'
    ),
    'gpt-4o-audio-preview': types.ProviderModelType(
      provider='openai',
      model='gpt-4o-audio-preview',
      provider_model_identifier='gpt-4o-audio-preview-2024-12-17'
    ),
    'gpt-4o-realtime-preview': types.ProviderModelType(
      provider='openai',
      model='gpt-4o-realtime-preview',
      provider_model_identifier='gpt-4o-realtime-preview-2024-12-17'
    ),
    'gpt-4o-mini': types.ProviderModelType(
      provider='openai',
      model='gpt-4o-mini',
      provider_model_identifier='gpt-4o-mini-2024-07-18'
    ),
    'gpt-4o-mini-audio-preview': types.ProviderModelType(
      provider='openai',
      model='gpt-4o-mini-audio-preview',
      provider_model_identifier='gpt-4o-mini-audio-preview-2024-12-17'
    ),
    'gpt-4o-mini-realtime-preview': types.ProviderModelType(
      provider='openai',
      model='gpt-4o-mini-realtime-preview',
      provider_model_identifier='gpt-4o-mini-realtime-preview-2024-12-17'
    ),
    'o1': types.ProviderModelType(
      provider='openai',
      model='o1',
      provider_model_identifier='o1-2024-12-17'
    ),
    'o1-pro': types.ProviderModelType(
      provider='openai',
      model='o1-pro',
      provider_model_identifier='o1-pro-2025-03-19'
    ),
    'o3': types.ProviderModelType(
      provider='openai',
      model='o3',
      provider_model_identifier='o3-2025-04-16'
    ),
    'o4-mini': types.ProviderModelType(
      provider='openai',
      model='o4-mini',
      provider_model_identifier='o4-mini-2025-04-16'
    ),
    'o3-mini': types.ProviderModelType(
      provider='openai',
      model='o3-mini',
      provider_model_identifier='o3-mini-2025-01-31'
    ),
    'o1-mini': types.ProviderModelType(
      provider='openai',
      model='o1-mini',
      provider_model_identifier='o1-mini-2024-09-12'
    ),
    'gpt-4o-mini-search-preview': types.ProviderModelType(
      provider='openai',
      model='gpt-4o-mini-search-preview',
      provider_model_identifier='gpt-4o-mini-search-preview-2025-03-11'
    ),
    'gpt-4o-search-preview': types.ProviderModelType(
      provider='openai',
      model='gpt-4o-search-preview',
      provider_model_identifier='gpt-4o-search-preview-2025-03-11'
    ),
    'computer-use-preview': types.ProviderModelType(
      provider='openai',
      model='computer-use-preview',
      provider_model_identifier='computer-use-preview-2025-03-11'
    ),
    'chatgpt-4o-latest': types.ProviderModelType(
      provider='openai',
      model='chatgpt-4o-latest',
      provider_model_identifier='chatgpt-4o-latest'
    ),
    'gpt-4-turbo': types.ProviderModelType(
      provider='openai',
      model='gpt-4-turbo',
      provider_model_identifier='gpt-4-turbo-2024-04-09'
    ),
    'gpt-4': types.ProviderModelType(
      provider='openai',
      model='gpt-4',
      provider_model_identifier='gpt-4-0613'
    ),
    'gpt-4-32k': types.ProviderModelType(
      provider='openai',
      model='gpt-4-32k',
      provider_model_identifier='gpt-4-32k'
    ),
    'gpt-3.5-turbo': types.ProviderModelType(
      provider='openai',
      model='gpt-3.5-turbo',
      provider_model_identifier='gpt-3.5-turbo-0125'
    ),
  }),

  # Claude models.
  # Models provided by Claude:
  # https://claude.ai/docs/models
  'claude': MappingProxyType({
    'sonnet':  types.ProviderModelType(
      provider='claude',
      model='sonnet',
      provider_model_identifier='claude-3-7-sonnet-20250219'
    ),
    'haiku':  types.ProviderModelType(
      provider='claude',
      model='haiku',
      provider_model_identifier='claude-3-5-haiku-20241022'
    ),
    '3.5-sonnet-v2': types.ProviderModelType(
      provider='claude',
      model='3.5-sonnet-v2',
      provider_model_identifier='claude-3-5-sonnet-20241022'
    ),
    '3.5-sonnet': types.ProviderModelType(
      provider='claude',
      model='3.5-sonnet',
      provider_model_identifier='claude-3-5-sonnet-20240620'
    ),
    'opus':  types.ProviderModelType(
      provider='claude',
      model='opus',
      provider_model_identifier='claude-3-opus-20240229'
    ),
    '3-sonnet': types.ProviderModelType(
      provider='claude',
      model='3-sonnet',
      provider_model_identifier='claude-3-sonnet-20240229'
    ),
    '3-haiku': types.ProviderModelType(
      provider='claude',
      model='3-haiku',
      provider_model_identifier='claude-3-haiku-20240307'
    ),
  }),

  # Gemini models.
  # Models provided by Gemini:
  # https://ai.google.dev/models/gemini
  'gemini': MappingProxyType({
    'gemini-1.5-pro': types.ProviderModelType(
      provider='gemini',
      model='gemini-1.5-pro',
      provider_model_identifier='gemini-1.5-pro'
    ),
    'gemini-1.5-flash-8b': types.ProviderModelType(
      provider='gemini',
      model='gemini-1.5-flash-8b',
      provider_model_identifier='gemini-1.5-flash-8b'
    ),
    'gemini-1.5-flash': types.ProviderModelType(
      provider='gemini',
      model='gemini-1.5-flash',
      provider_model_identifier='gemini-1.5-flash'
    ),
    'gemini-2.0-flash-lite': types.ProviderModelType(
      provider='gemini',
      model='gemini-2.0-flash-lite',
      provider_model_identifier='gemini-2.0-flash-lite'
    ),
    'gemini-2.0-flash': types.ProviderModelType(
      provider='gemini',
      model='gemini-2.0-flash',
      provider_model_identifier='gemini-2.0-flash'
    ),
    'gemini-2.5-pro-preview-03-25': types.ProviderModelType(
      provider='gemini',
      model='gemini-2.5-pro-preview-03-25',
      provider_model_identifier='gemini-2.5-pro-preview-03-25'
    ),
  }),

  # Cohere models.
  # Models provided by Cohere:
  # https://docs.cohere.com/docs/models
  'cohere': MappingProxyType({
    'command-a': types.ProviderModelType(
      provider='cohere',
      model='command-a',
      provider_model_identifier='command-a-03-2025'
    ),
    'command-r7b': types.ProviderModelType(
      provider='cohere',
      model='command-r7b',
      provider_model_identifier='command-r7b-12-2024'
    ),
    'command-r-plus': types.ProviderModelType(
      provider='cohere',
      model='command-r-plus',
      provider_model_identifier='command-r-plus'
    ),
    'command-r-08-2024': types.ProviderModelType(
      provider='cohere',
      model='command-r-08-2024',
      provider_model_identifier='command-r-08-2024'
    ),
    'command-r': types.ProviderModelType(
      provider='cohere',
      model='command-r',
      provider_model_identifier='command-r'
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
  }),

  # Databricks models.
  # Models provided by Databricks:
  # https://docs.databricks.com/en/machine-learning/foundation-models/index.html#provisioned-throughput-foundation-model-apis
  'databricks': MappingProxyType({
    'llama-4-maverick': types.ProviderModelType(
      provider='databricks',
      model='llama-4-maverick',
      provider_model_identifier='databricks-llama-4-maverick'
    ),
    'claude-3-7-sonnet': types.ProviderModelType(
      provider='databricks',
      model='claude-3-7-sonnet',
      provider_model_identifier='databricks-claude-3-7-sonnet'
    ),
    'meta-llama-3-1-8b-it': types.ProviderModelType(
      provider='databricks',
      model='meta-llama-3-1-8b-it',
      provider_model_identifier='databricks-meta-llama-3-1-8b-instruct'
    ),
    'meta-llama-3-3-70b-it': types.ProviderModelType(
      provider='databricks',
      model='meta-llama-3-3-70b-it',
      provider_model_identifier='databricks-meta-llama-3-3-70b-instruct'
    ),
    # TODO: This is extremely slow model. Until better filtering, it is not
    # included in the list.
    # 'meta-llama-3-1-405b-it': types.ProviderModelType(
    #   provider='databricks',
    #   model='meta-llama-3-1-405b-it',
    #   provider_model_identifier='databricks-meta-llama-3-1-405b-instruct'
    # ),
    'dbrx-it': types.ProviderModelType(
      provider='databricks',
      model='dbrx-it',
      provider_model_identifier='databricks-dbrx-instruct'
    ),
    'mixtral-8x7b-it': types.ProviderModelType(
      provider='databricks',
      model='mixtral-8x7b-it',
      provider_model_identifier='databricks-mixtral-8x7b-instruct'
    ),
  }),

  # Mistral models.
  # Models provided by Mistral:
  # https://docs.mistral.ai/platform/endpoints/
  'mistral': MappingProxyType({
    'ministral-3b': types.ProviderModelType(
      provider='mistral',
      model='ministral-3b',
      provider_model_identifier='ministral-3b-2410'
    ),
    'ministral-8b': types.ProviderModelType(
      provider='mistral',
      model='ministral-8b',
      provider_model_identifier='ministral-8b-2410'
    ),
    'open-mistral-7b': types.ProviderModelType(
      provider='mistral',
      model='open-mistral-7b',
      provider_model_identifier='open-mistral-7b'
    ),
    'open-mistral-nemo': types.ProviderModelType(
      provider='mistral',
      model='open-mistral-nemo',
      provider_model_identifier='open-mistral-nemo'
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
    'mistral-small': types.ProviderModelType(
      provider='mistral',
      model='mistral-small',
      provider_model_identifier='mistral-small-2503'
    ),
    'mistral-large': types.ProviderModelType(
      provider='mistral',
      model='mistral-large',
      provider_model_identifier='mistral-large-2411'
    ),
    'pixtral-large': types.ProviderModelType(
      provider='mistral',
      model='pixtral-large',
      provider_model_identifier='pixtral-large-2411'
    ),
    'codestral': types.ProviderModelType(
      provider='mistral',
      model='codestral',
      provider_model_identifier='codestral-2501'
    ),
    'pixtral-12b': types.ProviderModelType(
      provider='mistral',
      model='pixtral-12b',
      provider_model_identifier='pixtral-12b-2409'
    ),
    'mistral-saba': types.ProviderModelType(
      provider='mistral',
      model='mistral-saba',
      provider_model_identifier='mistral-saba-2502'
    ),
  }),

  # Hugging Face models suggested in:
  # https://huggingface.co/docs/inference-providers/en/tasks/chat-completion
  'huggingface': MappingProxyType({
    'gemma-2-2b-it': types.ProviderModelType(
      provider='huggingface',
      model='gemma-2-2b-it',
      provider_model_identifier='google/gemma-2-2b-it'
    ),
    'meta-llama-3.1-8b-it': types.ProviderModelType(
      provider='huggingface',
      model='meta-llama-3.1-8b-it',
      provider_model_identifier='meta-llama/Meta-Llama-3.1-8B-Instruct'
    ),
    'phi-4': types.ProviderModelType(
      provider='huggingface',
      model='phi-4',
      provider_model_identifier='microsoft/phi-4'
    ),
    'qwen3-32b': types.ProviderModelType(
      provider='huggingface',
      model='qwen3-32b',
      provider_model_identifier='Qwen/Qwen3-32B'
    ),
    'deepseek-r1': types.ProviderModelType(
      provider='huggingface',
      model='deepseek-r1',
      provider_model_identifier='deepseek-ai/DeepSeek-R1'
    ),
    'deepseek-v3': types.ProviderModelType(
      provider='huggingface',
      model='deepseek-v3',
      provider_model_identifier='deepseek-ai/DeepSeek-V3'
    ),
  }),

  'deepseek': MappingProxyType({
    'deepseek-v3': types.ProviderModelType(
      provider='deepseek',
      model='deepseek-v3',
      provider_model_identifier='deepseek-chat'
    ),
    'deepseek-r1': types.ProviderModelType(
      provider='deepseek',
      model='deepseek-r1',
      provider_model_identifier='deepseek-reasoner'
    ),
  }),

  'grok': MappingProxyType({
    'grok-3-beta': types.ProviderModelType(
      provider='grok',
      model='grok-3-beta',
      provider_model_identifier='grok-3-beta'
    ),
    'grok-3-fast-beta': types.ProviderModelType(
      provider='grok',
      model='grok-3-fast-beta',
      provider_model_identifier='grok-3-fast-beta'
    ),
    'grok-3-mini-beta': types.ProviderModelType(
      provider='grok',
      model='grok-3-mini-beta',
      provider_model_identifier='grok-3-mini-beta'
    ),
    'grok-3-mini-fast-beta': types.ProviderModelType(
      provider='grok',
      model='grok-3-mini-fast-beta',
      provider_model_identifier='grok-3-mini-fast-beta'
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

  'mock_slow_provider': MappingProxyType({
    'mock_slow_model': ALL_MODELS['mock_slow_provider']['mock_slow_model'],
  }),

  'openai': MappingProxyType({
    'gpt-4.1': ALL_MODELS['openai']['gpt-4.1'],
    'gpt-4.1-mini': ALL_MODELS['openai']['gpt-4.1-mini'],
    'gpt-4.1-nano': ALL_MODELS['openai']['gpt-4.1-nano'],
    'gpt-4.5-preview': ALL_MODELS['openai']['gpt-4.5-preview'],
    'gpt-4o': ALL_MODELS['openai']['gpt-4o'],
    'gpt-4o-mini': ALL_MODELS['openai']['gpt-4o-mini'],
    'o1': ALL_MODELS['openai']['o1'],
    'o1-pro': ALL_MODELS['openai']['o1-pro'],
    'o3': ALL_MODELS['openai']['o3'],
    'o4-mini': ALL_MODELS['openai']['o4-mini'],
    'o3-mini': ALL_MODELS['openai']['o3-mini'],
    'o1-mini': ALL_MODELS['openai']['o1-mini'],
    'gpt-4o-mini-search-preview': ALL_MODELS['openai']['gpt-4o-mini-search-preview'],
    'gpt-4o-search-preview': ALL_MODELS['openai']['gpt-4o-search-preview'],
    'chatgpt-4o-latest': ALL_MODELS['openai']['chatgpt-4o-latest'],
    'gpt-4-turbo': ALL_MODELS['openai']['gpt-4-turbo'],
    'gpt-4': ALL_MODELS['openai']['gpt-4'],
    'gpt-3.5-turbo': ALL_MODELS['openai']['gpt-3.5-turbo'],
  }),

  'claude': MappingProxyType({
    'sonnet': ALL_MODELS['claude']['sonnet'],
    'haiku': ALL_MODELS['claude']['haiku'],
    '3.5-sonnet-v2': ALL_MODELS['claude']['3.5-sonnet-v2'],
    '3.5-sonnet': ALL_MODELS['claude']['3.5-sonnet'],
    'opus': ALL_MODELS['claude']['opus'],
    '3-sonnet': ALL_MODELS['claude']['3-sonnet'],
    '3-haiku': ALL_MODELS['claude']['3-haiku'],
  }),

  'gemini': MappingProxyType({
    'gemini-1.5-pro': ALL_MODELS['gemini']['gemini-1.5-pro'],
    'gemini-1.5-flash-8b': ALL_MODELS['gemini']['gemini-1.5-flash-8b'],
    'gemini-1.5-flash': ALL_MODELS['gemini']['gemini-1.5-flash'],
    'gemini-2.0-flash-lite': ALL_MODELS['gemini']['gemini-2.0-flash-lite'],
    'gemini-2.0-flash': ALL_MODELS['gemini']['gemini-2.0-flash'],
    'gemini-2.5-pro-preview-03-25': ALL_MODELS['gemini']['gemini-2.5-pro-preview-03-25'],
  }),

  'cohere': MappingProxyType({
    'command-a': ALL_MODELS['cohere']['command-a'],
    'command-r7b': ALL_MODELS['cohere']['command-r7b'],
    'command-r-plus': ALL_MODELS['cohere']['command-r-plus'],
    'command-r-08-2024': ALL_MODELS['cohere']['command-r-08-2024'],
    'command-r': ALL_MODELS['cohere']['command-r'],
    'command': ALL_MODELS['cohere']['command'],
    'command-nightly': ALL_MODELS['cohere']['command-nightly'],
    'command-light': ALL_MODELS['cohere']['command-light'],
    'command-light-nightly': ALL_MODELS['cohere']['command-light-nightly'],
  }),

  'databricks': MappingProxyType({
    'llama-4-maverick': ALL_MODELS['databricks']['llama-4-maverick'],
    'claude-3-7-sonnet': ALL_MODELS['databricks']['claude-3-7-sonnet'],
    'meta-llama-3-1-8b-it': ALL_MODELS['databricks']['meta-llama-3-1-8b-it'],
    'meta-llama-3-3-70b-it': ALL_MODELS['databricks']['meta-llama-3-3-70b-it'],
    # TODO: This is extremely slow model. Until better filtering, it is not
    # included in the list.
    # 'meta-llama-3-1-405b-it': ALL_MODELS['databricks']['meta-llama-3-1-405b-it'],
    'dbrx-it': ALL_MODELS['databricks']['dbrx-it'],
    'mixtral-8x7b-it': ALL_MODELS['databricks']['mixtral-8x7b-it'],
  }),

  'mistral': MappingProxyType({
    'ministral-3b': ALL_MODELS['mistral']['ministral-3b'],
    'ministral-8b': ALL_MODELS['mistral']['ministral-8b'],
    'open-mistral-7b': ALL_MODELS['mistral']['open-mistral-7b'],
    'open-mistral-nemo': ALL_MODELS['mistral']['open-mistral-nemo'],
    'open-mixtral-8x7b': ALL_MODELS['mistral']['open-mixtral-8x7b'],
    'open-mixtral-8x22b': ALL_MODELS['mistral']['open-mixtral-8x22b'],
    'mistral-small': ALL_MODELS['mistral']['mistral-small'],
    'mistral-large': ALL_MODELS['mistral']['mistral-large'],
    'pixtral-large': ALL_MODELS['mistral']['pixtral-large'],
    'codestral': ALL_MODELS['mistral']['codestral'],
    'pixtral-12b': ALL_MODELS['mistral']['pixtral-12b'],
    'mistral-saba': ALL_MODELS['mistral']['mistral-saba'],
  }),

  'huggingface': MappingProxyType({
    'gemma-2-2b-it': ALL_MODELS['huggingface']['gemma-2-2b-it'],
    'meta-llama-3.1-8b-it': ALL_MODELS['huggingface']['meta-llama-3.1-8b-it'],
    'phi-4': ALL_MODELS['huggingface']['phi-4'],
    'qwen3-32b': ALL_MODELS['huggingface']['qwen3-32b'],
    'deepseek-r1': ALL_MODELS['huggingface']['deepseek-r1'],
    'deepseek-v3': ALL_MODELS['huggingface']['deepseek-v3'],
  }),

  'deepseek': MappingProxyType({
    'deepseek-v3': ALL_MODELS['deepseek']['deepseek-v3'],
    'deepseek-r1': ALL_MODELS['deepseek']['deepseek-r1'],
  }),

  'grok': MappingProxyType({
    'grok-3-beta': ALL_MODELS['grok']['grok-3-beta'],
    'grok-3-fast-beta': ALL_MODELS['grok']['grok-3-fast-beta'],
    'grok-3-mini-beta': ALL_MODELS['grok']['grok-3-mini-beta'],
    'grok-3-mini-fast-beta': ALL_MODELS['grok']['grok-3-mini-fast-beta'],
  }),
})

SMALL_GENERATE_TEXT_MODELS: Dict[
    str, Dict[str, types.ProviderModelType]] = MappingProxyType({
  'claude': MappingProxyType({
    '3-haiku': ALL_MODELS['claude']['3-haiku'],
  }),
  'cohere': MappingProxyType({
    'command-light': ALL_MODELS['cohere']['command-light'],
    'command-light-nightly': ALL_MODELS['cohere']['command-light-nightly'],
    'command-r': ALL_MODELS['cohere']['command-r'],
    'command-r-08-2024': ALL_MODELS['cohere']['command-r-08-2024'],
    'command-r7b': ALL_MODELS['cohere']['command-r7b'],
  }),
  'deepseek': MappingProxyType({
    'deepseek-v3': ALL_MODELS['deepseek']['deepseek-v3'],
  }),
  'gemini': MappingProxyType({
    'gemini-1.5-flash': ALL_MODELS['gemini']['gemini-1.5-flash'],
    'gemini-1.5-flash-8b': ALL_MODELS['gemini']['gemini-1.5-flash-8b'],
    'gemini-2.0-flash': ALL_MODELS['gemini']['gemini-2.0-flash'],
    'gemini-2.0-flash-lite': ALL_MODELS['gemini']['gemini-2.0-flash-lite'],
  }),
  'grok': MappingProxyType({
    'grok-3-mini-beta': ALL_MODELS['grok']['grok-3-mini-beta'],
  }),
  'mistral': MappingProxyType({
    'codestral': ALL_MODELS['mistral']['codestral'],
    'ministral-3b': ALL_MODELS['mistral']['ministral-3b'],
    'ministral-8b': ALL_MODELS['mistral']['ministral-8b'],
    'mistral-saba': ALL_MODELS['mistral']['mistral-saba'],
    'mistral-small': ALL_MODELS['mistral']['mistral-small'],
    'open-mistral-7b': ALL_MODELS['mistral']['open-mistral-7b'],
    'open-mistral-nemo': ALL_MODELS['mistral']['open-mistral-nemo'],
    'open-mixtral-8x7b': ALL_MODELS['mistral']['open-mixtral-8x7b'],
    'pixtral-12b': ALL_MODELS['mistral']['pixtral-12b'],
  }),
  'openai': MappingProxyType({
    'gpt-4.1-nano': ALL_MODELS['openai']['gpt-4.1-nano'],
    'gpt-4o-mini': ALL_MODELS['openai']['gpt-4o-mini'],
    'gpt-4o-mini-search-preview': ALL_MODELS['openai']['gpt-4o-mini-search-preview'],
  }),
})

MEDIUM_GENERATE_TEXT_MODELS: Dict[
    str, Dict[str, types.ProviderModelType]] = MappingProxyType({
  'claude': MappingProxyType({
    'haiku': ALL_MODELS['claude']['haiku'],
  }),
  'cohere': MappingProxyType({
    'command': ALL_MODELS['cohere']['command'],
    'command-nightly': ALL_MODELS['cohere']['command-nightly'],
  }),
  'deepseek': MappingProxyType({
    'deepseek-r1': ALL_MODELS['deepseek']['deepseek-r1'],
  }),
  'gemini': MappingProxyType({
    'gemini-1.5-pro': ALL_MODELS['gemini']['gemini-1.5-pro'],
  }),
  'grok': MappingProxyType({
    'grok-3-mini-fast-beta': ALL_MODELS['grok']['grok-3-mini-fast-beta'],
  }),
  'huggingface': MappingProxyType({
    'gemma-2-2b-it': ALL_MODELS['huggingface']['gemma-2-2b-it'],
    'meta-llama-3.1-8b-it': ALL_MODELS['huggingface']['meta-llama-3.1-8b-it'],
  }),
  'mistral': MappingProxyType({
    'mistral-large': ALL_MODELS['mistral']['mistral-large'],
    'open-mixtral-8x22b': ALL_MODELS['mistral']['open-mixtral-8x22b'],
    'pixtral-large': ALL_MODELS['mistral']['pixtral-large'],
  }),
  'openai': MappingProxyType({
    'gpt-3.5-turbo': ALL_MODELS['openai']['gpt-3.5-turbo'],
    'gpt-4.1-mini': ALL_MODELS['openai']['gpt-4.1-mini'],
    'o1-mini': ALL_MODELS['openai']['o1-mini'],
    'o3-mini': ALL_MODELS['openai']['o3-mini'],
    'o4-mini': ALL_MODELS['openai']['o4-mini'],
  }),
})

LARGE_GENERATE_TEXT_MODELS: Dict[
    str, Dict[str, types.ProviderModelType]] = MappingProxyType({
  'claude': MappingProxyType({
    '3-sonnet': ALL_MODELS['claude']['3-sonnet'],
    '3.5-sonnet': ALL_MODELS['claude']['3.5-sonnet'],
    '3.5-sonnet-v2': ALL_MODELS['claude']['3.5-sonnet-v2'],
    'opus': ALL_MODELS['claude']['opus'],
    'sonnet': ALL_MODELS['claude']['sonnet'],
  }),
  'cohere': MappingProxyType({
    'command-a': ALL_MODELS['cohere']['command-a'],
    'command-r-plus': ALL_MODELS['cohere']['command-r-plus'],
  }),
  'databricks': MappingProxyType({
    'claude-3-7-sonnet': ALL_MODELS['databricks']['claude-3-7-sonnet'],
    'dbrx-it': ALL_MODELS['databricks']['dbrx-it'],
    'llama-4-maverick': ALL_MODELS['databricks']['llama-4-maverick'],
    'meta-llama-3-1-8b-it': ALL_MODELS['databricks']['meta-llama-3-1-8b-it'],
    'meta-llama-3-3-70b-it': ALL_MODELS['databricks']['meta-llama-3-3-70b-it'],
    'mixtral-8x7b-it': ALL_MODELS['databricks']['mixtral-8x7b-it'],
  }),
  'gemini': MappingProxyType({
    'gemini-2.5-pro-preview-03-25': ALL_MODELS['gemini']['gemini-2.5-pro-preview-03-25'],
  }),
  'grok': MappingProxyType({
    'grok-3-beta': ALL_MODELS['grok']['grok-3-beta'],
    'grok-3-fast-beta': ALL_MODELS['grok']['grok-3-fast-beta'],
  }),
  'huggingface': MappingProxyType({
    'deepseek-r1': ALL_MODELS['huggingface']['deepseek-r1'],
    'deepseek-v3': ALL_MODELS['huggingface']['deepseek-v3'],
    'phi-4': ALL_MODELS['huggingface']['phi-4'],
    'qwen3-32b': ALL_MODELS['huggingface']['qwen3-32b'],
  }),
  'openai': MappingProxyType({
    'chatgpt-4o-latest': ALL_MODELS['openai']['chatgpt-4o-latest'],
    'gpt-4': ALL_MODELS['openai']['gpt-4'],
    'gpt-4-turbo': ALL_MODELS['openai']['gpt-4-turbo'],
    'gpt-4.1': ALL_MODELS['openai']['gpt-4.1'],
    'gpt-4.5-preview': ALL_MODELS['openai']['gpt-4.5-preview'],
    'gpt-4o': ALL_MODELS['openai']['gpt-4o'],
    'gpt-4o-search-preview': ALL_MODELS['openai']['gpt-4o-search-preview'],
    'o1': ALL_MODELS['openai']['o1'],
    'o1-pro': ALL_MODELS['openai']['o1-pro'],
    'o3': ALL_MODELS['openai']['o3'],
  }),
})

LARGEST_GENERATE_TEXT_MODELS: Dict[
    str, Dict[str, types.ProviderModelType]] = MappingProxyType({
  'claude': MappingProxyType({
    'sonnet': ALL_MODELS['claude']['sonnet'],
  }),
  'cohere': MappingProxyType({
    'command-a': ALL_MODELS['cohere']['command-a'],
  }),
  'databricks': MappingProxyType({
    'dbrx-it': ALL_MODELS['databricks']['dbrx-it'],
    'meta-llama-3-3-70b-it': ALL_MODELS['databricks']['meta-llama-3-3-70b-it'],
  }),
  'deepseek': MappingProxyType({
    'deepseek-r1': ALL_MODELS['deepseek']['deepseek-r1'],
  }),
  'gemini': MappingProxyType({
    'gemini-2.5-pro-preview-03-25': ALL_MODELS['gemini']['gemini-2.5-pro-preview-03-25'],
  }),
  'grok': MappingProxyType({
    'grok-3-beta': ALL_MODELS['grok']['grok-3-beta'],
  }),
  'huggingface': MappingProxyType({
    'phi-4': ALL_MODELS['huggingface']['phi-4'],
    'qwen3-32b': ALL_MODELS['huggingface']['qwen3-32b'],
  }),
  'mistral': MappingProxyType({
    'mistral-large': ALL_MODELS['mistral']['mistral-large'],
  }),
  'openai': MappingProxyType({
    'gpt-4.1': ALL_MODELS['openai']['gpt-4.1'],
  }),
})


def get_provider_model_config(
    model_identifier: types.ProviderModelIdentifierType
) -> types.ProviderModelType:
  if type_utils.is_provider_model_tuple(model_identifier):
    return ALL_MODELS[model_identifier[0]][model_identifier[1]]
  else:
    return model_identifier
