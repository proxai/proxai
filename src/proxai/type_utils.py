import proxai.types as types
import proxai.connectors.model_configs as model_configs
from typing import Any


def check_provider_model_identifier_type(
    provider_model_identifier: types.ProviderModelIdentifierType):
  """Check if provider model identifier is supported."""
  if isinstance(provider_model_identifier, types.ProviderModelType):
    provider = provider_model_identifier.provider
    model = provider_model_identifier.model
    if provider not in model_configs.ALL_MODELS:
      raise ValueError(
        f'Provider not supported: {provider}.\n'
        f'Supported providers: {model_configs.ALL_MODELS.keys()}')
    if model not in model_configs.ALL_MODELS[provider]:
      raise ValueError(
        f'Model not supported: {model}.\nSupported models: '
        f'{model_configs.ALL_MODELS[provider].keys()}')
    if provider_model_identifier != model_configs.ALL_MODELS[provider][model]:
      raise ValueError(
        'Mismatch between provider model identifier and model config.'
        f'Provider model identifier: {provider_model_identifier}'
        f'Model config: {model_configs.ALL_MODELS[provider][model]}')
  elif is_provider_model_tuple(provider_model_identifier):
    provider = provider_model_identifier[0]
    model = provider_model_identifier[1]
    if provider not in model_configs.ALL_MODELS:
      raise ValueError(
        f'Provider not supported: {provider}.\n'
        f'Supported providers: {model_configs.ALL_MODELS.keys()}')
    if model not in model_configs.ALL_MODELS[provider]:
      raise ValueError(
        f'Model not supported: {model}.\nSupported models: '
        f'{model_configs.ALL_MODELS[provider].keys()}')
  else:
    raise ValueError(
        f'Invalid provider model identifier: {provider_model_identifier}')


def check_messages_type(messages: types.MessagesType):
  """Check if messages type is supported."""
  for message in messages:
    if not isinstance(message, dict):
      raise ValueError(
          f'Each message in messages should be a dictionary. '
          f'Invalid message: {message}')
    if set(list(message.keys())) != {'role', 'content'}:
      raise ValueError(
          f'Each message should have keys "role" and "content". '
          f'Invalid message: {message}')
    if not isinstance(message['role'], str):
      raise ValueError(
          f'Role should be a string. Invalid role: {message["role"]}')
    if not isinstance(message['content'], str):
      raise ValueError(
          f'Content should be a string. Invalid content: {message["content"]}')
    if message['role'] not in ['user', 'assistant']:
      raise ValueError(
          'Role should be "user" or "assistant".\n'
          f'Invalid role: {message["role"]}')


def is_provider_model_tuple(value: Any) -> bool:
    return (type(value) == tuple
            and len(value) == 2
            and type(value[0]) == str
            and type(value[1]) == str)


def check_model_size_identifier_type(
    model_size_identifier: types.ModelSizeIdentifierType
) -> types.ModelSizeType:
  """Check if model size identifier is supported."""
  if isinstance(model_size_identifier, types.ModelSizeType):
    return model_size_identifier
  elif type(model_size_identifier) == str:
    valid_values = [size.value for size in types.ModelSizeType]
    if model_size_identifier not in valid_values:
      raise ValueError(
          'Model size should be proxai.types.ModelSizeType or one of the '
          'following strings: small, medium, large, largest\n'
          f'Invalid model size identifier: {model_size_identifier}')
    return types.ModelSizeType(model_size_identifier)
  raise ValueError(
        'Model size should be proxai.types.ModelSizeType or one of the '
        'following strings: small, medium, large, largest\n'
        f'Invalid model size identifier: {model_size_identifier}\n'
        f'Type: {type(model_size_identifier)}')
