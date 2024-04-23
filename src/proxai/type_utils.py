import datetime
import proxai.types as types


def check_model_type(model: types.ModelType):
  """Check if model type is supported."""
  provider, provider_model = model

  providers = set(item.value for item in types.Provider)
  if provider not in providers:
    raise ValueError(
      f'Provider not supported: {provider}. Supported providers: {providers}')

  provider_models = set(
      item.value for item in types.PROVIDER_MODEL_MAP[provider])
  if provider_model not in provider_models:
    raise ValueError(
      f'Model {model} not supported for provider {provider}.\n'
      f'Supported models: {provider_models}')


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
