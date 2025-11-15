import proxai.types as types


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
