import copy
import pydantic
import inspect

import proxai.types as types
import proxai.chat.chat_session as chat_session

def messages_param_to_chat(
    messages: types.MessagesParam | None) -> types.Chat | None:
  if messages is None:
    return None

  if type(messages) == list:
    return chat_session.Chat(messages=messages)
  elif type(messages) == dict:
    return chat_session.Chat(
        system_prompt=messages.get('system', None),
        messages=messages.get('messages', []))
  elif type(messages) != chat_session.Chat:
    raise ValueError(f'Invalid messages type: {type(messages)}')

  return messages


def output_format_param_to_output_format(
    output_format: types.OutputFormatParam | None
) -> types.OutputFormat:
  if not output_format:
    return types.OutputFormat(
        type=types.OutputFormatType.TEXT)

  if isinstance(output_format, types.OutputFormat):
    return output_format

  if type(output_format) == str:
    if output_format == 'text':
      return types.OutputFormat(
          type=types.OutputFormatType.TEXT)
    elif output_format == 'json':
      return types.OutputFormat(
          type=types.OutputFormatType.JSON)
    elif output_format == 'image':
      return types.OutputFormat(
          type=types.OutputFormatType.IMAGE)
    elif output_format == 'audio':
      return types.OutputFormat(
          type=types.OutputFormatType.AUDIO)
    elif output_format == 'video':
      return types.OutputFormat(
          type=types.OutputFormatType.VIDEO)
    else:
      raise ValueError(f'Invalid output format: {output_format}')
  elif (inspect.isclass(output_format) and
        issubclass(output_format, pydantic.BaseModel)):
    return types.OutputFormat(
        type=types.OutputFormatType.PYDANTIC,
        pydantic_class=output_format)

  raise ValueError(f'Invalid output format: {output_format}')


def check_messages_type(messages: types.MessagesType):
  """Check if messages type is supported."""
  for message in messages:
    if not isinstance(message, dict):
      raise ValueError(
          f'Each message in messages should be a dictionary. '
          f'Invalid message: {message}'
      )
    if set(message.keys()) != {'role', 'content'}:
      raise ValueError(
          f'Each message should have keys "role" and "content". '
          f'Invalid message: {message}'
      )
    if not isinstance(message['role'], str):
      raise ValueError(
          f'Role should be a string. Invalid role: {message["role"]}'
      )
    if not isinstance(message['content'], str):
      raise ValueError(
          f'Content should be a string. Invalid content: {message["content"]}'
      )
    if message['role'] not in ['user', 'assistant']:
      raise ValueError(
          'Role should be "user" or "assistant".\n'
          f'Invalid role: {message["role"]}'
      )


def check_model_size_identifier_type(
    model_size_identifier: types.ModelSizeIdentifierType
) -> types.ModelSizeType:
  """Check if model size identifier is supported."""
  if isinstance(model_size_identifier, types.ModelSizeType):
    return model_size_identifier
  elif isinstance(model_size_identifier, str):
    valid_values = [size.value for size in types.ModelSizeType]
    if model_size_identifier not in valid_values:
      raise ValueError(
          'Model size should be proxai.types.ModelSizeType or one of the '
          'following strings: small, medium, large, largest\n'
          f'Invalid model size identifier: {model_size_identifier}'
      )
    return types.ModelSizeType(model_size_identifier)
  raise ValueError(
      'Model size should be proxai.types.ModelSizeType or one of the '
      'following strings: small, medium, large, largest\n'
      f'Invalid model size identifier: {model_size_identifier}\n'
      f'Type: {type(model_size_identifier)}'
  )


def check_output_format_type_param(
    output_format_type: types.OutputFormatTypeParam
) -> types.OutputFormatType:
  """Check if output format type param is supported."""
  if isinstance(output_format_type, types.OutputFormatType):
    return output_format_type
  elif isinstance(output_format_type, str):
    normalized = output_format_type.upper()
    valid_values = [rt.value for rt in types.OutputFormatType]
    if normalized not in valid_values:
      valid_strings = ', '.join(
          rt.value.lower() for rt in types.OutputFormatType)
      raise ValueError(
          'Output format type should be proxai.types.OutputFormatType '
          f'or one of the following strings: {valid_strings}\n'
          f'Invalid output format type: {output_format_type}'
      )
    return types.OutputFormatType(normalized)
  raise ValueError(
      'Output format type should be proxai.types.OutputFormatType '
      'or one of the following strings: text, image, audio, video, '
      'json, pydantic, multi_modal\n'
      f'Invalid output format type: {output_format_type}\n'
      f'Type: {type(output_format_type)}'
  )


def check_input_format_type_param(
    input_format_type: types.InputFormatTypeParam
) -> types.InputFormatType:
  """Check if input format type param is supported."""
  if isinstance(input_format_type, types.InputFormatType):
    return input_format_type
  elif isinstance(input_format_type, str):
    normalized = input_format_type.upper()
    valid_values = [it.value for it in types.InputFormatType]
    if normalized not in valid_values:
      valid_strings = ', '.join(
          it.value.lower() for it in types.InputFormatType)
      raise ValueError(
          'Input format type should be proxai.types.InputFormatType '
          f'or one of the following strings: {valid_strings}\n'
          f'Invalid input format type: {input_format_type}'
      )
    return types.InputFormatType(normalized)
  raise ValueError(
      'Input format type should be proxai.types.InputFormatType '
      'or one of the following strings: text, image, document, '
      'audio, video, json, pydantic\n'
      f'Invalid input format type: {input_format_type}\n'
      f'Type: {type(input_format_type)}'
  )


def _normalize_chat_for_comparison(
    query_record: types.QueryRecord
) -> types.QueryRecord:
  """Strip file API metadata from chat MessageContent for comparison.

  Creates a shallow copy of the query record with deep-copied chat
  where provider_file_api_ids, provider_file_api_status, and filename
  are cleared from all MessageContent blocks. This ensures the
  equality check mirrors the hash.

  WARNING: Any field excluded here must also be excluded in
  hash_serializer._content_hash_dict(). These two functions define
  cache identity together — the hash finds the bucket, the equality
  check verifies the match. If they diverge, cache lookups will
  silently fail (hash matches but equality doesn't, or vice versa).

  See: hash_serializer._content_hash_dict()
  See: hash_serializer module docstring for full list of excluded fields
  """
  if query_record.chat is None:
    return query_record
  query_record = copy.copy(query_record)
  query_record.chat = query_record.chat.copy()
  for msg in query_record.chat.messages:
    if isinstance(msg.content, str):
      continue
    for mc in msg.content:
      mc.provider_file_api_ids = None
      mc.provider_file_api_status = None
      mc.proxdash_file_id = None
      mc.proxdash_file_status = None
      mc.filename = None
  return query_record


def is_query_record_equal(
    query_record_1: types.QueryRecord, query_record_2: types.QueryRecord
) -> bool:
  """Compare two query records for cache identity.

  Used by the cache pipeline after hash lookup to verify against
  hash collisions. Normalizes fields that are excluded from cache
  identity (Pydantic class values, connection_options flags, file
  API metadata) before comparing.

  WARNING: The normalization here must stay in sync with
  hash_serializer.get_query_record_hash(). See
  _normalize_chat_for_comparison() and hash_serializer module
  docstring for details.
  """
  def _normalize_output_format(qr: types.QueryRecord) -> types.QueryRecord:
    """Strip live pydantic_class, keep only name + json schema.

    Mirrors hash_serializer._hash_output_format: the live class is
    transport metadata — identity comes from class_name and
    class_json_schema which survive serialization.
    """
    if qr.output_format is None:
      return qr
    if qr.output_format.pydantic_class is None:
      return qr
    qr = copy.copy(qr)
    qr.output_format = types.OutputFormat(
        type=qr.output_format.type,
        pydantic_class=None,
        pydantic_class_name=qr.output_format.pydantic_class.__name__,
        pydantic_class_json_schema=(
            qr.output_format.pydantic_class.model_json_schema()
        ),
    )
    return qr

  query_record_1 = _normalize_output_format(query_record_1)
  query_record_2 = _normalize_output_format(query_record_2)

  # Normalize connection_options so equality mirrors the hash: only
  # `endpoint` is part of the query identity (see
  # hash_serializer._hash_connection_options). Without this, transient
  # per-call flags like `override_cache_value` on the stored record
  # would poison the equality check and break cache lookups after an
  # override write.
  if query_record_1.connection_options is not None:
    query_record_1 = copy.copy(query_record_1)
    query_record_1.connection_options = types.ConnectionOptions(
        endpoint=query_record_1.connection_options.endpoint
    )
  if query_record_2.connection_options is not None:
    query_record_2 = copy.copy(query_record_2)
    query_record_2.connection_options = types.ConnectionOptions(
        endpoint=query_record_2.connection_options.endpoint
    )

  # Normalize chat messages: strip file API metadata so equality
  # mirrors the hash (see hash_serializer._content_hash_dict).
  # Without this, different provider_file_api_ids on the same content
  # would break cache lookups.
  query_record_1 = _normalize_chat_for_comparison(query_record_1)
  query_record_2 = _normalize_chat_for_comparison(query_record_2)

  return query_record_1 == query_record_2


def _normalize_tag_param(param, tag_enum):
  """Normalize a tag param (single or list) to a list of enum values."""
  if param is None:
    return None
  if isinstance(param, (str, tag_enum)):
    param = [param]
  result = []
  for item in param:
    if isinstance(item, tag_enum):
      result.append(item)
    elif isinstance(item, str):
      try:
        result.append(tag_enum(item))
      except ValueError:
        result.append(tag_enum(item.upper()))
    else:
      raise ValueError(f'Invalid tag: {item}')
  return result


def create_input_format_type_list(
    tags: types.InputFormatTypeParam
) -> list[types.InputFormatType] | None:
  """Convert InputFormatTypeParam to list[InputFormatType]."""
  return _normalize_tag_param(tags, types.InputFormatType)


def create_output_format_type_list(
    tags: types.OutputFormatTypeParam
) -> list[types.OutputFormatType] | None:
  """Convert OutputFormatTypeParam to list[OutputFormatType]."""
  return _normalize_tag_param(tags, types.OutputFormatType)


def create_tool_tag_list(
    tags: types.ToolTagParam
) -> list[types.ToolTag] | None:
  """Convert ToolTagParam to list[ToolTag]."""
  return _normalize_tag_param(tags, types.ToolTag)


def create_feature_tag_list(
    features: types.FeatureTagParam
) -> list[types.FeatureTag] | None:
  """Convert feature tag param to list[FeatureTag]."""
  return _normalize_tag_param(features, types.FeatureTag)
