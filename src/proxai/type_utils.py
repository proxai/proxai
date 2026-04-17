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


def _raise_invalid_output_format_value_error(
    output_format: types.OutputFormatParam
) -> None:
  raise ValueError(
      'Please provide one of the followings:\n'
      ' - "json" as string for JSON output format\n'
      ' - dict for JSON schema output format\n'
      ' - pydantic.BaseModel for Pydantic output format\n'
      ' - proxai.types.OutputFormat for custom more advanced output '
      'format\n'
      'Check https://www.proxai.co/proxai-docs/advanced/response-format '
      'for more information.\n'
      f'Output format type: {type(output_format)}\n'
      f'Output format value: {output_format}'
  )


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
      'audio, video\n'
      f'Invalid input format type: {input_format_type}\n'
      f'Type: {type(input_format_type)}'
  )


def create_output_format(
    output_format: types.OutputFormatParam | None = None
) -> types.OutputFormat:
  """Convert various input formats to a standardized OutputFormat."""
  if output_format is None:
    return types.OutputFormat(type=types.OutputFormatType.TEXT)
  elif isinstance(output_format, str):
    if output_format == 'text':
      return types.OutputFormat(type=types.OutputFormatType.TEXT)
    if output_format == 'json':
      return types.OutputFormat(type=types.OutputFormatType.JSON)
    _raise_invalid_output_format_value_error(output_format)
  elif isinstance(output_format, dict):
    return types.OutputFormat(
        value=output_format, type=types.OutputFormatType.JSON_SCHEMA
    )
  elif (
      isinstance(output_format, type) and
      issubclass(output_format, pydantic.BaseModel)
  ):
    return types.OutputFormat(
        value=types.ResponseFormatPydanticValue(
            class_name=output_format.__name__, class_value=output_format
        ), type=types.OutputFormatType.PYDANTIC
    )
  elif isinstance(output_format, types.StructuredResponseFormat):
    if output_format.type == types.OutputFormatType.TEXT:
      return types.OutputFormat(type=types.OutputFormatType.TEXT)
    elif output_format.type == types.OutputFormatType.JSON:
      return types.OutputFormat(type=types.OutputFormatType.JSON)
    elif output_format.type == types.OutputFormatType.JSON_SCHEMA:
      return types.OutputFormat(
          value=output_format.schema,
          type=types.OutputFormatType.JSON_SCHEMA
      )
    elif output_format.type == types.OutputFormatType.PYDANTIC:
      return types.OutputFormat(
          value=types.ResponseFormatPydanticValue(
              class_name=output_format.schema.__name__,
              class_value=output_format.schema
          ), type=types.OutputFormatType.PYDANTIC
      )

  _raise_invalid_output_format_value_error(output_format)


def is_query_record_equal(
    query_record_1: types.QueryRecord, query_record_2: types.QueryRecord
) -> bool:
  """Compare two query records, handling Pydantic schemas specially."""
  if (
      query_record_1.output_format is not None and
      query_record_1.output_format.type == types.OutputFormatType.PYDANTIC
  ):
    pydantic_value_1 = query_record_1.output_format.value
    if pydantic_value_1.class_value is not None:
      query_record_1 = copy.deepcopy(query_record_1)
      query_record_1.output_format.value = types.ResponseFormatPydanticValue(
          class_name=pydantic_value_1.class_name, class_json_schema_value=(
              pydantic_value_1.class_value.model_json_schema()
          )
      )
      del query_record_1.output_format.value.class_value

  if (
      query_record_2.output_format is not None and
      query_record_2.output_format.type == types.OutputFormatType.PYDANTIC
  ):
    pydantic_value_2 = query_record_2.output_format.value
    if pydantic_value_2.class_value is not None:
      query_record_2 = copy.deepcopy(query_record_2)
      query_record_2.output_format.value = types.ResponseFormatPydanticValue(
          class_name=pydantic_value_2.class_name, class_json_schema_value=(
              pydantic_value_2.class_value.model_json_schema()
          )
      )
      del query_record_2.output_format.value.class_value

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

  return query_record_1 == query_record_2


def create_pydantic_instance_from_response(
    output_format: types.OutputFormat, response: types.Response
) -> pydantic.BaseModel:
  """Create pydantic instance from Response.

  If response.value already has the instance, return it.
  Otherwise, recreate from pydantic_metadata.instance_json_value.
  """
  if response.value is not None:
    return response.value
  elif (
      response.pydantic_metadata is not None and
      response.pydantic_metadata.instance_json_value is not None
  ):
    return output_format.value.class_value.model_validate(
        response.pydantic_metadata.instance_json_value
    )
  else:
    raise ValueError(
        'Response has no value (instance) or '
        'pydantic_metadata.instance_json_value. Please create an issue at '
        'https://github.com/proxai/proxai/issues.\n'
        f'Response: {response}'
    )


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
