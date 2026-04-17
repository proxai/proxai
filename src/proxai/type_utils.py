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


def response_format_param_to_response_format(
    response_format: types.ResponseFormatParam | None
) -> types.ResponseFormat:
  if not response_format:
    return types.ResponseFormat(
        type=types.ResponseFormatType.TEXT)
    
  if isinstance(response_format, types.ResponseFormat):
    return response_format

  if type(response_format) == str:
    if response_format == 'text':
      return types.ResponseFormat(
          type=types.ResponseFormatType.TEXT)
    elif response_format == 'json':
      return types.ResponseFormat(
          type=types.ResponseFormatType.JSON)
    elif response_format == 'image':
      return types.ResponseFormat(
          type=types.ResponseFormatType.IMAGE)
    elif response_format == 'audio':
      return types.ResponseFormat(
          type=types.ResponseFormatType.AUDIO)
    elif response_format == 'video':
      return types.ResponseFormat(
          type=types.ResponseFormatType.VIDEO)
    else:
      raise ValueError(f'Invalid response format: {response_format}')
  elif (inspect.isclass(response_format) and
        issubclass(response_format, pydantic.BaseModel)):
    return types.ResponseFormat(
        type=types.ResponseFormatType.PYDANTIC,
        pydantic_class=response_format)

  raise ValueError(f'Invalid response format: {response_format}')


def _raise_invalid_response_format_value_error(
    response_format: types.ResponseFormatParam
) -> None:
  raise ValueError(
      'Please provide one of the followings:\n'
      ' - "json" as string for JSON response format\n'
      ' - dict for JSON schema response format\n'
      ' - pydantic.BaseModel for Pydantic response format\n'
      ' - proxai.types.ResponseFormat for custom more advanced response '
      'format\n'
      'Check https://www.proxai.co/proxai-docs/advanced/response-format '
      'for more information.\n'
      f'Response format type: {type(response_format)}\n'
      f'Response format value: {response_format}'
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


def create_response_format(
    response_format: types.ResponseFormatParam | None = None
) -> types.ResponseFormat:
  """Convert various input formats to a standardized ResponseFormat."""
  if response_format is None:
    return types.ResponseFormat(type=types.ResponseFormatType.TEXT)
  elif isinstance(response_format, str):
    if response_format == 'text':
      return types.ResponseFormat(type=types.ResponseFormatType.TEXT)
    if response_format == 'json':
      return types.ResponseFormat(type=types.ResponseFormatType.JSON)
    _raise_invalid_response_format_value_error(response_format)
  elif isinstance(response_format, dict):
    return types.ResponseFormat(
        value=response_format, type=types.ResponseFormatType.JSON_SCHEMA
    )
  elif (
      isinstance(response_format, type) and
      issubclass(response_format, pydantic.BaseModel)
  ):
    return types.ResponseFormat(
        value=types.ResponseFormatPydanticValue(
            class_name=response_format.__name__, class_value=response_format
        ), type=types.ResponseFormatType.PYDANTIC
    )
  elif isinstance(response_format, types.StructuredResponseFormat):
    if response_format.type == types.ResponseFormatType.TEXT:
      return types.ResponseFormat(type=types.ResponseFormatType.TEXT)
    elif response_format.type == types.ResponseFormatType.JSON:
      return types.ResponseFormat(type=types.ResponseFormatType.JSON)
    elif response_format.type == types.ResponseFormatType.JSON_SCHEMA:
      return types.ResponseFormat(
          value=response_format.schema,
          type=types.ResponseFormatType.JSON_SCHEMA
      )
    elif response_format.type == types.ResponseFormatType.PYDANTIC:
      return types.ResponseFormat(
          value=types.ResponseFormatPydanticValue(
              class_name=response_format.schema.__name__,
              class_value=response_format.schema
          ), type=types.ResponseFormatType.PYDANTIC
      )

  _raise_invalid_response_format_value_error(response_format)


def is_query_record_equal(
    query_record_1: types.QueryRecord, query_record_2: types.QueryRecord
) -> bool:
  """Compare two query records, handling Pydantic schemas specially."""
  if (
      query_record_1.response_format is not None and
      query_record_1.response_format.type == types.ResponseFormatType.PYDANTIC
  ):
    pydantic_value_1 = query_record_1.response_format.value
    if pydantic_value_1.class_value is not None:
      query_record_1 = copy.deepcopy(query_record_1)
      query_record_1.response_format.value = types.ResponseFormatPydanticValue(
          class_name=pydantic_value_1.class_name, class_json_schema_value=(
              pydantic_value_1.class_value.model_json_schema()
          )
      )
      del query_record_1.response_format.value.class_value

  if (
      query_record_2.response_format is not None and
      query_record_2.response_format.type == types.ResponseFormatType.PYDANTIC
  ):
    pydantic_value_2 = query_record_2.response_format.value
    if pydantic_value_2.class_value is not None:
      query_record_2 = copy.deepcopy(query_record_2)
      query_record_2.response_format.value = types.ResponseFormatPydanticValue(
          class_name=pydantic_value_2.class_name, class_json_schema_value=(
              pydantic_value_2.class_value.model_json_schema()
          )
      )
      del query_record_2.response_format.value.class_value

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
    response_format: types.ResponseFormat, response: types.Response
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
    return response_format.value.class_value.model_validate(
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
