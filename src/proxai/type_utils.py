import copy
from typing import Optional
import proxai.types as types
import pydantic


def _raise_invalid_response_format_value_error(
    response_format: types.ResponseFormatParam) -> None:
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
        f'Response format value: {response_format}')


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


def create_response_format(
    response_format: Optional[types.ResponseFormatParam] = None
) -> types.ResponseFormat:
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
        value=response_format,
        type=types.ResponseFormatType.JSON_SCHEMA)
  elif (isinstance(response_format, type) and
        issubclass(response_format, pydantic.BaseModel)):
    return types.ResponseFormat(
        value=types.ResponseFormatPydanticValue(
            class_name=response_format.__name__,
            class_value=response_format),
        type=types.ResponseFormatType.PYDANTIC)
  elif isinstance(response_format, types.StructuredResponseFormat):
    if response_format.type == types.ResponseFormatType.TEXT:
      return types.ResponseFormat(
          type=types.ResponseFormatType.TEXT)
    elif response_format.type == types.ResponseFormatType.JSON:
      return types.ResponseFormat(
          type=types.ResponseFormatType.JSON)
    elif response_format.type == types.ResponseFormatType.JSON_SCHEMA:
      return types.ResponseFormat(
          value=response_format.schema,
          type=types.ResponseFormatType.JSON_SCHEMA)
    elif response_format.type == types.ResponseFormatType.PYDANTIC:
      return types.ResponseFormat(
          value=types.ResponseFormatPydanticValue(
              class_name=response_format.schema.__name__,
              class_value=response_format.schema),
          type=types.ResponseFormatType.PYDANTIC)

  _raise_invalid_response_format_value_error(response_format)


def is_query_record_equal(
    query_record_1: types.QueryRecord,
    query_record_2: types.QueryRecord) -> bool:
  if (query_record_1.response_format is not None and
      query_record_1.response_format.type == types.ResponseFormatType.PYDANTIC):
    pydantic_value_1 = query_record_1.response_format.value
    if pydantic_value_1.class_value is not None:
      query_record_1 = copy.deepcopy(query_record_1)
      query_record_1.response_format.value = types.ResponseFormatPydanticValue(
          class_name=pydantic_value_1.class_name,
          class_json_schema_value=(
              pydantic_value_1.class_value.model_json_schema()))
      del query_record_1.response_format.value.class_value

  if (query_record_2.response_format is not None and
      query_record_2.response_format.type == types.ResponseFormatType.PYDANTIC):
    pydantic_value_2 = query_record_2.response_format.value
    if pydantic_value_2.class_value is not None:
      query_record_2 = copy.deepcopy(query_record_2)
      query_record_2.response_format.value = types.ResponseFormatPydanticValue(
          class_name=pydantic_value_2.class_name,
          class_json_schema_value=(
              pydantic_value_2.class_value.model_json_schema()))
      del query_record_2.response_format.value.class_value

  return query_record_1 == query_record_2


def create_pydantic_instance_from_response(
    response_format: types.ResponseFormat,
    response: types.Response) -> pydantic.BaseModel:
  """Create pydantic instance from Response.

  If response.value already has the instance, return it.
  Otherwise, recreate from pydantic_metadata.instance_json_value.
  """
  if response.value is not None:
    return response.value
  elif (response.pydantic_metadata is not None and
        response.pydantic_metadata.instance_json_value is not None):
    return response_format.value.class_value.model_validate(
        response.pydantic_metadata.instance_json_value)
  else:
    raise ValueError(
        'Response has no value (instance) or '
        'pydantic_metadata.instance_json_value. Please create an issue at '
        'https://github.com/proxai/proxai/issues.\n'
        f'Response: {response}')


def create_feature_list_type(
    features: types.FeatureListParam
) -> types.FeatureListType:
  result_features = []
  for feature in features:
    if isinstance(feature, str):
      result_features.append(types.FeatureNameType(feature))
    elif isinstance(feature, types.FeatureNameType):
      result_features.append(feature)
    else:
      raise ValueError(f'Invalid feature: {feature}')
  return result_features
