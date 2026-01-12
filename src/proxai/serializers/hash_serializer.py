import hashlib
import json

import proxai.types as types

_SEPARATOR_CHAR = chr(255)
_HASH_LENGTH = 16


def _get_response_format_signature(
    response_format: types.ResponseFormat) -> str:
  signature_str = ''
  if response_format.type is not None:
    signature_str += response_format.type.value + _SEPARATOR_CHAR
  if response_format.value is not None:
    if isinstance(response_format.value, str):
      signature_str += response_format.value + _SEPARATOR_CHAR
    elif isinstance(response_format.value, dict):
      signature_str += json.dumps(
          response_format.value,
          sort_keys=True) + _SEPARATOR_CHAR
    elif isinstance(response_format.value, types.ResponseFormatPydanticValue):
      pydantic_value = response_format.value
      if pydantic_value.class_name is not None:
        signature_str += pydantic_value.class_name + _SEPARATOR_CHAR
      json_schema = None
      if pydantic_value.class_json_schema_value is not None:
        json_schema = pydantic_value.class_json_schema_value
      elif pydantic_value.class_value is not None:
        json_schema = pydantic_value.class_value.model_json_schema()
      if json_schema is not None:
        signature_str += json.dumps(
            json_schema,
            sort_keys=True) + _SEPARATOR_CHAR
    else:
      raise ValueError(
        'Unsupported response format value type: '
        f'{type(response_format.value)}')
  return signature_str


def get_query_record_hash(query_record: types.QueryRecord) -> str:
  signature_str = ''
  if query_record.call_type is not None:
    signature_str += query_record.call_type + _SEPARATOR_CHAR
  if query_record.provider_model is not None:
    signature_str += query_record.provider_model.provider + _SEPARATOR_CHAR
    signature_str += query_record.provider_model.model + _SEPARATOR_CHAR
    signature_str += (
        query_record.provider_model.provider_model_identifier + _SEPARATOR_CHAR)
  if query_record.prompt is not None:
    signature_str += query_record.prompt + _SEPARATOR_CHAR
  if query_record.system is not None:
    signature_str += query_record.system + _SEPARATOR_CHAR
  if query_record.messages is not None:
    for message in query_record.messages:
      signature_str += 'role:'+message['role'] + _SEPARATOR_CHAR
      signature_str += 'content:'+message['content'] + _SEPARATOR_CHAR
  if query_record.max_tokens is not None:
    signature_str += str(query_record.max_tokens) + _SEPARATOR_CHAR
  if query_record.temperature is not None:
    signature_str += str(query_record.temperature) + _SEPARATOR_CHAR
  if query_record.stop is not None:
    signature_str += str(query_record.stop) + _SEPARATOR_CHAR
  if query_record.response_format is not None:
    signature_str += _get_response_format_signature(
        query_record.response_format) + _SEPARATOR_CHAR
  if query_record.web_search is not None:
    signature_str += str(query_record.web_search) + _SEPARATOR_CHAR
  if query_record.feature_mapping_strategy is not None:
    signature_str += (
        query_record.feature_mapping_strategy.value + _SEPARATOR_CHAR)
  if query_record.chosen_endpoint is not None:
    signature_str += query_record.chosen_endpoint + _SEPARATOR_CHAR
  return hashlib.sha256(signature_str.encode()).hexdigest()[:_HASH_LENGTH]
