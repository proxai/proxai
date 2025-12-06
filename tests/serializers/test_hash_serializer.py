import proxai.types as types
import proxai.serializers.hash_serializer as hash_serializer
import pytest
import proxai.connectors.model_configs as model_configs
import pydantic
from typing import List, Optional


class UserModel(pydantic.BaseModel):
  name: str
  age: int


class AddressModel(pydantic.BaseModel):
  street: str
  city: str
  country: str


class UserWithAddressModel(pydantic.BaseModel):
  name: str
  email: Optional[str] = None
  address: AddressModel
  tags: List[str] = []


def _get_query_record_options():
  model_configs_instance = model_configs.ModelConfigs()
  return [
      {'call_type': types.CallType.GENERATE_TEXT},
      {'provider_model': model_configs_instance.get_provider_model(
          ('openai', 'gpt-4'))},
      {'prompt': 'Hello, world!'},
      {'system': 'Hello, system!'},
      {'messages': [{'role': 'user', 'content': 'Hello, user!'}]},
      {'max_tokens': 100},
      {'temperature': 0.5},
      {'stop': ['.', '?', '!']},
      {'call_type': types.CallType.GENERATE_TEXT,
       'provider_model': model_configs_instance.get_provider_model(
           ('openai', 'gpt-4')),
       'prompt': 'Hello, world!',
       'system': 'Hello, system!',
       'messages': [{'role': 'user', 'content': 'Hello, user!'}],
       'max_tokens': 100,
       'temperature': 0.5,
        'stop': ['.', '?', '!']},]


class TestBaseQueryCache:
  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_get_query_record_hash(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    query_hash_value = hash_serializer.get_query_record_hash(
        query_record=query_record)

    query_record_options['max_tokens'] = 222
    query_record_2 = types.QueryRecord(**query_record_options)
    query_hash_value_2 = hash_serializer.get_query_record_hash(
        query_record=query_record_2)

    assert query_hash_value != query_hash_value_2
    assert query_hash_value == (
        hash_serializer.get_query_record_hash(
            query_record=query_record))


def _create_pydantic_value(model_class):
  return types.ResponseFormatPydanticValue(
      class_name=model_class.__name__,
      class_value=model_class)


class TestResponseFormatHash:
  def test_text_response_format(self):
    response_format = types.ResponseFormat(
        type=types.ResponseFormatType.TEXT)
    query_record = types.QueryRecord(
        prompt='test',
        response_format=response_format)
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_json_response_format(self):
    response_format = types.ResponseFormat(
        type=types.ResponseFormatType.JSON)
    query_record = types.QueryRecord(
        prompt='test',
        response_format=response_format)
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_json_schema_dict_response_format(self):
    schema = {
        'type': 'object',
        'properties': {'name': {'type': 'string'}},
        'required': ['name']
    }
    response_format = types.ResponseFormat(
        value=schema,
        type=types.ResponseFormatType.JSON_SCHEMA)
    query_record = types.QueryRecord(
        prompt='test',
        response_format=response_format)
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_pydantic_response_format(self):
    response_format = types.ResponseFormat(
        value=_create_pydantic_value(UserModel),
        type=types.ResponseFormatType.PYDANTIC)
    query_record = types.QueryRecord(
        prompt='test',
        response_format=response_format)
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_nested_pydantic_response_format(self):
    response_format = types.ResponseFormat(
        value=_create_pydantic_value(UserWithAddressModel),
        type=types.ResponseFormatType.PYDANTIC)
    query_record = types.QueryRecord(
        prompt='test',
        response_format=response_format)
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_different_response_formats_produce_different_hashes(self):
    query_record_text = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.TEXT))

    query_record_json = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.JSON))

    query_record_pydantic = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            value=_create_pydantic_value(UserModel),
            type=types.ResponseFormatType.PYDANTIC))

    query_record_no_format = types.QueryRecord(prompt='test')

    hash_text = hash_serializer.get_query_record_hash(query_record_text)
    hash_json = hash_serializer.get_query_record_hash(query_record_json)
    hash_pydantic = hash_serializer.get_query_record_hash(query_record_pydantic)
    hash_no_format = hash_serializer.get_query_record_hash(query_record_no_format)

    assert hash_text != hash_json
    assert hash_text != hash_pydantic
    assert hash_text != hash_no_format
    assert hash_json != hash_pydantic
    assert hash_json != hash_no_format
    assert hash_pydantic != hash_no_format

  def test_different_pydantic_models_produce_different_hashes(self):
    query_record_user = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            value=_create_pydantic_value(UserModel),
            type=types.ResponseFormatType.PYDANTIC))

    query_record_address = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            value=_create_pydantic_value(AddressModel),
            type=types.ResponseFormatType.PYDANTIC))

    hash_user = hash_serializer.get_query_record_hash(query_record_user)
    hash_address = hash_serializer.get_query_record_hash(query_record_address)

    assert hash_user != hash_address

  def test_different_json_schemas_produce_different_hashes(self):
    schema_1 = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
    schema_2 = {'type': 'object', 'properties': {'age': {'type': 'integer'}}}

    query_record_1 = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            value=schema_1,
            type=types.ResponseFormatType.JSON_SCHEMA))

    query_record_2 = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            value=schema_2,
            type=types.ResponseFormatType.JSON_SCHEMA))

    hash_1 = hash_serializer.get_query_record_hash(query_record_1)
    hash_2 = hash_serializer.get_query_record_hash(query_record_2)

    assert hash_1 != hash_2

  def test_json_schema_key_order_does_not_affect_hash(self):
    schema_1 = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
    schema_2 = {'properties': {'name': {'type': 'string'}}, 'type': 'object'}

    query_record_1 = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            value=schema_1,
            type=types.ResponseFormatType.JSON_SCHEMA))

    query_record_2 = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            value=schema_2,
            type=types.ResponseFormatType.JSON_SCHEMA))

    hash_1 = hash_serializer.get_query_record_hash(query_record_1)
    hash_2 = hash_serializer.get_query_record_hash(query_record_2)

    assert hash_1 == hash_2

  def test_json_schema_with_unicode_characters(self):
    schema = {
        'type': 'object',
        'properties': {'名前': {'type': 'string', 'description': '用户名称 🎉'}},
        'required': ['名前']
    }
    response_format = types.ResponseFormat(
        value=schema,
        type=types.ResponseFormatType.JSON_SCHEMA)
    query_record = types.QueryRecord(
        prompt='test',
        response_format=response_format)
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_pydantic_class_name_stored(self):
    pydantic_value = _create_pydantic_value(UserModel)

    assert pydantic_value.class_name == 'UserModel'
    assert pydantic_value.class_value == UserModel

  def test_pydantic_hash_consistent_after_serialization(self):
    import proxai.serializers.type_serializer as type_serializer

    response_format = types.ResponseFormat(
        value=_create_pydantic_value(UserModel),
        type=types.ResponseFormatType.PYDANTIC)
    query_record = types.QueryRecord(
        prompt='test',
        response_format=response_format)

    hash_before = hash_serializer.get_query_record_hash(query_record)

    encoded = type_serializer.encode_query_record(query_record)
    decoded_query_record = type_serializer.decode_query_record(encoded)

    hash_after = hash_serializer.get_query_record_hash(decoded_query_record)

    assert hash_before == hash_after

  def test_pydantic_nested_hash_consistent_after_serialization(self):
    import proxai.serializers.type_serializer as type_serializer

    response_format = types.ResponseFormat(
        value=_create_pydantic_value(UserWithAddressModel),
        type=types.ResponseFormatType.PYDANTIC)
    query_record = types.QueryRecord(
        prompt='test',
        response_format=response_format)

    hash_before = hash_serializer.get_query_record_hash(query_record)

    encoded = type_serializer.encode_query_record(query_record)
    decoded_query_record = type_serializer.decode_query_record(encoded)

    hash_after = hash_serializer.get_query_record_hash(decoded_query_record)

    assert hash_before == hash_after

  def test_json_schema_hash_consistent_after_serialization(self):
    import proxai.serializers.type_serializer as type_serializer

    schema = {
        'type': 'object',
        'properties': {'name': {'type': 'string'}},
        'required': ['name']
    }
    response_format = types.ResponseFormat(
        value=schema,
        type=types.ResponseFormatType.JSON_SCHEMA)
    query_record = types.QueryRecord(
        prompt='test',
        response_format=response_format)

    hash_before = hash_serializer.get_query_record_hash(query_record)

    encoded = type_serializer.encode_query_record(query_record)
    decoded_query_record = type_serializer.decode_query_record(encoded)

    hash_after = hash_serializer.get_query_record_hash(decoded_query_record)

    assert hash_before == hash_after

  def test_pydantic_class_name_affects_hash(self):
    pydantic_value_1 = types.ResponseFormatPydanticValue(
        class_name='UserModel',
        class_value=UserModel)
    pydantic_value_2 = types.ResponseFormatPydanticValue(
        class_name='DifferentName',
        class_value=UserModel)

    query_record_1 = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            value=pydantic_value_1,
            type=types.ResponseFormatType.PYDANTIC))
    query_record_2 = types.QueryRecord(
        prompt='test',
        response_format=types.ResponseFormat(
            value=pydantic_value_2,
            type=types.ResponseFormatType.PYDANTIC))

    hash_1 = hash_serializer.get_query_record_hash(query_record_1)
    hash_2 = hash_serializer.get_query_record_hash(query_record_2)

    assert hash_1 != hash_2
