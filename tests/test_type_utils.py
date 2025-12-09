import datetime
import pydantic
import proxai.types as types
import proxai.type_utils as type_utils
import pytest


class SampleModel(pydantic.BaseModel):
  name: str
  age: int


class TestCheckMessagesType:
  def test_invalid_message_type(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type(['invalid_message'])

  def test_invalid_message_keys(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type([{'role': 'user'}])

  def test_invalid_role_type(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type([{'role': 1, 'content': 'content'}])

  def test_invalid_content_type(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type([{'role': 'user', 'content': 1}])

  def test_invalid_role(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type([
          {'role': 'invalid_role', 'content': 'content'}])

  def test_valid_message(self):
    type_utils.check_messages_type([
      {'role': 'user', 'content': 'content'}])


class TestCreateResponseFormat:
  def test_none_returns_text_format(self):
    result = type_utils.create_response_format(None)
    assert result.type == types.ResponseFormatType.TEXT

  def test_text_string_returns_text_format(self):
    result = type_utils.create_response_format('text')
    assert result.type == types.ResponseFormatType.TEXT

  def test_json_string_returns_json_format(self):
    result = type_utils.create_response_format('json')
    assert result.type == types.ResponseFormatType.JSON

  def test_dict_returns_json_schema_format(self):
    schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
    result = type_utils.create_response_format(schema)
    assert result.type == types.ResponseFormatType.JSON_SCHEMA
    assert result.value == schema

  def test_pydantic_model_returns_pydantic_format(self):
    result = type_utils.create_response_format(SampleModel)
    assert result.type == types.ResponseFormatType.PYDANTIC
    assert result.value.class_name == 'SampleModel'
    assert result.value.class_value == SampleModel

  def test_response_format_returns_same(self):
    response_format = types.ResponseFormat(type=types.ResponseFormatType.JSON)
    result = type_utils.create_response_format(response_format)
    assert result is response_format

  def test_invalid_string_raises_error(self):
    with pytest.raises(ValueError):
      type_utils.create_response_format('invalid')


class TestCreatePydanticInstanceFromResponsePydanticValue:
  def _create_response_format(self) -> types.ResponseFormat:
    return types.ResponseFormat(
        value=types.ResponseFormatPydanticValue(
            class_name=SampleModel.__name__,
            class_value=SampleModel),
        type=types.ResponseFormatType.PYDANTIC)

  def test_returns_instance_value_when_present(self):
    response_format = self._create_response_format()
    instance = SampleModel(name='John', age=30)
    response_pydantic_value = types.ResponsePydanticValue(
        class_name=SampleModel.__name__,
        instance_value=instance)

    result = type_utils.create_pydantic_instance_from_response_pydantic_value(
        response_format, response_pydantic_value)

    assert result is instance
    assert result.name == 'John'
    assert result.age == 30

  def test_creates_instance_from_json_value(self):
    response_format = self._create_response_format()
    response_pydantic_value = types.ResponsePydanticValue(
        class_name=SampleModel.__name__,
        instance_json_value={'name': 'Jane', 'age': 25})

    result = type_utils.create_pydantic_instance_from_response_pydantic_value(
        response_format, response_pydantic_value)

    assert isinstance(result, SampleModel)
    assert result.name == 'Jane'
    assert result.age == 25

  def test_prefers_instance_value_over_json_value(self):
    response_format = self._create_response_format()
    instance = SampleModel(name='John', age=30)
    response_pydantic_value = types.ResponsePydanticValue(
        class_name=SampleModel.__name__,
        instance_value=instance,
        instance_json_value={'name': 'Jane', 'age': 25})

    result = type_utils.create_pydantic_instance_from_response_pydantic_value(
        response_format, response_pydantic_value)

    assert result is instance
    assert result.name == 'John'

  def test_raises_error_when_no_value_present(self):
    response_format = self._create_response_format()
    response_pydantic_value = types.ResponsePydanticValue(
        class_name=SampleModel.__name__)

    with pytest.raises(ValueError) as exc_info:
      type_utils.create_pydantic_instance_from_response_pydantic_value(
          response_format, response_pydantic_value)

    assert 'no instance_value or instance_json_value' in str(exc_info.value)
