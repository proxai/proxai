import pydantic
import pytest

import proxai.type_utils as type_utils
import proxai.types as types


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
      type_utils.check_messages_type([{
          'role': 'invalid_role',
          'content': 'content'
      }])

  def test_valid_message(self):
    type_utils.check_messages_type([{'role': 'user', 'content': 'content'}])


class TestCreateOutputFormat:

  def test_none_returns_text_format(self):
    result = type_utils.create_output_format(None)
    assert result.type == types.OutputFormatType.TEXT

  def test_text_string_returns_text_format(self):
    result = type_utils.create_output_format('text')
    assert result.type == types.OutputFormatType.TEXT

  def test_json_string_returns_json_format(self):
    result = type_utils.create_output_format('json')
    assert result.type == types.OutputFormatType.JSON

  def test_dict_returns_json_schema_format(self):
    schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
    result = type_utils.create_output_format(schema)
    assert result.type == types.OutputFormatType.JSON_SCHEMA
    assert result.value == schema

  def test_pydantic_model_returns_pydantic_format(self):
    result = type_utils.create_output_format(SampleModel)
    assert result.type == types.OutputFormatType.PYDANTIC
    assert result.value.class_name == 'SampleModel'
    assert result.value.class_value == SampleModel

  def test_structured_output_format_text(self):
    output_format = types.StructuredOutputFormat(
        type=types.OutputFormatType.TEXT
    )
    result = type_utils.create_output_format(output_format)
    assert result.type == types.OutputFormatType.TEXT

  def test_structured_output_format_json(self):
    output_format = types.StructuredOutputFormat(
        type=types.OutputFormatType.JSON
    )
    result = type_utils.create_output_format(output_format)
    assert result.type == types.OutputFormatType.JSON

  def test_structured_output_format_json_schema(self):
    schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
    output_format = types.StructuredOutputFormat(
        schema=schema, type=types.OutputFormatType.JSON_SCHEMA
    )
    result = type_utils.create_output_format(output_format)
    assert result.type == types.OutputFormatType.JSON_SCHEMA
    assert result.value == schema

  def test_structured_output_format_pydantic(self):
    output_format = types.StructuredOutputFormat(
        schema=SampleModel, type=types.OutputFormatType.PYDANTIC
    )
    result = type_utils.create_output_format(output_format)
    assert result.type == types.OutputFormatType.PYDANTIC
    assert result.value.class_name == 'SampleModel'
    assert result.value.class_value == SampleModel

  def test_invalid_string_raises_error(self):
    with pytest.raises(ValueError):
      type_utils.create_output_format('invalid')


class TestCreatePydanticInstanceFromResponse:

  def _create_output_format(self) -> types.OutputFormat:
    return types.OutputFormat(
        value=types.OutputFormatPydanticValue(
            class_name=SampleModel.__name__, class_value=SampleModel
        ), type=types.OutputFormatType.PYDANTIC
    )

  def test_returns_instance_value_when_present(self):
    output_format = self._create_output_format()
    instance = SampleModel(name='John', age=30)
    response = types.Response(
        value=instance, type=types.ResponseType.PYDANTIC,
        pydantic_metadata=types.PydanticMetadataType(
            class_name=SampleModel.__name__
        )
    )

    result = type_utils.create_pydantic_instance_from_response(
        output_format, response
    )

    assert result is instance
    assert result.name == 'John'
    assert result.age == 30

  def test_creates_instance_from_json_value(self):
    output_format = self._create_output_format()
    response = types.Response(
        value=None, type=types.ResponseType.PYDANTIC,
        pydantic_metadata=types.PydanticMetadataType(
            class_name=SampleModel.__name__, instance_json_value={
                'name': 'Jane',
                'age': 25
            }
        )
    )

    result = type_utils.create_pydantic_instance_from_response(
        output_format, response
    )

    assert isinstance(result, SampleModel)
    assert result.name == 'Jane'
    assert result.age == 25

  def test_prefers_instance_value_over_json_value(self):
    output_format = self._create_output_format()
    instance = SampleModel(name='John', age=30)
    response = types.Response(
        value=instance, type=types.ResponseType.PYDANTIC,
        pydantic_metadata=types.PydanticMetadataType(
            class_name=SampleModel.__name__, instance_json_value={
                'name': 'Jane',
                'age': 25
            }
        )
    )

    result = type_utils.create_pydantic_instance_from_response(
        output_format, response
    )

    assert result is instance
    assert result.name == 'John'

  def test_raises_error_when_no_value_present(self):
    output_format = self._create_output_format()
    response = types.Response(
        value=None, type=types.ResponseType.PYDANTIC,
        pydantic_metadata=types.PydanticMetadataType(
            class_name=SampleModel.__name__
        )
    )

    with pytest.raises(ValueError) as exc_info:
      type_utils.create_pydantic_instance_from_response(
          output_format, response
      )

    assert 'no value (instance) or' in str(exc_info.value)


class TestOutputFormatParamToOutputFormat:

  def test_none_returns_text_format(self):
    result = type_utils.output_format_param_to_output_format(None)
    assert result.type == types.OutputFormatType.TEXT

  def test_text_string(self):
    result = type_utils.output_format_param_to_output_format('text')
    assert result.type == types.OutputFormatType.TEXT

  def test_json_string(self):
    result = type_utils.output_format_param_to_output_format('json')
    assert result.type == types.OutputFormatType.JSON

  def test_image_string(self):
    result = type_utils.output_format_param_to_output_format('image')
    assert result.type == types.OutputFormatType.IMAGE

  def test_audio_string(self):
    result = type_utils.output_format_param_to_output_format('audio')
    assert result.type == types.OutputFormatType.AUDIO

  def test_video_string(self):
    result = type_utils.output_format_param_to_output_format('video')
    assert result.type == types.OutputFormatType.VIDEO

  def test_invalid_string_raises_error(self):
    with pytest.raises(ValueError, match='Invalid output format'):
      type_utils.output_format_param_to_output_format('invalid')

  def test_pydantic_class(self):
    result = type_utils.output_format_param_to_output_format(SampleModel)
    assert result.type == types.OutputFormatType.PYDANTIC
    assert result.pydantic_class == SampleModel

  def test_output_format_passthrough(self):
    fmt = types.OutputFormat(type=types.OutputFormatType.JSON)
    result = type_utils.output_format_param_to_output_format(fmt)
    assert result is fmt

  def test_output_format_pydantic_passthrough(self):
    fmt = types.OutputFormat(
        type=types.OutputFormatType.PYDANTIC,
        pydantic_class=SampleModel)
    result = type_utils.output_format_param_to_output_format(fmt)
    assert result is fmt
    assert result.pydantic_class == SampleModel

  def test_invalid_type_raises_error(self):
    with pytest.raises(ValueError, match='Invalid output format'):
      type_utils.output_format_param_to_output_format(123)


class TestCreateFeatureListType:

  def test_string_features_converted_to_enum(self):
    features = ['prompt', 'messages', 'system']
    result = type_utils.create_feature_list_type(features)
    assert result == [
        types.FeatureNameType.PROMPT,
        types.FeatureNameType.MESSAGES,
        types.FeatureNameType.SYSTEM,
    ]

  def test_enum_features_kept_as_is(self):
    features = [
        types.FeatureNameType.PROMPT,
        types.FeatureNameType.MAX_TOKENS,
    ]
    result = type_utils.create_feature_list_type(features)
    assert result == [
        types.FeatureNameType.PROMPT,
        types.FeatureNameType.MAX_TOKENS,
    ]

  def test_mixed_string_and_enum_features(self):
    features = [
        'prompt',
        types.FeatureNameType.MESSAGES,
        'temperature',
    ]
    result = type_utils.create_feature_list_type(features)
    assert result == [
        types.FeatureNameType.PROMPT,
        types.FeatureNameType.MESSAGES,
        types.FeatureNameType.TEMPERATURE,
    ]

  def test_empty_list_returns_empty_list(self):
    result = type_utils.create_feature_list_type([])
    assert result == []

  def test_invalid_string_raises_error(self):
    with pytest.raises(ValueError):
      type_utils.create_feature_list_type(['invalid_feature'])

  def test_invalid_type_raises_error(self):
    with pytest.raises(ValueError):
      type_utils.create_feature_list_type([123])
