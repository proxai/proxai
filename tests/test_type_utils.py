import pydantic
import pytest

import proxai.chat.chat_session as chat_session
import proxai.chat.message as message
import proxai.chat.message_content as message_content
import proxai.type_utils as type_utils
import proxai.types as types


class SampleModel(pydantic.BaseModel):
  name: str
  age: int


_MODEL = types.ProviderModelType(
    provider='openai', model='gpt-4', provider_model_identifier='gpt-4'
)


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


class TestMessagesParamToChat:

  def test_none_returns_none(self):
    assert type_utils.messages_param_to_chat(None) is None

  def test_list_wraps_in_chat(self):
    msg = message.Message(role='user', content='hello')
    result = type_utils.messages_param_to_chat([msg])
    assert isinstance(result, chat_session.Chat)
    assert result.messages == [msg]

  def test_dict_builds_chat(self):
    msg = message.Message(role='user', content='hello')
    result = type_utils.messages_param_to_chat(
        {'system': 'be helpful', 'messages': [msg]}
    )
    assert result.system_prompt == 'be helpful'
    assert result.messages == [msg]

  def test_chat_passthrough(self):
    chat = chat_session.Chat()
    assert type_utils.messages_param_to_chat(chat) is chat

  def test_invalid_type_raises(self):
    with pytest.raises(ValueError):
      type_utils.messages_param_to_chat('not a chat')


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


class TestCheckModelSizeIdentifierType:

  def test_enum_passthrough(self):
    assert type_utils.check_model_size_identifier_type(
        types.ModelSizeType.SMALL
    ) == types.ModelSizeType.SMALL

  def test_string_converted(self):
    assert type_utils.check_model_size_identifier_type('small') == (
        types.ModelSizeType.SMALL
    )

  def test_invalid_raises(self):
    with pytest.raises(ValueError):
      type_utils.check_model_size_identifier_type('huge')


class TestCheckOutputFormatTypeParam:

  def test_enum_passthrough(self):
    assert type_utils.check_output_format_type_param(
        types.OutputFormatType.JSON
    ) == types.OutputFormatType.JSON

  def test_lowercase_string_normalized(self):
    assert type_utils.check_output_format_type_param('json') == (
        types.OutputFormatType.JSON
    )

  def test_invalid_raises(self):
    with pytest.raises(ValueError):
      type_utils.check_output_format_type_param('xml')


class TestCheckInputFormatTypeParam:

  def test_enum_passthrough(self):
    assert type_utils.check_input_format_type_param(
        types.InputFormatType.DOCUMENT
    ) == types.InputFormatType.DOCUMENT

  def test_lowercase_string_normalized(self):
    assert type_utils.check_input_format_type_param('document') == (
        types.InputFormatType.DOCUMENT
    )

  def test_invalid_raises(self):
    with pytest.raises(ValueError):
      type_utils.check_input_format_type_param('xml')


class TestCreateFeatureTagList:

  def test_none_returns_none(self):
    assert type_utils.create_feature_tag_list(None) is None

  def test_single_string_wrapped(self):
    assert type_utils.create_feature_tag_list('prompt') == [
        types.FeatureTag.PROMPT
    ]

  def test_mixed_str_and_enum(self):
    result = type_utils.create_feature_tag_list(
        ['prompt', types.FeatureTag.MESSAGES]
    )
    assert result == [types.FeatureTag.PROMPT, types.FeatureTag.MESSAGES]

  def test_invalid_string_raises(self):
    with pytest.raises(ValueError):
      type_utils.create_feature_tag_list(['not_a_tag'])


class TestTagListHelperWiring:

  def test_create_input_format_type_list(self):
    assert type_utils.create_input_format_type_list(['text']) == [
        types.InputFormatType.TEXT
    ]

  def test_create_output_format_type_list(self):
    assert type_utils.create_output_format_type_list(['json']) == [
        types.OutputFormatType.JSON
    ]

  def test_create_tool_tag_list(self):
    assert type_utils.create_tool_tag_list(['web_search']) == [
        types.ToolTag.WEB_SEARCH
    ]


class TestIsQueryRecordEqual:

  def test_identical_equal(self):
    qr_1 = types.QueryRecord(prompt='hello', system_prompt='be helpful')
    qr_2 = types.QueryRecord(prompt='hello', system_prompt='be helpful')
    assert type_utils.is_query_record_equal(qr_1, qr_2)

  def test_different_prompts_not_equal(self):
    qr_1 = types.QueryRecord(prompt='hello')
    qr_2 = types.QueryRecord(prompt='goodbye')
    assert not type_utils.is_query_record_equal(qr_1, qr_2)

  def test_pydantic_class_normalized(self):
    qr_live = types.QueryRecord(
        prompt='hi',
        output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
            pydantic_class=SampleModel,
        ),
    )
    qr_stored = types.QueryRecord(
        prompt='hi',
        output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
            pydantic_class_name='SampleModel',
            pydantic_class_json_schema=SampleModel.model_json_schema(),
        ),
    )
    assert type_utils.is_query_record_equal(qr_live, qr_stored)

  def test_connection_options_non_endpoint_fields_ignored(self):
    qr_1 = types.QueryRecord(
        prompt='hi',
        connection_options=types.ConnectionOptions(
            endpoint='ep', skip_cache=True, fallback_models=[_MODEL]
        ),
    )
    qr_2 = types.QueryRecord(
        prompt='hi',
        connection_options=types.ConnectionOptions(
            endpoint='ep', override_cache_value=True
        ),
    )
    assert type_utils.is_query_record_equal(qr_1, qr_2)

  def test_file_api_metadata_ignored(self):
    def _chat(ids):
      return chat_session.Chat(messages=[
          message.Message(
              role=message_content.MessageRoleType.USER,
              content=[
                  message_content.MessageContent(
                      type='image',
                      source='http://example.com/x',
                      provider_file_api_ids=ids,
                  )
              ],
          )
      ])

    qr_1 = types.QueryRecord(chat=_chat({'gemini': 'files/abc'}))
    qr_2 = types.QueryRecord(chat=_chat({'gemini': 'files/xyz'}))
    assert type_utils.is_query_record_equal(qr_1, qr_2)
