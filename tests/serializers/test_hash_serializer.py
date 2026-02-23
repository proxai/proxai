import pydantic
import pytest

import proxai.chat.chat_session as chat_session
import proxai.chat.message as message
import proxai.chat.message_content as message_content
import proxai.serializers.hash_serializer as hash_serializer
import proxai.types as types


_MODEL_1 = types.ProviderModelType(
    provider='openai', model='gpt-4', provider_model_identifier='gpt-4'
)


class UserModel(pydantic.BaseModel):
  name: str
  age: int


class AddressModel(pydantic.BaseModel):
  street: str
  city: str
  country: str


class UserWithAddressModel(pydantic.BaseModel):
  name: str
  email: str | None = None
  address: AddressModel
  tags: list[str] = []


def _get_query_record_options():
  return [{
      'prompt': 'Hello, world!'
  }, {
      'system_prompt': 'Hello, system!'
  }, {
      'chat':
          chat_session.Chat(messages=[
              message.Message(
                  role=message_content.MessageRoleType.USER,
                  content=[
                      message_content.MessageContent(
                          type='text', text='Hello, user!'
                      )
                  ]
              )
          ])
  }, {
      'parameters': types.ParameterType(max_tokens=100)
  }, {
      'parameters': types.ParameterType(temperature=0.5)
  }, {
      'parameters': types.ParameterType(stop=['.', '?', '!'])
  }, {
      'parameters': types.ParameterType(n=3)
  }, {
      'parameters': types.ParameterType(thinking=types.ThinkingType.HIGH)
  }, {
      'response_format':
          types.ResponseFormat(type=types.ResponseFormatType.TEXT)
  }, {
      'response_format':
          types.ResponseFormat(type=types.ResponseFormatType.JSON)
  }, {
      'tools': [types.Tools.WEB_SEARCH]
  }, {
      'provider_model': _MODEL_1
  }, {
      'connection_options':
          types.ConnectionOptions(endpoint='test_endpoint')
  }, {
      'prompt': 'Hello, world!',
      'system_prompt': 'Hello, system!',
      'chat':
          chat_session.Chat(messages=[
              message.Message(
                  role=message_content.MessageRoleType.USER,
                  content=[
                      message_content.MessageContent(
                          type='text', text='Hello, user!'
                      )
                  ]
              )
          ]),
      'provider_model': _MODEL_1,
      'parameters':
          types.ParameterType(
              max_tokens=100, temperature=0.5,
              stop=['.', '?', '!'], n=2,
              thinking=types.ThinkingType.MEDIUM
          ),
      'response_format':
          types.ResponseFormat(type=types.ResponseFormatType.JSON),
      'tools': [types.Tools.WEB_SEARCH],
      'connection_options':
          types.ConnectionOptions(endpoint='test_endpoint')
  }]


class TestBaseQueryCache:

  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_get_query_record_hash(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    query_hash_value = hash_serializer.get_query_record_hash(
        query_record=query_record
    )

    # Changing parameters produces a different hash
    query_record_options_copy = query_record_options.copy()
    if 'parameters' in query_record_options_copy:
      query_record_options_copy['parameters'] = types.ParameterType(
          max_tokens=222
      )
    else:
      query_record_options_copy['parameters'] = types.ParameterType(
          max_tokens=222
      )
    query_record_2 = types.QueryRecord(**query_record_options_copy)
    query_hash_value_2 = hash_serializer.get_query_record_hash(
        query_record=query_record_2
    )

    assert query_hash_value != query_hash_value_2
    # Same input produces same hash (deterministic)
    assert query_hash_value == (
        hash_serializer.get_query_record_hash(query_record=query_record)
    )


class TestResponseFormatHash:

  def test_text_response_format(self):
    response_format = types.ResponseFormat(type=types.ResponseFormatType.TEXT)
    query_record = types.QueryRecord(
        prompt='test', response_format=response_format
    )
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_json_response_format(self):
    response_format = types.ResponseFormat(type=types.ResponseFormatType.JSON)
    query_record = types.QueryRecord(
        prompt='test', response_format=response_format
    )
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_pydantic_response_format(self):
    response_format = types.ResponseFormat(
        type=types.ResponseFormatType.PYDANTIC,
        pydantic_class=UserModel
    )
    query_record = types.QueryRecord(
        prompt='test', response_format=response_format
    )
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_nested_pydantic_response_format(self):
    response_format = types.ResponseFormat(
        type=types.ResponseFormatType.PYDANTIC,
        pydantic_class=UserWithAddressModel
    )
    query_record = types.QueryRecord(
        prompt='test', response_format=response_format
    )
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_different_response_formats_produce_different_hashes(self):
    query_record_text = types.QueryRecord(
        prompt='test', response_format=types.ResponseFormat(
            type=types.ResponseFormatType.TEXT
        )
    )

    query_record_json = types.QueryRecord(
        prompt='test', response_format=types.ResponseFormat(
            type=types.ResponseFormatType.JSON
        )
    )

    query_record_pydantic = types.QueryRecord(
        prompt='test', response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            pydantic_class=UserModel
        )
    )

    query_record_no_format = types.QueryRecord(prompt='test')

    hash_text = hash_serializer.get_query_record_hash(query_record_text)
    hash_json = hash_serializer.get_query_record_hash(query_record_json)
    hash_pydantic = hash_serializer.get_query_record_hash(query_record_pydantic)
    hash_no_format = hash_serializer.get_query_record_hash(
        query_record_no_format
    )

    assert hash_text != hash_json
    assert hash_text != hash_pydantic
    assert hash_text != hash_no_format
    assert hash_json != hash_pydantic
    assert hash_json != hash_no_format
    assert hash_pydantic != hash_no_format

  def test_different_pydantic_models_produce_different_hashes(self):
    query_record_user = types.QueryRecord(
        prompt='test', response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            pydantic_class=UserModel
        )
    )

    query_record_address = types.QueryRecord(
        prompt='test', response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            pydantic_class=AddressModel
        )
    )

    hash_user = hash_serializer.get_query_record_hash(query_record_user)
    hash_address = hash_serializer.get_query_record_hash(query_record_address)

    assert hash_user != hash_address

  def test_pydantic_hash_consistent_after_serialization(self):
    import proxai.serializers.type_serializer as type_serializer

    response_format = types.ResponseFormat(
        type=types.ResponseFormatType.PYDANTIC,
        pydantic_class=UserModel
    )
    query_record = types.QueryRecord(
        prompt='test', response_format=response_format
    )

    hash_before = hash_serializer.get_query_record_hash(query_record)

    encoded = type_serializer.encode_query_record(query_record)
    decoded_query_record = type_serializer.decode_query_record(encoded)

    hash_after = hash_serializer.get_query_record_hash(decoded_query_record)

    assert hash_before == hash_after

  def test_pydantic_nested_hash_consistent_after_serialization(self):
    import proxai.serializers.type_serializer as type_serializer

    response_format = types.ResponseFormat(
        type=types.ResponseFormatType.PYDANTIC,
        pydantic_class=UserWithAddressModel
    )
    query_record = types.QueryRecord(
        prompt='test', response_format=response_format
    )

    hash_before = hash_serializer.get_query_record_hash(query_record)

    encoded = type_serializer.encode_query_record(query_record)
    decoded_query_record = type_serializer.decode_query_record(encoded)

    hash_after = hash_serializer.get_query_record_hash(decoded_query_record)

    assert hash_before == hash_after

  def test_pydantic_class_name_affects_hash(self):
    # Different pydantic classes with different names produce different hashes
    query_record_1 = types.QueryRecord(
        prompt='test', response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            pydantic_class=UserModel
        )
    )
    query_record_2 = types.QueryRecord(
        prompt='test', response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            pydantic_class=AddressModel
        )
    )

    hash_1 = hash_serializer.get_query_record_hash(query_record_1)
    hash_2 = hash_serializer.get_query_record_hash(query_record_2)

    assert hash_1 != hash_2


class TestChatHashing:

  def test_chat_hashing(self):
    chat = chat_session.Chat(messages=[
        message.Message(
            role=message_content.MessageRoleType.USER,
            content=[
                message_content.MessageContent(type='text', text='Hello')
            ]
        ),
        message.Message(
            role=message_content.MessageRoleType.ASSISTANT,
            content=[
                message_content.MessageContent(type='text', text='Hi there!')
            ]
        )
    ])
    query_record = types.QueryRecord(prompt='test', chat=chat)
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_chat_message_order_affects_hash(self):
    chat_1 = chat_session.Chat(messages=[
        message.Message(
            role=message_content.MessageRoleType.USER,
            content=[
                message_content.MessageContent(type='text', text='First')
            ]
        ),
        message.Message(
            role=message_content.MessageRoleType.ASSISTANT,
            content=[
                message_content.MessageContent(type='text', text='Second')
            ]
        )
    ])
    chat_2 = chat_session.Chat(messages=[
        message.Message(
            role=message_content.MessageRoleType.ASSISTANT,
            content=[
                message_content.MessageContent(type='text', text='Second')
            ]
        ),
        message.Message(
            role=message_content.MessageRoleType.USER,
            content=[
                message_content.MessageContent(type='text', text='First')
            ]
        )
    ])
    query_record_1 = types.QueryRecord(prompt='test', chat=chat_1)
    query_record_2 = types.QueryRecord(prompt='test', chat=chat_2)

    hash_1 = hash_serializer.get_query_record_hash(query_record_1)
    hash_2 = hash_serializer.get_query_record_hash(query_record_2)

    assert hash_1 != hash_2

  def test_chat_multimodal_content_hashing(self):
    chat = chat_session.Chat(messages=[
        message.Message(
            role=message_content.MessageRoleType.USER,
            content=[
                message_content.MessageContent(
                    type=message_content.ContentType.TEXT, text='Describe this'
                ),
                message_content.MessageContent(
                    type=message_content.ContentType.IMAGE,
                    source='https://example.com/img.png'
                )
            ]
        )
    ])
    query_record = types.QueryRecord(prompt='test', chat=chat)
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)


class TestParametersAffectHash:

  def test_temperature_affects_hash(self):
    qr_1 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(temperature=0.5)
    )
    qr_2 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(temperature=0.9)
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_2)
    )

  def test_max_tokens_affects_hash(self):
    qr_1 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(max_tokens=100)
    )
    qr_2 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(max_tokens=200)
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_2)
    )

  def test_n_affects_hash(self):
    qr_1 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(n=1)
    )
    qr_2 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(n=5)
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_2)
    )

  def test_thinking_affects_hash(self):
    qr_1 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(thinking=types.ThinkingType.LOW)
    )
    qr_2 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(thinking=types.ThinkingType.HIGH)
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_2)
    )

  def test_stop_affects_hash(self):
    qr_1 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(stop='.')
    )
    qr_2 = types.QueryRecord(
        prompt='test',
        parameters=types.ParameterType(stop='!')
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_2)
    )


class TestToolsAffectHash:

  def test_tools_affect_hash(self):
    qr_1 = types.QueryRecord(prompt='test')
    qr_2 = types.QueryRecord(
        prompt='test', tools=[types.Tools.WEB_SEARCH]
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_2)
    )


class TestConnectionOptionsAffectHash:

  def test_provider_model_affects_hash(self):
    model_1 = types.ProviderModelType(
        provider='openai', model='gpt-4', provider_model_identifier='gpt-4'
    )
    model_2 = types.ProviderModelType(
        provider='anthropic', model='claude', provider_model_identifier='claude'
    )
    qr_1 = types.QueryRecord(
        prompt='test', provider_model=model_1
    )
    qr_2 = types.QueryRecord(
        prompt='test', provider_model=model_2
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_2)
    )

  def test_endpoint_affects_hash(self):
    qr_1 = types.QueryRecord(
        prompt='test',
        connection_options=types.ConnectionOptions(
            endpoint='endpoint_a'
        )
    )
    qr_2 = types.QueryRecord(
        prompt='test',
        connection_options=types.ConnectionOptions(
            endpoint='endpoint_b'
        )
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_2)
    )


class TestSystemPromptAffectsHash:

  def test_system_prompt_affects_hash(self):
    qr_1 = types.QueryRecord(prompt='test', system_prompt='Be helpful.')
    qr_2 = types.QueryRecord(prompt='test', system_prompt='Be concise.')
    qr_3 = types.QueryRecord(prompt='test')
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_2)
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1) !=
        hash_serializer.get_query_record_hash(qr_3)
    )
