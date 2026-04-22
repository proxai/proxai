import os

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
      'output_format':
          types.OutputFormat(type=types.OutputFormatType.TEXT)
  }, {
      'output_format':
          types.OutputFormat(type=types.OutputFormatType.JSON)
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
      'output_format':
          types.OutputFormat(type=types.OutputFormatType.JSON),
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


class TestOutputFormatHash:

  def test_text_output_format(self):
    output_format = types.OutputFormat(type=types.OutputFormatType.TEXT)
    query_record = types.QueryRecord(
        prompt='test', output_format=output_format
    )
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_json_output_format(self):
    output_format = types.OutputFormat(type=types.OutputFormatType.JSON)
    query_record = types.QueryRecord(
        prompt='test', output_format=output_format
    )
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_pydantic_output_format(self):
    output_format = types.OutputFormat(
        type=types.OutputFormatType.PYDANTIC,
        pydantic_class=UserModel
    )
    query_record = types.QueryRecord(
        prompt='test', output_format=output_format
    )
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_nested_pydantic_output_format(self):
    output_format = types.OutputFormat(
        type=types.OutputFormatType.PYDANTIC,
        pydantic_class=UserWithAddressModel
    )
    query_record = types.QueryRecord(
        prompt='test', output_format=output_format
    )
    hash_value = hash_serializer.get_query_record_hash(query_record)

    assert len(hash_value) == hash_serializer._HASH_LENGTH
    assert hash_value == hash_serializer.get_query_record_hash(query_record)

  def test_different_output_formats_produce_different_hashes(self):
    query_record_text = types.QueryRecord(
        prompt='test', output_format=types.OutputFormat(
            type=types.OutputFormatType.TEXT
        )
    )

    query_record_json = types.QueryRecord(
        prompt='test', output_format=types.OutputFormat(
            type=types.OutputFormatType.JSON
        )
    )

    query_record_pydantic = types.QueryRecord(
        prompt='test', output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
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
        prompt='test', output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
            pydantic_class=UserModel
        )
    )

    query_record_address = types.QueryRecord(
        prompt='test', output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
            pydantic_class=AddressModel
        )
    )

    hash_user = hash_serializer.get_query_record_hash(query_record_user)
    hash_address = hash_serializer.get_query_record_hash(query_record_address)

    assert hash_user != hash_address

  def test_pydantic_hash_consistent_after_serialization(self):
    import proxai.serializers.type_serializer as type_serializer

    output_format = types.OutputFormat(
        type=types.OutputFormatType.PYDANTIC,
        pydantic_class=UserModel
    )
    query_record = types.QueryRecord(
        prompt='test', output_format=output_format
    )

    hash_before = hash_serializer.get_query_record_hash(query_record)

    encoded = type_serializer.encode_query_record(query_record)
    decoded_query_record = type_serializer.decode_query_record(encoded)

    hash_after = hash_serializer.get_query_record_hash(decoded_query_record)

    assert hash_before == hash_after

  def test_pydantic_nested_hash_consistent_after_serialization(self):
    import proxai.serializers.type_serializer as type_serializer

    output_format = types.OutputFormat(
        type=types.OutputFormatType.PYDANTIC,
        pydantic_class=UserWithAddressModel
    )
    query_record = types.QueryRecord(
        prompt='test', output_format=output_format
    )

    hash_before = hash_serializer.get_query_record_hash(query_record)

    encoded = type_serializer.encode_query_record(query_record)
    decoded_query_record = type_serializer.decode_query_record(encoded)

    hash_after = hash_serializer.get_query_record_hash(decoded_query_record)

    assert hash_before == hash_after

  def test_pydantic_class_name_affects_hash(self):
    # Different pydantic classes with different names produce different hashes
    query_record_1 = types.QueryRecord(
        prompt='test', output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
            pydantic_class=UserModel
        )
    )
    query_record_2 = types.QueryRecord(
        prompt='test', output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
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


def _user_chat(system_prompt=None):
  return chat_session.Chat(
      system_prompt=system_prompt,
      messages=[
          message.Message(
              role=message_content.MessageRoleType.USER,
              content=[
                  message_content.MessageContent(type='text', text='Hello'),
              ],
          )
      ],
  )


class TestChatSystemPromptAffectsHash:

  def test_chat_system_prompt_affects_hash(self):
    qr_1 = types.QueryRecord(chat=_user_chat(system_prompt='Be helpful.'))
    qr_2 = types.QueryRecord(chat=_user_chat(system_prompt='Be concise.'))
    qr_none = types.QueryRecord(chat=_user_chat())
    h1 = hash_serializer.get_query_record_hash(qr_1)
    h2 = hash_serializer.get_query_record_hash(qr_2)
    h_none = hash_serializer.get_query_record_hash(qr_none)
    assert h1 != h2
    assert h1 != h_none
    assert h2 != h_none

  def test_chat_system_prompt_deterministic(self):
    qr = types.QueryRecord(chat=_user_chat(system_prompt='Be helpful.'))
    h1 = hash_serializer.get_query_record_hash(qr)
    h2 = hash_serializer.get_query_record_hash(qr)
    assert h1 == h2


def _image_chat(**image_kwargs):
  return chat_session.Chat(messages=[
      message.Message(
          role=message_content.MessageRoleType.USER,
          content=[
              message_content.MessageContent(
                  type=message_content.ContentType.IMAGE, **image_kwargs
              ),
          ],
      )
  ])


def _doc_chat(**doc_kwargs):
  return chat_session.Chat(messages=[
      message.Message(
          role=message_content.MessageRoleType.USER,
          content=[
              message_content.MessageContent(
                  media_type='application/pdf', **doc_kwargs
              ),
          ],
      )
  ])


class TestMediaContentAffectsHash:

  def test_source_affects_hash(self):
    qr_1 = types.QueryRecord(
        chat=_image_chat(source='https://example.com/a.png')
    )
    qr_2 = types.QueryRecord(
        chat=_image_chat(source='https://example.com/b.png')
    )
    assert (
        hash_serializer.get_query_record_hash(qr_1)
        != hash_serializer.get_query_record_hash(qr_2)
    )

  def test_data_affects_hash(self):
    qr_1 = types.QueryRecord(chat=_image_chat(data=b'\x89PNG\r\n\x1a\nAAA'))
    qr_2 = types.QueryRecord(chat=_image_chat(data=b'\x89PNG\r\n\x1a\nBBB'))
    assert (
        hash_serializer.get_query_record_hash(qr_1)
        != hash_serializer.get_query_record_hash(qr_2)
    )

  def test_data_deterministic(self):
    payload = b'\x89PNG\r\n\x1a\nsame'
    qr_1 = types.QueryRecord(chat=_image_chat(data=payload))
    qr_2 = types.QueryRecord(chat=_image_chat(data=bytes(payload)))
    assert (
        hash_serializer.get_query_record_hash(qr_1)
        == hash_serializer.get_query_record_hash(qr_2)
    )

  def test_path_affects_hash(self, tmp_path):
    file_a = tmp_path / 'a.png'
    file_b = tmp_path / 'b.png'
    file_a.write_bytes(b'a')
    file_b.write_bytes(b'a')
    qr_a = types.QueryRecord(chat=_image_chat(path=str(file_a)))
    qr_b = types.QueryRecord(chat=_image_chat(path=str(file_b)))
    assert (
        hash_serializer.get_query_record_hash(qr_a)
        != hash_serializer.get_query_record_hash(qr_b)
    )

  def test_path_edit_invalidates_hash(self, tmp_path):
    file_path = tmp_path / 'img.png'
    file_path.write_bytes(b'first')
    qr = types.QueryRecord(chat=_image_chat(path=str(file_path)))
    hash_before = hash_serializer.get_query_record_hash(qr)

    # Append bytes so both mtime and size change.
    os.utime(file_path, (1_000_000, 1_000_000))
    hash_same_stat = hash_serializer.get_query_record_hash(qr)
    assert hash_same_stat != hash_before  # mtime differs from original

    file_path.write_bytes(b'first-and-more')
    hash_after_edit = hash_serializer.get_query_record_hash(qr)
    assert hash_after_edit != hash_same_stat

  def test_missing_path_does_not_raise(self, tmp_path):
    qr = types.QueryRecord(
        chat=_image_chat(path=str(tmp_path / 'does_not_exist.png'))
    )
    # Must not raise; falls back to hashing path string alone.
    hash_value = hash_serializer.get_query_record_hash(qr)
    assert len(hash_value) == hash_serializer._HASH_LENGTH

  def test_media_type_affects_hash(self):
    qr_png = types.QueryRecord(
        chat=_image_chat(
            source='https://example.com/file', media_type='image/png'
        )
    )
    qr_jpeg = types.QueryRecord(
        chat=_image_chat(
            source='https://example.com/file', media_type='image/jpeg'
        )
    )
    assert (
        hash_serializer.get_query_record_hash(qr_png)
        != hash_serializer.get_query_record_hash(qr_jpeg)
    )

  def test_content_type_affects_hash(self):
    image_chat = chat_session.Chat(messages=[
        message.Message(
            role=message_content.MessageRoleType.USER,
            content=[
                message_content.MessageContent(
                    type=message_content.ContentType.IMAGE,
                    source='https://example.com/file',
                ),
            ],
        )
    ])
    doc_chat = chat_session.Chat(messages=[
        message.Message(
            role=message_content.MessageRoleType.USER,
            content=[
                message_content.MessageContent(
                    type=message_content.ContentType.DOCUMENT,
                    source='https://example.com/file',
                ),
            ],
        )
    ])
    assert (
        hash_serializer.get_query_record_hash(
            types.QueryRecord(chat=image_chat)
        )
        != hash_serializer.get_query_record_hash(
            types.QueryRecord(chat=doc_chat)
        )
    )


_GEMINI_MODEL = types.ProviderModelType(
    provider='gemini', model='gemini-2.5-flash',
    provider_model_identifier='gemini-2.5-flash')
_CLAUDE_MODEL = types.ProviderModelType(
    provider='claude', model='claude-sonnet-4-6',
    provider_model_identifier='claude-sonnet-4-6')


class TestFileApiMetadataExcludedFromHash:

  def test_file_ids_excluded_for_path_content(self, tmp_path):
    file_path = tmp_path / 'doc.pdf'
    file_path.write_bytes(b'pdf-content')

    qr_without = types.QueryRecord(
        chat=_doc_chat(path=str(file_path)),
        provider_model=_GEMINI_MODEL)
    hash_without = hash_serializer.get_query_record_hash(qr_without)

    mc_with = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc123'},
        provider_file_api_status={
            'gemini': message_content.FileUploadMetadata(
                file_id='files/abc123',
                state=message_content.FileUploadState.ACTIVE)})
    qr_with = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_with])]),
        provider_model=_GEMINI_MODEL)
    hash_with = hash_serializer.get_query_record_hash(qr_with)

    assert hash_without == hash_with

  def test_file_ids_excluded_for_data_content(self):
    data = b'\x89PNG-test-data'

    qr_without = types.QueryRecord(
        chat=_image_chat(data=data),
        provider_model=_GEMINI_MODEL)
    hash_without = hash_serializer.get_query_record_hash(qr_without)

    mc_with = message_content.MessageContent(
        type='image', data=data,
        provider_file_api_ids={'gemini': 'files/abc'},
        provider_file_api_status={
            'gemini': message_content.FileUploadMetadata(
                file_id='files/abc',
                state=message_content.FileUploadState.ACTIVE)})
    qr_with = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_with])]),
        provider_model=_GEMINI_MODEL)
    hash_with = hash_serializer.get_query_record_hash(qr_with)

    assert hash_without == hash_with

  def test_file_ids_excluded_for_source_content(self):
    qr_without = types.QueryRecord(
        chat=_image_chat(source='https://example.com/img.png'),
        provider_model=_GEMINI_MODEL)
    hash_without = hash_serializer.get_query_record_hash(qr_without)

    mc_with = message_content.MessageContent(
        type='image', source='https://example.com/img.png',
        provider_file_api_ids={'gemini': 'files/abc'},
        provider_file_api_status={
            'gemini': message_content.FileUploadMetadata(
                file_id='files/abc',
                state=message_content.FileUploadState.ACTIVE)})
    qr_with = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_with])]),
        provider_model=_GEMINI_MODEL)
    hash_with = hash_serializer.get_query_record_hash(qr_with)

    assert hash_without == hash_with

  def test_different_file_ids_same_path_same_hash(self, tmp_path):
    file_path = tmp_path / 'doc.pdf'
    file_path.write_bytes(b'same-content')

    mc_1 = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})
    mc_2 = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/xyz'})

    qr_1 = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_1])]),
        provider_model=_GEMINI_MODEL)
    qr_2 = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_2])]),
        provider_model=_GEMINI_MODEL)

    assert (hash_serializer.get_query_record_hash(qr_1)
            == hash_serializer.get_query_record_hash(qr_2))

  def test_adding_provider_file_ids_does_not_change_hash(self, tmp_path):
    file_path = tmp_path / 'doc.pdf'
    file_path.write_bytes(b'content')

    mc = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf')
    qr = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc])]),
        provider_model=_GEMINI_MODEL)
    hash_before = hash_serializer.get_query_record_hash(qr)

    mc.provider_file_api_ids = {'gemini': 'files/abc'}
    mc.provider_file_api_status = {
        'gemini': message_content.FileUploadMetadata(
            file_id='files/abc',
            state=message_content.FileUploadState.ACTIVE)}
    hash_after = hash_serializer.get_query_record_hash(qr)

    assert hash_before == hash_after

  def test_multi_provider_upload_does_not_change_hash(self, tmp_path):
    file_path = tmp_path / 'doc.pdf'
    file_path.write_bytes(b'content')

    mc = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf')
    qr = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc])]),
        provider_model=_GEMINI_MODEL)

    mc.provider_file_api_ids = {'gemini': 'files/abc'}
    hash_gemini_only = hash_serializer.get_query_record_hash(qr)

    mc.provider_file_api_ids = {
        'gemini': 'files/abc', 'claude': 'file-xyz',
        'openai': 'file-123'}
    hash_all_providers = hash_serializer.get_query_record_hash(qr)

    assert hash_gemini_only == hash_all_providers

  def test_filename_excluded_from_hash(self, tmp_path):
    file_path = tmp_path / 'doc.pdf'
    file_path.write_bytes(b'content')

    mc_1 = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf',
        filename='report.pdf')
    mc_2 = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf',
        filename='document.pdf')
    mc_3 = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf')

    def _hash(mc):
      return hash_serializer.get_query_record_hash(
          types.QueryRecord(
              chat=chat_session.Chat(messages=[
                  message.Message(
                      role=message_content.MessageRoleType.USER,
                      content=[mc])]),
              provider_model=_GEMINI_MODEL))

    assert _hash(mc_1) == _hash(mc_2) == _hash(mc_3)


class TestProxDashFieldsExcludedFromHash:

  def test_proxdash_fields_excluded_for_path_content(self, tmp_path):
    file_path = tmp_path / 'doc.pdf'
    file_path.write_bytes(b'pdf-content')

    mc_without = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf')
    qr_without = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_without])]),
        provider_model=_GEMINI_MODEL)
    hash_without = hash_serializer.get_query_record_hash(qr_without)

    mc_with = message_content.MessageContent(
        path=str(file_path), media_type='application/pdf',
        proxdash_file_id='clxyz123',
        proxdash_file_status=message_content.ProxDashFileStatus(
            file_id='clxyz123',
            s3_key='files/user1/clxyz123',
            upload_confirmed=True))
    qr_with = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_with])]),
        provider_model=_GEMINI_MODEL)
    hash_with = hash_serializer.get_query_record_hash(qr_with)

    assert hash_without == hash_with

  def test_proxdash_fields_excluded_for_remote_only_content(self):
    mc_without = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})
    qr_without = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_without])]),
        provider_model=_GEMINI_MODEL)
    hash_without = hash_serializer.get_query_record_hash(qr_without)

    mc_with = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'},
        proxdash_file_id='clxyz123',
        proxdash_file_status=message_content.ProxDashFileStatus(
            file_id='clxyz123',
            upload_confirmed=True))
    qr_with = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_with])]),
        provider_model=_GEMINI_MODEL)
    hash_with = hash_serializer.get_query_record_hash(qr_with)

    assert hash_without == hash_with


class TestRemoteOnlyContentHash:

  def test_remote_only_uses_provider_file_id(self):
    mc = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})
    qr = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc])]),
        provider_model=_GEMINI_MODEL)
    hash_value = hash_serializer.get_query_record_hash(qr)
    assert len(hash_value) == hash_serializer._HASH_LENGTH

  def test_remote_only_different_files_different_hash(self):
    mc_1 = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})
    mc_2 = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/xyz'})

    def _hash(mc):
      return hash_serializer.get_query_record_hash(
          types.QueryRecord(
              chat=chat_session.Chat(messages=[
                  message.Message(
                      role=message_content.MessageRoleType.USER,
                      content=[mc])]),
              provider_model=_GEMINI_MODEL))

    assert _hash(mc_1) != _hash(mc_2)

  def test_remote_only_same_file_same_hash(self):
    mc_1 = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})
    mc_2 = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})

    def _hash(mc):
      return hash_serializer.get_query_record_hash(
          types.QueryRecord(
              chat=chat_session.Chat(messages=[
                  message.Message(
                      role=message_content.MessageRoleType.USER,
                      content=[mc])]),
              provider_model=_GEMINI_MODEL))

    assert _hash(mc_1) == _hash(mc_2)

  def test_remote_only_adding_second_provider_stable(self):
    mc = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})
    qr = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc])]),
        provider_model=_GEMINI_MODEL)
    hash_before = hash_serializer.get_query_record_hash(qr)

    mc.provider_file_api_ids['claude'] = 'file-xyz'
    hash_after = hash_serializer.get_query_record_hash(qr)

    assert hash_before == hash_after

  def test_remote_only_different_provider_different_hash(self):
    mc_gemini = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})
    mc_claude = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'claude': 'file-xyz'})

    qr_gemini = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_gemini])]),
        provider_model=_GEMINI_MODEL)
    qr_claude = types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc_claude])]),
        provider_model=_CLAUDE_MODEL)

    assert (hash_serializer.get_query_record_hash(qr_gemini)
            != hash_serializer.get_query_record_hash(qr_claude))


class TestStringMessageContent:
  """Cover the isinstance(msg.content, str) branch in _hash_chat.

  Message.content is typed str | list[MessageContent | str]; bare strings
  are NOT auto-wrapped by Message.__post_init__, so the hash path at
  hash_serializer.py:96-98 is the one that runs for them.
  """

  def _chat_with_string_content(self, text):
    return chat_session.Chat(messages=[
        message.Message(
            role=message_content.MessageRoleType.USER,
            content=text,
        )
    ])

  def test_string_content_deterministic(self):
    qr = types.QueryRecord(chat=self._chat_with_string_content('Hello'))
    h1 = hash_serializer.get_query_record_hash(qr)
    h2 = hash_serializer.get_query_record_hash(qr)
    assert h1 == h2
    assert len(h1) == hash_serializer._HASH_LENGTH

  def test_string_content_affects_hash(self):
    qr_1 = types.QueryRecord(chat=self._chat_with_string_content('Hello'))
    qr_2 = types.QueryRecord(chat=self._chat_with_string_content('Goodbye'))
    assert (
        hash_serializer.get_query_record_hash(qr_1)
        != hash_serializer.get_query_record_hash(qr_2)
    )

  def test_string_content_differs_from_list_content(self):
    # Same text, different content shapes (bare str vs. [MessageContent]) —
    # the hash takes different branches, so the hashes must differ.
    qr_str = types.QueryRecord(chat=self._chat_with_string_content('Hello'))
    qr_list = types.QueryRecord(chat=chat_session.Chat(messages=[
        message.Message(
            role=message_content.MessageRoleType.USER,
            content=[
                message_content.MessageContent(type='text', text='Hello')
            ],
        )
    ]))
    assert (
        hash_serializer.get_query_record_hash(qr_str)
        != hash_serializer.get_query_record_hash(qr_list)
    )


class TestRemoteOnlyFileIdsFallback:
  """Cover the `elif file_ids` branch in _content_hash_dict.

  When remote-only content has provider_file_api_ids but the query's
  provider is None or not present in the ids dict, _content_hash_dict
  stores the full dict under 'provider_file_ids' instead of a single
  'provider_file_id'.
  """

  def _qr(self, mc, provider_model):
    return types.QueryRecord(
        chat=chat_session.Chat(messages=[
            message.Message(
                role=message_content.MessageRoleType.USER,
                content=[mc])]),
        provider_model=provider_model)

  def test_remote_only_no_provider_model_uses_all_ids(self):
    # provider_model=None → provider resolves to None → `elif file_ids`
    # branch fires and embeds the full dict.
    mc = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})
    qr = self._qr(mc, provider_model=None)
    hash_value = hash_serializer.get_query_record_hash(qr)
    assert len(hash_value) == hash_serializer._HASH_LENGTH

  def test_remote_only_no_provider_model_different_ids_differ(self):
    mc_1 = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/abc'})
    mc_2 = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'gemini': 'files/xyz'})
    assert (
        hash_serializer.get_query_record_hash(self._qr(mc_1, None))
        != hash_serializer.get_query_record_hash(self._qr(mc_2, None))
    )

  def test_remote_only_cross_provider_uses_all_ids(self):
    # Query targets gemini, but the content only has claude's file id.
    # Provider is not in file_ids → `elif file_ids` branch fires.
    mc = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'claude': 'file-xyz'})
    qr = self._qr(mc, provider_model=_GEMINI_MODEL)
    hash_value = hash_serializer.get_query_record_hash(qr)
    assert len(hash_value) == hash_serializer._HASH_LENGTH

  def test_remote_only_cross_provider_adding_ids_changes_hash(self):
    # In the fallback branch the entire file_ids dict is part of the
    # hash identity, so adding a second provider's id MUST change it.
    # This is the behavioral difference from the `provider in file_ids`
    # branch exercised by test_remote_only_adding_second_provider_stable.
    mc_1 = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'claude': 'file-xyz'})
    mc_2 = message_content.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={
            'claude': 'file-xyz', 'openai': 'file-123'})
    assert (
        hash_serializer.get_query_record_hash(
            self._qr(mc_1, provider_model=_GEMINI_MODEL))
        != hash_serializer.get_query_record_hash(
            self._qr(mc_2, provider_model=_GEMINI_MODEL))
    )


class TestConnectionOptionsExclusion:
  """Only ConnectionOptions.endpoint contributes to the hash.

  fallback_models, suppress_provider_errors, skip_cache, and
  override_cache_value are intentionally excluded — the cache identity
  is the logical request, not per-call retry/cache-policy knobs. If
  anyone accidentally includes one of them in _hash_connection_options,
  these tests fail.
  """

  def _hash(self, **connection_kwargs):
    return hash_serializer.get_query_record_hash(
        types.QueryRecord(
            prompt='test',
            connection_options=types.ConnectionOptions(**connection_kwargs),
        )
    )

  def test_fallback_models_excluded(self):
    model_a = types.ProviderModelType(
        provider='openai', model='gpt-4', provider_model_identifier='gpt-4')
    model_b = types.ProviderModelType(
        provider='claude', model='opus',
        provider_model_identifier='claude-opus')
    assert self._hash() == self._hash(fallback_models=[model_a])
    assert self._hash(fallback_models=[model_a]) == (
        self._hash(fallback_models=[model_a, model_b])
    )

  def test_suppress_provider_errors_excluded(self):
    assert self._hash() == self._hash(suppress_provider_errors=True)
    assert self._hash(suppress_provider_errors=True) == (
        self._hash(suppress_provider_errors=False)
    )

  def test_skip_cache_excluded(self):
    assert self._hash() == self._hash(skip_cache=True)
    assert self._hash(skip_cache=True) == self._hash(skip_cache=False)

  def test_override_cache_value_excluded(self):
    assert self._hash() == self._hash(override_cache_value=True)
    assert self._hash(override_cache_value=True) == (
        self._hash(override_cache_value=False)
    )
