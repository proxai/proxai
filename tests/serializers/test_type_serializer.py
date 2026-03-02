import datetime

import pydantic
import pytest

import proxai.chat.chat_session as chat_session
import proxai.chat.message as message
import proxai.chat.message_content as message_content
import proxai.serializers.hash_serializer as hash_serializer
import proxai.serializers.type_serializer as type_serializer
import proxai.types as types


_MODEL_1 = types.ProviderModelType(
    provider='openai', model='gpt-4', provider_model_identifier='gpt-4'
)
_MODEL_2 = types.ProviderModelType(
    provider='openai', model='o3-mini', provider_model_identifier='o3-mini'
)
_MODEL_3 = types.ProviderModelType(
    provider='claude', model='opus-4',
    provider_model_identifier='claude-opus-4'
)
_MODEL_4 = types.ProviderModelType(
    provider='claude', model='sonnet-4',
    provider_model_identifier='claude-sonnet-4'
)
_MODEL_5 = types.ProviderModelType(
    provider='openai', model='gpt-4o-mini',
    provider_model_identifier='gpt-4o-mini'
)


def _get_provider_model_type_options():
  return [
      {
          'provider': 'openai',
          'model': 'gpt-4',
          'provider_model_identifier': 'gpt-4'
      },
  ]



def _get_provider_model_pricing_type_options():
  return [
      {},
      {
          'input_token_cost': 0.001
      },
      {
          'output_token_cost': 0.002
      },
      {
          'input_token_cost': 0.001,
          'output_token_cost': 0.002
      },
      {
          'input_token_cost': 0.0,
          'output_token_cost': 0.0
      },
      {
          'input_token_cost': 1.5,
          'output_token_cost': 0.5
      },
  ]


def _get_feature_support_type_options():
  return [
      types.FeatureSupportType.SUPPORTED,
      types.FeatureSupportType.BEST_EFFORT,
      types.FeatureSupportType.NOT_SUPPORTED,
  ]


def _get_parameter_config_type_options():
  return [
      {},
      {
          'temperature': types.FeatureSupportType.SUPPORTED
      },
      {
          'max_tokens': types.FeatureSupportType.BEST_EFFORT
      },
      {
          'stop': types.FeatureSupportType.NOT_SUPPORTED
      },
      {
          'n': types.FeatureSupportType.SUPPORTED
      },
      {
          'thinking': types.FeatureSupportType.BEST_EFFORT
      },
      {
          'temperature': types.FeatureSupportType.SUPPORTED,
          'max_tokens': types.FeatureSupportType.SUPPORTED,
          'stop': types.FeatureSupportType.BEST_EFFORT,
          'n': types.FeatureSupportType.NOT_SUPPORTED,
          'thinking': types.FeatureSupportType.SUPPORTED
      },
  ]


def _get_tool_config_type_options():
  return [
      {},
      {
          'web_search': types.FeatureSupportType.SUPPORTED
      },
      {
          'web_search': types.FeatureSupportType.NOT_SUPPORTED
      },
  ]


def _get_response_format_config_type_options():
  return [
      {},
      {
          'text': types.FeatureSupportType.SUPPORTED
      },
      {
          'image': types.FeatureSupportType.NOT_SUPPORTED
      },
      {
          'json': types.FeatureSupportType.SUPPORTED
      },
      {
          'pydantic': types.FeatureSupportType.BEST_EFFORT
      },
      {
          'text': types.FeatureSupportType.SUPPORTED,
          'image': types.FeatureSupportType.NOT_SUPPORTED,
          'audio': types.FeatureSupportType.NOT_SUPPORTED,
          'video': types.FeatureSupportType.NOT_SUPPORTED,
          'json': types.FeatureSupportType.SUPPORTED,
          'pydantic': types.FeatureSupportType.BEST_EFFORT,
          'multi_modal': types.FeatureSupportType.NOT_SUPPORTED
      },
  ]


def _get_feature_config_type_options():
  return [
      {},
      {
          'prompt': types.FeatureSupportType.SUPPORTED
      },
      {
          'messages': types.FeatureSupportType.SUPPORTED
      },
      {
          'system_prompt': types.FeatureSupportType.BEST_EFFORT
      },
      {
          'add_system_to_messages': True
      },
      {
          'add_system_to_messages': False
      },
      {
          'parameters': types.ParameterConfigType(
              temperature=types.FeatureSupportType.SUPPORTED
          )
      },
      {
          'tools': types.ToolConfigType(
              web_search=types.FeatureSupportType.SUPPORTED
          )
      },
      {
          'response_format': types.ResponseFormatConfigType(
              text=types.FeatureSupportType.SUPPORTED,
              json=types.FeatureSupportType.SUPPORTED
          )
      },
      {
          'prompt': types.FeatureSupportType.SUPPORTED,
          'messages': types.FeatureSupportType.SUPPORTED,
          'system_prompt': types.FeatureSupportType.BEST_EFFORT,
          'add_system_to_messages': True,
          'parameters': types.ParameterConfigType(
              temperature=types.FeatureSupportType.SUPPORTED,
              max_tokens=types.FeatureSupportType.SUPPORTED,
              stop=types.FeatureSupportType.BEST_EFFORT,
              n=types.FeatureSupportType.NOT_SUPPORTED,
              thinking=types.FeatureSupportType.SUPPORTED
          ),
          'tools': types.ToolConfigType(
              web_search=types.FeatureSupportType.SUPPORTED
          ),
          'response_format': types.ResponseFormatConfigType(
              text=types.FeatureSupportType.SUPPORTED,
              image=types.FeatureSupportType.NOT_SUPPORTED,
              audio=types.FeatureSupportType.NOT_SUPPORTED,
              video=types.FeatureSupportType.NOT_SUPPORTED,
              json=types.FeatureSupportType.SUPPORTED,
              pydantic=types.FeatureSupportType.BEST_EFFORT,
              multi_modal=types.FeatureSupportType.NOT_SUPPORTED
          )
      },
  ]


def _get_provider_model_metadata_type_options():
  return [
      {},
      {
          'call_type': types.CallType.TEXT
      },
      {
          'is_recommended': True
      },
      {
          'is_recommended': False
      },
      {
          'model_size_tags': []
      },
      {
          'model_size_tags': [types.ModelSizeType.SMALL]
      },
      {
          'model_size_tags': [types.ModelSizeType.MEDIUM]
      },
      {
          'model_size_tags': [types.ModelSizeType.LARGE]
      },
      {
          'model_size_tags': [types.ModelSizeType.LARGEST]
      },
      {
          'model_size_tags': [
              types.ModelSizeType.SMALL, types.ModelSizeType.MEDIUM
          ]
      },
      {
          'model_size_tags': [
              types.ModelSizeType.LARGE, types.ModelSizeType.LARGEST
          ]
      },
      {
          'tags': []
      },
      {
          'tags': ['tag1']
      },
      {
          'tags': ['tag1', 'tag2', 'tag3']
      },
      {
          'call_type': types.CallType.TEXT,
          'is_recommended': True,
          'model_size_tags': [types.ModelSizeType.LARGE],
          'tags': ['tag1', 'tag2']
      },
  ]


def _get_provider_model_config_options():
  return [
      {
          'provider_model': _MODEL_1,
          'pricing': types.ProviderModelPricingType(),
          'features': types.FeatureConfigType(),
          'metadata': types.ProviderModelMetadataType()
      },
      {
          'provider_model': _MODEL_1,
          'pricing':
              types.ProviderModelPricingType(
                  input_token_cost=0.001, output_token_cost=0.002
              ),
          'features': types.FeatureConfigType(),
          'metadata': types.ProviderModelMetadataType()
      },
      {
          'provider_model': _MODEL_1,
          'pricing': types.ProviderModelPricingType(),
          'features': types.FeatureConfigType(
              prompt=types.FeatureSupportType.SUPPORTED
          ),
          'metadata': types.ProviderModelMetadataType()
      },
      {
          'provider_model': _MODEL_1,
          'pricing': types.ProviderModelPricingType(),
          'features': types.FeatureConfigType(),
          'metadata':
              types.ProviderModelMetadataType(
                  call_type=types.CallType.TEXT, is_recommended=True
              )
      },
      {
          'provider_model': _MODEL_3,
          'pricing':
              types.ProviderModelPricingType(
                  input_token_cost=0.003, output_token_cost=0.001
              ),
          'features': types.FeatureConfigType(
              prompt=types.FeatureSupportType.SUPPORTED,
              messages=types.FeatureSupportType.SUPPORTED,
              system_prompt=types.FeatureSupportType.BEST_EFFORT,
              parameters=types.ParameterConfigType(
                  temperature=types.FeatureSupportType.SUPPORTED,
                  max_tokens=types.FeatureSupportType.SUPPORTED
              )
          ),
          'metadata':
              types.ProviderModelMetadataType(
                  call_type=types.CallType.TEXT, is_recommended=True,
                  model_size_tags=[types.ModelSizeType.LARGEST],
                  tags=['production', 'recommended']
              )
      },
  ]


def _get_model_configs_schema_metadata_type_options():
  return [
      {},
      {
          'version': '1.0.0'
      },
      {
          'released_at': datetime.datetime.now(datetime.timezone.utc)
      },
      {
          'min_proxai_version': '>=1.8.12'
      },
      {
          'config_origin': types.ConfigOriginType.BUILT_IN
      },
      {
          'config_origin': types.ConfigOriginType.PROXDASH
      },
      {
          'release_notes': 'Initial release'
      },
      {
          'version': '1.0.0',
          'released_at': datetime.datetime.now(datetime.timezone.utc),
          'config_origin': types.ConfigOriginType.BUILT_IN,
          'release_notes': 'Initial release'
      },
      {
          'version': '2.1.0',
          'released_at': datetime.datetime.now(datetime.timezone.utc),
          'min_proxai_version': '>=2.0.0',
          'config_origin': types.ConfigOriginType.PROXDASH,
          'release_notes': 'Added new models and updated pricing'
      },
  ]


def _get_model_registry_options():
  _default_config = types.ProviderModelConfig(
      provider_model=_MODEL_1,
      pricing=types.ProviderModelPricingType(),
      features=types.FeatureConfigType(),
      metadata=types.ProviderModelMetadataType()
  )
  return [{
      'metadata': types.ModelConfigsSchemaMetadataType(version='1.0.0'),
      'default_model_priority_list': [_MODEL_1],
      'provider_model_configs': {
          'openai': {
              'gpt-4': _default_config
          }
      }
  }, {
      'metadata':
          types.ModelConfigsSchemaMetadataType(
              version='1.0.0',
              released_at=datetime.datetime.now(datetime.timezone.utc),
              config_origin=types.ConfigOriginType.BUILT_IN,
              release_notes='Initial release'
          ),
      'default_model_priority_list': [_MODEL_1],
      'provider_model_configs': {
          'openai': {
              'gpt-4': _default_config
          }
      }
  }, {
      'metadata': types.ModelConfigsSchemaMetadataType(version='1.0.0'),
      'default_model_priority_list': [_MODEL_1, _MODEL_3, _MODEL_2],
      'provider_model_configs': {
          'openai': {
              'gpt-4':
                  types.ProviderModelConfig(
                      provider_model=_MODEL_1,
                      pricing=types.ProviderModelPricingType(
                          input_token_cost=0.001, output_token_cost=0.002
                      ),
                      features=types.FeatureConfigType(
                          prompt=types.FeatureSupportType.SUPPORTED,
                          messages=types.FeatureSupportType.SUPPORTED
                      ),
                      metadata=types.ProviderModelMetadataType(
                          call_type=types.CallType.TEXT, is_recommended=True,
                          model_size_tags=[types.ModelSizeType.LARGE]
                      )
                  )
          }
      }
  }, {
      'metadata':
          types.ModelConfigsSchemaMetadataType(
              version='2.1.0',
              released_at=datetime.datetime.now(datetime.timezone.utc),
              min_proxai_version='>=2.0.0',
              config_origin=types.ConfigOriginType.PROXDASH,
              release_notes='Added new models and updated pricing'
          ),
      'default_model_priority_list': [_MODEL_1, _MODEL_3, _MODEL_2],
      'provider_model_configs': {
          'openai': {
              'gpt-4':
                  types.ProviderModelConfig(
                      provider_model=_MODEL_1,
                      pricing=types.ProviderModelPricingType(
                          input_token_cost=0.001, output_token_cost=0.002
                      ),
                      features=types.FeatureConfigType(
                          prompt=types.FeatureSupportType.SUPPORTED,
                          messages=types.FeatureSupportType.SUPPORTED,
                          parameters=types.ParameterConfigType(
                              temperature=types.FeatureSupportType.SUPPORTED,
                              max_tokens=types.FeatureSupportType.SUPPORTED
                          )
                      ),
                      metadata=types.ProviderModelMetadataType(
                          call_type=types.CallType.TEXT, is_recommended=True,
                          model_size_tags=[types.ModelSizeType.LARGE]
                      )
                  ),
              'o3-mini':
                  types.ProviderModelConfig(
                      provider_model=_MODEL_2,
                      pricing=types.ProviderModelPricingType(),
                      features=types.FeatureConfigType(),
                      metadata=types.ProviderModelMetadataType()
                  )
          },
          'claude': {
              'opus-4':
                  types.ProviderModelConfig(
                      provider_model=_MODEL_3,
                      pricing=types.ProviderModelPricingType(),
                      features=types.FeatureConfigType(),
                      metadata=types.ProviderModelMetadataType(
                          is_recommended=True,
                          model_size_tags=[types.ModelSizeType.LARGEST]
                      )
                  ),
              'sonnet-4':
                  types.ProviderModelConfig(
                      provider_model=_MODEL_4,
                      pricing=types.ProviderModelPricingType(),
                      features=types.FeatureConfigType(),
                      metadata=types.ProviderModelMetadataType()
                  )
          }
      }
  }]


def _get_parameter_type_options():
  return [
      {},
      {
          'temperature': 0.5
      },
      {
          'max_tokens': 100
      },
      {
          'stop': 'stop_word'
      },
      {
          'stop': ['stop1', 'stop2']
      },
      {
          'n': 3
      },
      {
          'thinking': types.ThinkingType.LOW
      },
      {
          'thinking': types.ThinkingType.MEDIUM
      },
      {
          'thinking': types.ThinkingType.HIGH
      },
      {
          'temperature': 0.7,
          'max_tokens': 200,
          'stop': ['stop1'],
          'n': 2,
          'thinking': types.ThinkingType.HIGH
      },
  ]


def _get_connection_options_options():
  return [
      {},
      {
          'fallback_models': [_MODEL_1]
      },
      {
          'fallback_models': [_MODEL_1, _MODEL_3]
      },
      {
          'suppress_provider_errors': True
      },
      {
          'suppress_provider_errors': False
      },
      {
          'endpoint': 'some_endpoint'
      },
      {
          'skip_cache': True
      },
      {
          'override_cache_value': True
      },
      {
          'fallback_models': [_MODEL_1, _MODEL_2],
          'suppress_provider_errors': True,
          'endpoint': 'test_endpoint',
          'skip_cache': True,
          'override_cache_value': False
      },
  ]


class _UserModel(pydantic.BaseModel):
  name: str
  age: int


class _AddressModel(pydantic.BaseModel):
  street: str
  city: str
  country: str


class _UserWithAddressModel(pydantic.BaseModel):
  name: str
  email: str | None = None
  address: _AddressModel
  tags: list[str] = []


def _get_response_format_options():
  return [
      {
          'type': types.ResponseFormatType.TEXT
      },
      {
          'type': types.ResponseFormatType.IMAGE
      },
      {
          'type': types.ResponseFormatType.JSON
      },
      {
          'type': types.ResponseFormatType.PYDANTIC
      },
      {
          'type': types.ResponseFormatType.PYDANTIC,
          'pydantic_class': _UserModel
      },
      {
          'type': types.ResponseFormatType.PYDANTIC,
          'pydantic_class': _UserWithAddressModel
      },
  ]


def _get_choice_type_options():
  return [
      {},
      {
          'output_text': 'Hello, world!'
      },
      {
          'output_image':
              message_content.MessageContent(
                  type=message_content.ContentType.IMAGE,
                  source='https://example.com/img.png'
              )
      },
      {
          'output_audio':
              message_content.MessageContent(
                  type=message_content.ContentType.AUDIO,
                  source='https://example.com/audio.mp3'
              )
      },
      {
          'output_video':
              message_content.MessageContent(
                  type=message_content.ContentType.VIDEO,
                  source='https://example.com/video.mp4'
              )
      },
      {
          'output_json': {'key': 'value'}
      },
      {
          'output_json': {'name': 'test', 'items': [1, 2, 3]}
      },
      {
          'content': 'Hello, world!'
      },
      {
          'content': ['text1', 'text2']
      },
      {
          'content': [
              message_content.MessageContent(
                  type=message_content.ContentType.TEXT, text='hello'
              )
          ]
      },
      {
          'content': [
              'plain text',
              message_content.MessageContent(
                  type=message_content.ContentType.TEXT, text='rich text'
              )
          ]
      },
      {
          'output_text': 'text result',
          'output_json': {'key': 'value'},
          'content': [
              message_content.MessageContent(
                  type=message_content.ContentType.TEXT, text='hello'
              )
          ]
      },
  ]


def _get_usage_type_options():
  return [
      {},
      {
          'input_tokens': 100
      },
      {
          'output_tokens': 200
      },
      {
          'total_tokens': 300
      },
      {
          'estimated_cost': 50
      },
      {
          'input_tokens': 100,
          'output_tokens': 200,
          'total_tokens': 300,
          'estimated_cost': 50
      },
  ]


def _get_tool_usage_type_options():
  return [
      {},
      {
          'web_search_count': 3
      },
      {
          'web_search_citations': ['https://example.com']
      },
      {
          'web_search_count': 5,
          'web_search_citations': [
              'https://example.com', 'https://test.com'
          ]
      },
  ]


def _get_timestamp_type_options():
  return [
      {},
      {
          'start_utc_date': datetime.datetime.now(datetime.timezone.utc)
      },
      {
          'end_utc_date': datetime.datetime.now(datetime.timezone.utc)
      },
      {
          'local_time_offset_minute': -300
      },
      {
          'response_time': datetime.timedelta(seconds=1.5)
      },
      {
          'cache_response_time': datetime.timedelta(seconds=0.01)
      },
      {
          'start_utc_date': datetime.datetime.now(datetime.timezone.utc),
          'end_utc_date': datetime.datetime.now(datetime.timezone.utc),
          'local_time_offset_minute': -300,
          'response_time': datetime.timedelta(seconds=2.0),
          'cache_response_time': datetime.timedelta(seconds=0.05)
      },
  ]


def _get_result_record_options():
  return [
      {
          'status': types.ResultStatusType.SUCCESS
      },
      {
          'status': types.ResultStatusType.FAILED
      },
      {
          'role': types.MessageRoleType.ASSISTANT
      },
      {
          'output_text': 'Hello, world!'
      },
      {
          'output_image': b'image_data_bytes'
      },
      {
          'output_audio': b'audio_data_bytes'
      },
      {
          'output_video': b'video_data_bytes'
      },
      {
          'output_json': {'key': 'value'}
      },
      {
          'output_json': {'name': 'test', 'items': [1, 2, 3]}
      },
      {
          'content': 'Hello, world!'
      },
      {
          'content': [
              'text part',
              message_content.MessageContent(
                  type=message_content.ContentType.TEXT, text='rich part'
              )
          ]
      },
      {
          'choices': [types.ChoiceType(content='choice 1')]
      },
      {
          'choices': [
              types.ChoiceType(content='choice 1'),
              types.ChoiceType(content='choice 2')
          ]
      },
      {
          'error': 'Error message'
      },
      {
          'error_traceback': 'Traceback (most recent call last):\n  File...'
      },
      {
          'usage': types.UsageType(input_tokens=100, output_tokens=200)
      },
      {
          'tool_usage': types.ToolUsageType(web_search_count=3)
      },
      {
          'timestamp':
              types.TimeStampType(
                  start_utc_date=datetime.datetime.now(datetime.timezone.utc),
                  response_time=datetime.timedelta(seconds=1)
              )
      },
      {
          'status': types.ResultStatusType.SUCCESS,
          'role': types.MessageRoleType.ASSISTANT,
          'output_text': 'Full response',
          'output_image': b'img_bytes',
          'output_json': {'result': 'ok'},
          'content': [
              message_content.MessageContent(
                  type=message_content.ContentType.TEXT, text='hello'
              )
          ],
          'choices': [types.ChoiceType(content='choice 1')],
          'usage':
              types.UsageType(
                  input_tokens=100, output_tokens=200,
                  total_tokens=300, estimated_cost=50
              ),
          'tool_usage':
              types.ToolUsageType(
                  web_search_count=2,
                  web_search_citations=['https://example.com']
              ),
          'timestamp':
              types.TimeStampType(
                  start_utc_date=datetime.datetime.now(datetime.timezone.utc),
                  end_utc_date=datetime.datetime.now(datetime.timezone.utc),
                  local_time_offset_minute=-300,
                  response_time=datetime.timedelta(seconds=1.5),
                  cache_response_time=datetime.timedelta(seconds=0.01)
              )
      },
  ]


def _get_query_record_options():
  return [
      {
          'prompt': 'Hello, world!'
      },
      {
          'chat':
              chat_session.Chat(messages=[
                  message.Message(
                      role=message_content.MessageRoleType.USER,
                      content=[
                          message_content.MessageContent(
                              type='text', text='Hello'
                          )
                      ]
                  )
              ])
      },
      {
          'system_prompt': 'You are a helpful assistant.'
      },
      {
          'provider_model': _MODEL_1
      },
      {
          'provider_model': _MODEL_3
      },
      {
          'parameters': types.ParameterType(temperature=0.5)
      },
      {
          'parameters': types.ParameterType(max_tokens=100)
      },
      {
          'tools': [types.Tools.WEB_SEARCH]
      },
      {
          'response_format':
              types.ResponseFormat(type=types.ResponseFormatType.TEXT)
      },
      {
          'response_format':
              types.ResponseFormat(type=types.ResponseFormatType.JSON)
      },
      {
          'response_format':
              types.ResponseFormat(type=types.ResponseFormatType.PYDANTIC)
      },
      {
          'connection_options':
              types.ConnectionOptions(
                  fallback_models=[_MODEL_1],
                  suppress_provider_errors=True,
                  endpoint='some_endpoint'
              )
      },
      {
          'hash_value': 'some_hash_value'
      },
      {
          'prompt': 'Hello, world!',
          'chat':
              chat_session.Chat(messages=[
                  message.Message(
                      role=message_content.MessageRoleType.USER,
                      content=[
                          message_content.MessageContent(
                              type='text', text='Hi'
                          )
                      ]
                  ),
                  message.Message(
                      role=message_content.MessageRoleType.ASSISTANT,
                      content=[
                          message_content.MessageContent(
                              type='text', text='Hello!'
                          )
                      ]
                  )
              ]),
          'system_prompt': 'Be helpful.',
          'provider_model': _MODEL_1,
          'parameters':
              types.ParameterType(
                  temperature=0.7, max_tokens=200, stop=['stop1'], n=2
              ),
          'tools': [types.Tools.WEB_SEARCH],
          'response_format':
              types.ResponseFormat(type=types.ResponseFormatType.TEXT),
          'connection_options':
              types.ConnectionOptions(
                  fallback_models=[_MODEL_1, _MODEL_2],
                  suppress_provider_errors=True,
                  endpoint='test_endpoint',
                  skip_cache=False,
                  override_cache_value=True
              ),
          'hash_value': 'test_hash'
      },
  ]


def _get_connection_metadata_options():
  return [
      {},
      {
          'result_source': types.ResultSource.CACHE
      },
      {
          'result_source': types.ResultSource.PROVIDER
      },
      {
          'cache_hit': True
      },
      {
          'cache_hit': False
      },
      {
          'cache_look_fail_reason':
              types.CacheLookFailReason.CACHE_NOT_FOUND
      },
      {
          'cache_look_fail_reason':
              types.CacheLookFailReason.UNIQUE_RESPONSE_LIMIT_NOT_REACHED
      },
      {
          'endpoint_used': 'some_endpoint'
      },
      {
          'failed_fallback_models': [_MODEL_1]
      },
      {
          'failed_fallback_models': [_MODEL_1, _MODEL_3]
      },
      {
          'feature_mapping_strategy': types.FeatureMappingStrategy.BEST_EFFORT
      },
      {
          'feature_mapping_strategy': types.FeatureMappingStrategy.STRICT
      },
      {
          'result_source': types.ResultSource.CACHE,
          'cache_hit': True
      },
      {
          'result_source': types.ResultSource.PROVIDER,
          'cache_hit': False,
          'cache_look_fail_reason':
              types.CacheLookFailReason.CACHE_NOT_MATCHED,
          'endpoint_used': 'test_endpoint',
          'failed_fallback_models': [_MODEL_2, _MODEL_4],
          'feature_mapping_strategy': types.FeatureMappingStrategy.STRICT
      },
  ]


def _get_call_record_options():
  return [
      {
          'query': types.QueryRecord(prompt='test')
      },
      {
          'result':
              types.ResultRecord(
                  status=types.ResultStatusType.SUCCESS,
                  content='Hello'
              )
      },
      {
          'connection':
              types.ConnectionMetadata(
                  result_source=types.ResultSource.CACHE, cache_hit=True
              )
      },
      {
          'connection':
              types.ConnectionMetadata(
                  result_source=types.ResultSource.PROVIDER,
                  cache_hit=False,
                  endpoint_used='test_endpoint',
                  failed_fallback_models=[_MODEL_2],
                  feature_mapping_strategy=(
                      types.FeatureMappingStrategy.BEST_EFFORT
                  )
              )
      },
      {
          'query': types.QueryRecord(prompt='test'),
          'result':
              types.ResultRecord(
                  status=types.ResultStatusType.SUCCESS,
                  content='Hello'
              ),
          'connection':
              types.ConnectionMetadata(
                  result_source=types.ResultSource.CACHE, cache_hit=True
              )
      },
  ]


def _get_cache_record_options():
  return [
      {
          'query': types.QueryRecord(prompt='test')
      },
      {
          'results': [
              types.ResultRecord(
                  status=types.ResultStatusType.SUCCESS,
                  content='Hello, world!'
              )
          ]
      },
      {
          'shard_id': 0
      },
      {
          'shard_id': 'backlog'
      },
      {
          'last_access_time': datetime.datetime.now()
      },
      {
          'call_count': 1
      },
      {
          'query': types.QueryRecord(prompt='test'),
          'results': [
              types.ResultRecord(
                  status=types.ResultStatusType.SUCCESS,
                  content='Hello, world!'
              )
          ],
          'shard_id': 0,
          'last_access_time': datetime.datetime.now(),
          'call_count': 1
      },
  ]


def _get_light_cache_record_options():
  return [
      {
          'query_hash': 'hash_value'
      },
      {
          'results_count': 1
      },
      {
          'shard_id': 0
      },
      {
          'last_access_time': datetime.datetime.now()
      },
      {
          'call_count': 1
      },
      {
          'query_hash': 'hash_value',
          'results_count': 1,
          'shard_id': 0,
          'last_access_time': datetime.datetime.now(),
          'call_count': 1
      },
  ]


def _get_logging_options_options():
  return [
      {},
      {
          'logging_path': 'logging_path'
      },
      {
          'stdout': True
      },
      {
          'hide_sensitive_content': True
      },
  ]


def _get_cache_options_options():
  return [
      {},
      {
          'cache_path': 'cache_path'
      },
      {
          'unique_response_limit': 1
      },
      {
          'retry_if_error_cached': True
      },
      {
          'clear_query_cache_on_connect': True
      },
      {
          'clear_model_cache_on_connect': True
      },
      {
          'disable_model_cache': True
      },
      {
          'model_cache_duration': 3600
      },
      {
          'cache_path': 'cache_path',
          'unique_response_limit': 5,
          'retry_if_error_cached': True,
          'clear_query_cache_on_connect': True,
          'clear_model_cache_on_connect': True,
          'disable_model_cache': False,
          'model_cache_duration': 7200
      },
  ]


def _get_proxdash_options_options():
  return [
      {},
      {
          'stdout': True
      },
      {
          'hide_sensitive_content': True
      },
      {
          'disable_proxdash': True
      },
      {
          'api_key': 'test_api_key'
      },
      {
          'base_url': 'https://test.example.com'
      },
      {
          'stdout': True,
          'hide_sensitive_content': True,
          'disable_proxdash': False,
          'api_key': 'my_api_key',
          'base_url': 'https://api.proxai.com'
      },
  ]


def _get_run_options_options():
  return [
      {},
      {
          'run_type': types.RunType.TEST
      },
      {
          'hidden_run_key': 'hidden_run_key'
      },
      {
          'experiment_path': 'experiment_path'
      },
      {
          'root_logging_path': 'root_logging_path'
      },
      {
          'default_model_cache_path': 'default_model_cache_path'
      },
      {
          'logging_options':
              types.LoggingOptions(
                  logging_path='logging_path', stdout=True,
                  hide_sensitive_content=True
              )
      },
      {
          'cache_options':
              types.CacheOptions(
                  cache_path='cache_path', unique_response_limit=1,
                  retry_if_error_cached=True, clear_query_cache_on_connect=True,
                  clear_model_cache_on_connect=True
              )
      },
      {
          'proxdash_options':
              types.ProxDashOptions(
                  stdout=True, hide_sensitive_content=True,
                  disable_proxdash=True
              )
      },
      {
          'allow_multiprocessing': True
      },
      {
          'model_test_timeout': 25
      },
      {
          'feature_mapping_strategy': types.FeatureMappingStrategy.BEST_EFFORT
      },
      {
          'feature_mapping_strategy': types.FeatureMappingStrategy.STRICT
      },
      {
          'suppress_provider_errors': True
      },
      {
          'run_type': types.RunType.TEST,
          'hidden_run_key': 'hidden_run_key',
          'experiment_path': 'experiment_path',
          'root_logging_path': 'root_logging_path',
          'default_model_cache_path': 'default_model_cache_path',
          'logging_options':
              types.LoggingOptions(
                  logging_path='logging_path', stdout=True,
                  hide_sensitive_content=True
              ),
          'cache_options':
              types.CacheOptions(
                  cache_path='cache_path', unique_response_limit=1,
                  retry_if_error_cached=True, clear_query_cache_on_connect=True,
                  clear_model_cache_on_connect=True
              ),
          'proxdash_options':
              types.ProxDashOptions(
                  stdout=True, hide_sensitive_content=True,
                  disable_proxdash=True
              ),
          'allow_multiprocessing': True,
          'model_test_timeout': 25,
          'feature_mapping_strategy': types.FeatureMappingStrategy.STRICT
      },
  ]


def _get_model_status_options():
  return [{}, {
      'unprocessed_models': {_MODEL_1}
  }, {
      'working_models': {_MODEL_1, _MODEL_2}
  }, {
      'failed_models': {_MODEL_1, _MODEL_2, _MODEL_3}
  }, {
      'filtered_models': {_MODEL_1, _MODEL_2, _MODEL_3, _MODEL_4}
  }, {
      'provider_queries': {
          _MODEL_1:
              types.CallRecord(
                  query=types.QueryRecord(prompt='Hello'),
                  result=types.ResultRecord(
                      status=types.ResultStatusType.SUCCESS,
                      content='Hello, world!'
                  )
              )
      }
  }, {
      'unprocessed_models': {_MODEL_1},
      'working_models': {_MODEL_2},
      'failed_models': {_MODEL_3},
      'filtered_models': {_MODEL_4}
  }, {
      'unprocessed_models': {_MODEL_1, _MODEL_2},
      'working_models': {_MODEL_2, _MODEL_3},
      'failed_models': {_MODEL_3, _MODEL_4},
      'filtered_models': {_MODEL_4, _MODEL_1},
      'provider_queries': {
          _MODEL_1:
              types.CallRecord(
                  query=types.QueryRecord(prompt='Hello'),
                  result=types.ResultRecord(
                      status=types.ResultStatusType.SUCCESS,
                      content='Hello, world!'
                  )
              )
      }
  }]


def _get_provider_model_configs_mapping_type_options():
  return [
      {
          'openai': {
              'gpt-4':
                  types.ProviderModelConfig(
                      provider_model=_MODEL_1,
                      pricing=types.ProviderModelPricingType(),
                      features=types.FeatureConfigType(),
                      metadata=types.ProviderModelMetadataType()
                  )
          }
      },
      {
          'openai': {
              'gpt-4':
                  types.ProviderModelConfig(
                      provider_model=_MODEL_1,
                      pricing=types.ProviderModelPricingType(
                          input_token_cost=0.001, output_token_cost=0.002
                      ),
                      features=types.FeatureConfigType(),
                      metadata=types.ProviderModelMetadataType()
                  )
          },
          'claude': {
              'opus-4':
                  types.ProviderModelConfig(
                      provider_model=_MODEL_3,
                      pricing=types.ProviderModelPricingType(),
                      features=types.FeatureConfigType(),
                      metadata=types.ProviderModelMetadataType(
                          is_recommended=True
                      )
                  )
          }
      },
  ]


def _get_recommended_models_mapping_type_options():
  return [{
      'openai': [_MODEL_1]
  }, {
      'openai': [_MODEL_1],
      'claude': [_MODEL_3]
  }, {
      'openai': [_MODEL_1, _MODEL_2],
      'claude': [_MODEL_3, _MODEL_4]
  }]


def _get_call_type_mapping_type_options():
  return [{
      types.CallType.TEXT: [_MODEL_1]
  }, {
      types.CallType.TEXT: [_MODEL_1, _MODEL_4]
  }, {
      types.CallType.TEXT: [_MODEL_1, _MODEL_2],
      types.CallType.IMAGE: [_MODEL_3, _MODEL_4]
  }]


def _get_model_size_mapping_type_options():
  return [{
      types.ModelSizeType.SMALL: [_MODEL_5]
  }, {
      types.ModelSizeType.SMALL: [_MODEL_5],
      types.ModelSizeType.LARGE: [_MODEL_1]
  }, {
      types.ModelSizeType.SMALL: [_MODEL_5],
      types.ModelSizeType.LARGE: [_MODEL_1, _MODEL_3]
  }]


def _get_default_model_priority_list_options():
  return [
      [_MODEL_1],
      [_MODEL_1, _MODEL_3],
      [_MODEL_1, _MODEL_3, _MODEL_2]
  ]


class TestTypeSerializer:

  @pytest.mark.parametrize(
      'provider_model_type_options', _get_provider_model_type_options()
  )
  def test_encode_decode_provider_model_type(self, provider_model_type_options):
    # Test successful encode/decode round-trip
    provider_model_type = types.ProviderModelType(
        **provider_model_type_options
    )
    encoded_provider_model_type = type_serializer.encode_provider_model_type(
        provider_model_type=provider_model_type
    )
    decoded_provider_model_type = type_serializer.decode_provider_model_type(
        record=encoded_provider_model_type
    )
    assert provider_model_type == decoded_provider_model_type

    # Test validation: missing provider
    invalid_record = encoded_provider_model_type.copy()
    del invalid_record['provider']
    with pytest.raises(ValueError, match='Provider not found in record'):
      type_serializer.decode_provider_model_type(record=invalid_record)

    # Test validation: missing model
    invalid_record = encoded_provider_model_type.copy()
    del invalid_record['model']
    with pytest.raises(ValueError, match='Model not found in record'):
      type_serializer.decode_provider_model_type(record=invalid_record)

    # Test validation: missing provider_model_identifier
    invalid_record = encoded_provider_model_type.copy()
    del invalid_record['provider_model_identifier']
    with pytest.raises(
        ValueError, match='Provider model identifier not found in record'
    ):
      type_serializer.decode_provider_model_type(record=invalid_record)

  @pytest.mark.parametrize(
      'provider_model_pricing_type_options',
      _get_provider_model_pricing_type_options()
  )
  def test_encode_decode_provider_model_pricing_type(
      self, provider_model_pricing_type_options
  ):
    provider_model_pricing_type = types.ProviderModelPricingType(
        **provider_model_pricing_type_options
    )
    encoded_provider_model_pricing_type = (
        type_serializer.encode_provider_model_pricing_type(
            provider_model_pricing_type=provider_model_pricing_type
        )
    )
    decoded_provider_model_pricing_type = (
        type_serializer.decode_provider_model_pricing_type(
            record=encoded_provider_model_pricing_type
        )
    )
    assert provider_model_pricing_type == decoded_provider_model_pricing_type

  @pytest.mark.parametrize(
      'feature_support_type_option', _get_feature_support_type_options()
  )
  def test_encode_decode_feature_support_type(
      self, feature_support_type_option
  ):
    encoded_feature_support_type = (
        type_serializer.encode_feature_support_type(
            feature_support_type=feature_support_type_option
        )
    )
    decoded_feature_support_type = (
        type_serializer.decode_feature_support_type(
            value=encoded_feature_support_type
        )
    )
    assert feature_support_type_option == decoded_feature_support_type

  @pytest.mark.parametrize(
      'parameter_config_type_options', _get_parameter_config_type_options()
  )
  def test_encode_decode_parameter_config_type(
      self, parameter_config_type_options
  ):
    parameter_config_type = types.ParameterConfigType(
        **parameter_config_type_options
    )
    encoded_parameter_config_type = (
        type_serializer.encode_parameter_config_type(
            parameter_config_type=parameter_config_type
        )
    )
    decoded_parameter_config_type = (
        type_serializer.decode_parameter_config_type(
            record=encoded_parameter_config_type
        )
    )
    assert parameter_config_type == decoded_parameter_config_type

  @pytest.mark.parametrize(
      'tool_config_type_options', _get_tool_config_type_options()
  )
  def test_encode_decode_tool_config_type(self, tool_config_type_options):
    tool_config_type = types.ToolConfigType(**tool_config_type_options)
    encoded_tool_config_type = (
        type_serializer.encode_tool_config_type(
            tool_config_type=tool_config_type
        )
    )
    decoded_tool_config_type = (
        type_serializer.decode_tool_config_type(
            record=encoded_tool_config_type
        )
    )
    assert tool_config_type == decoded_tool_config_type

  @pytest.mark.parametrize(
      'response_format_config_type_options',
      _get_response_format_config_type_options()
  )
  def test_encode_decode_response_format_config_type(
      self, response_format_config_type_options
  ):
    response_format_config_type = types.ResponseFormatConfigType(
        **response_format_config_type_options
    )
    encoded_response_format_config_type = (
        type_serializer.encode_response_format_config_type(
            response_format_config_type=response_format_config_type
        )
    )
    decoded_response_format_config_type = (
        type_serializer.decode_response_format_config_type(
            record=encoded_response_format_config_type
        )
    )
    assert response_format_config_type == decoded_response_format_config_type

  @pytest.mark.parametrize(
      'feature_config_type_options', _get_feature_config_type_options()
  )
  def test_encode_decode_feature_config_type(
      self, feature_config_type_options
  ):
    feature_config_type = types.FeatureConfigType(
        **feature_config_type_options
    )
    encoded_feature_config_type = (
        type_serializer.encode_feature_config_type(
            feature_config_type=feature_config_type
        )
    )
    decoded_feature_config_type = (
        type_serializer.decode_feature_config_type(
            record=encoded_feature_config_type
        )
    )
    assert feature_config_type == decoded_feature_config_type

  @pytest.mark.parametrize(
      'provider_model_metadata_type_options',
      _get_provider_model_metadata_type_options()
  )
  def test_encode_decode_provider_model_metadata_type(
      self, provider_model_metadata_type_options
  ):
    provider_model_metadata_type = types.ProviderModelMetadataType(
        **provider_model_metadata_type_options
    )
    encoded_provider_model_metadata_type = (
        type_serializer.encode_provider_model_metadata_type(
            provider_model_metadata_type=provider_model_metadata_type
        )
    )
    decoded_provider_model_metadata_type = (
        type_serializer.decode_provider_model_metadata_type(
            record=encoded_provider_model_metadata_type
        )
    )
    assert provider_model_metadata_type == decoded_provider_model_metadata_type

  @pytest.mark.parametrize(
      'provider_model_config_options',
      _get_provider_model_config_options()
  )
  def test_encode_decode_provider_model_config(
      self, provider_model_config_options
  ):
    provider_model_config = types.ProviderModelConfig(
        **provider_model_config_options
    )
    encoded_provider_model_config = (
        type_serializer.encode_provider_model_config(
            provider_model_config=provider_model_config
        )
    )
    decoded_provider_model_config = (
        type_serializer.decode_provider_model_config(
            record=encoded_provider_model_config
        )
    )
    assert provider_model_config == decoded_provider_model_config

  @pytest.mark.parametrize(
      'model_configs_schema_metadata_type_options',
      _get_model_configs_schema_metadata_type_options()
  )
  def test_encode_decode_model_configs_schema_metadata_type(
      self, model_configs_schema_metadata_type_options
  ):
    model_configs_schema_metadata_type = types.ModelConfigsSchemaMetadataType(
        **model_configs_schema_metadata_type_options
    )
    encoded_model_configs_schema_metadata_type = (
        type_serializer.encode_model_configs_schema_metadata_type(
            model_configs_schema_metadata_type=(
                model_configs_schema_metadata_type
            )
        )
    )
    decoded_model_configs_schema_metadata_type = (
        type_serializer.decode_model_configs_schema_metadata_type(
            record=encoded_model_configs_schema_metadata_type
        )
    )
    assert model_configs_schema_metadata_type == (
        decoded_model_configs_schema_metadata_type
    )

  @pytest.mark.parametrize(
      'model_registry_options', _get_model_registry_options()
  )
  def test_encode_decode_model_registry(self, model_registry_options):
    model_registry = types.ModelRegistry(**model_registry_options)
    encoded_model_registry = (
        type_serializer.encode_model_registry(
            model_registry=model_registry
        )
    )
    decoded_model_registry = (
        type_serializer.decode_model_registry(
            record=encoded_model_registry
        )
    )
    assert model_registry == decoded_model_registry

  @pytest.mark.parametrize(
      'provider_model_configs_mapping_type_options',
      _get_provider_model_configs_mapping_type_options()
  )
  def test_encode_decode_provider_model_configs_mapping_type(
      self, provider_model_configs_mapping_type_options
  ):
    encoded_provider_model_configs_mapping_type = (
        type_serializer.encode_provider_model_configs_mapping_type(
            provider_model_configs=provider_model_configs_mapping_type_options
        )
    )
    decoded_provider_model_configs_mapping_type = (
        type_serializer.decode_provider_model_configs_mapping_type(
            record=encoded_provider_model_configs_mapping_type
        )
    )
    assert provider_model_configs_mapping_type_options == (
        decoded_provider_model_configs_mapping_type
    )

  @pytest.mark.parametrize(
      'recommended_models_mapping_type_options',
      _get_recommended_models_mapping_type_options()
  )
  def test_encode_decode_recommended_models_mapping_type(
      self, recommended_models_mapping_type_options
  ):
    encoded_recommended_models_mapping_type = (
        type_serializer.encode_recommended_models_mapping_type(
            recommended_models=recommended_models_mapping_type_options
        )
    )
    decoded_recommended_models_mapping_type = (
        type_serializer.decode_recommended_models_mapping_type(
            record=encoded_recommended_models_mapping_type
        )
    )
    assert recommended_models_mapping_type_options == (
        decoded_recommended_models_mapping_type
    )

  @pytest.mark.parametrize(
      'call_type_mapping_type_options',
      _get_call_type_mapping_type_options()
  )
  def test_encode_decode_call_type_mapping_type(
      self, call_type_mapping_type_options
  ):
    encoded_call_type_mapping_type = (
        type_serializer.encode_call_type_mapping_type(
            call_type_mapping=call_type_mapping_type_options
        )
    )
    decoded_call_type_mapping_type = (
        type_serializer.decode_call_type_mapping_type(
            record=encoded_call_type_mapping_type
        )
    )
    assert call_type_mapping_type_options == decoded_call_type_mapping_type

  @pytest.mark.parametrize(
      'model_size_mapping_type_options',
      _get_model_size_mapping_type_options()
  )
  def test_encode_decode_model_size_mapping_type(
      self, model_size_mapping_type_options
  ):
    encoded_model_size_mapping_type = (
        type_serializer.encode_model_size_mapping_type(
            model_size_mapping=model_size_mapping_type_options
        )
    )
    decoded_model_size_mapping_type = (
        type_serializer.decode_model_size_mapping_type(
            record=encoded_model_size_mapping_type
        )
    )
    assert model_size_mapping_type_options == decoded_model_size_mapping_type

  @pytest.mark.parametrize(
      'default_model_priority_list_options',
      _get_default_model_priority_list_options()
  )
  def test_encode_decode_default_model_priority_list(
      self, default_model_priority_list_options
  ):
    encoded_default_model_priority_list = (
        type_serializer.encode_default_model_priority_list(
            default_model_priority_list=default_model_priority_list_options
        )
    )
    decoded_default_model_priority_list = (
        type_serializer.decode_default_model_priority_list(
            record=encoded_default_model_priority_list
        )
    )
    assert default_model_priority_list_options == (
        decoded_default_model_priority_list
    )

  @pytest.mark.parametrize(
      'parameter_type_options', _get_parameter_type_options()
  )
  def test_encode_decode_parameter_type(self, parameter_type_options):
    parameter_type = types.ParameterType(**parameter_type_options)
    encoded_parameter_type = type_serializer.encode_parameter_type(
        parameter_type=parameter_type
    )
    decoded_parameter_type = type_serializer.decode_parameter_type(
        record=encoded_parameter_type
    )
    assert parameter_type == decoded_parameter_type

  @pytest.mark.parametrize(
      'connection_options_options', _get_connection_options_options()
  )
  def test_encode_decode_connection_options(self, connection_options_options):
    connection_options = types.ConnectionOptions(**connection_options_options)
    encoded_connection_options = type_serializer.encode_connection_options(
        connection_options=connection_options
    )
    decoded_connection_options = type_serializer.decode_connection_options(
        record=encoded_connection_options
    )
    assert connection_options == decoded_connection_options

  @pytest.mark.parametrize(
      'response_format_options', _get_response_format_options()
  )
  def test_encode_decode_response_format(self, response_format_options):
    response_format = types.ResponseFormat(**response_format_options)
    encoded = type_serializer.encode_response_format(
        response_format=response_format
    )
    decoded = type_serializer.decode_response_format(record=encoded)
    assert decoded.type == response_format.type
    # pydantic_class cannot be reconstructed from serialized form
    assert decoded.pydantic_class is None
    if response_format.pydantic_class is not None:
      assert 'pydantic_class_name' in encoded
      assert encoded['pydantic_class_name'] == (
          response_format.pydantic_class.__name__
      )
      assert 'pydantic_class_json_schema' in encoded
      # Verify metadata survives round-trip
      assert decoded.pydantic_class_name == (
          response_format.pydantic_class.__name__
      )
      assert decoded.pydantic_class_json_schema == (
          response_format.pydantic_class.model_json_schema()
      )

  @pytest.mark.parametrize('choice_type_options', _get_choice_type_options())
  def test_encode_decode_choice_type(self, choice_type_options):
    choice_type = types.ChoiceType(**choice_type_options)
    encoded_choice_type = type_serializer.encode_choice_type(
        choice_type=choice_type
    )
    decoded_choice_type = type_serializer.decode_choice_type(
        record=encoded_choice_type
    )
    assert choice_type == decoded_choice_type

  @pytest.mark.parametrize('usage_type_options', _get_usage_type_options())
  def test_encode_decode_usage_type(self, usage_type_options):
    usage_type = types.UsageType(**usage_type_options)
    encoded_usage_type = type_serializer.encode_usage_type(
        usage_type=usage_type
    )
    decoded_usage_type = type_serializer.decode_usage_type(
        record=encoded_usage_type
    )
    assert usage_type == decoded_usage_type

  @pytest.mark.parametrize(
      'tool_usage_type_options', _get_tool_usage_type_options()
  )
  def test_encode_decode_tool_usage_type(self, tool_usage_type_options):
    tool_usage_type = types.ToolUsageType(**tool_usage_type_options)
    encoded_tool_usage_type = type_serializer.encode_tool_usage_type(
        tool_usage_type=tool_usage_type
    )
    decoded_tool_usage_type = type_serializer.decode_tool_usage_type(
        record=encoded_tool_usage_type
    )
    assert tool_usage_type == decoded_tool_usage_type

  @pytest.mark.parametrize(
      'timestamp_type_options', _get_timestamp_type_options()
  )
  def test_encode_decode_timestamp_type(self, timestamp_type_options):
    timestamp_type = types.TimeStampType(**timestamp_type_options)
    encoded_timestamp_type = type_serializer.encode_timestamp_type(
        timestamp_type=timestamp_type
    )
    decoded_timestamp_type = type_serializer.decode_timestamp_type(
        record=encoded_timestamp_type
    )
    assert timestamp_type == decoded_timestamp_type

  @pytest.mark.parametrize(
      'result_record_options', _get_result_record_options()
  )
  def test_encode_decode_result_record(self, result_record_options):
    result_record = types.ResultRecord(**result_record_options)
    encoded_result_record = type_serializer.encode_result_record(
        result_record=result_record
    )
    decoded_result_record = type_serializer.decode_result_record(
        record=encoded_result_record
    )
    assert result_record == decoded_result_record

  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_encode_decode_query_record(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    encoded_query_record = type_serializer.encode_query_record(
        query_record=query_record
    )
    decoded_query_record = type_serializer.decode_query_record(
        record=encoded_query_record
    )
    assert query_record == decoded_query_record

  @pytest.mark.parametrize(
      'connection_metadata_options', _get_connection_metadata_options()
  )
  def test_encode_decode_connection_metadata(
      self, connection_metadata_options
  ):
    connection_metadata = types.ConnectionMetadata(
        **connection_metadata_options
    )
    encoded_connection_metadata = (
        type_serializer.encode_connection_metadata(
            connection_metadata=connection_metadata
        )
    )
    decoded_connection_metadata = (
        type_serializer.decode_connection_metadata(
            record=encoded_connection_metadata
        )
    )
    assert connection_metadata == decoded_connection_metadata

  @pytest.mark.parametrize('call_record_options', _get_call_record_options())
  def test_encode_decode_call_record(self, call_record_options):
    call_record = types.CallRecord(**call_record_options)
    encoded_call_record = type_serializer.encode_call_record(
        call_record=call_record
    )
    decoded_call_record = type_serializer.decode_call_record(
        record=encoded_call_record
    )
    assert call_record == decoded_call_record

  @pytest.mark.parametrize('cache_record_options', _get_cache_record_options())
  def test_encode_decode_cache_record(self, cache_record_options):
    cache_record = types.CacheRecord(**cache_record_options)
    encoded_cache_record = type_serializer.encode_cache_record(
        cache_record=cache_record
    )
    decoded_cache_record = type_serializer.decode_cache_record(
        record=encoded_cache_record
    )
    assert cache_record == decoded_cache_record

  @pytest.mark.parametrize(
      'light_cache_record_options', _get_light_cache_record_options()
  )
  def test_encode_decode_light_cache_record(self, light_cache_record_options):
    light_cache_record = types.LightCacheRecord(**light_cache_record_options)
    encoded_light_cache_record = (
        type_serializer.encode_light_cache_record(
            light_cache_record=light_cache_record
        )
    )
    decoded_light_cache_record = (
        type_serializer.decode_light_cache_record(
            record=encoded_light_cache_record
        )
    )
    assert light_cache_record == decoded_light_cache_record

  @pytest.mark.parametrize(
      'logging_options_options', _get_logging_options_options()
  )
  def test_encode_decode_logging_options(self, logging_options_options):
    logging_options = types.LoggingOptions(**logging_options_options)
    encoded_logging_options = type_serializer.encode_logging_options(
        logging_options=logging_options
    )
    decoded_logging_options = type_serializer.decode_logging_options(
        record=encoded_logging_options
    )
    assert logging_options == decoded_logging_options

  @pytest.mark.parametrize(
      'cache_options_options', _get_cache_options_options()
  )
  def test_encode_decode_cache_options(self, cache_options_options):
    cache_options = types.CacheOptions(**cache_options_options)
    encoded_cache_options = type_serializer.encode_cache_options(
        cache_options=cache_options
    )
    decoded_cache_options = type_serializer.decode_cache_options(
        record=encoded_cache_options
    )
    assert cache_options == decoded_cache_options

  @pytest.mark.parametrize(
      'proxdash_options_options', _get_proxdash_options_options()
  )
  def test_encode_decode_proxdash_options(self, proxdash_options_options):
    proxdash_options = types.ProxDashOptions(**proxdash_options_options)
    encoded_proxdash_options = type_serializer.encode_proxdash_options(
        proxdash_options=proxdash_options
    )
    decoded_proxdash_options = type_serializer.decode_proxdash_options(
        record=encoded_proxdash_options
    )
    assert proxdash_options == decoded_proxdash_options

  @pytest.mark.parametrize('run_options_options', _get_run_options_options())
  def test_encode_decode_run_options(self, run_options_options):
    run_options = types.RunOptions(**run_options_options)
    encoded_run_options = type_serializer.encode_run_options(
        run_options=run_options
    )
    decoded_run_options = type_serializer.decode_run_options(
        record=encoded_run_options
    )
    assert run_options == decoded_run_options

  @pytest.mark.parametrize('model_status_options', _get_model_status_options())
  def test_encode_decode_model_status(self, model_status_options):
    model_status = types.ModelStatus(**model_status_options)
    encoded_model_status = type_serializer.encode_model_status(
        model_status=model_status
    )
    decoded_model_status = type_serializer.decode_model_status(
        record=encoded_model_status
    )
    assert model_status == decoded_model_status

  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_get_query_record_hash(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    query_hash_value = hash_serializer.get_query_record_hash(
        query_record=query_record
    )

    query_record_options_copy = query_record_options.copy()
    query_record_options_copy['prompt'] = 'different_prompt_for_hash_test'
    query_record_2 = types.QueryRecord(**query_record_options_copy)
    query_hash_value_2 = hash_serializer.get_query_record_hash(
        query_record=query_record_2
    )

    assert query_hash_value != query_hash_value_2
    assert query_hash_value == (
        hash_serializer.get_query_record_hash(query_record=query_record)
    )

  def test_encode_decode_response_format_hash_consistency(self):
    response_format = types.ResponseFormat(
        type=types.ResponseFormatType.PYDANTIC,
        pydantic_class=_UserModel
    )
    query_record = types.QueryRecord(
        prompt='test', response_format=response_format
    )
    hash_before = hash_serializer.get_query_record_hash(query_record)

    encoded = type_serializer.encode_query_record(query_record)
    decoded_query_record = type_serializer.decode_query_record(encoded)

    hash_after = hash_serializer.get_query_record_hash(decoded_query_record)
    assert hash_before == hash_after

  def test_encode_decode_choice_type_with_pydantic(self):
    user = _UserModel(name='John', age=30)
    choice_type = types.ChoiceType(
        output_text='John is 30',
        output_pydantic=user,
        content=[
            message_content.MessageContent(
                type=message_content.ContentType.TEXT, text='John is 30'
            )
        ]
    )
    encoded = type_serializer.encode_choice_type(choice_type=choice_type)
    decoded = type_serializer.decode_choice_type(record=encoded)
    # output_pydantic cannot be reconstructed from serialized form
    assert decoded.output_pydantic is None
    assert decoded.output_text == choice_type.output_text
    assert decoded.content == choice_type.content
    # Verify pydantic metadata was encoded
    assert 'output_pydantic' in encoded
    assert encoded['output_pydantic']['class_name'] == '_UserModel'
    assert encoded['output_pydantic']['instance_json_value'] == {
        'name': 'John', 'age': 30
    }

  def test_encode_decode_result_record_with_pydantic(self):
    user = _UserModel(name='Jane', age=25)
    result_record = types.ResultRecord(
        status=types.ResultStatusType.SUCCESS,
        output_text='Jane is 25',
        output_pydantic=user
    )
    encoded = type_serializer.encode_result_record(
        result_record=result_record
    )
    decoded = type_serializer.decode_result_record(record=encoded)
    # output_pydantic cannot be reconstructed from serialized form
    assert decoded.output_pydantic is None
    assert decoded.status == result_record.status
    assert decoded.output_text == result_record.output_text
    # Verify pydantic metadata was encoded
    assert 'output_pydantic' in encoded
    assert encoded['output_pydantic']['class_name'] == '_UserModel'
    assert encoded['output_pydantic']['instance_json_value'] == {
        'name': 'Jane', 'age': 25
    }
