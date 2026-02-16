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


def _get_provider_model_type_options():
  return [
      {
          'provider': 'openai',
          'model': 'gpt-4',
          'provider_model_identifier': 'gpt-4'
      },
  ]


def _get_provider_model_identifier_options():
  return [
      _MODEL_1,
      _MODEL_3,
      ('openai', 'gpt-4'),
      ('claude', 'sonnet-4'),
  ]


def _get_provider_model_pricing_type_options():
  return [
      {},
      {
          'per_response_token_cost': 0.001
      },
      {
          'per_query_token_cost': 0.002
      },
      {
          'per_response_token_cost': 0.001,
          'per_query_token_cost': 0.002
      },
      {
          'per_response_token_cost': 0.0,
          'per_query_token_cost': 0.0
      },
      {
          'per_response_token_cost': 1.5,
          'per_query_token_cost': 0.5
      },
  ]


def _get_endpoint_feature_info_type_options():
  return [
      {},
      {
          'supported': []
      },
      {
          'supported': ['feature1']
      },
      {
          'supported': ['feature1', 'feature2', 'feature3']
      },
      {
          'best_effort': []
      },
      {
          'best_effort': ['feature1']
      },
      {
          'best_effort': ['feature1', 'feature2']
      },
      {
          'not_supported': []
      },
      {
          'not_supported': ['feature1']
      },
      {
          'not_supported': ['feature1', 'feature2', 'feature3']
      },
      {
          'supported': ['feature1'],
          'best_effort': ['feature2'],
          'not_supported': ['feature3']
      },
  ]


def _get_feature_mapping_type_options():
  return [
      {},
      {
          types.FeatureNameType.PROMPT: types.EndpointFeatureInfoType()
      },
      {
          types.FeatureNameType.PROMPT:
              types.EndpointFeatureInfoType(supported=['value1'])
      },
      {
          types.FeatureNameType.MESSAGES:
              types.EndpointFeatureInfoType(best_effort=['value1', 'value2'])
      },
      {
          types.FeatureNameType.SYSTEM_PROMPT:
              types.EndpointFeatureInfoType(not_supported=['value1'])
      },
      {
          types.FeatureNameType.PROMPT:
              types.EndpointFeatureInfoType(
                  supported=['value1'], best_effort=['value2'],
                  not_supported=['value3']
              )
      },
      {
          types.FeatureNameType.PROMPT:
              types.EndpointFeatureInfoType(supported=['value1']),
          types.FeatureNameType.MESSAGES:
              types.EndpointFeatureInfoType(best_effort=['value2'])
      },
      {
          types.FeatureNameType.PROMPT:
              types.EndpointFeatureInfoType(supported=['value1']),
          types.FeatureNameType.MESSAGES:
              types.EndpointFeatureInfoType(best_effort=['value2']),
          types.FeatureNameType.SYSTEM_PROMPT:
              types.EndpointFeatureInfoType(not_supported=['value3']),
          types.FeatureNameType.MAX_TOKENS:
              types.EndpointFeatureInfoType(
                  supported=['value4'], best_effort=['value5'],
                  not_supported=['value6']
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
          'is_featured': True
      },
      {
          'is_featured': False
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
          'is_default_candidate': True
      },
      {
          'is_default_candidate': False
      },
      {
          'default_candidate_priority': 1
      },
      {
          'default_candidate_priority': 100
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
          'is_featured': True,
          'model_size_tags': [types.ModelSizeType.LARGE],
          'is_default_candidate': True,
          'default_candidate_priority': 5,
          'tags': ['tag1', 'tag2']
      },
  ]


def _get_provider_model_config_type_options():
  return [
      {},
      {
          'provider_model': _MODEL_1
      },
      {
          'pricing':
              types.ProviderModelPricingType(
                  per_response_token_cost=0.001, per_query_token_cost=0.002
              )
      },
      {
          'features': {
              types.FeatureNameType.PROMPT:
                  types.EndpointFeatureInfoType(not_supported=['feature1'])
          }
      },
      {
          'metadata':
              types.ProviderModelMetadataType(
                  call_type=types.CallType.TEXT, is_featured=True
              )
      },
      {
          'provider_model': _MODEL_3,
          'pricing':
              types.ProviderModelPricingType(
                  per_response_token_cost=0.003, per_query_token_cost=0.001
              ),
          'features': {
              types.FeatureNameType.PROMPT:
                  types.EndpointFeatureInfoType(
                      not_supported=['feature1', 'feature2']
                  )
          },
          'metadata':
              types.ProviderModelMetadataType(
                  call_type=types.CallType.TEXT, is_featured=True,
                  model_size_tags=[types.ModelSizeType.LARGEST],
                  is_default_candidate=True, default_candidate_priority=10,
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


def _get_model_configs_schema_version_config_type_options():
  return [{}, {
      'provider_model_configs': {
          'openai': {
              'gpt-4':
                  types.ProviderModelConfigType(provider_model=_MODEL_1)
          }
      }
  }, {
      'provider_model_configs': {
          'openai': {
              'gpt-4':
                  types.ProviderModelConfigType(
                      provider_model=_MODEL_1,
                      pricing=types.ProviderModelPricingType(
                          per_response_token_cost=0.001,
                          per_query_token_cost=0.002
                      )
                  )
          },
          'claude': {
              'opus-4':
                  types.ProviderModelConfigType(
                      provider_model=_MODEL_3,
                      metadata=types.ProviderModelMetadataType(
                          is_featured=True
                      )
                  )
          }
      }
  }, {
      'featured_models': {
          'openai': [_MODEL_1]
      }
  }, {
      'featured_models': {
          'openai': [_MODEL_1],
          'claude': [('claude', 'opus-4')]
      }
  }, {
      'models_by_call_type': {
          types.CallType.TEXT: {
              'openai': [_MODEL_1],
              'claude': [('claude', 'sonnet-4')]
          }
      }
  }, {
      'models_by_size': {
          types.ModelSizeType.SMALL: [('openai', 'gpt-4o-mini')],
          types.ModelSizeType.LARGE: [_MODEL_1, ('claude', 'opus-4')]
      }
  }, {
      'default_model_priority_list': [
          _MODEL_1, ('claude', 'opus-4'), ('openai', 'o3-mini')
      ]
  }, {
      'provider_model_configs': {
          'openai': {
              'gpt-4':
                  types.ProviderModelConfigType(
                      provider_model=_MODEL_1,
                      pricing=types.ProviderModelPricingType(
                          per_response_token_cost=0.001,
                          per_query_token_cost=0.002
                      ), features={
                          types.FeatureNameType.PROMPT:
                              types.EndpointFeatureInfoType(
                                  not_supported=['feature1']
                              )
                      }, metadata=types.ProviderModelMetadataType(
                          call_type=types.CallType.TEXT,
                          is_featured=True,
                          model_size_tags=[types.ModelSizeType.LARGE]
                      )
                  ),
              'o3-mini':
                  types.ProviderModelConfigType(provider_model=_MODEL_2)
          },
          'claude': {
              'opus-4':
                  types.ProviderModelConfigType(
                      provider_model=_MODEL_3,
                      metadata=types.ProviderModelMetadataType(
                          is_featured=True,
                          model_size_tags=[types.ModelSizeType.LARGEST]
                      )
                  ),
              'sonnet-4':
                  types.ProviderModelConfigType(provider_model=_MODEL_4)
          }
      },
      'featured_models': {
          'openai': [_MODEL_1],
          'claude': [('claude', 'opus-4')]
      },
      'models_by_call_type': {
          types.CallType.TEXT: {
              'openai': [_MODEL_1],
              'claude': [('claude', 'sonnet-4')]
          }
      },
      'models_by_size': {
          types.ModelSizeType.SMALL: [('openai', 'gpt-4o-mini')],
          types.ModelSizeType.LARGE: [_MODEL_1, ('claude', 'opus-4')]
      },
      'default_model_priority_list': [
          _MODEL_1, ('claude', 'opus-4'), ('openai', 'o3-mini')
      ]
  }]


def _get_model_configs_schema_type_options():
  return [{}, {
      'metadata': types.ModelConfigsSchemaMetadataType(version='1.0.0')
  }, {
      'metadata':
          types.ModelConfigsSchemaMetadataType(
              version='1.0.0',
              released_at=datetime.datetime.now(datetime.timezone.utc),
              config_origin=types.ConfigOriginType.BUILT_IN,
              release_notes='Initial release'
          )
  }, {
      'version_config':
          types.ModelConfigsSchemaVersionConfigType(
              provider_model_configs={
                  'openai': {
                      'gpt-4':
                          types.ProviderModelConfigType(
                              provider_model=_MODEL_1
                          )
                  }
              }
          )
  }, {
      'version_config':
          types.ModelConfigsSchemaVersionConfigType(
              featured_models={
                  'openai': [_MODEL_1]
              }
          )
  }, {
      'metadata':
          types.ModelConfigsSchemaMetadataType(
              version='1.0.0',
              released_at=datetime.datetime.now(datetime.timezone.utc)
          ),
      'version_config':
          types.ModelConfigsSchemaVersionConfigType(
              provider_model_configs={
                  'openai': {
                      'gpt-4':
                          types.ProviderModelConfigType(
                              provider_model=_MODEL_1
                          )
                  }
              }
          )
  }, {
      'metadata':
          types.ModelConfigsSchemaMetadataType(
              version='2.1.0',
              released_at=datetime.datetime.now(datetime.timezone.utc),
              min_proxai_version='>=2.0.0',
              config_origin=types.ConfigOriginType.PROXDASH,
              release_notes='Added new models and updated pricing'
          ),
      'version_config':
          types.ModelConfigsSchemaVersionConfigType(
              provider_model_configs={
                  'openai': {
                      'gpt-4':
                          types.ProviderModelConfigType(
                              provider_model=_MODEL_1,
                              pricing=types.ProviderModelPricingType(
                                  per_response_token_cost=0.001,
                                  per_query_token_cost=0.002
                              ), features={
                                  types.FeatureNameType.PROMPT:
                                      types.EndpointFeatureInfoType(
                                          not_supported=['feature1']
                                      )
                              }, metadata=types.ProviderModelMetadataType(
                                  call_type=types.CallType.TEXT,
                                  is_featured=True,
                                  model_size_tags=[types.ModelSizeType.LARGE]
                              )
                          ),
                      'o3-mini':
                          types.ProviderModelConfigType(
                              provider_model=_MODEL_2
                          )
                  },
                  'claude': {
                      'opus-4':
                          types.ProviderModelConfigType(
                              provider_model=_MODEL_3,
                              metadata=types.ProviderModelMetadataType(
                                  is_featured=True,
                                  model_size_tags=[types.ModelSizeType.LARGEST]
                              )
                          ),
                      'sonnet-4':
                          types.ProviderModelConfigType(
                              provider_model=_MODEL_4
                          )
                  }
              }, featured_models={
                  'openai': [_MODEL_1],
                  'claude': [('claude', 'opus-4')]
              }, models_by_call_type={
                  types.CallType.TEXT: {
                      'openai': [_MODEL_1],
                      'claude': [('claude', 'sonnet-4')]
                  }
              }, models_by_size={
                  types.ModelSizeType.SMALL: [('openai', 'gpt-4o-mini')],
                  types.ModelSizeType.LARGE: [_MODEL_1, ('claude', 'opus-4')]
              }, default_model_priority_list=[
                  _MODEL_1, ('claude', 'opus-4'), ('openai', 'o3-mini')
              ]
          )
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
          'provider_model': _MODEL_1
      },
      {
          'feature_mapping_strategy': types.FeatureMappingStrategy.BEST_EFFORT
      },
      {
          'feature_mapping_strategy': types.FeatureMappingStrategy.STRICT
      },
      {
          'chosen_endpoint': 'some_endpoint'
      },
      {
          'provider_model': _MODEL_3,
          'feature_mapping_strategy': types.FeatureMappingStrategy.STRICT,
          'chosen_endpoint': 'test_endpoint'
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
          'content': 'Full response',
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
                      content='Hello'
                  )
              ])
      },
      {
          'system_prompt': 'You are a helpful assistant.'
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
                  provider_model=_MODEL_1,
                  feature_mapping_strategy=(
                      types.FeatureMappingStrategy.BEST_EFFORT
                  )
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
                      content='Hi'
                  ),
                  message.Message(
                      role=message_content.MessageRoleType.ASSISTANT,
                      content='Hello!'
                  )
              ]),
          'system_prompt': 'Be helpful.',
          'parameters':
              types.ParameterType(
                  temperature=0.7, max_tokens=200, stop=['stop1'], n=2
              ),
          'tools': [types.Tools.WEB_SEARCH],
          'response_format':
              types.ResponseFormat(type=types.ResponseFormatType.TEXT),
          'connection_options':
              types.ConnectionOptions(
                  provider_model=_MODEL_1,
                  feature_mapping_strategy=(
                      types.FeatureMappingStrategy.STRICT
                  ),
                  chosen_endpoint='test_endpoint'
              ),
          'hash_value': 'test_hash'
      },
  ]


def _get_cache_metadata_options():
  return [
      {},
      {
          'cache_hit': True
      },
      {
          'cache_hit': False
      },
      {
          'result_source': types.ResultSource.CACHE
      },
      {
          'result_source': types.ResultSource.PROVIDER
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
          'cache_hit': True,
          'result_source': types.ResultSource.CACHE
      },
      {
          'cache_hit': False,
          'result_source': types.ResultSource.PROVIDER,
          'cache_look_fail_reason':
              types.CacheLookFailReason.CACHE_NOT_MATCHED
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
          'cache':
              types.CacheMetadata(
                  cache_hit=True, result_source=types.ResultSource.CACHE
              )
      },
      {
          'query': types.QueryRecord(prompt='test'),
          'result':
              types.ResultRecord(
                  status=types.ResultStatusType.SUCCESS,
                  content='Hello'
              ),
          'cache':
              types.CacheMetadata(
                  cache_hit=True, result_source=types.ResultSource.CACHE
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


def _get_provider_model_configs_type_options():
  return [
      {
          'openai': {
              'gpt-4':
                  types.ProviderModelConfigType(provider_model=_MODEL_1)
          }
      },
      {
          'openai': {
              'gpt-4':
                  types.ProviderModelConfigType(
                      provider_model=_MODEL_1,
                      pricing=types.ProviderModelPricingType(
                          per_response_token_cost=0.001,
                          per_query_token_cost=0.002
                      )
                  )
          },
          'claude': {
              'opus-4':
                  types.ProviderModelConfigType(
                      provider_model=_MODEL_3,
                      metadata=types.ProviderModelMetadataType(
                          is_featured=True
                      )
                  )
          }
      },
  ]


def _get_featured_models_type_options():
  return [{
      'openai': [_MODEL_1]
  }, {
      'openai': [_MODEL_1],
      'claude': [('claude', 'opus-4')]
  }, {
      'openai': [_MODEL_1, ('openai', 'o3-mini')],
      'claude': [('claude', 'opus-4'), _MODEL_4]
  }]


def _get_models_by_call_type_type_options():
  return [{
      types.CallType.TEXT: {
          'openai': [_MODEL_1]
      }
  }, {
      types.CallType.TEXT: {
          'openai': [_MODEL_1],
          'claude': [('claude', 'sonnet-4')]
      }
  }, {
      types.CallType.TEXT: {
          'openai': [_MODEL_1, ('openai', 'o3-mini')],
          'claude': [('claude', 'sonnet-4'), _MODEL_3]
      }
  }]


def _get_models_by_size_type_options():
  return [{
      types.ModelSizeType.SMALL: [('openai', 'gpt-4o-mini')]
  }, {
      types.ModelSizeType.SMALL: [('openai', 'gpt-4o-mini')],
      types.ModelSizeType.LARGE: [_MODEL_1]
  }, {
      types.ModelSizeType.SMALL: [('openai', 'gpt-4o-mini')],
      types.ModelSizeType.LARGE: [_MODEL_1, ('claude', 'opus-4')]
  }]


def _get_default_model_priority_list_type_options():
  return [
      [_MODEL_1],
      [_MODEL_1, ('claude', 'opus-4')],
      [_MODEL_1, ('claude', 'opus-4'), _MODEL_2]
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
      'provider_model_identifier', _get_provider_model_identifier_options()
  )
  def test_encode_decode_provider_model_identifier(
      self, provider_model_identifier
  ):
    encoded_provider_model_identifier = (
        type_serializer.encode_provider_model_identifier(
            provider_model_identifier=provider_model_identifier
        )
    )
    decoded_provider_model_identifier = (
        type_serializer.decode_provider_model_identifier(
            record=encoded_provider_model_identifier
        )
    )
    assert provider_model_identifier == decoded_provider_model_identifier

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
      'endpoint_feature_info_type_options',
      _get_endpoint_feature_info_type_options()
  )
  def test_encode_decode_endpoint_feature_info_type(
      self, endpoint_feature_info_type_options
  ):
    endpoint_feature_info_type = types.EndpointFeatureInfoType(
        **endpoint_feature_info_type_options
    )
    encoded_endpoint_feature_info_type = (
        type_serializer.encode_endpoint_feature_info_type(
            endpoint_feature_info_type=endpoint_feature_info_type
        )
    )
    decoded_endpoint_feature_info_type = (
        type_serializer.decode_endpoint_feature_info_type(
            record=encoded_endpoint_feature_info_type
        )
    )
    assert endpoint_feature_info_type == decoded_endpoint_feature_info_type

  @pytest.mark.parametrize(
      'feature_mapping_type_options', _get_feature_mapping_type_options()
  )
  def test_encode_decode_feature_mapping_type(
      self, feature_mapping_type_options
  ):
    encoded_feature_mapping_type = (
        type_serializer.encode_feature_mapping_type(
            feature_mapping=feature_mapping_type_options
        )
    )
    decoded_feature_mapping_type = (
        type_serializer.decode_feature_mapping_type(
            record=encoded_feature_mapping_type
        )
    )
    assert feature_mapping_type_options == decoded_feature_mapping_type

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
      'provider_model_config_type_options',
      _get_provider_model_config_type_options()
  )
  def test_encode_decode_provider_model_config_type(
      self, provider_model_config_type_options
  ):
    provider_model_config_type = types.ProviderModelConfigType(
        **provider_model_config_type_options
    )
    encoded_provider_model_config_type = (
        type_serializer.encode_provider_model_config_type(
            provider_model_config_type=provider_model_config_type
        )
    )
    decoded_provider_model_config_type = (
        type_serializer.decode_provider_model_config_type(
            record=encoded_provider_model_config_type
        )
    )
    assert provider_model_config_type == decoded_provider_model_config_type

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
      'model_configs_schema_version_config_type_options',
      _get_model_configs_schema_version_config_type_options()
  )
  def test_encode_decode_model_configs_schema_version_config_type(
      self, model_configs_schema_version_config_type_options
  ):
    model_configs_schema_version_config_type = (
        types.ModelConfigsSchemaVersionConfigType(
            **model_configs_schema_version_config_type_options
        )
    )
    encoded_model_configs_schema_version_config_type = (
        type_serializer.encode_model_configs_schema_version_config_type(
            model_configs_schema_version_config_type=(
                model_configs_schema_version_config_type
            )
        )
    )
    decoded_model_configs_schema_version_config_type = (
        type_serializer.decode_model_configs_schema_version_config_type(
            record=encoded_model_configs_schema_version_config_type
        )
    )
    assert model_configs_schema_version_config_type == (
        decoded_model_configs_schema_version_config_type
    )

  @pytest.mark.parametrize(
      'model_configs_schema_type_options',
      _get_model_configs_schema_type_options()
  )
  def test_encode_decode_model_configs_schema_type(
      self, model_configs_schema_type_options
  ):
    model_configs_schema_type = types.ModelConfigsSchemaType(
        **model_configs_schema_type_options
    )
    encoded_model_configs_schema_type = (
        type_serializer.encode_model_configs_schema_type(
            model_configs_schema_type=model_configs_schema_type
        )
    )
    decoded_model_configs_schema_type = (
        type_serializer.decode_model_configs_schema_type(
            record=encoded_model_configs_schema_type
        )
    )
    assert model_configs_schema_type == decoded_model_configs_schema_type

  @pytest.mark.parametrize(
      'provider_model_configs_type_options',
      _get_provider_model_configs_type_options()
  )
  def test_encode_decode_provider_model_configs_type(
      self, provider_model_configs_type_options
  ):
    encoded_provider_model_configs_type = (
        type_serializer.encode_provider_model_configs_type(
            provider_model_configs=provider_model_configs_type_options
        )
    )
    decoded_provider_model_configs_type = (
        type_serializer.decode_provider_model_configs_type(
            record=encoded_provider_model_configs_type
        )
    )
    assert provider_model_configs_type_options == (
        decoded_provider_model_configs_type
    )

  @pytest.mark.parametrize(
      'featured_models_type_options', _get_featured_models_type_options()
  )
  def test_encode_decode_featured_models_type(
      self, featured_models_type_options
  ):
    encoded_featured_models_type = (
        type_serializer.encode_featured_models_type(
            featured_models=featured_models_type_options
        )
    )
    decoded_featured_models_type = (
        type_serializer.decode_featured_models_type(
            record=encoded_featured_models_type
        )
    )
    assert featured_models_type_options == decoded_featured_models_type

  @pytest.mark.parametrize(
      'models_by_call_type_type_options',
      _get_models_by_call_type_type_options()
  )
  def test_encode_decode_models_by_call_type_type(
      self, models_by_call_type_type_options
  ):
    encoded_models_by_call_type_type = (
        type_serializer.encode_models_by_call_type_type(
            models_by_call_type=models_by_call_type_type_options
        )
    )
    decoded_models_by_call_type_type = (
        type_serializer.decode_models_by_call_type_type(
            record=encoded_models_by_call_type_type
        )
    )
    assert models_by_call_type_type_options == decoded_models_by_call_type_type

  @pytest.mark.parametrize(
      'models_by_size_type_options', _get_models_by_size_type_options()
  )
  def test_encode_decode_models_by_size_type(self, models_by_size_type_options):
    encoded_models_by_size_type = (
        type_serializer.encode_models_by_size_type(
            models_by_size=models_by_size_type_options
        )
    )
    decoded_models_by_size_type = (
        type_serializer.decode_models_by_size_type(
            record=encoded_models_by_size_type
        )
    )
    assert models_by_size_type_options == decoded_models_by_size_type

  @pytest.mark.parametrize(
      'default_model_priority_list_type_options',
      _get_default_model_priority_list_type_options()
  )
  def test_encode_decode_default_model_priority_list_type(
      self, default_model_priority_list_type_options
  ):
    encoded_default_model_priority_list_type = (
        type_serializer.encode_default_model_priority_list_type(
            default_model_priority_list=(
                default_model_priority_list_type_options
            )
        )
    )
    decoded_default_model_priority_list_type = (
        type_serializer.decode_default_model_priority_list_type(
            record=encoded_default_model_priority_list_type
        )
    )
    assert default_model_priority_list_type_options == (
        decoded_default_model_priority_list_type
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
      'cache_metadata_options', _get_cache_metadata_options()
  )
  def test_encode_decode_cache_metadata(self, cache_metadata_options):
    cache_metadata = types.CacheMetadata(**cache_metadata_options)
    encoded_cache_metadata = type_serializer.encode_cache_metadata(
        cache_metadata=cache_metadata
    )
    decoded_cache_metadata = type_serializer.decode_cache_metadata(
        record=encoded_cache_metadata
    )
    assert cache_metadata == decoded_cache_metadata

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
