import shutil
from pprint import pprint
import os
import pydantic

import proxai as px
import proxai.types as types


_MODEL_CONFIGS = {
    'openai': {
        'default_model': ('openai', 'gpt-4o'),
        'failing_model': ('mock_failing_provider', 'mock_failing_model'),
        'thinking_model': ('openai', 'o3'),
        'image_model': ('openai', 'dall-e-3'),
        'audio_model': ('openai', 'tts-1'),
        'video_model': ('openai', 'sora-2'),
        'web_search_supported': True,
    },
    'gemini': {
        'default_model': ('gemini', 'gemini-3-flash-preview'),
        'failing_model': ('mock_failing_provider', 'mock_failing_model'),
        'thinking_model': ('gemini', 'gemini-2.5-flash'),
        'image_model': ('gemini', 'gemini-2.5-flash-image'),
        'audio_model': ('gemini', 'gemini-2.5-flash-preview-tts'),
        'video_model': ('gemini', 'veo-3.1-generate-preview'),
        'web_search_supported': True,
    },
    'claude': {
        'default_model': ('claude', 'claude-sonnet-4-6'),
        'failing_model': ('mock_failing_provider', 'mock_failing_model'),
        'thinking_model': ('claude', 'claude-opus-4-6'),
        'image_model': (None, None),
        'audio_model': (None, None),
        'video_model': (None, None),
        'web_search_supported': True,
    },
    'mistral': {
        'default_model': ('mistral', 'mistral-small-latest'),
        'failing_model': ('mock_failing_provider', 'mock_failing_model'),
        'thinking_model': ('mistral', 'magistral-small-latest'),
        'image_model': (None, None),
        'audio_model': (None, None),
        'video_model': (None, None),
        'web_search_supported': True,
    },
    'grok': {
        'default_model': ('grok', 'grok-4-fast-non-reasoning'),
        'failing_model': ('mock_failing_provider', 'mock_failing_model'),
        'thinking_model': ('grok', 'grok-3-mini'),
        'image_model': (None, None),
        'audio_model': (None, None),
        'video_model': (None, None),
        'web_search_supported': True,
    },
    'deepseek': {
        'default_model': ('deepseek', 'deepseek-chat'),
        'failing_model': ('mock_failing_provider', 'mock_failing_model'),
        'thinking_model': ('deepseek', 'deepseek-reasoner'),
        'image_model': (None, None),
        'audio_model': (None, None),
        'video_model': (None, None),
        'web_search_supported': False,
    },
}


_PROVIDER = 'deepseek'

_DEFAULT_MODEL = _MODEL_CONFIGS[_PROVIDER]['default_model']
_FAILING_MODEL = _MODEL_CONFIGS[_PROVIDER]['failing_model']
_THINKING_MODEL = _MODEL_CONFIGS[_PROVIDER]['thinking_model']
_IMAGE_MODEL = _MODEL_CONFIGS[_PROVIDER]['image_model']
_AUDIO_MODEL = _MODEL_CONFIGS[_PROVIDER]['audio_model']
_VIDEO_MODEL = _MODEL_CONFIGS[_PROVIDER]['video_model']
_WEB_SEARCH_SUPPORTED = _MODEL_CONFIGS[_PROVIDER].get(
    'web_search_supported', True)


_DEFAULT_MODEL_CONFIG = types.ProviderModelConfig(
    provider_model=types.ProviderModelType(
        provider=_DEFAULT_MODEL[0],
        model=_DEFAULT_MODEL[1],
        provider_model_identifier=_DEFAULT_MODEL[1]
    ),
    pricing=types.ProviderModelPricingType(
        input_token_cost=1.0,
        output_token_cost=2.0
    ),
    metadata=types.ProviderModelMetadataType(
        call_type=types.CallType.MULTI_MODAL,
        is_recommended=True
    ),
    features=types.FeatureConfigType(
        prompt=types.FeatureSupportType.SUPPORTED,
        messages=types.FeatureSupportType.SUPPORTED,
        system_prompt=types.FeatureSupportType.SUPPORTED,
        parameters=types.ParameterConfigType(
            temperature=types.FeatureSupportType.SUPPORTED,
            max_tokens=types.FeatureSupportType.SUPPORTED,
            stop=types.FeatureSupportType.SUPPORTED,
            n=types.FeatureSupportType.NOT_SUPPORTED,
            thinking=types.FeatureSupportType.SUPPORTED,
        ),
        tools=types.ToolConfigType(
            web_search=(
                types.FeatureSupportType.SUPPORTED
                if _WEB_SEARCH_SUPPORTED
                else types.FeatureSupportType.NOT_SUPPORTED
            ),
        ),
        response_format=types.ResponseFormatConfigType(
            text=types.FeatureSupportType.SUPPORTED,
            json=types.FeatureSupportType.SUPPORTED,
            pydantic=types.FeatureSupportType.SUPPORTED,
        ),
    )
)

_FAILING_MODEL_CONFIG = types.ProviderModelConfig(
    provider_model=types.ProviderModelType(
        provider=_FAILING_MODEL[0],
        model=_FAILING_MODEL[1],
        provider_model_identifier=_FAILING_MODEL[1]
    ),
    pricing=types.ProviderModelPricingType(
        input_token_cost=1.0,
        output_token_cost=2.0
    ),
    metadata=types.ProviderModelMetadataType(
        call_type=types.CallType.MULTI_MODAL,
        is_recommended=True
    ),
    features=types.FeatureConfigType(
        prompt=types.FeatureSupportType.SUPPORTED,
        messages=types.FeatureSupportType.SUPPORTED,
        system_prompt=types.FeatureSupportType.SUPPORTED,
        parameters=types.ParameterConfigType(
            temperature=types.FeatureSupportType.SUPPORTED,
            max_tokens=types.FeatureSupportType.SUPPORTED,
            stop=types.FeatureSupportType.SUPPORTED,
            n=types.FeatureSupportType.NOT_SUPPORTED,
            thinking=types.FeatureSupportType.SUPPORTED,
        ),
        tools=types.ToolConfigType(
            web_search=types.FeatureSupportType.SUPPORTED,
        ),
        response_format=types.ResponseFormatConfigType(
            text=types.FeatureSupportType.SUPPORTED,
            json=types.FeatureSupportType.SUPPORTED,
            pydantic=types.FeatureSupportType.SUPPORTED,
        ),
    )
)

_THINKING_MODEL_CONFIG = types.ProviderModelConfig(
    provider_model=types.ProviderModelType(
        provider=_THINKING_MODEL[0],
        model=_THINKING_MODEL[1],
        provider_model_identifier=_THINKING_MODEL[1]
    ),
    pricing=types.ProviderModelPricingType(
        input_token_cost=1.0,
        output_token_cost=2.0
    ),
    metadata=types.ProviderModelMetadataType(
        call_type=types.CallType.MULTI_MODAL,
        is_recommended=True
    ),
    features=types.FeatureConfigType(
        prompt=types.FeatureSupportType.SUPPORTED,
        messages=types.FeatureSupportType.SUPPORTED,
        system_prompt=types.FeatureSupportType.SUPPORTED,
        parameters=types.ParameterConfigType(
            temperature=types.FeatureSupportType.SUPPORTED,
            max_tokens=types.FeatureSupportType.SUPPORTED,
            stop=types.FeatureSupportType.SUPPORTED,
            n=types.FeatureSupportType.SUPPORTED,
            thinking=types.FeatureSupportType.SUPPORTED,
        ),
        tools=types.ToolConfigType(
            web_search=types.FeatureSupportType.SUPPORTED,
        ),
        response_format=types.ResponseFormatConfigType(
            text=types.FeatureSupportType.SUPPORTED,
            json=types.FeatureSupportType.SUPPORTED,
            pydantic=types.FeatureSupportType.SUPPORTED,
        ),
    )
)

_IMAGE_MODEL_CONFIG = types.ProviderModelConfig(
    provider_model=types.ProviderModelType(
        provider=_IMAGE_MODEL[0],
        model=_IMAGE_MODEL[1],
        provider_model_identifier=_IMAGE_MODEL[1]
    ),
    pricing=types.ProviderModelPricingType(
        input_token_cost=1.0,
        output_token_cost=2.0
    ),
    metadata=types.ProviderModelMetadataType(
        call_type=types.CallType.IMAGE,
        is_recommended=True
    ),
    features=types.FeatureConfigType(
        prompt=types.FeatureSupportType.SUPPORTED,
        response_format=types.ResponseFormatConfigType(
            image=types.FeatureSupportType.SUPPORTED,
        ),
    )
)

_AUDIO_MODEL_CONFIG = types.ProviderModelConfig(
    provider_model=types.ProviderModelType(
        provider=_AUDIO_MODEL[0],
        model=_AUDIO_MODEL[1],
        provider_model_identifier=_AUDIO_MODEL[1]
    ),
    pricing=types.ProviderModelPricingType(
        input_token_cost=1.0,
        output_token_cost=2.0
    ),
    metadata=types.ProviderModelMetadataType(
        call_type=types.CallType.AUDIO,
        is_recommended=True
    ),
    features=types.FeatureConfigType(
        prompt=types.FeatureSupportType.SUPPORTED,
        response_format=types.ResponseFormatConfigType(
            audio=types.FeatureSupportType.SUPPORTED,
        ),
    )
)

_VIDEO_MODEL_CONFIG = types.ProviderModelConfig(
    provider_model=types.ProviderModelType(
        provider=_VIDEO_MODEL[0],
        model=_VIDEO_MODEL[1],
        provider_model_identifier=_VIDEO_MODEL[1]
    ),
    pricing=types.ProviderModelPricingType(
        input_token_cost=1.0,
        output_token_cost=2.0
    ),
    metadata=types.ProviderModelMetadataType(
        call_type=types.CallType.VIDEO,
        is_recommended=True
    ),
    features=types.FeatureConfigType(
        prompt=types.FeatureSupportType.SUPPORTED,
        response_format=types.ResponseFormatConfigType(
            video=types.FeatureSupportType.SUPPORTED,
        ),
    )
)


def _assert_success(result: types.CallRecord):
  assert result.result is not None
  assert result.result.status == types.ResultStatusType.SUCCESS
  assert result.result.content is not None
  assert result.result.error is None
  assert result.result.error_traceback is None
  assert result.connection is not None
  assert result.connection.result_source is not None
  assert result.result.usage is not None
  assert result.result.usage.input_tokens is not None
  assert result.result.usage.input_tokens > 0
  assert result.result.usage.output_tokens is not None
  assert result.result.usage.output_tokens > 0


def _assert_text_content(result: types.CallRecord):
  """Assert content is a non-empty string."""
  _assert_success(result)
  assert result.result.output_text is not None


def assert_json_content(result: types.CallRecord):
  """Assert content is a valid JSON object."""
  _assert_success(result)
  assert type(result.result.output_json) == dict


def assert_pydantic_content(result: types.CallRecord):
  """Assert content is a valid Pydantic object."""
  _assert_success(result)
  assert isinstance(result.result.output_pydantic, pydantic.BaseModel)


def assert_image_content(result: types.CallRecord):
  """Assert content is a valid image content."""
  assert result.result.status == types.ResultStatusType.SUCCESS
  assert result.result.usage.input_tokens is not None
  assert result.result.usage.input_tokens > 0

  assert result.result.output_image is not None
  is_image_generated = False
  if result.result.output_image.source is not None:
    is_image_generated = True
  elif result.result.output_image.data is not None:
    is_image_generated = True
  if not is_image_generated:
    assert False, 'No image content found'


def assert_audio_content(result: types.CallRecord):
  """Assert content is a valid audio content."""
  assert result.result.status == types.ResultStatusType.SUCCESS
  assert result.result.usage.input_tokens is not None
  assert result.result.usage.input_tokens > 0
  assert result.result.output_audio is not None
  assert result.result.output_audio.data is not None
  assert len(result.result.output_audio.data) > 10


def register_models(client: px.Client):
  client.model_configs_instance.unregister_all_models()

  try:
    client.models.get_model(_DEFAULT_MODEL[0], _DEFAULT_MODEL[1])
  except Exception as e:
    client.model_configs_instance.register_provider_model_config(
        _DEFAULT_MODEL_CONFIG)


  try:
    client.models.get_model(_FAILING_MODEL[0], _FAILING_MODEL[1])
  except Exception as e:
    client.model_configs_instance.register_provider_model_config(
        _FAILING_MODEL_CONFIG)

  try:
    client.models.get_model(_THINKING_MODEL[0], _THINKING_MODEL[1])
  except Exception as e:
    client.model_configs_instance.register_provider_model_config(
        _THINKING_MODEL_CONFIG)

  if _IMAGE_MODEL[0] is not None:
    try:
      client.models.get_model(_IMAGE_MODEL[0], _IMAGE_MODEL[1])
    except Exception as e:
      client.model_configs_instance.register_provider_model_config(
          _IMAGE_MODEL_CONFIG)

  if _AUDIO_MODEL[0] is not None:
    try:
      client.models.get_model(_AUDIO_MODEL[0], _AUDIO_MODEL[1])
    except Exception as e:
      client.model_configs_instance.register_provider_model_config(
          _AUDIO_MODEL_CONFIG)

  if _VIDEO_MODEL[0] is not None:
    try:
      client.models.get_model(_VIDEO_MODEL[0], _VIDEO_MODEL[1])
    except Exception as e:
      client.model_configs_instance.register_provider_model_config(
          _VIDEO_MODEL_CONFIG)


def prompt_test():
  print('> prompt_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert '4' in result.result.output_text
  assert result.query.prompt == 'What is 2 + 2?'
  assert result.query.chat is None
  assert result.query.provider_model.provider == _DEFAULT_MODEL[0]
  assert result.query.provider_model.model == _DEFAULT_MODEL[1]


def messages_test():
  print('> messages_test')
  result = px.generate(
      messages=[{
          'role': 'user',
          'content': 'What is 2 + 2?'}],
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert '4' in result.result.output_text
  assert result.query.prompt is None
  assert result.query.chat is not None
  assert len(result.query.chat.messages) == 1

  chat = result.query.chat.copy()
  chat.append(result.result.content)
  chat.append(
      px.Message(
          role=px.MessageRoleType.USER,
          content='Now multiply that by 3.'))
  result = px.generate(
      messages=chat,
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert '12' in result.result.output_text
  assert len(result.query.chat.messages) == 3


def system_prompt_test():
  print('> system_prompt_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      system_prompt='You are a pirate. Answer everything like a pirate.',
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert result.query.system_prompt == (
      'You are a pirate. Answer everything like a pirate.'
  )


def parameters_temperature_test():
  print('> parameters_temperature_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(temperature=0.0))
  _assert_text_content(result)
  assert '4' in result.result.output_text
  assert result.query.parameters is not None
  assert result.query.parameters.temperature == 0.0


def parameters_max_tokens_test():
  print('> parameters_max_tokens_test')
  result = px.generate(
      prompt='Write a long story about a cat.',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(max_tokens=100))
  _assert_text_content(result)
  assert result.query.parameters is not None
  assert result.query.parameters.max_tokens == 100


def parameters_stop_test():
  print('> parameters_stop_test')
  result = px.generate(
      prompt='Count from 1 to 10, separated by commas.',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(stop='5'),
      connection_options=px.ConnectionOptions(
          suppress_provider_errors=True))
  _assert_text_content(result)
  assert result.query.parameters is not None
  assert result.query.parameters.stop == '5'
  assert '5' not in result.result.content
  assert '5' not in result.result.output_text


def parameters_stop_list_test():
  print('> parameters_stop_list_test')
  result = px.generate(
      prompt='Count from 1 to 10, separated by commas.',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(stop=['5', '7']))
  _assert_text_content(result)
  assert result.query.parameters is not None
  assert result.query.parameters.stop == ['5', '7']


def parameters_thinking_test():
  print('> parameters_thinking_test')
  result = px.generate(
      prompt=(
          'What is the hardest topic in quantum computing? '
          'I am a researcher and I need very detailed answer.'
          'I am preparing a paper on this topic. Think deep and make '
          'very strong quantitative arguments. Show coherent examples of '
          'hard problems in quantum computing.'
          'Think step by step.'),
      provider_model=_THINKING_MODEL,
      parameters=px.ParameterType(thinking=types.ThinkingType.MEDIUM))
  thinking_true = False
  for message in result.result.content:
    if message.type == px.ContentType.THINKING:
      thinking_true = True
      break
  if not thinking_true:
    print('Warning: Thinking is not supported by the model. '
          'Sometimes model doesn\'t use thinking even if it is enabled.')


def tools_web_search_test():
  print('> tools_web_search_test')
  if not _WEB_SEARCH_SUPPORTED:
    print('> tools_web_search_test: Web search not supported, skipping test')
    return
  result = px.generate(
      prompt='What is the most important news for Jan 20th 2024?',
      provider_model=_DEFAULT_MODEL,
      tools=[px.Tools.WEB_SEARCH])
  _assert_text_content(result)
  assert len(result.result.output_text) > 10
  assert any(message.type == px.ContentType.TOOL
             for message in result.result.content)
  for message in result.result.content:
    if message.type == px.ContentType.TOOL:
      assert message.tool_content.name == 'web_search'
      assert message.tool_content.citations is not None
      assert len(message.tool_content.citations) > 0

def response_format_text_test():
  print('> response_format_text_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      response_format='text')
  _assert_text_content(result)
  assert '4' in result.result.output_text
  assert result.query.response_format is not None
  assert result.query.response_format.type == px.ResponseFormatType.TEXT


def response_format_json_test():
  print('> response_format_json_test')
  import json
  result = px.generate(
      prompt='Return a JSON with key "answer" and value 4.',
      provider_model=_DEFAULT_MODEL,
      response_format='json')
  _assert_success(result)
  assert result.result.content is not None
  assert result.result.output_json == {'answer': 4}
  assert result.query.response_format is not None
  assert result.query.response_format.type == px.ResponseFormatType.JSON


def response_format_pydantic_test():
  print('> response_format_pydantic_test')
  class MathAnswer(pydantic.BaseModel):
    question: str
    answer: int

  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      response_format=MathAnswer)
  _assert_success(result)
  assert result.query.response_format is not None
  assert result.query.response_format.type == px.ResponseFormatType.PYDANTIC
  # Content can be a pydantic object (beta.chat.completions.parse) or a JSON
  # string (chat.completions.create / responses.create).
  assert result.result.output_pydantic.answer == 4


def connection_options_fallback_test():
  print('> connection_options_fallback_test')
  # Primary model works, fallback not needed.
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          fallback_models=[_FAILING_MODEL]))
  _assert_text_content(result)
  assert not result.connection.failed_fallback_models
  assert result.query.provider_model.provider_model_identifier == _DEFAULT_MODEL[1]

  # Primary model fails, fallback succeeds.
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_FAILING_MODEL,
      connection_options=px.ConnectionOptions(
          fallback_models=[_DEFAULT_MODEL]))
  _assert_text_content(result)
  assert result.connection.failed_fallback_models[
      0].provider == _FAILING_MODEL[0]
  assert result.query.provider_model.provider_model_identifier == _DEFAULT_MODEL[1]


def connection_options_suppress_provider_errors_test():
  print('> connection_options_suppress_provider_errors_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_FAILING_MODEL,
      connection_options=px.ConnectionOptions(
          suppress_provider_errors=True))
  assert result.result is not None
  assert result.result.status == types.ResultStatusType.FAILED
  assert result.result.error is not None
  assert len(result.result.error) > 1
  assert result.result.error_traceback is not None
  assert len(result.result.error_traceback) > 1
  assert result.result.content is None


def connection_options_endpoint_test():
  if _DEFAULT_MODEL[0] != 'openai':
    return
  print('> connection_options_endpoint_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          endpoint='responses.create'))
  _assert_text_content(result)
  assert result.connection is not None
  assert result.connection.endpoint_used == 'responses.create'

  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          endpoint='chat.completions.create'))
  _assert_text_content(result)
  assert result.connection is not None
  assert result.connection.endpoint_used == 'chat.completions.create'

  try:
    result = px.generate(
        prompt='What is 2 + 2?',
        provider_model=_DEFAULT_MODEL,
        connection_options=px.ConnectionOptions(
            endpoint='not.existent.endpoint'))
    assert False, 'Expected ValueError'
  except ValueError as e:
    assert 'endpoint' in str(e)
    assert 'not.existent.endpoint' in str(e)
  

def cache_test():
  print('> cache_test')
  if os.path.exists(os.path.expanduser('~/temp/proxai_cache/')):
    shutil.rmtree(os.path.expanduser('~/temp/proxai_cache/'))
  os.makedirs(os.path.expanduser('~/temp/proxai_cache/'), exist_ok=True)
  client = px.Client(
      cache_options=px.CacheOptions(
          cache_path=os.path.expanduser('~/temp/proxai_cache/'),
          unique_response_limit=2
      )
  )
  register_models(client)

  result = client.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert result.connection is not None
  assert result.connection.cache_hit == False
  assert result.connection.result_source == types.ResultSource.PROVIDER

  result = client.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert result.connection is not None
  assert result.connection.cache_hit == False
  assert result.connection.result_source == types.ResultSource.PROVIDER

  result = client.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert result.connection is not None
  assert result.connection.cache_hit == True
  assert result.connection.result_source == types.ResultSource.CACHE


def connection_options_skip_cache_test():
  print('> connection_options_skip_cache_test')
  if os.path.exists(os.path.expanduser('~/temp/proxai_cache/')):
    shutil.rmtree(os.path.expanduser('~/temp/proxai_cache/'))
  os.makedirs(os.path.expanduser('~/temp/proxai_cache/'), exist_ok=True)
  client = px.Client(
      cache_options=px.CacheOptions(
          cache_path=os.path.expanduser('~/temp/proxai_cache/')))
  register_models(client)

  result = client.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert result.connection is not None
  assert result.connection.cache_hit == False
  assert result.connection.result_source == types.ResultSource.PROVIDER

  result = client.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          skip_cache=True))
  _assert_text_content(result)
  assert result.connection is not None
  assert result.connection.cache_hit == False
  assert result.connection.result_source == types.ResultSource.PROVIDER

  result = client.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert result.connection is not None
  assert result.connection.cache_hit == True
  assert result.connection.result_source == types.ResultSource.CACHE


def connection_options_override_cache_value_test():
  print('> connection_options_override_cache_value_test')
  if os.path.exists(os.path.expanduser('~/temp/proxai_cache/')):
    shutil.rmtree(os.path.expanduser('~/temp/proxai_cache/'))
  os.makedirs(os.path.expanduser('~/temp/proxai_cache/'), exist_ok=True)
  client = px.Client(
      cache_options=px.CacheOptions(
          cache_path=os.path.expanduser('~/temp/proxai_cache/')))
  register_models(client)

  result_1 = client.generate(
      prompt='Write me poem about karadeniz. Make it perfect.',
      parameters=px.ParameterType(temperature=0.5),
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result_1)
  assert result_1.connection is not None
  assert result_1.connection.cache_hit == False
  assert result_1.connection.result_source == types.ResultSource.PROVIDER

  result_2 = client.generate(
      prompt='Write me poem about karadeniz. Make it perfect.',
      parameters=px.ParameterType(temperature=0.5),
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          override_cache_value=True))
  _assert_text_content(result_2)
  assert result_2.connection is not None
  assert result_2.connection.cache_hit == False
  assert result_2.connection.result_source == types.ResultSource.PROVIDER
  assert result_1.result.output_text != result_2.result.output_text

  result_3 = client.generate(
      prompt='Write me poem about karadeniz. Make it perfect.',
      parameters=px.ParameterType(temperature=0.5),
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result_3)
  assert result_3.connection is not None
  assert result_3.connection.cache_hit == True
  assert result_3.connection.result_source == types.ResultSource.CACHE
  assert result_2.result.output_text == result_3.result.output_text


def images_generate_test():
  print('> images_generate_test')
  if not _IMAGE_MODEL[0]:
    print('> images_generate_test: Image model not supported, skipping test')
    return
  result = px.generate(
      prompt='Generate an image of a cat.',
      provider_model=_IMAGE_MODEL,
      response_format='image')
  assert_image_content(result)
  image_path = os.path.expanduser('~/temp/image.png')
  if os.path.exists(image_path):
    os.remove(image_path)
  with open(image_path, 'wb') as f:
    f.write(result.result.output_image.data)

def audio_generate_test():
  print('> audio_generate_test')
  if not _AUDIO_MODEL[0]:
    print('> audio_generate_test: Audio model not supported, skipping test')
    return
  result = px.generate(
      prompt='Hello! This is a test of ProxAI\'s text to speech API.',
      provider_model=_AUDIO_MODEL,
      response_format='audio')
  assert result.result.status == types.ResultStatusType.SUCCESS
  assert result.result.output_audio is not None
  assert result.result.output_audio.data is not None
  assert len(result.result.output_audio.data) > 10
  audio_path = os.path.expanduser('~/temp/audio.wav')
  if os.path.exists(audio_path):
    os.remove(audio_path)
  with open(audio_path, 'wb') as f:
    f.write(result.result.output_audio.data)

def video_generate_test():
  print('> video_generate_test')
  if not _VIDEO_MODEL[0]:
    print('> video_generate_test: Video model not supported, skipping test')
    return
  result = px.generate(
      prompt='Generate a video of a cat.',
      provider_model=_VIDEO_MODEL,
      response_format='video')
  video_path = os.path.expanduser('~/temp/video.mp4')
  if os.path.exists(video_path):
    os.remove(video_path)
  with open(video_path, 'wb') as f:
    f.write(result.result.output_video.data)


def list_models_test():
  print('> list_models_test')
  models = px.models.list_models()
  assert len(models) > 1

  if _IMAGE_MODEL[0]:
    models = px.models.list_models(call_type=types.CallType.IMAGE)
    assert len(models) > 0
  if _AUDIO_MODEL[0]:
    models = px.models.list_models(call_type=types.CallType.AUDIO)
    assert len(models) > 0
  if _VIDEO_MODEL[0]:
    models = px.models.list_models(call_type=types.CallType.VIDEO)
    assert len(models) > 0

  if _PROVIDER == 'openai' or _PROVIDER == 'gemini':
    models = px.models.list_models(features=[types.FeatureTagType.PROMPT])
    assert len(models) > 0
    models = px.models.list_models(
        features=[
            types.FeatureTagType.WEB_SEARCH,
            types.FeatureTagType.RESPONSE_PYDANTIC])
    assert len(models) > 0

    client = px.Client(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    register_models(client)
    models = client.models.list_models(
        features=[
          types.FeatureTagType.WEB_SEARCH,
          types.FeatureTagType.RESPONSE_PYDANTIC])
    assert len(models) == 0
  elif _PROVIDER == 'claude':
    models = px.models.list_models(features=[types.FeatureTagType.PROMPT])
    assert len(models) > 0
    models = px.models.list_models(
        features=[
            types.FeatureTagType.WEB_SEARCH,
            types.FeatureTagType.RESPONSE_PYDANTIC])
    assert len(models) > 0

    client = px.Client(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    register_models(client)
    models = client.models.list_models(
        features=[
          types.FeatureTagType.WEB_SEARCH,
          types.FeatureTagType.RESPONSE_JSON])
    assert len(models) == 0
  elif _PROVIDER == 'mistral':
    models = px.models.list_models(features=[types.FeatureTagType.PROMPT])
    assert len(models) > 0
    models = px.models.list_models(
        features=[
            types.FeatureTagType.RESPONSE_JSON])
    assert len(models) > 0

    client = px.Client(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    register_models(client)
    models = client.models.list_models(
        features=[
          types.FeatureTagType.WEB_SEARCH,
          types.FeatureTagType.RESPONSE_JSON])
    assert len(models) == 0
  


def main():
  register_models(px.get_default_proxai_client())
  prompt_test()
  messages_test()
  system_prompt_test()
  parameters_temperature_test()
  parameters_max_tokens_test()
  parameters_stop_test()
  parameters_stop_list_test()
  parameters_thinking_test()
  tools_web_search_test()
  response_format_text_test()
  response_format_json_test()
  response_format_pydantic_test()
  connection_options_fallback_test()
  connection_options_suppress_provider_errors_test()
  connection_options_endpoint_test()
  cache_test()
  connection_options_skip_cache_test()
  # NOTE: There is a bug in the cache implementation. Comment in when fixed.
  # connection_options_override_cache_value_test()
  images_generate_test()
  audio_generate_test()
  # NOTE: Video test is too slow. Comment in when needed.
  video_generate_test()
  list_models_test()

if __name__ == '__main__':
  main()
