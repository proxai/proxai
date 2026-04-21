"""Quick manual test for ProxDash upload_call_record API.

Usage:
  poetry run python3 examples/proxdash_test.py
  poetry run python3 examples/proxdash_test.py --test text_input
  poetry run python3 examples/proxdash_test.py --test all
"""

import argparse
import os
from pprint import pprint

import pydantic

import proxai as px
import proxai.serializers.type_serializer as type_serializer
import proxai.types as types

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'refactoring_test_assets')
_MAIN_MODEL = ('gemini', 'gemini-3-flash-preview')


def _get_model_config(
    provider: str,
    model: str,
    provider_model_identifier: str,
    web_search: bool = False,
    input_format: list[str] | None = None,
    output_format: list[types.OutputFormatType] | None = None,
):
  S = types.FeatureSupportType.SUPPORTED
  NS = types.FeatureSupportType.NOT_SUPPORTED
  if input_format is None:
    input_format = ['text', 'json', 'pydantic']
  if output_format is None:
    output_format = [types.OutputFormatType.TEXT]

  web_search_supported = S if web_search else NS
  text_supported = (S if types.OutputFormatType.TEXT in output_format else NS)
  json_supported = (S if types.OutputFormatType.JSON in output_format else NS)
  pydantic_supported = (
      S if types.OutputFormatType.PYDANTIC in output_format else NS
  )
  image_supported = (S if types.OutputFormatType.IMAGE in output_format else NS)
  audio_supported = (S if types.OutputFormatType.AUDIO in output_format else NS)
  video_supported = (S if types.OutputFormatType.VIDEO in output_format else NS)
  multi_modal_supported = (
      S if types.OutputFormatType.MULTI_MODAL in output_format else NS
  )

  BE = types.FeatureSupportType.BEST_EFFORT
  text_input = S if 'text' in input_format else NS
  image_input = S if 'image' in input_format else NS
  document_input = S if 'document' in input_format else NS
  audio_input = S if 'audio' in input_format else NS
  video_input = S if 'video' in input_format else NS
  json_input = BE if 'json' in input_format else NS
  pydantic_input = BE if 'pydantic' in input_format else NS

  return types.ProviderModelConfig(
      provider_model=types.ProviderModelType(
          provider=provider, model=model,
          provider_model_identifier=provider_model_identifier
      ), pricing=types.ProviderModelPricingType(
          input_token_cost=1.0, output_token_cost=2.0
      ), metadata=types.ProviderModelMetadataType(is_recommended=True),
      features=types.FeatureConfigType(
          prompt=S,
          messages=S,
          system_prompt=S,
          parameters=types.ParameterConfigType(
              temperature=S, max_tokens=S, stop=S, n=NS, thinking=NS
          ),
          tools=types.ToolConfigType(web_search=web_search_supported),
          output_format=types.OutputFormatConfigType(
              text=text_supported,
              json=json_supported,
              pydantic=pydantic_supported,
              image=image_supported,
              audio=audio_supported,
              video=video_supported,
              multi_modal=multi_modal_supported,
          ),
          input_format=types.InputFormatConfigType(
              text=text_input,
              image=image_input,
              document=document_input,
              audio=audio_input,
              video=video_input,
              json=json_input,
              pydantic=pydantic_input,
          ),
      )
  )


def register_models(client: px.Client):
  client.model_configs_instance.unregister_all_models()

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='mock_failing_provider',
          model='mock_failing_model',
          provider_model_identifier='mock_failing_model',
          web_search=False,
          output_format=[
              types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC
          ],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='gpt-4o',
          provider_model_identifier='gpt-4o',
          web_search=True,
          input_format=['text', 'image', 'document', 'json', 'pydantic'],
          output_format=[
              types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC
          ],
      )
  )
  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='o3',
          provider_model_identifier='o3',
          web_search=False,
          input_format=['text', 'image', 'document', 'json', 'pydantic'],
          output_format=[
              types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC
          ],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='dall-e-3',
          provider_model_identifier='dall-e-3',
          web_search=False,
          output_format=[types.OutputFormatType.IMAGE],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='tts-1',
          provider_model_identifier='tts-1',
          web_search=False,
          output_format=[types.OutputFormatType.AUDIO],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='sora-2',
          provider_model_identifier='sora-2',
          web_search=False,
          output_format=[types.OutputFormatType.VIDEO],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='gemini-3-flash-preview',
          provider_model_identifier='gemini-3-flash-preview',
          web_search=True,
          input_format=[
              'text', 'image', 'json', 'pydantic', 'document', 'audio', 'video'
          ],
          output_format=[
              types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC
          ],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='gemini-2.5-flash',
          provider_model_identifier='gemini-2.5-flash',
          web_search=False,
          input_format=[
              'text', 'image', 'document', 'audio', 'video', 'json', 'pydantic'
          ],
          output_format=[
              types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC
          ],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='gemini-2.5-flash-image',
          provider_model_identifier='gemini-2.5-flash-image',
          web_search=False,
          output_format=[types.OutputFormatType.IMAGE],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='gemini-2.5-flash-preview-tts',
          provider_model_identifier='gemini-2.5-flash-preview-tts',
          web_search=False,
          output_format=[types.OutputFormatType.AUDIO],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='veo-3.1-generate-preview',
          provider_model_identifier='veo-3.1-generate-preview',
          web_search=False,
          output_format=[types.OutputFormatType.VIDEO],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='claude',
          model='claude-sonnet-4-6',
          provider_model_identifier='claude-sonnet-4-6',
          web_search=True,
          input_format=['text', 'image', 'document', 'json', 'pydantic'],
          output_format=[
              types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC
          ],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='claude',
          model='claude-opus-4-6',
          provider_model_identifier='claude-opus-4-6',
          web_search=False,
          input_format=['text', 'image', 'document', 'json', 'pydantic'],
          output_format=[
              types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC
          ],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='deepseek',
          model='deepseek-chat',
          provider_model_identifier='deepseek-chat',
          web_search=False,
          input_format=['text', 'json', 'pydantic'],
          output_format=[
              types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC
          ],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='deepseek',
          model='deepseek-reasoner',
          provider_model_identifier='deepseek-reasoner',
          web_search=False,
          input_format=['text', 'json', 'pydantic'],
          output_format=[
              types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC
          ],
      )
  )

  client.model_configs_instance.override_default_model_priority_list([
      px.models.get_model('gemini', 'gemini-3-flash-preview'),
      px.models.get_model('openai', 'gpt-4o'),
      px.models.get_model('claude', 'claude-sonnet-4-6'),
  ])


def _asset(filename):
  return os.path.join(_ASSETS_DIR, filename)


def _truncate_data(obj, max_len=80):
  """Truncate long string/bytes values in nested dicts/lists for display."""
  if isinstance(obj, dict):
    return {k: _truncate_data(v, max_len) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_truncate_data(v, max_len) for v in obj]
  if isinstance(obj, str) and len(obj) > max_len:
    return obj[:max_len] + f'... ({len(obj)} chars)'
  return obj


def _print_result(call_record):
  data = type_serializer.encode_call_record(call_record)
  pprint(_truncate_data(data))
  print(
      '> Check latest request in ProxDash: '
      'http://localhost:3000/dashboard/requests'
  )
  input('> Press Enter to continue...')


# --- Input format tests ---


def test_text_input():
  """Test text prompt input."""
  print('\n=== test_text_input ===')
  call_record = px.generate(
      prompt='Say hello in one word.', provider_model=_MAIN_MODEL
  )
  _print_result(call_record)


def test_json_input():
  """Test JSON content input."""
  print('\n=== test_json_input ===')
  call_record = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.JSON,
                  json={
                      'title': 'This is a love letter to cats!',
                      'description':
                          ('I want to explain my feelings in json format.'),
                  },
              ),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is this about?',
              ),
          ],
      }], provider_model=_MAIN_MODEL
  )
  _print_result(call_record)


def test_pydantic_input():
  """Test pydantic content input."""
  print('\n=== test_pydantic_input ===')

  class CatDescription(pydantic.BaseModel):
    name: str = 'Whiskers'
    color: str = 'orange'
    favorite_food: str = 'tuna'

  call_record = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.PYDANTIC_INSTANCE,
                  pydantic_content=types.PydanticContent(
                      instance_value=CatDescription(),
                  ),
              ),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is this about?',
              ),
          ],
      }], provider_model=_MAIN_MODEL
  )
  _print_result(call_record)


def test_markdown_input():
  """Test markdown document input."""
  print('\n=== test_markdown_input ===')
  call_record = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.DOCUMENT,
                  path=_asset('cat.md'),
                  media_type='text/markdown',
              ),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is inside this document?',
              ),
          ],
      }], provider_model=_MAIN_MODEL
  )
  _print_result(call_record)


def test_image_input():
  """Test image input."""
  print('\n=== test_image_input ===')
  call_record = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.IMAGE,
                  path=_asset('cat.jpeg'),
                  media_type='image/jpeg',
              ),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is in this image?',
              ),
          ],
      }], provider_model=_MAIN_MODEL
  )
  _print_result(call_record)


def test_audio_input():
  """Test audio input."""
  print('\n=== test_audio_input ===')
  call_record = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.AUDIO,
                  path=_asset('cat.mp3'),
                  media_type='audio/mpeg',
              ),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is this audio about?',
              ),
          ],
      }], provider_model=_MAIN_MODEL
  )
  _print_result(call_record)


def test_video_input():
  """Test video input."""
  print('\n=== test_video_input ===')
  call_record = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.VIDEO,
                  path=_asset('cat.mp4'),
                  media_type='video/mp4',
              ),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is in this video?',
              ),
          ],
      }], provider_model=_MAIN_MODEL
  )
  _print_result(call_record)


# --- Output format tests ---


def test_json_output():
  """Test JSON output format."""
  print('\n=== test_json_output ===')
  call_record = px.generate(
      prompt='Return a JSON with key "answer" and value 4.',
      provider_model=_MAIN_MODEL, output_format='json'
  )
  _print_result(call_record)


def test_pydantic_output():
  """Test pydantic output format."""
  print('\n=== test_pydantic_output ===')

  class MathAnswer(pydantic.BaseModel):
    question: str
    answer: int

  call_record = px.generate(
      prompt='What is 2 + 2?', provider_model=_MAIN_MODEL,
      output_format=MathAnswer
  )
  _print_result(call_record)


def test_image_output():
  """Test image output format."""
  print('\n=== test_image_output ===')
  call_record = px.generate(
      prompt='Make a funny cartoon cat in a living room.',
      provider_model=('gemini', 'gemini-2.5-flash-image'),
      output_format='image',
  )
  _print_result(call_record)


def test_audio_output():
  """Test audio output format."""
  print('\n=== test_audio_output ===')
  call_record = px.generate(
      prompt='Say hello world in a cheerful voice.',
      provider_model=('gemini', 'gemini-2.5-flash-preview-tts'),
      output_format='audio',
  )
  _print_result(call_record)


def test_video_output():
  """Test video output format."""
  print('\n=== test_video_output ===')
  call_record = px.generate(
      prompt='A cat playing with a ball of yarn.',
      provider_model=('gemini', 'veo-3.1-generate-preview'),
      output_format='video',
  )
  _print_result(call_record)


TEST_SEQUENCE = [
    # ('text_input', test_text_input),
    # ('json_input', test_json_input),
    # ('pydantic_input', test_pydantic_input),
    # ('markdown_input', test_markdown_input),
    # ('image_input', test_image_input),
    # ('audio_input', test_audio_input),
    # ('video_input', test_video_input),
    # ('json_output', test_json_output),
    # ('pydantic_output', test_pydantic_output),
    ('image_output', test_image_output),
    # ('audio_output', test_audio_output),
    # ('video_output', test_video_output),
]
TEST_MAP = dict(TEST_SEQUENCE)


def main():
  parser = argparse.ArgumentParser(description='ProxDash manual test')
  test_names = [name for name, _ in TEST_SEQUENCE]
  parser.add_argument(
      '--test', default='all',
      help=f'Test to run: {", ".join(test_names)}, or "all"'
  )
  args = parser.parse_args()

  px.connect(
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          base_url='http://localhost:3001',
          api_key='pjlfi0h-mo8mrm56-fgsvgftdk78',
      ),
      cache_options=px.CacheOptions(cache_path='/tmp/proxai_cache',),
  )
  register_models(px.get_default_proxai_client())

  if args.test == 'all':
    for _name, test_fn in TEST_SEQUENCE:
      test_fn()
  else:
    if args.test not in TEST_MAP:
      print(f'Unknown test: {args.test}')
      print(f'Available: {", ".join(test_names)}')
      return
    TEST_MAP[args.test]()


if __name__ == '__main__':
  main()
