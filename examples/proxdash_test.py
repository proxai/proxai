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
_MAIN_MODEL = ('gemini', 'gemini-3-flash')


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


def test_markdown_input_grok():
  """Test markdown input with grok (best effort - reads file as inline text)."""
  print('\n=== test_markdown_input_grok ===')
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
      }], provider_model=('grok', 'grok-3')
  )
  _print_result(call_record)


def test_markdown_input_mock():
  """Test markdown input with mock provider (best effort)."""
  print('\n=== test_markdown_input_mock ===')
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
      }], provider_model=('mock_provider', 'mock_model')
  )
  _print_result(call_record)


def test_markdown_input_mock_failing():
  """Test markdown input with mock failing provider."""
  print('\n=== test_markdown_input_mock_failing ===')
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
      }],
      provider_model=('mock_failing_provider', 'mock_failing_model'),
      connection_options=px.ConnectionOptions(
          suppress_provider_errors=True,
      ),
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
      provider_model=('gemini', 'gemini-2.5-flash-tts'),
      output_format='audio',
  )
  _print_result(call_record)


def test_video_output():
  """Test video output format."""
  print('\n=== test_video_output ===')
  call_record = px.generate(
      prompt='A cat playing with a ball of yarn.',
      provider_model=('gemini', 'veo-3.1-generate'),
      output_format='video',
  )
  _print_result(call_record)


TEST_SEQUENCE = [
    ('text_input', test_text_input),
    ('json_input', test_json_input),
    ('pydantic_input', test_pydantic_input),
    ('markdown_input', test_markdown_input),
    ('markdown_input_grok', test_markdown_input_grok),
    ('markdown_input_mock', test_markdown_input_mock),
    ('markdown_input_mock_failing', test_markdown_input_mock_failing),
    ('image_input', test_image_input),
    ('audio_input', test_audio_input),
    ('video_input', test_video_input),
    ('json_output', test_json_output),
    ('pydantic_output', test_pydantic_output),
    ('image_output', test_image_output),
    ('audio_output', test_audio_output),
    ('video_output', test_video_output),
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
      cache_options=px.CacheOptions(
          cache_path='/tmp/proxai_cache',
          # clear_query_cache_on_connect=True,
      ),
      provider_call_options=px.ProviderCallOptions(
          allow_parallel_file_operations=True,
      ),
  )

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
