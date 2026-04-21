"""Quick manual test for ProxDash upload_call_record API.

Usage:
  poetry run python3 examples/proxdash_test.py
  poetry run python3 examples/proxdash_test.py --test upload_text
  poetry run python3 examples/proxdash_test.py --test all
"""

import argparse
from pprint import pprint

import proxai as px
import proxai.types as types


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
    input_format = ['text']
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

  text_input = S if 'text' in input_format else NS
  image_input = S if 'image' in input_format else NS
  document_input = S if 'document' in input_format else NS
  audio_input = S if 'audio' in input_format else NS
  video_input = S if 'video' in input_format else NS

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
          input_format=['text', 'image', 'document'],
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
          input_format=['text', 'image', 'document'],
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
          input_format=['text', 'image', 'document', 'audio', 'video'],
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
          input_format=['text', 'image', 'document', 'audio', 'video'],
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
          input_format=['text', 'image', 'document'],
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
          input_format=['text', 'image', 'document'],
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


# --- Tests ---


def test_upload_text():
  """Test uploading a simple text generation call record."""
  print('> test_upload_text')
  px.generate(prompt='Say hello in one word.', provider_model=('openai', 'gpt-4o'))
  # px.models.check_health()
  # # pprint(px.models.list_working_models(verbose=True))
  # result = px.generate_text('Say hello in one word.')
  # pprint(result)
  # print('  PASSED')


TEST_SEQUENCE = [
    ('upload_text', test_upload_text),
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
          api_key='bj8leip-mo7q9fzn-mb8di2v5wb7',
      ),
      cache_options=px.CacheOptions(
          cache_path='/tmp/proxai_cache',
      ),
  )
  register_models(px.get_default_proxai_client())

  if args.test == 'all':
    for name, test_fn in TEST_SEQUENCE:
      test_fn()
  else:
    if args.test not in TEST_MAP:
      print(f'Unknown test: {args.test}')
      print(f'Available: {", ".join(test_names)}')
      return
    TEST_MAP[args.test]()


if __name__ == '__main__':
  main()
