import argparse
import os
import subprocess
import tempfile
import urllib.request
from pprint import pprint
from pydantic import BaseModel
import proxai as px
import proxai.types as types


def list_models_examples():

  def _names(models):
    return {m.model for m in models}

  def _model_names(models):
    return sorted(m.model for m in models)

  print('> list_models_examples')
  print()

  # --- output_format filtering ---
  print('>> px.models.list_models()')
  text_models = px.models.list_models()
  print(f'    Default (text output): {_model_names(text_models)}')
  text_names = _names(text_models)
  assert 'gpt-4o' in text_names
  assert 'gemini-2.5-flash' in text_names
  assert 'sonnet-4.6' in text_names
  assert 'dall-e-3' not in text_names
  assert 'tts-1' not in text_names
  assert 'sora-2' not in text_names

  print()
  print('>> px.models.list_models(output_format=...)')
  image_models = px.models.list_models(
      output_format=types.OutputFormatType.IMAGE,
      recommended_only=False,
  )
  print(f'    IMAGE:    {_model_names(image_models)}')
  image_names = _names(image_models)
  assert 'dall-e-3' in image_names
  assert 'gemini-2.5-flash-image' in image_names
  assert 'gpt-4o' not in image_names

  audio_models = px.models.list_models(
      output_format=types.OutputFormatType.AUDIO,
      recommended_only=False,
  )
  print(f'    AUDIO:    {_model_names(audio_models)}')
  assert 'tts-1' in _names(audio_models)
  assert 'gemini-2.5-flash-tts' in _names(audio_models)
  assert 'gpt-4o' not in _names(audio_models)

  video_models = px.models.list_models(
      output_format=types.OutputFormatType.VIDEO,
      recommended_only=False,
  )
  print(f'    VIDEO:    {_model_names(video_models)}')
  assert 'sora-2' in _names(video_models)
  assert 'veo-3.1-generate' in _names(video_models)
  assert 'gpt-4o' not in _names(video_models)

  json_models = px.models.list_models(output_format=types.OutputFormatType.JSON)
  print(f'    JSON:     {_model_names(json_models)}')
  json_names = _names(json_models)
  assert 'gpt-4o' in json_names
  assert 'dall-e-3' not in json_names

  # --- input_format filtering ---
  print()
  print('>> px.models.list_models(input_format=...)')
  image_input = px.models.list_models(input_format=types.InputFormatType.IMAGE)
  print(f'    IMAGE:    {_model_names(image_input)}')
  image_input_names = _names(image_input)
  assert 'gpt-4o' in image_input_names
  assert 'sonnet-4.6' in image_input_names
  assert 'gemini-3-flash' in image_input_names
  assert 'deepseek-v4-flash' not in image_input_names
  assert 'dall-e-3' not in image_input_names

  audio_input = px.models.list_models(input_format=types.InputFormatType.AUDIO)
  print(f'    AUDIO:    {_model_names(audio_input)}')
  audio_input_names = _names(audio_input)
  assert 'gemini-3-flash' in audio_input_names
  assert 'gpt-4o' not in audio_input_names

  doc_input = px.models.list_models(input_format=types.InputFormatType.DOCUMENT)
  print(f'    DOCUMENT: {_model_names(doc_input)}')
  doc_input_names = _names(doc_input)
  assert 'gpt-4o' in doc_input_names
  assert 'sonnet-4.6' in doc_input_names

  # --- combined input + output filtering ---
  print()
  print('>> px.models.list_models(input_format=IMAGE,'
        ' output_format=JSON)')
  image_in_json_out = px.models.list_models(
      input_format=types.InputFormatType.IMAGE,
      output_format=types.OutputFormatType.JSON
  )
  print(f'    Result:   {_model_names(image_in_json_out)}')
  combined_names = _names(image_in_json_out)
  assert 'gpt-4o' in combined_names
  assert 'gemini-3-flash' in combined_names
  assert 'deepseek-v4-flash' not in combined_names
  assert 'dall-e-3' not in combined_names

  # --- tool_tags filtering ---
  print()
  print('>> px.models.list_models(tool_tags=...)')
  web_search_models = px.models.list_models(tool_tags=types.ToolTag.WEB_SEARCH)
  print(f'    WEB_SEARCH: {_model_names(web_search_models)}')
  ws_names = _names(web_search_models)
  assert 'gpt-4o' in ws_names
  assert 'gemini-3-flash' in ws_names
  assert 'sonnet-4.6' in ws_names
  assert 'deepseek-v4-flash' not in ws_names
  assert 'grok-3' not in ws_names

  # --- feature_tags filtering ---
  print()
  print('>> px.models.list_models(feature_tags=...)')
  thinking_models = px.models.list_models(
      feature_tags=types.FeatureTag.THINKING
  )
  print(f'    THINKING: {_model_names(thinking_models)}')
  thinking_names = _names(thinking_models)
  assert 'o3' in thinking_names
  assert 'opus-4.6' in thinking_names
  assert 'gpt-4o' not in thinking_names

  # --- list_providers ---
  print()
  print('>> px.models.list_providers()')
  providers = px.models.list_providers()
  print(f'    Providers: {providers}')
  assert 'openai' in providers
  assert 'gemini' in providers

  # --- list_provider_models ---
  print()
  print('>> px.models.list_provider_models(provider)')
  openai_models = px.models.list_provider_models('openai')
  print(f'    openai: {_model_names(openai_models)}')
  openai_names = _names(openai_models)
  assert 'gpt-4o' in openai_names
  assert 'o3' in openai_names
  assert 'gemini-2.5-flash' not in openai_names

  gemini_models = px.models.list_provider_models('gemini')
  print(f'    gemini: {_model_names(gemini_models)}')
  assert 'gemini-3-flash' in _names(gemini_models)

  # --- get_model ---
  print()
  print('>> px.models.get_model(provider, model)')
  model = px.models.get_model('openai', 'gpt-4o')
  print(f'    get_model("openai", "gpt-4o") = {model}')
  assert model.provider == 'openai'
  assert model.model == 'gpt-4o'

  print()


def check_health_use():
  print('> Default client check health')
  px.models.check_health(verbose=True)

  print('> Custom client check health with no multi processing and timeout=1')
  client = px.Client(
      model_probe_options=px.ModelProbeOptions(
          allow_multiprocessing=False,
          timeout=1,
      ),
  )
  client.models.check_health(verbose=True)


def plain_alias_function_use():
  print('> plain_alias_function_use')

  print('* generate_text:')
  print('OpenAI:')
  response = px.generate_text(
      prompt='What is the capital of France and which AI provider are you?',
      provider_model=('openai', 'gpt-4o')
  )
  print(response)

  print('Gemini:')
  response = px.generate_text(
      prompt='What is the capital of France and which AI provider are you?',
      provider_model=('gemini', 'gemini-3-flash')
  )
  print(response)

  print('* generate_json:')
  print('OpenAI:')
  response = px.generate_json(
      prompt='Give me the list of biggest cities in France.',
      provider_model=('openai', 'gpt-4o')
  )
  pprint(response)

  print('Gemini:')
  response = px.generate_json(
      prompt='Give me the list of biggest cities in France.',
      provider_model=('gemini', 'gemini-3-flash')
  )
  pprint(response)

  print('* generate_pydantic:')

  class BiggestCities(BaseModel):
    cities: list[str]

  print('OpenAI:')
  response = px.generate_pydantic(
      prompt='Give me the list of biggest cities in France.',
      output_format=BiggestCities, provider_model=('openai', 'gpt-4o')
  )
  print(response)

  print('Gemini:')
  response = px.generate_pydantic(
      prompt='Give me the list of biggest cities in France.',
      output_format=BiggestCities,
      provider_model=('gemini', 'gemini-3-flash')
  )
  print(response)


def _open_image_response(response: types.MessageContent | None):
  if not response:
    print('> No response')
    return

  suffix = '.png'
  if response.media_type and '/' in response.media_type:
    ext = response.media_type.split('/', 1)[1]
    if ext == 'jpeg':
      ext = 'jpg'
    suffix = f'.{ext}'

  tmp_path = None
  try:
    if response.data:
      with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.data)
        tmp_path = tmp.name
    elif response.source:
      with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        with urllib.request.urlopen(response.source) as http_resp:
          tmp.write(http_resp.read())
        tmp_path = tmp.name
    else:
      print('> No image data or source')
      return

    print(f'> Opening image preview: {tmp_path}')
    subprocess.run(['qlmanage', '-p', tmp_path], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)
  finally:
    if tmp_path and os.path.exists(tmp_path):
      os.unlink(tmp_path)


def image_alias_function_use():
  print('> image_alias_function_use')

  print('* generate_image:')
  print('OpenAI:')
  response = px.generate_image(
      prompt='Make funny cartoon cat in living room.',
      provider_model=('openai', 'dall-e-3')
  )
  _open_image_response(response)

  print('Gemini:')
  response = px.generate_image(
      prompt='Make funny cartoon cat in living room.',
      provider_model=('gemini', 'gemini-2.5-flash-image')
  )
  _open_image_response(response)


def set_model_use():
  print('> set_model_use')

  print('* set_model text models:')

  print('OpenAI:')
  px.set_model(('openai', 'gpt-4o'))
  response = px.generate_text(prompt='Which provider model are you?')
  print(response)
  assert 'openai' in response.lower()

  print('Gemini:')
  px.set_model(('gemini', 'gemini-3-flash'))
  response = px.generate_text(prompt='Which provider model are you?')
  print(response)
  assert 'gemini' in response.lower() or 'google' in response.lower()

  print('* set_model image models:')
  print('OpenAI:')
  px.set_model(generate_image=('openai', 'dall-e-3'))
  response = px.generate_image(prompt='Make funny cartoon cat in living room.')
  _open_image_response(response)

  print('Gemini:')
  px.set_model(generate_image=('gemini', 'gemini-2.5-flash-image'))
  response = px.generate_image(prompt='Make funny cartoon cat in living room.')
  _open_image_response(response)


TEST_SEQUENCE = [
    ('list_models_examples', list_models_examples),
    ('check_health_use', check_health_use),
    ('plain_alias_function_use', plain_alias_function_use),
    ('image_alias_function_use', image_alias_function_use),
    ('set_model_use', set_model_use),
]
TEST_MAP = dict(TEST_SEQUENCE)


def main():
  parser = argparse.ArgumentParser(description='Alias API manual test')
  test_names = [name for name, _ in TEST_SEQUENCE]
  parser.add_argument(
      '--test', default='all',
      help=f'Test to run: {", ".join(test_names)}, or "all"')
  args = parser.parse_args()

  if args.test == 'all':
    for name, test_fn in TEST_SEQUENCE:
      try:
        test_fn()
      except Exception as e:
        print(f'  FAILED [{name}]: {e}')
  else:
    if args.test not in TEST_MAP:
      print(f'Unknown test: {args.test}')
      print(f'Available: {", ".join(test_names)}')
      return
    TEST_MAP[args.test]()

if __name__ == "__main__":
  main()
