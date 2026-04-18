import os
import subprocess
import tempfile
import urllib.request
from pprint import pprint
from pydantic import BaseModel
import proxai as px
import proxai.types as types


def _get_model_config(
    provider: str,
    model: str,
    provider_model_identifier: str,
    web_search: bool = False,
    input_format: list[str] | None = None,
    output_format: list[types.OutputFormatType] = [types.OutputFormatType.TEXT],
):
  """Get a model config for a given parameters."""
  S = types.FeatureSupportType.SUPPORTED
  NS = types.FeatureSupportType.NOT_SUPPORTED
  if input_format is None:
    input_format = ['text']

  web_search_supported = S if web_search else NS
  text_supported = (
      S if types.OutputFormatType.TEXT in output_format else NS)
  json_supported = (
      S if types.OutputFormatType.JSON in output_format else NS)
  pydantic_supported = (
      S if types.OutputFormatType.PYDANTIC in output_format else NS)
  image_supported = (
      S if types.OutputFormatType.IMAGE in output_format else NS)
  audio_supported = (
      S if types.OutputFormatType.AUDIO in output_format else NS)
  video_supported = (
      S if types.OutputFormatType.VIDEO in output_format else NS)
  multi_modal_supported = (
      S if types.OutputFormatType.MULTI_MODAL in output_format
      else NS)
  return types.ProviderModelConfig(
      provider_model=types.ProviderModelType(
          provider=provider,
          model=model,
          provider_model_identifier=provider_model_identifier
      ),
      pricing=types.ProviderModelPricingType(
          input_token_cost=1.0,
          output_token_cost=2.0
      ),
      metadata=types.ProviderModelMetadataType(
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
              web_search=web_search_supported,
          ),
          input_format=types.InputFormatConfigType(
              text=S if 'text' in input_format else NS,
              image=S if 'image' in input_format else NS,
              document=S if 'document' in input_format else NS,
              audio=S if 'audio' in input_format else NS,
              video=S if 'video' in input_format else NS,
              json=S if 'json' in input_format else NS,
              pydantic=S if 'pydantic' in input_format else NS,
          ),
          output_format=types.OutputFormatConfigType(
              text=text_supported,
              json=json_supported,
              pydantic=pydantic_supported,
              image=image_supported,
              audio=audio_supported,
              video=video_supported,
              multi_modal=multi_modal_supported,
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
              types.OutputFormatType.TEXT,
              types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC],
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
              types.OutputFormatType.TEXT,
              types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC],
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
              types.OutputFormatType.TEXT,
              types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC],
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
              types.OutputFormatType.TEXT,
              types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC],
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
              types.OutputFormatType.TEXT,
              types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC],
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
              types.OutputFormatType.TEXT,
              types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC],
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
              types.OutputFormatType.TEXT,
              types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='deepseek',
          model='deepseek-chat',
          provider_model_identifier='deepseek-chat',
          web_search=False,
          output_format=[
              types.OutputFormatType.TEXT,
              types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='deepseek',
          model='deepseek-reasoner',
          provider_model_identifier='deepseek-reasoner',
          web_search=False,
          output_format=[
              types.OutputFormatType.TEXT,
              types.OutputFormatType.JSON,
              types.OutputFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.override_default_model_priority_list([
      px.models.get_model('gemini', 'gemini-3-flash-preview'),
      px.models.get_model('openai', 'gpt-4o'),
      px.models.get_model('claude', 'claude-sonnet-4-6'),
  ])


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
  assert 'dall-e-3' not in text_names
  assert 'tts-1' not in text_names
  assert 'sora-2' not in text_names

  print()
  print('>> px.models.list_models(output_format=...)')
  image_models = px.models.list_models(
      output_format=types.OutputFormatType.IMAGE)
  print(f'    IMAGE:    {_model_names(image_models)}')
  image_names = _names(image_models)
  assert 'dall-e-3' in image_names
  assert 'gemini-2.5-flash-image' in image_names
  assert 'gpt-4o' not in image_names

  audio_models = px.models.list_models(
      output_format=types.OutputFormatType.AUDIO)
  print(f'    AUDIO:    {_model_names(audio_models)}')
  assert 'tts-1' in _names(audio_models)
  assert 'gpt-4o' not in _names(audio_models)

  video_models = px.models.list_models(
      output_format=types.OutputFormatType.VIDEO)
  print(f'    VIDEO:    {_model_names(video_models)}')
  assert 'sora-2' in _names(video_models)
  assert 'gpt-4o' not in _names(video_models)

  json_models = px.models.list_models(
      output_format=types.OutputFormatType.JSON)
  print(f'    JSON:     {_model_names(json_models)}')
  json_names = _names(json_models)
  assert 'gpt-4o' in json_names
  assert 'dall-e-3' not in json_names
  assert json_names == text_names

  # --- input_format filtering ---
  print()
  print('>> px.models.list_models(input_format=...)')
  image_input = px.models.list_models(
      input_format=types.InputFormatType.IMAGE)
  print(f'    IMAGE:    {_model_names(image_input)}')
  image_input_names = _names(image_input)
  assert 'gpt-4o' in image_input_names
  assert 'claude-sonnet-4-6' in image_input_names
  assert 'gemini-3-flash-preview' in image_input_names
  assert 'deepseek-chat' not in image_input_names
  assert 'dall-e-3' not in image_input_names

  audio_input = px.models.list_models(
      input_format=types.InputFormatType.AUDIO)
  print(f'    AUDIO:    {_model_names(audio_input)}')
  audio_input_names = _names(audio_input)
  assert 'gemini-3-flash-preview' in audio_input_names
  assert 'gpt-4o' not in audio_input_names

  doc_input = px.models.list_models(
      input_format=types.InputFormatType.DOCUMENT)
  print(f'    DOCUMENT: {_model_names(doc_input)}')
  assert _names(doc_input) == image_input_names

  # --- combined input + output filtering ---
  print()
  print('>> px.models.list_models(input_format=IMAGE,'
        ' output_format=JSON)')
  image_in_json_out = px.models.list_models(
      input_format=types.InputFormatType.IMAGE,
      output_format=types.OutputFormatType.JSON)
  print(f'    Result:   {_model_names(image_in_json_out)}')
  combined_names = _names(image_in_json_out)
  assert 'gpt-4o' in combined_names
  assert 'gemini-3-flash-preview' in combined_names
  assert 'deepseek-chat' not in combined_names
  assert 'dall-e-3' not in combined_names

  # --- tool_tags filtering ---
  print()
  print('>> px.models.list_models(tool_tags=...)')
  web_search_models = px.models.list_models(
      tool_tags=types.ToolTag.WEB_SEARCH)
  print(f'    WEB_SEARCH: {_model_names(web_search_models)}')
  ws_names = _names(web_search_models)
  assert 'gpt-4o' in ws_names
  assert 'gemini-3-flash-preview' in ws_names
  assert 'claude-sonnet-4-6' in ws_names
  assert 'o3' not in ws_names
  assert 'deepseek-chat' not in ws_names

  # --- feature_tags filtering ---
  print()
  print('>> px.models.list_models(feature_tags=...)')
  thinking_models = px.models.list_models(
      feature_tags=types.FeatureTag.THINKING)
  print(f'    THINKING: {_model_names(thinking_models)}')
  assert len(thinking_models) > 0
  assert 'gpt-4o' in _names(thinking_models)

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
  assert 'gemini-3-flash-preview' in _names(gemini_models)

  # --- get_model ---
  print()
  print('>> px.models.get_model(provider, model)')
  model = px.models.get_model('openai', 'gpt-4o')
  print(f'    get_model("openai", "gpt-4o") = {model}')
  assert model.provider == 'openai'
  assert model.model == 'gpt-4o'

  print()


def check_health_use():
  px.check_health(verbose=True)
  px.models.list_working_models(verbose=True)


def plain_alias_function_use():
  print('> plain_alias_function_use')

  print('* generate_text:')
  print('OpenAI:')
  response = px.generate_text(
      prompt='What is the capital of France and which AI provider are you?',
      provider_model=('openai', 'gpt-4o'))
  print(response)

  print('Gemini:')
  response = px.generate_text(
      prompt='What is the capital of France and which AI provider are you?',
      provider_model=('gemini', 'gemini-3-flash-preview'))
  print(response)

  print('* generate_json:')
  print('OpenAI:')
  response = px.generate_json(
      prompt='Give me the list of biggest cities in France.',
      provider_model=('openai', 'gpt-4o'))
  pprint(response)

  print('Gemini:')
  response = px.generate_json(
      prompt='Give me the list of biggest cities in France.',
      provider_model=('gemini', 'gemini-3-flash-preview'))
  pprint(response)

  print('* generate_pydantic:')
  class BiggestCities(BaseModel):
    cities: list[str]

  print('OpenAI:')
  response = px.generate_pydantic(
      prompt='Give me the list of biggest cities in France.',
      output_format=BiggestCities,
      provider_model=('openai', 'gpt-4o'))
  print(response)

  print('Gemini:')
  response = px.generate_pydantic(
      prompt='Give me the list of biggest cities in France.',
      output_format=BiggestCities,
      provider_model=('gemini', 'gemini-3-flash-preview'))
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
      with tempfile.NamedTemporaryFile(
          delete=False, suffix=suffix) as tmp:
        tmp.write(response.data)
        tmp_path = tmp.name
    elif response.source:
      with tempfile.NamedTemporaryFile(
          delete=False, suffix=suffix) as tmp:
        with urllib.request.urlopen(response.source) as http_resp:
          tmp.write(http_resp.read())
        tmp_path = tmp.name
    else:
      print('> No image data or source')
      return

    print(f'> Opening image preview: {tmp_path}')
    subprocess.run(
        ['qlmanage', '-p', tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False)
  finally:
    if tmp_path and os.path.exists(tmp_path):
      os.unlink(tmp_path)


def image_alias_function_use():
  print('> image_alias_function_use')

  print('* generate_image:')
  print('OpenAI:')
  response = px.generate_image(
      prompt='Make funny cartoon cat in living room.',
      provider_model=('openai', 'dall-e-3'))
  _open_image_response(response)

  print('Gemini:')
  response = px.generate_image(
      prompt='Make funny cartoon cat in living room.',
      provider_model=('gemini', 'gemini-2.5-flash-image'))
  _open_image_response(response)


def set_model_use():
  print('> set_model_use')

  print('* set_model text models:')

  print('OpenAI:')
  px.set_model(('openai', 'gpt-4o'))
  response = px.generate_text(
      prompt='Which provider model are you?')
  print(response)
  assert 'openai' in response.lower()

  print('Gemini:')
  px.set_model(('gemini', 'gemini-3-flash-preview'))
  response = px.generate_text(
      prompt='Which provider model are you?')
  print(response)
  assert 'gemini' in response.lower() or 'google' in response.lower()

  print('* set_model image models:')
  print('OpenAI:')
  px.set_model(generate_image=('openai', 'dall-e-3'))
  response = px.generate_image(
      prompt='Make funny cartoon cat in living room.')
  _open_image_response(response)

  print('Gemini:')
  px.set_model(generate_image=('gemini', 'gemini-2.5-flash-image'))
  response = px.generate_image(
      prompt='Make funny cartoon cat in living room.')
  _open_image_response(response)


def main():
  register_models(px.get_default_proxai_client())
  list_models_examples()
  check_health_use()
#   plain_alias_function_use()
#   image_alias_function_use()
#   set_model_use()


if __name__ == "__main__":
  main()