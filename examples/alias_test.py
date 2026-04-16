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
    call_type: types.CallType = types.CallType.MULTI_MODAL,
    response_format: list[types.ResponseFormatType] = [types.ResponseFormatType.TEXT],
):
  """Get a model config for a given parameters."""

  web_search_supported = (
      types.FeatureSupportType.SUPPORTED
      if web_search
      else types.FeatureSupportType.NOT_SUPPORTED
  )

  text_supported = (
      types.FeatureSupportType.SUPPORTED
      if types.ResponseFormatType.TEXT in response_format
      else types.FeatureSupportType.NOT_SUPPORTED
  )
  json_supported = (
      types.FeatureSupportType.SUPPORTED
      if types.ResponseFormatType.JSON in response_format
      else types.FeatureSupportType.NOT_SUPPORTED
  )
  pydantic_supported = (
      types.FeatureSupportType.SUPPORTED
      if types.ResponseFormatType.PYDANTIC in response_format
      else types.FeatureSupportType.NOT_SUPPORTED
  )
  image_supported = (
      types.FeatureSupportType.SUPPORTED
      if types.ResponseFormatType.IMAGE in response_format
      else types.FeatureSupportType.NOT_SUPPORTED
  )
  audio_supported = (
      types.FeatureSupportType.SUPPORTED
      if types.ResponseFormatType.AUDIO in response_format
      else types.FeatureSupportType.NOT_SUPPORTED
  )
  video_supported = (
      types.FeatureSupportType.SUPPORTED
      if types.ResponseFormatType.VIDEO in response_format
      else types.FeatureSupportType.NOT_SUPPORTED
  )
  multi_modal_supported = (
      types.FeatureSupportType.SUPPORTED
      if types.ResponseFormatType.MULTI_MODAL in response_format
      else types.FeatureSupportType.NOT_SUPPORTED
  )
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
          call_type=call_type,
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
          response_format=types.ResponseFormatConfigType(
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
          call_type=types.CallType.MULTI_MODAL,
          response_format=[
              types.ResponseFormatType.TEXT,
              types.ResponseFormatType.JSON,
              types.ResponseFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='gpt-4o',
          provider_model_identifier='gpt-4o',
          web_search=True,
          call_type=types.CallType.MULTI_MODAL,
          response_format=[
              types.ResponseFormatType.TEXT,
              types.ResponseFormatType.JSON,
              types.ResponseFormatType.PYDANTIC],
      )
  )
  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='o3',
          provider_model_identifier='o3',
          web_search=False,
          call_type=types.CallType.MULTI_MODAL,
          response_format=[
              types.ResponseFormatType.TEXT,
              types.ResponseFormatType.JSON,
              types.ResponseFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='dall-e-3',
          provider_model_identifier='dall-e-3',
          web_search=False,
          call_type=types.CallType.IMAGE,
          response_format=[types.ResponseFormatType.IMAGE],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='tts-1',
          provider_model_identifier='tts-1',
          web_search=False,
          call_type=types.CallType.AUDIO,
          response_format=[types.ResponseFormatType.AUDIO],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='openai',
          model='sora-2',
          provider_model_identifier='sora-2',
          web_search=False,
          call_type=types.CallType.VIDEO,
          response_format=[types.ResponseFormatType.VIDEO],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='gemini-3-flash-preview',
          provider_model_identifier='gemini-3-flash-preview',
          web_search=True,
          call_type=types.CallType.MULTI_MODAL,
          response_format=[
              types.ResponseFormatType.TEXT,
              types.ResponseFormatType.JSON,
              types.ResponseFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='gemini-2.5-flash',
          provider_model_identifier='gemini-2.5-flash',
          web_search=False,
          call_type=types.CallType.MULTI_MODAL,
          response_format=[
              types.ResponseFormatType.TEXT,
              types.ResponseFormatType.JSON,
              types.ResponseFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='gemini-2.5-flash-image',
          provider_model_identifier='gemini-2.5-flash-image',
          web_search=False,
          call_type=types.CallType.IMAGE,
          response_format=[types.ResponseFormatType.IMAGE],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='gemini-2.5-flash-preview-tts',
          provider_model_identifier='gemini-2.5-flash-preview-tts',
          web_search=False,
          call_type=types.CallType.AUDIO,
          response_format=[types.ResponseFormatType.AUDIO],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='gemini',
          model='veo-3.1-generate-preview',
          provider_model_identifier='veo-3.1-generate-preview',
          web_search=False,
          call_type=types.CallType.VIDEO,
          response_format=[types.ResponseFormatType.VIDEO],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='claude',
          model='claude-sonnet-4-6',
          provider_model_identifier='claude-sonnet-4-6',
          web_search=True,
          call_type=types.CallType.MULTI_MODAL,
          response_format=[
              types.ResponseFormatType.TEXT,
              types.ResponseFormatType.JSON,
              types.ResponseFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='claude',
          model='claude-opus-4-6',
          provider_model_identifier='claude-opus-4-6',
          web_search=False,
          call_type=types.CallType.MULTI_MODAL,
          response_format=[
              types.ResponseFormatType.TEXT,
              types.ResponseFormatType.JSON,
              types.ResponseFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='deepseek',
          model='deepseek-chat',
          provider_model_identifier='deepseek-chat',
          web_search=False,
          call_type=types.CallType.MULTI_MODAL,
          response_format=[
              types.ResponseFormatType.TEXT,
              types.ResponseFormatType.JSON,
              types.ResponseFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.register_provider_model_config(
      _get_model_config(
          provider='deepseek',
          model='deepseek-reasoner',
          provider_model_identifier='deepseek-reasoner',
          web_search=False,
          call_type=types.CallType.MULTI_MODAL,
          response_format=[
              types.ResponseFormatType.TEXT,
              types.ResponseFormatType.JSON,
              types.ResponseFormatType.PYDANTIC],
      )
  )

  client.model_configs_instance.override_default_model_priority_list([
      px.models.get_model('gemini', 'gemini-3-flash-preview'),
      px.models.get_model('openai', 'gpt-4o'),
      px.models.get_model('claude', 'claude-sonnet-4-6'),
  ])


def list_models_examples():
  def _print_models(models: list[types.ProviderModelType]):
    for model in models:
      print(f'{model.provider:>25} - {model.model}')
  print('> list_models_examples')
  print('* Default list_models')
  models = px.models.list_models()
  _print_models(models)
  print('* Multi-modal models')
  models = px.models.list_models(call_type=types.CallType.MULTI_MODAL)
  _print_models(models)
  print('* Image models')
  image_models = px.models.list_models(call_type=types.CallType.IMAGE)
  _print_models(image_models)
  print('* Audio models')
  audio_models = px.models.list_models(call_type=types.CallType.AUDIO)
  _print_models(audio_models)
  print('* Video models')
  video_models = px.models.list_models(call_type=types.CallType.VIDEO)
  _print_models(video_models)


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
      response_format=BiggestCities,
      provider_model=('openai', 'gpt-4o'))
  print(response)

  print('Gemini:')
  response = px.generate_pydantic(
      prompt='Give me the list of biggest cities in France.',
      response_format=BiggestCities,
      provider_model=('gemini', 'gemini-3-flash-preview'))
  print(response)

def image_alias_function_use():
  print('> image_alias_function_use')

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

  # print('* set_model image models:')
  # print('OpenAI:')
  # px.set_model(provider_model=('openai', 'dall-e-3'), )
  # response = px.generate_image(
  #     prompt='Make funny cartoon cat in living room.')
  # print(response)

  # print('Gemini:')
  # px.set_model(('gemini', 'gemini-2.5-flash-image'))
  # response = px.generate_image(
  #     prompt='Make funny cartoon cat in living room.')
  # print(response)


def main():
  register_models(px.get_default_proxai_client())
  list_models_examples()
  # plain_alias_function_use()
  image_alias_function_use()
  # set_model_use()


if __name__ == "__main__":
  main()