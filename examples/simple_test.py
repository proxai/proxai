import random
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

import proxai as px


def simple_model_test():
  result = px.generate(
      'When is the first galatasaray and fenerbahce?')
  pprint(result)


def simple_cache_test():
  px.connect(
      experiment_path='simple_test/run_1',
      cache_path=f'{Path.home()}/proxai_cache/')

  random_int = random.randint(1, 1000000)

  def _test_function():
    result = px.generate_text(
        'This is a test message to check if the cache is working or '
        f'not. {random_int}',
        provider_model=('openai', 'gpt-5.1'),
        extensive_return=True)
    print('Response    :', result.response_record.response.value)
    print('Source      :', result.response_source)
    print('Fail reason :', result.look_fail_reason)
    return result

  print('1:')
  _test_function()
  time.sleep(1)

  print('2:')
  _test_function()
  time.sleep(1)

  print('3:')
  _test_function()
  time.sleep(1)


def list_models():
  # client = px.Client(
  #     provider_call_options=px.ProviderCallOptions(
  #         feature_mapping_strategy=px.FeatureMappingStrategy.STRICT))
  # models = client.models.list_provider_models(provider='gemini', feature_tags=['thinking'])
  # print(len(models))
  # for model in models:
  #   print(model)

  # client = px.Client(
  #     provider_call_options=px.ProviderCallOptions(
  #         feature_mapping_strategy=px.FeatureMappingStrategy.STRICT))
  # client = px.Client()
  # models = client.models.list_models(output_format='audio')
  # print(len(models))
  # for model in models:
  #   print(model)

  for provider in px.models.list_providers():
    print(f'#### {provider} ####')
    for size in ['small', 'medium', 'large', 'largest']:
      models = px.models.list_provider_models(provider, model_size=size)
      print(f'  === {size.upper()} ({len(models)}) ===')
      for model in models:
        print(f'    {model}')

    all_models = px.models.list_provider_models(provider)
    untagged = [
        m for m in all_models
        if not px.models.get_model_config(
            m.provider, m.model).metadata.model_size_tags
    ]
    print(f'  === NO SIZE TAGS ({len(untagged)}) ===')
    for model in untagged:
      print(f'    {model}')
    print()


def check_health():
  px.models.check_health(verbose=True)


def proxdash_test():
  client = px.Client(
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          base_url='http://localhost:3001',
          api_key='yf5ak72-mogdeiah-tmui8htgtgn',
      ),
  )
  client.models.list_models()


def main():
  # simple_model_test()
  # simple_cache_test()
  list_models()
  # check_health()
  # proxdash_test()

if __name__ == '__main__':
  main()
