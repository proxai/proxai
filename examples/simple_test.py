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


def image_cache_test():
  client = px.Client(
      logging_options=px.LoggingOptions(
          stdout=True,
      ),
      cache_options=px.CacheOptions(
          cache_path=f'{Path.home()}/proxai_cache_2/'))

  pprint(client.models.list_models(output_format='image'))

  result = client.generate_image(
      'Make funny cartoon cat in living room.',
      provider_model=('gemini', 'gemini-2.5-flash-image'))
  print(result.data[:100])


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
  client = px.Client(
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          base_url='http://localhost:3001',
          api_key='yf5ak72-mogdeiah-tmui8htgtgn',
      ),
  )
  client.models.check_health(verbose=True)


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
  image_cache_test()
  # list_models()
  # check_health()
  # proxdash_test()

if __name__ == '__main__':
  main()
