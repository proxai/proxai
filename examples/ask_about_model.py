"""Examples of asking about model properties."""
import collections
import functools
from pathlib import Path
import random
import datetime
import proxai as px
import os
import proxai.types as px_types
from pprint import pprint

_BREAK_CACHES = True
_ONLY_LARGEST_MODELS = False


def get_models(verbose=True):
  provider_models = px.models.get_all_models(
      only_largest_models=_ONLY_LARGEST_MODELS,
      verbose=True)
  grouped_models = collections.defaultdict(list)
  for provider_model in provider_models:
    grouped_models[provider_model.provider].append(provider_model.model)
  if verbose:
    print('Available models:')
    for provider, models in grouped_models.items():
      print(f'{provider}:')
      for model in models:
        print(f'   {model}')
    print()
  return provider_models


def test_query(break_caches: bool=False):
  prompt = ''
  if break_caches:
    prompt = ('Please ignore this but I really like number'
              f' {random.randint(1, 1000000)}.\n')
  prompt += 'Which company created you and what is your model name?'
  try:
    return px.generate_text(prompt)
  except Exception as e:
    error_message = str(e).strip().split('\n')[-1]
    print(f'<ERROR>: {error_message}')
    return 'ERROR'


def print_summary():
  summary_json = px.get_summary(json=True)
  pprint(summary_json)
  print()


def run_tests(provider_models, query_func):
  print(f'{"PROVIDER":10} | {"MODEL":45} | {"DURATION":13} | {"RESPONSE"}')
  for provider_model in provider_models:
    px.set_model(generate_text=(provider_model.provider, provider_model.model))
    start_time = datetime.datetime.now()
    response = query_func()
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    response = response.strip().split('\n')[0][:100] + (
      '...' if len(response) > 100 else '')
    print(f'{provider_model.provider:10} | {provider_model.model:45} | '
          f'{duration:10.0f} ms | {response}')


def main():
  px.connect(
      experiment_path='ask_about_model/run_1',
      logging_path=f'{Path.home()}/proxai_log/',
      cache_path=f'{Path.home()}/proxai_cache/',
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          hide_sensitive_content=True))
  provider_models = get_models()
  run_tests(provider_models, functools.partial(test_query, break_caches=_BREAK_CACHES))


if __name__ == '__main__':
  main()
