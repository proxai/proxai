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
_ONLY_LARGEST_MODELS = True


def get_models(verbose=True):
  models = px.models.generate_text(
      only_largest_models=_ONLY_LARGEST_MODELS,
      verbose=True)
  grouped_models = collections.defaultdict(list)
  for provider, model in models:
    grouped_models[provider].append(model)
  if verbose:
    print('Available models:')
    for provider, provider_models in grouped_models.items():
      print(f'{provider}:')
      for provider_model in provider_models:
        print(f'   {provider_model}')
    print()
  return models


def test_query(break_caches: bool=False):
  prompt = ''
  if break_caches:
    prompt = ('Please ignore this but I really like number'
              f' {random.randint(1, 1000000)}.\n')
  prompt += 'Which company created you and what is your model name?'
  try:
    return px.generate_text(prompt)
  except Exception as e:
    print(str(e))
    return 'ERROR'


def print_summary():
  summary_json = px.get_summary(json=True)
  pprint(summary_json)
  print()


def run_tests(models, query_func):
  print(f'{"PROVIDER":10} | {"MODEL":45} | {"DURATION":13} | {"RESPONSE"}')
  for provider, provider_model in models:
    px.set_model(generate_text=(provider, provider_model))
    start_time = datetime.datetime.now()
    response = query_func()
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    response = response.strip().split('\n')[0][:100] + (
      '...' if len(response) > 100 else '')
    print(f'{provider:10} | {provider_model:45} | {duration:10.0f} ms | {response}')


def main():
  cache_path = f'{Path.home()}/proxai_cache/'
  logging_path = f'{Path.home()}/proxai_log/ask_about_model/'
  os.makedirs(cache_path, exist_ok=True)
  os.makedirs(logging_path, exist_ok=True)
  px.connect(
      experiment_path='ask_about_model/run_3',
      cache_path=cache_path,
      logging_path=logging_path,
      logging_options=px.LoggingOptions(
          proxdash_stdout=True,
          hide_sensitive_content=True))
  models = get_models()
  run_tests(models, functools.partial(test_query, break_caches=_BREAK_CACHES))


if __name__ == '__main__':
  main()
