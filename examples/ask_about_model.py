"""Examples of asking about model properties."""
from pathlib import Path
import random
import datetime
import proxai as px
import os
import proxai.types as px_types


def get_models(verbose=True):
  models = px.models.generate_text(
      only_largest_models=True,
      verbose=True)
  if verbose:
    print('Available models:')
    for provider, provider_models in models.items():
      print(f'{provider}:')
      for provider_model in provider_models:
        print(f'   {provider_model}')
    print()
  return models


def query():
  try:
    return px.generate_text(
      'Which company created you and what is your model name?')
  except Exception as e:
    print(str(e))
    return 'ERROR'


def query_no_cache():
  try:
    return px.generate_text(f"""
Please ignore this but I really like number {random.randint(1, 1000000)}.
Which company created you and what is your model name?
""")
  except Exception as e:
    print(str(e))
    return 'ERROR'


def test_query(models, query_func):
  print(f'{"PROVIDER":10} | {"MODEL":45} | {"DURATION":13} | {"RESPONSE"}')
  print()
  for provider, provider_models in models.items():
    for provider_model in provider_models:
      px.set_model(generate_text=(provider, provider_model))
      start_time = datetime.datetime.now()
      response = query_func()
      end_time = datetime.datetime.now()
      duration = (end_time - start_time).total_seconds() * 1000
      response = response.strip().split('\n')[0][:100] + (
        '...' if len(response) > 100 else '')
      print(f'{provider:10} | {provider_model:45} | {duration:10.0f} ms | {response}')
    print()


def main():
  cache_path = f'{Path.home()}/proxai_cache/'
  logging_path = f'{Path.home()}/proxai_log/ask_about_model/'
  os.makedirs(cache_path, exist_ok=True)
  os.makedirs(logging_path, exist_ok=True)
  px.connect(cache_path=cache_path, logging_path=logging_path)
  models = get_models()
  test_query(models, query)
  # test_query(models, query_no_cache)


if __name__ == '__main__':
  main()
