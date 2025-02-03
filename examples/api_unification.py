"""Examples of asking about model properties."""
import collections
import functools
from pathlib import Path
import random
import datetime
import proxai as px
import os
import proxai.types as px_types

TOTAL_CACHE_HITS = 0
TOTAL_PROVIDER_CALLS = 0


def basic_test(hard_start=False):
  options = [
      {'name': 'Basic Test / 1st Run',
       'strict_feature_test': False,
       'retry_if_error_cached': False,
       'use_cache': not hard_start},
      {'name': 'Basic Test / 2nd Run',
       'strict_feature_test': False,
       'retry_if_error_cached': False,
       'use_cache': True},
      {'name': 'Basic Test / 1st Run',
       'strict_feature_test': False,
       'retry_if_error_cached': True,
       'use_cache': True},
      {'name': 'Basic Test / 2nd Run',
       'strict_feature_test': False,
       'retry_if_error_cached': True,
       'use_cache': True},
  ]
  px_generate_text = functools.partial(
      px.generate_text,
      prompt='Which company created you and what is your model name?')
  return options, px_generate_text


def feature_complete_test(hard_start=False):
  options = [
      {'name': 'Feature Complete Test / 1st Run',
       'strict_feature_test': False,
       'retry_if_error_cached': True,
       'use_cache': not hard_start},
      {'name': 'Feature Complete Test / 2nd Run',
       'strict_feature_test': False,
       'retry_if_error_cached': True,
       'use_cache': True},
      {'name': 'Feature Complete Test / 1st Run',
       'strict_feature_test': True,
       'retry_if_error_cached': True,
       'use_cache': True},
      {'name': 'Feature Complete Test / 2nd Run',
       'strict_feature_test': True,
       'retry_if_error_cached': True,
       'use_cache': True},
  ]
  px_generate_text = functools.partial(
      px.generate_text,
      system='Answer all questions in French.',
      messages=[
        {'role': 'user', 'content': 'Hello!'},
        {'role': 'assistant', 'content': 'Bonjour!'},
        {'role': 'user',
         'content': 'Which company created you and what is your model name?'}],
      max_tokens=100,
      temperature=1.0,
      stop=['.'])
  return options, px_generate_text


def make_query(
    provider,
    model,
    use_cache,
    px_generate_text):
  global TOTAL_CACHE_HITS, TOTAL_PROVIDER_CALLS
  try:
    logging_record: px_types.LoggingRecord = px_generate_text(
        provider=provider,
        model=model,
        use_cache=use_cache,
        extensive_return=True)
    if logging_record.response_source == px_types.ResponseSource.CACHE:
      TOTAL_CACHE_HITS += 1
    else:
      TOTAL_PROVIDER_CALLS += 1
    return logging_record.response_record.response
  except Exception as e:
    return f'ERROR: {str(e)}'


def run_queries(
    retry_if_error_cached = False,
    strict_feature_test = False,
    use_cache = False,
    px_generate_text = None):
  cache_path = f'{Path.home()}/proxai_cache/'
  logging_path = f'{Path.home()}/proxai_log/api_unification/'
  os.makedirs(cache_path, exist_ok=True)
  os.makedirs(logging_path, exist_ok=True)
  px.connect(
      experiment_path='api_unification/run_1',
      logging_path=logging_path,
      cache_options=px_types.CacheOptions(
          path=cache_path,
          unique_response_limit=1,
          retry_if_error_cached=retry_if_error_cached),
      strict_feature_test=strict_feature_test)
  models = px.models.generate_text(verbose=True)
  print(f'{"PROVIDER":10} | {"MODEL":45} | {"DURATION":13} | {"RESPONSE"}')
  for provider, provider_model in models:
    start_time = datetime.datetime.now()
    response = make_query(
        provider=provider,
        model=provider_model,
        use_cache=use_cache,
        px_generate_text=px_generate_text)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    response = response.strip().split('\n')[0][:100] + (
      '...' if len(response) > 100 else '')
    print(f'{provider:10} | {provider_model:45} | {duration:10.0f} ms | {response}')


def run_tests(options, px_generate_text):
  for idx, option in enumerate(options):
    print(f'\n\n{idx+1}.{option["name"]}')
    print(f'{option["retry_if_error_cached"]=}')
    print(f'{option["strict_feature_test"]=}')
    print(f'{option["use_cache"]=}')
    print()
    total_cache_hits = TOTAL_CACHE_HITS
    total_provider_calls = TOTAL_PROVIDER_CALLS
    run_queries(
        retry_if_error_cached=option['retry_if_error_cached'],
        strict_feature_test=option['strict_feature_test'],
        use_cache=option['use_cache'],
        px_generate_text=px_generate_text)
    print()
    print(f'Cache Hits     : {TOTAL_CACHE_HITS - total_cache_hits}')
    print(f'Provider Calls : {TOTAL_PROVIDER_CALLS - total_provider_calls}')
    print(f'Total Cache Hits     : {TOTAL_CACHE_HITS}')
    print(f'Total Provider Calls : {TOTAL_PROVIDER_CALLS}')
    input('Press Enter to continue...')


def main():
  options, px_generate_text = basic_test(
      hard_start=False)
  run_tests(options, px_generate_text)

  options, px_generate_text = feature_complete_test(
      hard_start=False)
  run_tests(options, px_generate_text)


if __name__ == '__main__':
  main()
