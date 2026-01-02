from pathlib import Path
import proxai as px
import random
import time
from pprint import pprint
from dataclasses import asdict


def simple_model_test():
  random_int = random.randint(1, 1000000)
  result = px.generate_text(
      'When is the first galatasaray and fenerbahce?',
      provider_model=('cohere', 'command-a'),
      extensive_return=True)
  pprint(asdict(result))


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
  result = _test_function()
  time.sleep(1)

  print('2:')
  result = _test_function()
  time.sleep(1)

  print('3:')
  result = _test_function()
  time.sleep(1)


def list_models():
  px.connect(
      experiment_path='simple_test/run_1',
      logging_path=f'{Path.home()}/proxai_log/',
      cache_path=f'{Path.home()}/proxai_cache/')
  model_status = px.models.list_models(
      model_size='small',
      verbose=True,
      return_all=True)
  from pprint import pprint
  pprint(model_status.working_models)
  pprint(model_status.failed_models)

def check_health():
  px.connect(
      experiment_path='simple_test/run_1',
      logging_path=f'{Path.home()}/proxai_log/',
      cache_path=f'{Path.home()}/proxai_cache/')
  px.check_health(verbose=True)


def main():
  simple_model_test()
  # simple_cache_test()
  # list_models()
  # check_health()

if __name__ == '__main__':
  main()
