from pathlib import Path
import proxai as px
import random
import time
from pprint import pprint
from dataclasses import asdict


def simple_model_test():
  random_int = random.randint(1, 1000000)
  result = px.generate_text(
      'This is a test message to check if the cache is working or '
      f'not. {random_int}',
      extensive_return=True)
  pprint(asdict(result))


def simple_cache_test():
  px.connect(
      experiment_path='simple_test/run_1',
      logging_path=f'{Path.home()}/proxai_log/',
      cache_path=f'{Path.home()}/proxai_cache/',
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          base_url='http://localhost:3001',
          api_key='bry2oe2-m9xev24i-q2pjebcpc9'))
  random_int = random.randint(1, 1000000)
  result = px.generate_text(
      'This is a test message to check if the cache is working or '
      f'not. {random_int}')
  print(f'1: {result}')
  time.sleep(1)

  result = px.generate_text(
      'This is a test message to check if the cache is working or '
      f'not. {random_int}')
  print(f'2: {result}')
  time.sleep(1)

  result = px.generate_text(
      'This is a test message to check if the cache is working or '
      f'not. {random_int}')
  print(f'3: {result}')


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
  # simple_model_test()
  # simple_cache_test()
  # list_models()
  check_health()

if __name__ == '__main__':
  main()
