from pathlib import Path
import proxai as px
import random
import time
from pprint import pprint


def simple_model_test():
  random_int = random.randint(1, 1000000)
  result = px.generate_text(
      'This is a test message to check if the cache is working or '
      f'not. {random_int}',
      provider_model=('grok', 'grok-3-mini-fast-beta'),
      extensive_return=True)
  pprint(result)


def simple_cache_test():
  px.connect(
      experiment_path='simple_test/run_1',
      logging_path=f'{Path.home()}/proxai_log/',
      cache_path=f'{Path.home()}/proxai_cache/',
      proxdash_options=px.ProxDashOptions(stdout=True))
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


def main():
  simple_model_test()
  simple_cache_test()


if __name__ == '__main__':
  main()
