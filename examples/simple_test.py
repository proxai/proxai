import os
from pathlib import Path
import proxai as px
import random
import time


def main():
  cache_path = f'{Path.home()}/proxai_cache/'
  logging_path = f'{Path.home()}/proxai_log/ask_about_model/'
  os.makedirs(cache_path, exist_ok=True)
  os.makedirs(logging_path, exist_ok=True)
  px.connect(
      experiment_path='simple_test/run_1',
      cache_path=cache_path,
      logging_path=logging_path,
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


if __name__ == '__main__':
  main()
