from pathlib import Path
import proxai as px
import datetime
from pprint import pprint
from dataclasses import asdict

def main():
  pprint(asdict(px.get_current_options()))

  start_time = datetime.datetime.now()
  print(f'len = {len(px.models.generate_text())}')
  end_time = datetime.datetime.now()
  print(f'> First call Time taken: {end_time - start_time}')

  start_time = datetime.datetime.now()
  print(f'len = {len(px.models.generate_text())}')
  end_time = datetime.datetime.now()
  print(f'> Second call Time taken: {end_time - start_time}')

  px.connect(
      experiment_path='simple_test/run_1',
      logging_path=f'{Path.home()}/proxai_log/',
      cache_path=f'{Path.home()}/proxai_cache/',
      proxdash_options=px.ProxDashOptions(stdout=True))

  print()
  pprint(px.get_current_options())

  start_time = datetime.datetime.now()
  print(f'len = {len(px.models.generate_text())}')
  end_time = datetime.datetime.now()
  print(f'> Third call Time taken: {end_time - start_time}')

  start_time = datetime.datetime.now()
  print(f'len = {len(px.models.generate_text())}')
  end_time = datetime.datetime.now()
  print(f'> Fourth call Time taken: {end_time - start_time}')



if __name__ == '__main__':
  main()
