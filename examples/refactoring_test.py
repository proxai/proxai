from dataclasses import asdict
from pprint import pprint

import proxai as px


def simple_model_test():
  result = px.generate_text(
      'When is the first galatasaray and fenerbahce?',
      provider_model=('openai', 'gpt-5.2'))
  pprint(result)
  print(result.result.error)
  print(result.result.error_traceback)


def main():
  simple_model_test()


if __name__ == '__main__':
  main()
