import proxai as px


def main():
  px.connect(
      experiment_name='cache_test/run_1',
      logging_options=px.LoggingOptions(
          proxdash_stdout=True))
  result = px.generate_text(
      'This is a test message to check if the cache is working or not.')
  print(f'1: {result}')
  result = px.generate_text(
      'I am really happy about writing codes.')
  print(f'2: {result}')
  result = px.generate_text(
      'Especially, good APIs makes me happy.')
  print(f'3: {result}')


if __name__ == '__main__':
  main()
