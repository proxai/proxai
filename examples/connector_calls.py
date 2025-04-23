import proxai as px
from pathlib import Path
import random

def base_call():
  response = px.generate_text(
      prompt='What is the capital of France? '
      'Give some basic information about the city.')
  print('Base call response: ', response.replace('\n', ' ')[:100])


def advanced_call():
  response = px.generate_text(
      messages=[
          {'role': 'user', 'content': 'What is the capital of France? Give some basic information about the city.'},
          {'role': 'assistant', 'content': f'Paris is the capital of France. {random.randint(1, 100)}. It has a population of 2.1 million. {random.randint(1, 1000000)}.'},
          {'role': 'user', 'content': 'What is the capital of United Kingdom? Give some basic information about the city.'},
      ],
      system='Add random number after each sentences. Use couple of sentences.',
      max_tokens=500,
      temperature=0.7,
      stop='STOP',
      )
  print('Advanced call response: ', response.replace('\n', ' ')[:100])


if __name__ == '__main__':
  px.connect(
      experiment_path='connector_calls/run_1',
      logging_path=f'{Path.home()}/proxai_log/',
      cache_path=f'{Path.home()}/proxai_cache/',
      cache_options=px.CacheOptions(
          clear_model_cache_on_connect=True,
          clear_query_cache_on_connect=True,
      ),
      proxdash_options=px.ProxDashOptions(stdout=True))
  models = px.models.list_provider_models(provider='mistral')
  print('--- Base call ---')
  for model in models:
    print(f'Model: {model}')
    px.set_model(model)
    base_call()

  print('--- Advanced call ---')
  for model in models:
    print(f'Model: {model}')
    px.set_model(model)
    advanced_call()
