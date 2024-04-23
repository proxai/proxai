"""Examples of asking about model properties."""
import collections
import functools
from pathlib import Path
import random
import datetime
import proxai as px
import os
import proxai.types as px_types

_PROVIDERS = [
    'openai',
    'claude',
    'gemini',
    'cohere',
    'databricks',
    'mistral',
    'hugging_face']
_BREAK_CACHES = True
_HISTORY = True


def get_models(verbose=True):
  models = px.models.generate_text(
      only_largest_models=True,
      verbose=True)
  grouped_models = collections.defaultdict(list)
  for provider, model in models:
    grouped_models[provider].append(model)
  if verbose:
    print('Available models:')
    for provider, provider_models in grouped_models.items():
      print(f'{provider}:')
      for provider_model in provider_models:
        print(f'   {provider_model}')
    print()
  return models


def test_query(
    model: px_types.ModelType,
    break_caches: bool=False,
    history: bool=True):
  prompt = ''
  messages = None
  if break_caches:
    prompt = ('Please ignore this but I really like number'
              f' {random.randint(1, 1000000)}.\n')
  prompt += 'Which company created you and what is your model name?'
  if history:
    messages = [
        {'role': 'user', 'content': 'Hello!'},
        {'role': 'assistant', 'content': 'Bonjour!'},
        {'role': 'user', 'content': prompt}]
    prompt = None
  stop = ['.']
  if model[0] == 'mistral':
    stop = None
  return px.generate_text(
      model=model,
      prompt=prompt,
      system='Answer all questions in French.',
      messages=messages,
      max_tokens=100,
      temperature=0.1,
      stop=stop,
      use_cache=False)


def run_tests(models, query_func):
  print(f'{"PROVIDER":10} | {"MODEL":45} | {"DURATION":13} | {"RESPONSE"}')
  for model in models:
    provider, provider_model = model
    if provider not in _PROVIDERS:
      continue
    start_time = datetime.datetime.now()
    response = query_func(model=model)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    response = response.strip().split('\n')[0][:100] + (
      '...' if len(response) > 100 else '')
    print(f'{provider:10} | {provider_model:45} | {duration:10.0f} ms | {response}')


def main():
  cache_path = f'{Path.home()}/proxai_cache/'
  logging_path = f'{Path.home()}/proxai_log/ask_about_model/'
  os.makedirs(cache_path, exist_ok=True)
  os.makedirs(logging_path, exist_ok=True)
  px.connect(cache_path=cache_path, logging_path=logging_path)
  models = get_models()
  run_tests(models, functools.partial(
      test_query,
      break_caches=_BREAK_CACHES,
      history=_HISTORY))


if __name__ == '__main__':
  main()
