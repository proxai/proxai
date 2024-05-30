import os
from pathlib import Path
import proxai as px
import proxai.types as px_types

_UNIQUE_RESPONSE_LIMIT = 3
_REPEAT_COUNT = 6


if __name__ == '__main__':
  cache_path = f'{Path.home()}/proxai_cache/'
  logging_path = f'{Path.home()}/proxai_log/math_problems/'
  os.makedirs(cache_path, exist_ok=True)
  os.makedirs(logging_path, exist_ok=True)
  px.connect(
      cache_path=cache_path,
      cache_options=px_types.CacheOptions(
          unique_response_limit=_UNIQUE_RESPONSE_LIMIT),
      logging_path=logging_path)

  models = px.models.generate_text(only_largest_models=True, verbose=True)
  for model in models:
    provider, provider_model = model
    print(f'{provider:>10} - {provider_model}')
    px.set_model(generate_text=model)
    for idx in range(_REPEAT_COUNT):
      print(f'Try {idx + 1}:')
      try:
        result = px.generate_text(
            'Give me one sentence that will make me laugh.')
      except Exception as e:
        print('Error:', str(e))
        continue
      print(f'Result: {result}')
