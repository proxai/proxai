import random
from pathlib import Path
import proxai as px
import proxai.types as px_types

_UNIQUE_RESPONSE_LIMIT = 3
_REPEAT_COUNT = 6


if __name__ == '__main__':
  px.connect(
      logging_path=f'{Path.home()}/proxai_log/',
      cache_path=f'{Path.home()}/proxai_cache/',
      cache_options=px_types.CacheOptions(
          unique_response_limit=_UNIQUE_RESPONSE_LIMIT))

  random_seed = random.randint(0, 1000000)
  models = px.models.generate_text(only_largest_models=True, verbose=True)
  for model in models:
    provider, provider_model = model
    print(f'{provider:>10} - {provider_model}')
    px.set_model(generate_text=model)
    for idx in range(_REPEAT_COUNT):
      print(f'Try {idx + 1}: ', end=' ')
      try:
        result = px.generate_text(
            'Give me one sentence that will make me laugh.'
            f' You have {random_seed} seconds left!',
            temperature=0.8,
            extensive_return=True)
      except Exception as e:
        print('[Provider] <Error>')
        continue
      print(f'{"["+result.response_source+"]":>10} '
            f'{result.response_record.response.strip()}')
