import random
from pathlib import Path
import proxai as px
import proxai.types as px_types

_UNIQUE_RESPONSE_LIMIT = 3
_REPEAT_COUNT = 6
_RANDOM_SEED = 42


if __name__ == '__main__':
  px.connect(
      logging_path=f'{Path.home()}/proxai_log/',
      cache_path=f'{Path.home()}/proxai_cache/',
      cache_options=px_types.CacheOptions(
          unique_response_limit=_UNIQUE_RESPONSE_LIMIT,
          clear_query_cache_on_connect=True))

  provider_models = px.models.get_all_models(
      only_largest_models=True, verbose=True)
  for provider_model in provider_models:
    print(f'{provider_model.provider:>10} - {provider_model.model}')
    px.set_model(generate_text=provider_model)
    for idx in range(_REPEAT_COUNT):
      print(f'Try {idx + 1}: ', end=' ')
      try:
        result = px.generate_text(
            'Give me one sentence that will make me laugh.'
            f' You have {_RANDOM_SEED} seconds left!',
            temperature=0.8,
            extensive_return=True)
      except Exception as e:
        print('[Provider] <Error>')
        continue
      print(f'{"["+result.response_source+"]":>10} '
            f'{result.response_record.response.strip()}')
