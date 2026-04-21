"""ProxDash Files API integration test.

Tests px.files.* operations with ProxDash connected — alone and
combined with providers.

Usage:
  poetry run python3 examples/proxdash_files_api_test.py
  poetry run python3 examples/proxdash_files_api_test.py --test upload_proxdash_pdf
  poetry run python3 examples/proxdash_files_api_test.py --test all
"""

import argparse
import os
import time

import proxai as px
import proxai.types as types

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'refactoring_test_assets')


def _asset(filename):
  return os.path.join(_ASSETS_DIR, filename)


def _print_media(media):
  print(f'  proxdash_file_id: {media.proxdash_file_id}')
  if media.proxdash_file_status:
    pd = media.proxdash_file_status
    print(
        f'  proxdash_file_status: s3_key={pd.s3_key}, '
        f'confirmed={pd.upload_confirmed}'
    )
  if media.provider_file_api_ids:
    for provider, file_id in media.provider_file_api_ids.items():
      status = media.provider_file_api_status.get(provider)
      state = status.state.value if status and status.state else '?'
      print(f'  {provider}: id={file_id}, state={state}')
  else:
    print('  providers: (none)')


def _upload_and_verify(media, providers, expect_proxdash=True):
  start = time.time()
  px.files.upload(media=media, providers=providers)
  elapsed = time.time() - start
  _print_media(media)
  print(f'  Time: {elapsed:.2f}s')
  if expect_proxdash:
    assert media.proxdash_file_id is not None, (
        'Expected proxdash_file_id to be set'
    )
  for p in providers:
    assert p in media.provider_file_api_ids, (
        f'Expected provider_file_api_ids[{p}] to be set'
    )
  return elapsed


# --- 1. ProxDash-only upload ---


def test_upload_proxdash_pdf():
  print('\n=== upload_proxdash_pdf ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  _upload_and_verify(media, providers=[])
  assert not media.provider_file_api_ids


def test_upload_proxdash_image():
  print('\n=== upload_proxdash_image ===')
  media = px.MessageContent(path=_asset('cat.jpeg'), media_type='image/jpeg')
  _upload_and_verify(media, providers=[])
  assert not media.provider_file_api_ids


def test_upload_proxdash_markdown():
  print('\n=== upload_proxdash_markdown ===')
  media = px.MessageContent(path=_asset('cat.md'), media_type='text/markdown')
  _upload_and_verify(media, providers=[])
  assert not media.provider_file_api_ids


def test_upload_proxdash_audio():
  print('\n=== upload_proxdash_audio ===')
  media = px.MessageContent(path=_asset('cat.mp3'), media_type='audio/mpeg')
  _upload_and_verify(media, providers=[])
  assert not media.provider_file_api_ids


# --- 2. ProxDash + single provider ---


def test_upload_proxdash_gemini_pdf():
  print('\n=== upload_proxdash_gemini_pdf ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  _upload_and_verify(media, providers=['gemini'])
  assert 'gemini' in media.provider_file_api_ids


def test_upload_proxdash_openai_pdf():
  print('\n=== upload_proxdash_openai_pdf ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  _upload_and_verify(media, providers=['openai'])
  assert 'openai' in media.provider_file_api_ids


def test_upload_proxdash_claude_pdf():
  print('\n=== upload_proxdash_claude_pdf ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  _upload_and_verify(media, providers=['claude'])
  assert 'claude' in media.provider_file_api_ids


def test_upload_proxdash_mistral_pdf():
  print('\n=== upload_proxdash_mistral_pdf ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  _upload_and_verify(media, providers=['mistral'])
  assert 'mistral' in media.provider_file_api_ids


# --- 3. ProxDash + multi provider ---


def test_upload_proxdash_multi():
  print('\n=== upload_proxdash_multi ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  _upload_and_verify(media, providers=['gemini', 'claude'])
  assert 'gemini' in media.provider_file_api_ids
  assert 'claude' in media.provider_file_api_ids


# --- 4. Parallel vs sequential ---


def test_upload_parallel():
  print('\n=== upload_parallel ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  elapsed = _upload_and_verify(media, providers=['gemini', 'claude'])
  print(f'  Parallel time: {elapsed:.2f}s')


def test_upload_sequential():
  print('\n=== upload_sequential ===')
  px.connect(
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          base_url='http://localhost:3001',
          api_key='pjlfi0h-mo8mrm56-fgsvgftdk78',
      ),
      cache_options=px.CacheOptions(cache_path='/tmp/proxai_cache'),
      provider_call_options=px.ProviderCallOptions(
          allow_parallel_file_operations=False,
      ),
  )
  _register_models()
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  elapsed = _upload_and_verify(media, providers=['gemini', 'claude'])
  print(f'  Sequential time: {elapsed:.2f}s')
  # Reconnect with parallel for remaining tests.
  _connect_default()
  _register_models()


# --- 5. List with ProxDash deduplication ---


def test_list_proxdash_only():
  print('\n=== list_proxdash_only ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  px.files.upload(media=media, providers=[])
  assert media.proxdash_file_id is not None

  results = px.files.list(providers=['gemini'])
  pd_results = [r for r in results if r.proxdash_file_id]
  print(f'  Total results: {len(results)}')
  print(f'  ProxDash results: {len(pd_results)}')
  for r in pd_results[:3]:
    print(
        f'    proxdash_id={r.proxdash_file_id}, '
        f'providers={list(r.provider_file_api_ids.keys()) if r.provider_file_api_ids else []}'
    )


def test_list_proxdash_gemini():
  print('\n=== list_proxdash_gemini ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  px.files.upload(media=media, providers=['gemini'])
  assert media.proxdash_file_id is not None
  gemini_file_id = media.provider_file_api_ids['gemini']

  results = px.files.list(providers=['gemini'])
  pd_results = [r for r in results if r.proxdash_file_id]
  provider_only = [r for r in results if not r.proxdash_file_id]

  print(f'  Total results: {len(results)}')
  print(f'  ProxDash results: {len(pd_results)}')
  print(f'  Provider-only results: {len(provider_only)}')

  # The file should appear in ProxDash results with gemini ID.
  matching = [
      r for r in pd_results if r.provider_file_api_ids and
      r.provider_file_api_ids.get('gemini') == gemini_file_id
  ]
  print(f'  ProxDash results with matching gemini ID: {len(matching)}')

  # The same gemini file_id should NOT appear in provider-only results.
  duplicates = [
      r for r in provider_only if r.provider_file_api_ids and
      r.provider_file_api_ids.get('gemini') == gemini_file_id
  ]
  print(f'  Duplicates in provider-only: {len(duplicates)}')
  assert len(duplicates
            ) == 0, (f'Expected no duplicates, found {len(duplicates)}')


# --- 6. Download via ProxDash ---


def test_download_proxdash():
  print('\n=== download_proxdash ===')
  with open(_asset('cat.pdf'), 'rb') as f:
    original_data = f.read()
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  px.files.upload(media=media, providers=[])
  assert media.proxdash_file_id is not None

  # Clear local data so download must come from ProxDash.
  media.data = None
  media.path = None
  px.files.download(media=media)
  assert media.data is not None
  assert len(media.data) == len(original_data)
  print(f'  Downloaded {len(media.data)} bytes from ProxDash')
  print(f'  Match: {media.data == original_data}')


# --- 7. Delete ---


def test_delete_proxdash_gemini():
  print('\n=== delete_proxdash_gemini ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  px.files.upload(media=media, providers=['gemini'])
  assert media.proxdash_file_id is not None
  assert 'gemini' in media.provider_file_api_ids
  print('  Before remove:')
  _print_media(media)

  px.files.remove(media=media)
  print('  After remove:')
  _print_media(media)
  assert media.proxdash_file_id is None
  assert media.proxdash_file_status is None
  assert media.provider_file_api_ids == {}


def test_delete_proxdash_only():
  print('\n=== delete_proxdash_only ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf'
  )
  px.files.upload(media=media, providers=[])
  assert media.proxdash_file_id is not None
  print('  Before remove:')
  _print_media(media)

  px.files.remove(media=media)
  print('  After remove:')
  _print_media(media)
  assert media.proxdash_file_id is None


# --- Runner ---


def _get_model_config(
    provider,
    model,
    provider_model_identifier,
    input_format=None,
):
  S = types.FeatureSupportType.SUPPORTED
  NS = types.FeatureSupportType.NOT_SUPPORTED
  if input_format is None:
    input_format = ['text', 'document']
  return types.ProviderModelConfig(
      provider_model=types.ProviderModelType(
          provider=provider, model=model,
          provider_model_identifier=provider_model_identifier
      ), pricing=types.ProviderModelPricingType(
          input_token_cost=1.0, output_token_cost=2.0
      ), metadata=types.ProviderModelMetadataType(is_recommended=True),
      features=types.FeatureConfigType(
          prompt=S,
          messages=S,
          system_prompt=S,
          parameters=types.ParameterConfigType(
              temperature=S, max_tokens=S, stop=S, n=NS, thinking=NS
          ),
          output_format=types.OutputFormatConfigType(text=S),
          input_format=types.InputFormatConfigType(
              text=S if 'text' in input_format else NS,
              image=S if 'image' in input_format else NS,
              document=S if 'document' in input_format else NS,
              audio=S if 'audio' in input_format else NS,
              video=S if 'video' in input_format else NS,
          ),
      )
  )


def _register_models():
  client = px.get_default_proxai_client()
  client.model_configs_instance.unregister_all_models()
  for provider, model in [
      ('gemini', 'gemini-2.5-flash'),
      ('claude', 'claude-sonnet-4-6'),
      ('openai', 'gpt-4o'),
      ('mistral', 'mistral-small-latest'),
  ]:
    if provider == 'gemini':
      fmt = ['text', 'document', 'image', 'audio', 'video']
    else:
      fmt = ['text', 'document', 'image']
    client.model_configs_instance.register_provider_model_config(
        _get_model_config(provider, model, model, input_format=fmt)
    )


def _connect_default():
  px.connect(
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          base_url='http://localhost:3001',
          api_key='pjlfi0h-mo8mrm56-fgsvgftdk78',
      ),
      cache_options=px.CacheOptions(cache_path='/tmp/proxai_cache'),
  )


TEST_SEQUENCE = [
    # 1. ProxDash-only
    ('upload_proxdash_pdf', test_upload_proxdash_pdf),
    ('upload_proxdash_image', test_upload_proxdash_image),
    ('upload_proxdash_markdown', test_upload_proxdash_markdown),
    ('upload_proxdash_audio', test_upload_proxdash_audio),
    # 2. ProxDash + single provider
    ('upload_proxdash_gemini_pdf', test_upload_proxdash_gemini_pdf),
    ('upload_proxdash_openai_pdf', test_upload_proxdash_openai_pdf),
    ('upload_proxdash_claude_pdf', test_upload_proxdash_claude_pdf),
    ('upload_proxdash_mistral_pdf', test_upload_proxdash_mistral_pdf),
    # 3. ProxDash + multi provider
    ('upload_proxdash_multi', test_upload_proxdash_multi),
    # 4. Parallel vs sequential
    ('upload_parallel', test_upload_parallel),
    ('upload_sequential', test_upload_sequential),
    # 5. List
    ('list_proxdash_only', test_list_proxdash_only),
    ('list_proxdash_gemini', test_list_proxdash_gemini),
    # 6. Download
    ('download_proxdash', test_download_proxdash),
    # 7. Delete
    ('delete_proxdash_gemini', test_delete_proxdash_gemini),
    ('delete_proxdash_only', test_delete_proxdash_only),
]
TEST_MAP = dict(TEST_SEQUENCE)


def main():
  parser = argparse.ArgumentParser(description='ProxDash Files API manual test')
  test_names = [name for name, _ in TEST_SEQUENCE]
  parser.add_argument(
      '--test', default='all',
      help=f'Test to run: {", ".join(test_names)}, or "all"'
  )
  args = parser.parse_args()

  _connect_default()
  _register_models()

  if args.test == 'all':
    for _name, test_fn in TEST_SEQUENCE:
      try:
        test_fn()
      except Exception as e:
        print(f'  FAILED: {e}')
  else:
    if args.test not in TEST_MAP:
      print(f'Unknown test: {args.test}')
      print(f'Available: {", ".join(test_names)}')
      return
    TEST_MAP[args.test]()


if __name__ == '__main__':
  main()
