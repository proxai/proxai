"""Quick manual test for px.files.upload() API.

Usage:
  poetry run python3 examples/files_api_test.py
  poetry run python3 examples/files_api_test.py --test single_gemini_pdf
  poetry run python3 examples/files_api_test.py --test single_gemini_image
  poetry run python3 examples/files_api_test.py --test multi_parallel
  poetry run python3 examples/files_api_test.py --test all
"""

import argparse
import json
import os
import time

import proxai as px
from proxai.chat.message_content import FileUploadState
from proxai.connectors.files import FileUploadError

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'refactoring_test_assets')
OUTPUT_DIR = os.path.expanduser('~/temp')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'uploaded_media_contents.json')


def _asset(filename):
  return os.path.join(_ASSETS_DIR, filename)


def _ensure_output_dir():
  os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_result(media, label):
  _ensure_output_dir()
  results = []
  if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
      results = json.load(f)
  results.append({
      'label': label,
      'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
      'media_content': media.to_dict(),
  })
  with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
  print(f'  Saved to {OUTPUT_FILE}')


def _print_upload_result(media):
  if media.provider_file_api_ids:
    for provider, file_id in media.provider_file_api_ids.items():
      status = media.provider_file_api_status.get(provider)
      state = status.state.value if status and status.state else '?'
      size = status.size_bytes if status else '?'
      print(f'  {provider}: id={file_id}, state={state}, size={size}')
  else:
    print('  No uploads completed.')


def _upload_and_print(media, providers, label):
  start = time.time()
  px.files.upload(media=media, providers=providers)
  elapsed = time.time() - start
  _print_upload_result(media)
  print(f'  Time: {elapsed:.2f}s')
  _save_result(media, label)
  return elapsed


# --- Helpers ---

def _test_upload_success(provider, asset_file, mime_type, label):
  print(f'\n=== Single Upload: {provider} {asset_file} ===')
  media = px.MessageContent(path=_asset(asset_file), media_type=mime_type)
  _upload_and_print(media, [provider], label)
  assert media.provider_file_api_status[provider].state == FileUploadState.ACTIVE


def _test_upload_expect_error(provider, asset_file, mime_type, label):
  print(f'\n=== Single Upload: {provider} {asset_file} (expect error) ===')
  media = px.MessageContent(path=_asset(asset_file), media_type=mime_type)
  try:
    px.files.upload(media=media, providers=[provider])
    _save_result(media, label + '_unexpected_success')
    raise AssertionError(
        f"Expected FileUploadError for {provider} with "
        f"{asset_file}, but upload succeeded.")
  except FileUploadError as e:
    print(f'  Expected error: {e}')
    assert provider in e.errors
    assert media.provider_file_api_status[provider].state == FileUploadState.FAILED


# --- Single provider: document tests (all 4 should succeed) ---

def test_gemini_pdf():
  _test_upload_success('gemini', 'cat.pdf', 'application/pdf', 'gemini_pdf')

def test_claude_pdf():
  _test_upload_success('claude', 'cat.pdf', 'application/pdf', 'claude_pdf')

def test_openai_pdf():
  _test_upload_success('openai', 'cat.pdf', 'application/pdf', 'openai_pdf')

def test_mistral_pdf():
  _test_upload_success('mistral', 'cat.pdf', 'application/pdf', 'mistral_pdf')


# --- Single provider: image tests ---

def test_gemini_image():
  _test_upload_success('gemini', 'cat.jpeg', 'image/jpeg', 'gemini_image')

def test_claude_image():
  _test_upload_success('claude', 'cat.jpeg', 'image/jpeg', 'claude_image')

def test_openai_image():
  _test_upload_success('openai', 'cat.jpeg', 'image/jpeg', 'openai_image')

def test_mistral_image():
  _test_upload_success('mistral', 'cat.jpeg', 'image/jpeg', 'mistral_image')


# --- Single provider: audio tests ---

def test_gemini_audio():
  _test_upload_success('gemini', 'cat.mp3', 'audio/mpeg', 'gemini_audio')

def test_claude_audio():
  _test_upload_success('claude', 'cat.mp3', 'audio/mpeg', 'claude_audio')

def test_openai_audio():
  _test_upload_success('openai', 'cat.mp3', 'audio/mpeg', 'openai_audio')

def test_mistral_audio_fail():
  _test_upload_expect_error(
      'mistral', 'cat.mp3', 'audio/mpeg', 'mistral_audio_fail')


# --- Single provider: video tests ---

def test_gemini_video():
  _test_upload_success('gemini', 'cat.mp4', 'video/mp4', 'gemini_video')

def test_claude_video():
  _test_upload_success('claude', 'cat.mp4', 'video/mp4', 'claude_video')

def test_openai_video():
  _test_upload_success('openai', 'cat.mp4', 'video/mp4', 'openai_video')

def test_mistral_video_fail():
  _test_upload_expect_error(
      'mistral', 'cat.mp4', 'video/mp4', 'mistral_video_fail')


# --- Multi-provider tests ---

def test_multi_sequential():
  print('\n=== Multi Upload: Sequential (parallel=False) ===')
  px.connect(
      provider_call_options=px.ProviderCallOptions(
          allow_parallel_file_operations=False))
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf')
  providers = ['gemini', 'claude', 'openai']
  start = time.time()
  px.files.upload(media=media, providers=providers)
  elapsed = time.time() - start
  _print_upload_result(media)
  print(f'  Time: {elapsed:.2f}s (sequential)')
  for p in providers:
    assert p in media.provider_file_api_ids
  _save_result(media, 'multi_sequential')
  px.reset_state()


def test_multi_parallel():
  print('\n=== Multi Upload: Parallel (parallel=True, default) ===')
  px.connect(
      provider_call_options=px.ProviderCallOptions(
          allow_parallel_file_operations=True))
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf')
  providers = ['gemini', 'claude', 'openai']
  start = time.time()
  px.files.upload(media=media, providers=providers)
  elapsed = time.time() - start
  _print_upload_result(media)
  print(f'  Time: {elapsed:.2f}s (parallel)')
  for p in providers:
    assert p in media.provider_file_api_ids
  _save_result(media, 'multi_parallel')
  px.reset_state()


def test_multi_parallel_mixed_media():
  print('\n=== Multi Upload: Parallel mixed (gemini OK, mistral error) ===')
  px.connect()
  media = px.MessageContent(
      path=_asset('cat.mp3'), media_type='audio/mpeg')
  try:
    px.files.upload(media=media, providers=['gemini', 'mistral'])
    _save_result(media, 'multi_parallel_mixed_unexpected_success')
    raise AssertionError(
        "Expected FileUploadError for partial failure, "
        "but all uploads succeeded.")
  except FileUploadError as e:
    print(f'  Partial failure (expected): {e}')
    assert 'mistral' in e.errors
    assert e.media.provider_file_api_status['gemini'].state == FileUploadState.ACTIVE
    assert e.media.provider_file_api_status['mistral'].state == FileUploadState.FAILED
    _save_result(e.media, 'multi_parallel_mixed_partial')
    print(f'  gemini: OK (id={e.media.provider_file_api_ids.get("gemini")})')
    print(f'  mistral: FAILED')
  px.reset_state()


# --- Remove helpers ---

def _test_remove_success(provider, asset_file, mime_type):
  print(f'\n=== Remove: {provider} {asset_file} ===')
  media = px.MessageContent(path=_asset(asset_file), media_type=mime_type)
  px.files.upload(media=media, providers=[provider])
  file_id = media.provider_file_api_ids[provider]
  print(f'  Uploaded: {file_id}')

  px.files.remove(media=media, providers=[provider])
  assert provider not in media.provider_file_api_ids
  assert provider not in media.provider_file_api_status
  print(f'  Removed OK')


# --- Single provider remove tests ---

def test_remove_gemini():
  _test_remove_success('gemini', 'cat.pdf', 'application/pdf')

def test_remove_claude():
  _test_remove_success('claude', 'cat.pdf', 'application/pdf')

def test_remove_openai():
  _test_remove_success('openai', 'cat.pdf', 'application/pdf')

def test_remove_mistral():
  _test_remove_success('mistral', 'cat.pdf', 'application/pdf')


# --- Multi-provider remove tests ---

def test_remove_all():
  print('\n=== Remove: All providers (default) ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf')
  px.files.upload(media=media, providers=['gemini', 'claude', 'openai'])
  assert len(media.provider_file_api_ids) == 3
  print(f'  Uploaded to: {list(media.provider_file_api_ids.keys())}')

  px.files.remove(media=media)
  assert media.provider_file_api_ids == {}
  assert media.provider_file_api_status == {}
  print('  Removed all OK')


def test_remove_selective():
  print('\n=== Remove: Selective (gemini only, keep claude) ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf')
  px.files.upload(media=media, providers=['gemini', 'claude'])
  assert len(media.provider_file_api_ids) == 2
  print(f'  Uploaded to: {list(media.provider_file_api_ids.keys())}')

  px.files.remove(media=media, providers=['gemini'])
  assert 'gemini' not in media.provider_file_api_ids
  assert 'claude' in media.provider_file_api_ids
  print(f'  Remaining: {list(media.provider_file_api_ids.keys())}')

  px.files.remove(media=media)
  assert media.provider_file_api_ids == {}
  print('  Cleaned up remaining OK')


# --- List helpers ---

# Assets each provider supports for upload.
_PROVIDER_ASSETS = {
    'gemini': [
        ('cat.pdf', 'application/pdf'),
        ('cat.jpeg', 'image/jpeg'),
        ('cat.webp', 'image/webp'),
        ('cat.mp3', 'audio/mpeg'),
        ('cat.mp4', 'video/mp4'),
    ],
    'claude': [
        ('cat.pdf', 'application/pdf'),
        ('cat.jpeg', 'image/jpeg'),
        ('cat.webp', 'image/webp'),
        ('cat.mp3', 'audio/mpeg'),
        ('cat.mp4', 'video/mp4'),
    ],
    'openai': [
        ('cat.pdf', 'application/pdf'),
        ('cat.jpeg', 'image/jpeg'),
        ('cat.webp', 'image/webp'),
        ('cat.mp3', 'audio/mpeg'),
        ('cat.mp4', 'video/mp4'),
    ],
    'mistral': [
        ('cat.pdf', 'application/pdf'),
        ('cat.jpeg', 'image/jpeg'),
        ('cat.webp', 'image/webp'),
    ],
}


def _test_list_provider(provider):
  assets = _PROVIDER_ASSETS[provider]
  print(f'\n=== List: {provider} ({len(assets)} files) ===')

  medias = []
  for asset_file, mime_type in assets:
    media = px.MessageContent(
        path=_asset(asset_file), media_type=mime_type)
    px.files.upload(media=media, providers=[provider])
    medias.append(media)
  uploaded_ids = {
      m.provider_file_api_ids[provider] for m in medias}
  print(f'  Uploaded {len(uploaded_ids)} files')

  files = px.files.list(providers=[provider])
  listed_ids = {
      f.provider_file_api_ids[provider] for f in files}
  for uid in uploaded_ids:
    assert uid in listed_ids, (
        f'Uploaded file {uid} not found in list results')
  print(f'  Listed {len(files)} files, all {len(uploaded_ids)} found')

  for media in medias:
    px.files.remove(media=media, providers=[provider])
  print(f'  Cleaned up {len(medias)} files')


# --- Single provider list tests ---

def test_list_gemini():
  _test_list_provider('gemini')

def test_list_claude():
  _test_list_provider('claude')

def test_list_openai():
  _test_list_provider('openai')

def test_list_mistral():
  _test_list_provider('mistral')


# --- Multi-provider list tests ---

def test_list_all():
  print('\n=== List: All providers ===')
  medias = []
  for provider, assets in _PROVIDER_ASSETS.items():
    asset_file, mime_type = assets[0]
    media = px.MessageContent(
        path=_asset(asset_file), media_type=mime_type)
    px.files.upload(media=media, providers=[provider])
    medias.append((provider, media))
  print(f'  Uploaded to: {[p for p, _ in medias]}')

  files = px.files.list()
  providers_in_results = set()
  for f in files:
    providers_in_results.update(f.provider_file_api_ids.keys())
  for provider, _ in medias:
    assert provider in providers_in_results
  print(f'  Listed {len(files)} files from {sorted(providers_in_results)}')

  for _, media in medias:
    px.files.remove(media=media)
  print('  Cleaned up OK')


def test_list_with_limit():
  print('\n=== List: With limit_per_provider=1 ===')
  medias = []
  for asset_file, mime_type in _PROVIDER_ASSETS['gemini'][:2]:
    media = px.MessageContent(
        path=_asset(asset_file), media_type=mime_type)
    px.files.upload(media=media, providers=['gemini'])
    medias.append(media)
  print(f'  Uploaded {len(medias)} files to gemini')

  files = px.files.list(providers=['gemini'], limit_per_provider=1)
  assert len(files) == 1
  print(f'  Listed {len(files)} file (limit=1)')

  for media in medias:
    px.files.remove(media=media, providers=['gemini'])
  print('  Cleaned up OK')


# --- Serialization round-trip test ---

def test_serialization_round_trip():
  print('\n=== Serialization Round-Trip ===')
  media = px.MessageContent(
      path=_asset('cat.pdf'), media_type='application/pdf')
  px.files.upload(media=media, providers=['gemini'])

  d = media.to_dict()
  json_str = json.dumps(d)
  restored = px.MessageContent.from_dict(json.loads(json_str))

  assert restored.provider_file_api_ids == media.provider_file_api_ids
  gemini_status = restored.provider_file_api_status['gemini']
  assert gemini_status.file_id == media.provider_file_api_status['gemini'].file_id
  assert gemini_status.state == FileUploadState.ACTIVE
  print('  Round-trip OK')
  print(f'  file_id: {gemini_status.file_id}')
  print(f'  uri: {gemini_status.uri}')

  px.files.remove(media=media)
  print('  Cleaned up OK')


# --- Cleanup ---

def test_cleanup_all():
  """Best-effort cleanup of all files tracked in the JSON output."""
  print('\n=== Cleanup: Remove all tracked files ===')
  if not os.path.exists(OUTPUT_FILE):
    print('  No output file found, nothing to clean up.')
    return

  with open(OUTPUT_FILE, 'r') as f:
    results = json.load(f)

  succeeded = 0
  failed = 0
  for entry in results:
    mc_dict = entry.get('media_content', {})
    if not mc_dict.get('provider_file_api_ids'):
      continue
    media = px.MessageContent.from_dict(mc_dict)
    providers = list(media.provider_file_api_ids.keys())
    try:
      px.files.remove(media=media)
      print(f'  removed {providers}')
      succeeded += len(providers)
    except Exception:
      removed = [p for p in providers if p not in media.provider_file_api_ids]
      remaining = [p for p in providers if p in media.provider_file_api_ids]
      print(f'  removed {removed} failed {remaining}')
      succeeded += len(removed)
      failed += len(remaining)

  print(f'  Done: {succeeded} removed, {failed} failed')

  os.remove(OUTPUT_FILE)
  print(f'  Cleared {OUTPUT_FILE}')


# --- Runner ---

TEST_SEQUENCE = [
    ('gemini_pdf', test_gemini_pdf),
    ('claude_pdf', test_claude_pdf),
    ('openai_pdf', test_openai_pdf),
    ('mistral_pdf', test_mistral_pdf),

    ('gemini_image', test_gemini_image),
    ('claude_image', test_claude_image),
    ('openai_image', test_openai_image),
    ('mistral_image', test_mistral_image),

    ('gemini_audio', test_gemini_audio),
    ('claude_audio', test_claude_audio),
    ('openai_audio', test_openai_audio),
    ('mistral_audio_fail', test_mistral_audio_fail),

    ('gemini_video', test_gemini_video),
    ('claude_video', test_claude_video),
    ('openai_video', test_openai_video),
    ('mistral_video_fail', test_mistral_video_fail),

    ('multi_sequential', test_multi_sequential),
    ('multi_parallel', test_multi_parallel),
    ('multi_parallel_mixed_media', test_multi_parallel_mixed_media),

    ('remove_gemini', test_remove_gemini),
    ('remove_claude', test_remove_claude),
    ('remove_openai', test_remove_openai),
    ('remove_mistral', test_remove_mistral),
    ('remove_all', test_remove_all),
    ('remove_selective', test_remove_selective),

    ('list_gemini', test_list_gemini),
    ('list_claude', test_list_claude),
    ('list_openai', test_list_openai),
    ('list_mistral', test_list_mistral),
    ('list_all', test_list_all),
    ('list_with_limit', test_list_with_limit),

    ('serialization', test_serialization_round_trip),
    ('cleanup_all', test_cleanup_all),
]
TEST_MAP = dict(TEST_SEQUENCE)


def main():
  parser = argparse.ArgumentParser(description='Files API manual test')
  test_names = [name for name, _ in TEST_SEQUENCE]
  parser.add_argument(
      '--test', default='all',
      help=f'Test to run: {", ".join(test_names)}, or "all"')
  args = parser.parse_args()

  px.connect()

  if args.test == 'all':
    for name, test_fn in TEST_SEQUENCE:
      test_fn()
      px.reset_state()
      px.connect()
  else:
    if args.test not in TEST_MAP:
      print(f'Unknown test: {args.test}')
      print(f'Available: {", ".join(test_names)}')
      return
    TEST_MAP[args.test]()

  if os.path.exists(OUTPUT_FILE):
    print(f'\nResults saved to: {OUTPUT_FILE}')


if __name__ == '__main__':
  main()
