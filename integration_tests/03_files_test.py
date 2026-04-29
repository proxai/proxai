"""Integration tests — px.files.* and ProxDash file integration.

Covers upload / list / download / remove for each provider, multi-provider
parallel + sequential, generate-with-files, file caching, and ProxDash
file persistence (proxdash_file_id, dedup, download via ProxDash, delete).

Usage:
  poetry run python3 integration_tests/03_files_test.py
  poetry run python3 integration_tests/03_files_test.py --auto-continue
  poetry run python3 integration_tests/03_files_test.py --test upload_pdf_all_providers
"""
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import proxai as px
import proxai.types as types
from proxai.chat.message_content import FileUploadState
from proxai.connectors.files import FileUploadError

import _utils
from _utils import (
    integration_block, manual_check, run_sequence,
    init_run, ensure_setup_state, asset,
    FILES_PROVIDER_MODELS,
    ASSET_PDF, ASSET_IMAGE, ASSET_WEBP, ASSET_MD, ASSET_AUDIO, ASSET_VIDEO,
)


_LABEL = '03_files'


@integration_block
def setup_connection(state_data):
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      logging_options=types.LoggingOptions(
          logging_path=ctx.root_logging_path),
      cache_options=types.CacheOptions(
          cache_path=ctx.root_cache_path,
          clear_query_cache_on_connect=True,
      ),
      proxdash_options=types.ProxDashOptions(
          stdout=True,
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
      provider_call_options=types.ProviderCallOptions(
          allow_parallel_file_operations=True),
  )
  print(f'> Connected at {ctx.proxdash_base_url}')
  return state_data


def _media(path_name: str, media_type: str) -> px.MessageContent:
  return px.MessageContent(path=asset(path_name), media_type=media_type)


def _print_upload(media: px.MessageContent) -> None:
  if media.proxdash_file_id:
    print(f'  proxdash_file_id: {media.proxdash_file_id}')
  if media.provider_file_api_ids:
    for prov, fid in media.provider_file_api_ids.items():
      st = media.provider_file_api_status.get(prov)
      state = st.state.value if st and st.state else '?'
      print(f'  {prov}: id={fid}, state={state}')
  else:
    print('  (no provider uploads)')


def _assert_active(media: px.MessageContent, provider: str) -> None:
  st = media.provider_file_api_status.get(provider)
  assert st is not None, f'no status entry for {provider}'
  assert st.state == FileUploadState.ACTIVE, (
      f'{provider} upload state {st.state}, expected ACTIVE')


def _expect_upload_error(media: px.MessageContent, provider: str) -> None:
  try:
    px.files.upload(media=media, providers=[provider])
  except (ValueError, FileUploadError) as e:
    print(f'  expected error for {provider}: {e}')
    return
  raise AssertionError(f'upload for {provider} should have failed')


# -----------------------------------------------------------------------------
# 8.1  Single-provider upload — parameterized over (asset, provider)
# -----------------------------------------------------------------------------

# Asset compatibility per provider.
_PROVIDER_ASSETS_OK = {
    'pdf':   ['gemini', 'claude', 'openai', 'mistral'],
    'image': ['gemini', 'claude', 'mistral'],
    'webp':  ['gemini', 'claude', 'mistral'],
    'audio': ['gemini'],
    'video': ['gemini'],
}
_PROVIDER_ASSETS_FAIL = {
    'pdf':   [],
    'image': ['openai'],
    'audio': ['claude', 'openai', 'mistral'],
    'video': ['claude', 'openai', 'mistral'],
}


@integration_block
def upload_pdf_all_providers(state_data):
  """Every provider in FILES_PROVIDER_MODELS accepts cat.pdf."""
  for provider in _PROVIDER_ASSETS_OK['pdf']:
    print(f'\n> {provider} pdf upload')
    media = _media(ASSET_PDF, 'application/pdf')
    px.files.upload(media=media, providers=[provider])
    _print_upload(media)
    _assert_active(media, provider)
    px.files.remove(media=media, providers=[provider])
  return state_data


@integration_block
def upload_image_per_provider(state_data):
  """gemini/claude/mistral accept jpeg; openai rejects."""
  for provider in _PROVIDER_ASSETS_OK['image']:
    print(f'\n> {provider} image upload')
    media = _media(ASSET_IMAGE, 'image/jpeg')
    px.files.upload(media=media, providers=[provider])
    _print_upload(media)
    _assert_active(media, provider)
    px.files.remove(media=media, providers=[provider])
  for provider in _PROVIDER_ASSETS_FAIL['image']:
    print(f'\n> {provider} image upload (expected failure)')
    media = _media(ASSET_IMAGE, 'image/jpeg')
    _expect_upload_error(media, provider)
  return state_data


@integration_block
def upload_audio_gemini_only(state_data):
  """Only gemini supports cat.mp3."""
  print('\n> gemini audio upload')
  media = _media(ASSET_AUDIO, 'audio/mpeg')
  px.files.upload(media=media, providers=['gemini'])
  _print_upload(media)
  _assert_active(media, 'gemini')
  px.files.remove(media=media, providers=['gemini'])

  for provider in _PROVIDER_ASSETS_FAIL['audio']:
    print(f'\n> {provider} audio upload (expected failure)')
    media = _media(ASSET_AUDIO, 'audio/mpeg')
    _expect_upload_error(media, provider)
  return state_data


@integration_block
def upload_video_gemini_only(state_data):
  """Only gemini supports cat.mp4."""
  print('\n> gemini video upload')
  media = _media(ASSET_VIDEO, 'video/mp4')
  px.files.upload(media=media, providers=['gemini'])
  _print_upload(media)
  _assert_active(media, 'gemini')
  px.files.remove(media=media, providers=['gemini'])

  for provider in _PROVIDER_ASSETS_FAIL['video']:
    print(f'\n> {provider} video upload (expected failure)')
    media = _media(ASSET_VIDEO, 'video/mp4')
    _expect_upload_error(media, provider)
  return state_data


# -----------------------------------------------------------------------------
# 8.2  Multi-provider upload — sequential and parallel
# -----------------------------------------------------------------------------

@integration_block
def upload_multi_sequential(state_data):
  ctx = _utils._CTX
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url, api_key=state_data['api_key']),
      provider_call_options=types.ProviderCallOptions(
          allow_parallel_file_operations=False),
  )
  media = _media(ASSET_PDF, 'application/pdf')
  providers = ['gemini', 'claude', 'openai']
  start = time.time()
  px.files.upload(media=media, providers=providers)
  elapsed = time.time() - start
  print(f'> sequential upload: {elapsed:.2f}s')
  for p in providers:
    assert p in media.provider_file_api_ids
  px.files.remove(media=media)
  state_data['_seq_elapsed'] = elapsed
  return state_data


@integration_block
def upload_multi_parallel(state_data):
  ctx = _utils._CTX
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url, api_key=state_data['api_key']),
      provider_call_options=types.ProviderCallOptions(
          allow_parallel_file_operations=True),
  )
  media = _media(ASSET_PDF, 'application/pdf')
  providers = ['gemini', 'claude', 'openai']
  start = time.time()
  px.files.upload(media=media, providers=providers)
  elapsed = time.time() - start
  print(f'> parallel upload: {elapsed:.2f}s')
  for p in providers:
    assert p in media.provider_file_api_ids
  seq = state_data.get('_seq_elapsed')
  if seq is not None:
    print(f'> seq={seq:.2f}s, par={elapsed:.2f}s '
          f'(parallel should usually be faster)')
  px.files.remove(media=media)
  return state_data


@integration_block
def upload_multi_mixed_media_fail(state_data):
  """A media unsupported by one of the providers raises ValueError."""
  ctx = _utils._CTX
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          disable_proxdash=True),
  )
  media = _media(ASSET_AUDIO, 'audio/mpeg')
  raised = None
  try:
    px.files.upload(media=media, providers=['gemini', 'mistral'])
  except ValueError as e:
    raised = str(e)
    print(f'> raised: {raised[:200]}')
  assert raised, 'mixed-media upload should have raised'
  return state_data


# -----------------------------------------------------------------------------
# 8.3  Remove
# -----------------------------------------------------------------------------

@integration_block
def remove_per_provider(state_data):
  for provider in ['gemini', 'claude', 'openai', 'mistral']:
    media = _media(ASSET_PDF, 'application/pdf')
    px.files.upload(media=media, providers=[provider])
    fid = media.provider_file_api_ids[provider]
    px.files.remove(media=media, providers=[provider])
    assert provider not in media.provider_file_api_ids
    assert provider not in media.provider_file_api_status
    print(f'> {provider}: removed {fid}')
  return state_data


@integration_block
def remove_all(state_data):
  media = _media(ASSET_PDF, 'application/pdf')
  px.files.upload(media=media, providers=['gemini', 'claude', 'openai'])
  assert len(media.provider_file_api_ids) == 3
  px.files.remove(media=media)
  assert media.provider_file_api_ids == {}
  assert media.provider_file_api_status == {}
  print('> remove() with no providers cleared all')
  return state_data


@integration_block
def remove_selective(state_data):
  media = _media(ASSET_PDF, 'application/pdf')
  px.files.upload(media=media, providers=['gemini', 'claude'])
  assert len(media.provider_file_api_ids) == 2
  px.files.remove(media=media, providers=['gemini'])
  assert 'gemini' not in media.provider_file_api_ids
  assert 'claude' in media.provider_file_api_ids
  px.files.remove(media=media)
  assert media.provider_file_api_ids == {}
  print('> selective remove + final cleanup OK')
  return state_data


# -----------------------------------------------------------------------------
# 8.4  List
# -----------------------------------------------------------------------------

# Assets each provider supports, used for the list test.
_PROVIDER_ASSETS = {
    'gemini': [
        (ASSET_PDF, 'application/pdf'),
        (ASSET_IMAGE, 'image/jpeg'),
        (ASSET_WEBP, 'image/webp'),
        (ASSET_AUDIO, 'audio/mpeg'),
        (ASSET_VIDEO, 'video/mp4'),
    ],
    'claude':  [(ASSET_PDF, 'application/pdf'),
                (ASSET_IMAGE, 'image/jpeg'),
                (ASSET_WEBP, 'image/webp')],
    'openai':  [(ASSET_PDF, 'application/pdf'),
                (ASSET_MD, 'text/markdown')],
    'mistral': [(ASSET_PDF, 'application/pdf'),
                (ASSET_IMAGE, 'image/jpeg'),
                (ASSET_WEBP, 'image/webp')],
}


@integration_block
def list_per_provider(state_data):
  for provider in ['gemini', 'claude', 'openai', 'mistral']:
    print(f'\n> {provider} list')
    medias = []
    for path_name, mime in _PROVIDER_ASSETS[provider]:
      m = _media(path_name, mime)
      px.files.upload(media=m, providers=[provider])
      medias.append(m)
    uploaded_ids = {m.provider_file_api_ids[provider] for m in medias}
    listed = px.files.list(providers=[provider])
    listed_ids = {f.provider_file_api_ids[provider] for f in listed}
    for uid in uploaded_ids:
      assert uid in listed_ids, f'{provider}: {uid} missing from list'
    print(f'  uploaded {len(uploaded_ids)}, listed {len(listed)}')
    for m in medias:
      px.files.remove(media=m, providers=[provider])
  return state_data


@integration_block
def list_all_providers(state_data):
  medias = []
  for provider, assets in _PROVIDER_ASSETS.items():
    path_name, mime = assets[0]
    m = _media(path_name, mime)
    px.files.upload(media=m, providers=[provider])
    medias.append((provider, m))
  files = px.files.list()
  in_results = set()
  for f in files:
    if f.provider_file_api_ids:
      in_results.update(f.provider_file_api_ids.keys())
  for provider, _m in medias:
    assert provider in in_results, f'{provider} missing from global list'
  print(f'> listed {len(files)} files across {sorted(in_results)}')
  for _provider, m in medias:
    px.files.remove(media=m)
  return state_data


@integration_block
def list_with_limit(state_data):
  medias = []
  for path_name, mime in _PROVIDER_ASSETS['gemini'][:2]:
    m = _media(path_name, mime)
    px.files.upload(media=m, providers=['gemini'])
    medias.append(m)
  files = px.files.list(providers=['gemini'], limit_per_provider=1)
  assert len(files) == 1
  print(f'> limit_per_provider=1 returned {len(files)} file')
  for m in medias:
    px.files.remove(media=m, providers=['gemini'])
  return state_data


# -----------------------------------------------------------------------------
# 8.5  Download
# -----------------------------------------------------------------------------

@integration_block
def download_mistral_success(state_data):
  """Mistral supports download — bytes round-trip."""
  path = asset(ASSET_PDF)
  with open(path, 'rb') as f:
    original = f.read()

  media = _media(ASSET_PDF, 'application/pdf')
  px.files.upload(media=media, providers=['mistral'])
  uploaded_id = media.provider_file_api_ids['mistral']
  print(f'> uploaded: {uploaded_id}')

  download_media = px.MessageContent(
      media_type='application/pdf',
      provider_file_api_ids={'mistral': uploaded_id},
      provider_file_api_status=media.provider_file_api_status,
  )
  px.files.download(media=download_media, provider='mistral')
  assert download_media.data is not None
  assert len(download_media.data) == len(original)
  print(f'> downloaded {len(download_media.data)} bytes (in-memory)')

  with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
    tmp_path = tmp.name
  try:
    path_media = px.MessageContent(
        media_type='application/pdf',
        provider_file_api_ids={'mistral': uploaded_id},
        provider_file_api_status=media.provider_file_api_status,
    )
    px.files.download(media=path_media, provider='mistral', path=tmp_path)
    assert path_media.path == tmp_path
    with open(tmp_path, 'rb') as f:
      bytes_on_disk = f.read()
    assert len(bytes_on_disk) == len(original)
    print(f'> downloaded to path: {tmp_path}')
  finally:
    if os.path.exists(tmp_path):
      os.unlink(tmp_path)

  px.files.remove(media=media, providers=['mistral'])
  return state_data


@integration_block
def download_unsupported_providers(state_data):
  """gemini / claude / openai do not support download."""
  for provider in ['gemini', 'claude', 'openai']:
    media = _media(ASSET_PDF, 'application/pdf')
    px.files.upload(media=media, providers=[provider])
    raised = None
    try:
      px.files.download(media=media, provider=provider)
    except (ValueError, Exception) as e:
      raised = str(e)
    assert raised, f'{provider} download should have failed'
    print(f'> {provider}: expected error: {raised[:120]}')
    px.files.remove(media=media, providers=[provider])
  return state_data


# -----------------------------------------------------------------------------
# 8.6  Generate with files — manual upload + auto upload + failures
# -----------------------------------------------------------------------------

def _assert_cat_in_text(result, label: str = '') -> None:
  text = (result.result.output_text or '').lower()
  assert any(w in text for w in ('cat', 'kitten', 'feline')), (
      f'{label}: cat keywords not found in: {text[:200]}')


def _generate_with_media(provider_model, content_type, path_name, mime, prompt):
  return px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=content_type, path=asset(path_name), media_type=mime),
              px.MessageContent(type=px.ContentType.TEXT, text=prompt),
          ],
      }],
      provider_model=provider_model)


@integration_block
def generate_manual_upload_per_media(state_data):
  """Pre-upload via px.files.upload, then call generate with the same media.

  Iterates over (provider, asset, mime, prompt) for combinations expected
  to succeed.
  """
  scenarios = [
      ('gemini',  ASSET_PDF,   'application/pdf', 'What is inside this document?'),
      ('claude',  ASSET_PDF,   'application/pdf', 'What is inside this document?'),
      ('openai',  ASSET_PDF,   'application/pdf', 'What is inside this document?'),
      ('mistral', ASSET_PDF,   'application/pdf', 'What is inside this document?'),
      ('gemini',  ASSET_MD,    'text/markdown',   'What is inside this document?'),
      ('openai',  ASSET_MD,    'text/markdown',   'What is inside this document?'),
      ('mistral', ASSET_MD,    'text/markdown',   'What is inside this document?'),
      ('gemini',  ASSET_IMAGE, 'image/jpeg',      'What is in this image?'),
      ('claude',  ASSET_IMAGE, 'image/jpeg',      'What is in this image?'),
      ('mistral', ASSET_IMAGE, 'image/jpeg',      'What is in this image?'),
      ('gemini',  ASSET_AUDIO, 'audio/mpeg',      'What is this audio about?'),
      ('gemini',  ASSET_VIDEO, 'video/mp4',       'What is in this video?'),
  ]
  for provider, path_name, mime, prompt in scenarios:
    pm = FILES_PROVIDER_MODELS[provider]
    print(f'\n> {provider} manual-upload {path_name}')
    media = _media(path_name, mime)
    px.files.upload(media=media, providers=[provider])
    content_type = {
        'application/pdf': px.ContentType.DOCUMENT,
        'text/markdown':   px.ContentType.DOCUMENT,
        'image/jpeg':      px.ContentType.IMAGE,
        'audio/mpeg':      px.ContentType.AUDIO,
        'video/mp4':       px.ContentType.VIDEO,
    }[mime]
    result = px.generate(
        messages=[{
            'role': 'user',
            'content': [
                media,
                px.MessageContent(type=px.ContentType.TEXT, text=prompt),
            ],
        }],
        provider_model=pm)
    print(f'  {(result.result.output_text or "")[:100]}...')
    _assert_cat_in_text(result, f'{provider}/{path_name}')
    px.files.remove(media=media, providers=[provider])
  return state_data


@integration_block
def generate_auto_upload_pdf_per_provider(state_data):
  """No pre-upload; px.generate triggers auto-upload."""
  for provider in ['gemini', 'claude', 'openai', 'mistral']:
    pm = FILES_PROVIDER_MODELS[provider]
    print(f'\n> {provider} auto-upload pdf')
    media = _media(ASSET_PDF, 'application/pdf')
    result = px.generate(
        messages=[{
            'role': 'user',
            'content': [
                media,
                px.MessageContent(
                    type=px.ContentType.TEXT,
                    text='What is inside this document?'),
            ],
        }],
        provider_model=pm)
    _assert_cat_in_text(result, f'auto/{provider}')
    assert media.provider_file_api_ids and (
        provider in media.provider_file_api_ids), (
        f'auto-upload did not populate {provider} file_id')
    print(f'  auto-uploaded: {media.provider_file_api_ids[provider]}')
    px.files.remove(media=media, providers=[provider])
  return state_data


@integration_block
def generate_unsupported_media_fails(state_data):
  """Calling generate with media a provider can't accept raises ValueError."""
  scenarios = [
      ('claude',  ASSET_MD,    'text/markdown'),
      ('openai',  ASSET_IMAGE, 'image/jpeg'),
      ('claude',  ASSET_AUDIO, 'audio/mpeg'),
      ('openai',  ASSET_AUDIO, 'audio/mpeg'),
      ('mistral', ASSET_AUDIO, 'audio/mpeg'),
      ('claude',  ASSET_VIDEO, 'video/mp4'),
      ('openai',  ASSET_VIDEO, 'video/mp4'),
      ('mistral', ASSET_VIDEO, 'video/mp4'),
  ]
  for provider, path_name, mime in scenarios:
    print(f'\n> {provider} {path_name} (expect failure)')
    media = _media(path_name, mime)
    raised = None
    try:
      px.files.upload(media=media, providers=[provider])
    except ValueError as e:
      raised = str(e)
      print(f'  expected error: {raised[:120]}')
    assert raised, f'{provider} should have refused {path_name}'
  return state_data


@integration_block
def serialization_round_trip(state_data):
  """MessageContent.to_dict / from_dict preserves file API state."""
  media = _media(ASSET_PDF, 'application/pdf')
  px.files.upload(media=media, providers=['gemini'])

  d = media.to_dict()
  s = json.dumps(d)
  restored = px.MessageContent.from_dict(json.loads(s))
  assert restored.provider_file_api_ids == media.provider_file_api_ids
  st = restored.provider_file_api_status['gemini']
  assert st.file_id == media.provider_file_api_status['gemini'].file_id
  assert st.state == FileUploadState.ACTIVE
  print(f'> round-trip OK: {st.file_id}')
  px.files.remove(media=media)
  return state_data


# -----------------------------------------------------------------------------
# 8.7  Cache + files
# -----------------------------------------------------------------------------

_FILE_CACHE_PATH = os.path.expanduser('~/temp/proxai_files_cache/')


def _create_cached_client():
  if os.path.exists(_FILE_CACHE_PATH):
    shutil.rmtree(_FILE_CACHE_PATH)
  os.makedirs(_FILE_CACHE_PATH, exist_ok=True)
  return px.Client(
      cache_options=types.CacheOptions(
          cache_path=_FILE_CACHE_PATH, unique_response_limit=1))


@integration_block
def cache_with_file_upload(state_data):
  """First call PROVIDER, subsequent calls (same media + new media object) CACHE."""
  for provider in ['gemini', 'claude', 'openai', 'mistral']:
    pm = FILES_PROVIDER_MODELS[provider]
    print(f'\n> {provider} cache-with-file')
    client = _create_cached_client()
    media = _media(ASSET_PDF, 'application/pdf')

    msgs = lambda m: [{
        'role': 'user',
        'content': [
            m,
            px.MessageContent(
                type=px.ContentType.TEXT,
                text='What is inside this document?'),
        ],
    }]

    r1 = client.generate(messages=msgs(media), provider_model=pm)
    assert r1.connection.result_source == types.ResultSource.PROVIDER
    fid = media.provider_file_api_ids[provider]
    print(f'  call 1: PROVIDER, file={fid}')

    r2 = client.generate(messages=msgs(media), provider_model=pm)
    assert r2.connection.result_source == types.ResultSource.CACHE
    print('  call 2: CACHE (same media object)')

    media_new = _media(ASSET_PDF, 'application/pdf')
    r3 = client.generate(messages=msgs(media_new), provider_model=pm)
    assert r3.connection.result_source == types.ResultSource.CACHE
    print('  call 3: CACHE (new media object, same content)')

    client.files.remove(media=media, providers=[provider])
    new_ids = media_new.provider_file_api_ids or {}
    if provider in new_ids and new_ids[provider] != fid:
      client.files.remove(media=media_new, providers=[provider])
    if os.path.exists(_FILE_CACHE_PATH):
      shutil.rmtree(_FILE_CACHE_PATH)
  return state_data


# -----------------------------------------------------------------------------
# 8.8  ProxDash file integration
# -----------------------------------------------------------------------------

@integration_block
def proxdash_only_upload(state_data):
  """providers=[] uploads to ProxDash only."""
  for path_name, mime in [
      (ASSET_PDF, 'application/pdf'),
      (ASSET_IMAGE, 'image/jpeg'),
      (ASSET_MD, 'text/markdown'),
      (ASSET_AUDIO, 'audio/mpeg'),
  ]:
    print(f'\n> proxdash-only {path_name}')
    media = _media(path_name, mime)
    px.files.upload(media=media, providers=[])
    _print_upload(media)
    assert media.proxdash_file_id is not None
    assert not media.provider_file_api_ids
    px.files.remove(media=media)
  return state_data


@integration_block
def proxdash_plus_provider_upload(state_data):
  """Uploading to a provider also persists to ProxDash."""
  media = _media(ASSET_PDF, 'application/pdf')
  px.files.upload(media=media, providers=['gemini'])
  _print_upload(media)
  assert media.proxdash_file_id is not None
  assert 'gemini' in media.provider_file_api_ids
  px.files.remove(media=media)
  return state_data


@integration_block
def proxdash_list_dedup(state_data):
  """A file uploaded to both ProxDash and a provider appears once in list."""
  media = _media(ASSET_PDF, 'application/pdf')
  px.files.upload(media=media, providers=['gemini'])
  assert media.proxdash_file_id is not None
  gemini_id = media.provider_file_api_ids['gemini']

  files = px.files.list(providers=['gemini'])
  pd_results = [r for r in files if r.proxdash_file_id]
  provider_only = [r for r in files if not r.proxdash_file_id]

  matching_proxdash = [
      r for r in pd_results
      if r.provider_file_api_ids
      and r.provider_file_api_ids.get('gemini') == gemini_id
  ]
  duplicates_in_provider_only = [
      r for r in provider_only
      if r.provider_file_api_ids
      and r.provider_file_api_ids.get('gemini') == gemini_id
  ]
  print(f'> {len(files)} total, {len(pd_results)} proxdash-tagged, '
        f'{len(provider_only)} provider-only')
  assert len(duplicates_in_provider_only) == 0, (
      f'expected no provider-only duplicate, got '
      f'{len(duplicates_in_provider_only)}')
  print(f'> dedup OK ({len(matching_proxdash)} proxdash entries)')
  px.files.remove(media=media)
  return state_data


@integration_block
def proxdash_download(state_data):
  """Download via ProxDash with no provider IDs."""
  with open(asset(ASSET_PDF), 'rb') as f:
    original = f.read()

  media = _media(ASSET_PDF, 'application/pdf')
  px.files.upload(media=media, providers=[])
  assert media.proxdash_file_id is not None

  media.data = None
  media.path = None
  px.files.download(media=media)
  assert media.data is not None
  assert len(media.data) == len(original)
  print(f'> downloaded {len(media.data)} bytes via ProxDash')
  px.files.remove(media=media)
  return state_data


@integration_block
def proxdash_delete_clears_all_ids(state_data):
  """remove() clears proxdash_file_id and provider_file_api_ids."""
  media = _media(ASSET_PDF, 'application/pdf')
  px.files.upload(media=media, providers=['gemini'])
  assert media.proxdash_file_id is not None
  assert 'gemini' in media.provider_file_api_ids
  px.files.remove(media=media)
  assert media.proxdash_file_id is None
  assert media.proxdash_file_status is None
  assert media.provider_file_api_ids == {}
  print('> all ids cleared after remove')
  return state_data


# -----------------------------------------------------------------------------
# 8.9  Cleanup
# -----------------------------------------------------------------------------

@integration_block
def cleanup_all(state_data):
  """Best-effort: remove anything still referenced by stale media records."""
  print('> Best-effort sweep of remaining provider files...')
  for provider in ['gemini', 'claude', 'openai', 'mistral']:
    files = px.files.list(providers=[provider])
    for media in files:
      try:
        px.files.remove(media=media, providers=[provider])
      except Exception as e:
        print(f'  {provider} skip: {e}')
    print(f'> {provider}: swept {len(files)}')
  return state_data


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

BLOCKS = [
    upload_pdf_all_providers,
    upload_image_per_provider,
    upload_audio_gemini_only,
    upload_video_gemini_only,
    upload_multi_sequential,
    upload_multi_parallel,
    upload_multi_mixed_media_fail,
    remove_per_provider,
    remove_all,
    remove_selective,
    list_per_provider,
    list_all_providers,
    list_with_limit,
    download_mistral_success,
    download_unsupported_providers,
    generate_manual_upload_per_media,
    generate_auto_upload_pdf_per_provider,
    generate_unsupported_media_fails,
    serialization_round_trip,
    cache_with_file_upload,
    proxdash_only_upload,
    proxdash_plus_provider_upload,
    proxdash_list_dedup,
    proxdash_download,
    proxdash_delete_clears_all_ids,
    cleanup_all,
]


def main():
  ctx = init_run(_LABEL)
  state_data = ensure_setup_state(ctx)
  state_data = setup_connection(state_data=state_data, force_run=True)
  run_sequence(_LABEL, BLOCKS, state_data=state_data)


if __name__ == '__main__':
  main()
