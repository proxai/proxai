"""Tests for FilesManager upload and remove orchestration logic."""

from unittest.mock import MagicMock, patch

import pytest

import proxai.connections.api_key_manager as api_key_manager
import proxai.connections.proxdash as proxdash
import proxai.connectors.file_helpers as file_helpers
import proxai.connectors.files as files_module
import proxai.connectors.model_configs as model_configs
import proxai.types as types
from proxai.chat.message_content import (
    FileUploadMetadata,
    FileUploadState,
    MessageContent,
    ProxDashFileStatus,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.delenv(api_key, raising=False)
  yield


def _make_api_key_manager(monkeypatch, providers):
  for provider in providers:
    for key in model_configs.PROVIDER_KEY_MAP.get(provider, ()):
      monkeypatch.setenv(key, f'test-{key}')
  return api_key_manager.ApiKeyManager(
      init_from_params=api_key_manager.ApiKeyManagerParams()
  )


def _make_files_manager(
    monkeypatch, providers, parallel=True, proxdash_connection=None
):
  akm = _make_api_key_manager(monkeypatch, providers)
  return files_module.FilesManager(
      init_from_params=files_module.FilesManagerParams(
          api_key_manager=akm,
          provider_call_options=types.
          ProviderCallOptions(allow_parallel_file_operations=parallel,),
          proxdash_connection=proxdash_connection,
      )
  )


def _make_media(**kwargs):
  defaults = dict(path='/tmp/test.pdf', media_type='application/pdf')
  defaults.update(kwargs)
  return MessageContent(**defaults)


def _fake_upload_metadata(provider):
  return FileUploadMetadata(
      file_id=f'file-{provider}-123',
      provider=provider,
      state=FileUploadState.ACTIVE,
  )


def _fake_upload_fn(file_path, file_data, filename, mime_type, token_map):
  return FileUploadMetadata(
      file_id='file-fake-123',
      state=FileUploadState.ACTIVE,
  )


def _failing_upload_fn(file_path, file_data, filename, mime_type, token_map):
  raise RuntimeError('upload failed')


def _fake_remove_fn(file_id, token_map):
  pass


def _failing_remove_fn(file_id, token_map):
  raise RuntimeError('remove failed')


# --- Upload validation ---


class TestUploadValidation:

  def test_rejects_text_type(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = MessageContent(type='text', text='hello')
    with pytest.raises(ValueError, match='media content type'):
      mgr.upload(media=media, providers=['gemini'])

  def test_rejects_no_path_or_data(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = MessageContent(
        type='document', source='https://example.com/doc.pdf',
        media_type='application/pdf'
    )
    with pytest.raises(ValueError, match="'path' or 'data'"):
      mgr.upload(media=media, providers=['gemini'])

  def test_rejects_unsupported_provider(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    with pytest.raises(ValueError, match='does not support'):
      mgr.upload(media=media, providers=['deepseek'])

  def test_rejects_missing_api_key(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, [])
    media = _make_media()
    with pytest.raises(ValueError, match='No API key'):
      mgr.upload(media=media, providers=['gemini'])


# --- Upload execution ---


class TestUploadExecution:

  def test_single_provider_success(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    dispatch = {'gemini': _fake_upload_fn}
    with patch.dict(file_helpers.UPLOAD_DISPATCH, dispatch):
      mgr.upload(media=media, providers=['gemini'])
    assert 'gemini' in media.provider_file_api_ids
    assert media.provider_file_api_ids['gemini'] == 'file-fake-123'
    assert media.provider_file_api_status['gemini'].state == (
        FileUploadState.ACTIVE
    )
    assert media.provider_file_api_status['gemini'].provider == 'gemini'

  def test_multi_provider_success(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini', 'claude'])
    media = _make_media()
    dispatch = {'gemini': _fake_upload_fn, 'claude': _fake_upload_fn}
    with patch.dict(file_helpers.UPLOAD_DISPATCH, dispatch):
      mgr.upload(media=media, providers=['gemini', 'claude'])
    assert len(media.provider_file_api_ids) == 2

  def test_partial_failure(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini', 'claude'], parallel=False)
    media = _make_media()
    dispatch = {'gemini': _fake_upload_fn, 'claude': _failing_upload_fn}
    with patch.dict(file_helpers.UPLOAD_DISPATCH, dispatch):
      with pytest.raises(files_module.FileUploadError) as exc_info:
        mgr.upload(media=media, providers=['gemini', 'claude'])
    assert 'claude' in exc_info.value.errors
    assert 'gemini' in media.provider_file_api_ids
    assert media.provider_file_api_status['claude'].state == (
        FileUploadState.FAILED
    )

  def test_upload_with_data_instead_of_path(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = MessageContent(data=b'test-data', media_type='application/pdf')
    dispatch = {'gemini': _fake_upload_fn}
    with patch.dict(file_helpers.UPLOAD_DISPATCH, dispatch):
      mgr.upload(media=media, providers=['gemini'])
    assert 'gemini' in media.provider_file_api_ids


# --- Remove validation ---


class TestRemoveValidation:

  def test_rejects_empty_providers_list(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    with pytest.raises(ValueError, match='at least one provider'):
      mgr.remove(media=media, providers=[])

  def test_rejects_no_uploaded_providers(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    with pytest.raises(ValueError, match='No uploaded providers'):
      mgr.remove(media=media)

  def test_rejects_provider_not_on_media(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    media.provider_file_api_ids = {'claude': 'file-123'}
    media.provider_file_api_status = {'claude': _fake_upload_metadata('claude')}
    with pytest.raises(ValueError, match='No file_id found'):
      mgr.remove(media=media, providers=['gemini'])


# --- Remove execution ---


class TestRemoveExecution:

  def _make_uploaded_media(self, providers):
    media = _make_media()
    media.provider_file_api_ids = {}
    media.provider_file_api_status = {}
    for p in providers:
      media.provider_file_api_ids[p] = f'file-{p}-123'
      media.provider_file_api_status[p] = _fake_upload_metadata(p)
    return media

  def test_remove_single_provider(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = self._make_uploaded_media(['gemini'])
    dispatch = {'gemini': _fake_remove_fn}
    with patch.dict(file_helpers.REMOVE_DISPATCH, dispatch):
      mgr.remove(media=media, providers=['gemini'])
    assert media.provider_file_api_ids == {}
    assert media.provider_file_api_status == {}

  def test_remove_all_defaults_to_all_providers(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini', 'claude'])
    media = self._make_uploaded_media(['gemini', 'claude'])
    dispatch = {'gemini': _fake_remove_fn, 'claude': _fake_remove_fn}
    with patch.dict(file_helpers.REMOVE_DISPATCH, dispatch):
      mgr.remove(media=media)
    assert media.provider_file_api_ids == {}
    assert media.provider_file_api_status == {}

  def test_remove_selective_keeps_other_providers(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini', 'claude'])
    media = self._make_uploaded_media(['gemini', 'claude'])
    dispatch = {'gemini': _fake_remove_fn, 'claude': _fake_remove_fn}
    with patch.dict(file_helpers.REMOVE_DISPATCH, dispatch):
      mgr.remove(media=media, providers=['gemini'])
    assert 'gemini' not in media.provider_file_api_ids
    assert 'claude' in media.provider_file_api_ids
    assert 'gemini' not in media.provider_file_api_status
    assert 'claude' in media.provider_file_api_status

  def test_remove_partial_failure(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini', 'claude'], parallel=False)
    media = self._make_uploaded_media(['gemini', 'claude'])
    dispatch = {'gemini': _fake_remove_fn, 'claude': _failing_remove_fn}
    with patch.dict(file_helpers.REMOVE_DISPATCH, dispatch):
      with pytest.raises(files_module.FileRemoveError) as exc_info:
        mgr.remove(media=media)
    assert 'claude' in exc_info.value.errors
    assert 'gemini' not in media.provider_file_api_ids
    assert 'claude' in media.provider_file_api_ids


# --- List helpers ---


def _fake_list_fn(token_map, limit=100):
  return [
      FileUploadMetadata(
          file_id='file-1', filename='doc.pdf', mime_type='application/pdf',
          state=FileUploadState.ACTIVE
      ),
      FileUploadMetadata(
          file_id='file-2', filename='img.png', mime_type='image/png',
          state=FileUploadState.ACTIVE
      ),
  ]


def _fake_list_single_fn(token_map, limit=100):
  return [
      FileUploadMetadata(
          file_id='file-single', filename='report.pdf',
          mime_type='application/pdf', state=FileUploadState.ACTIVE
      ),
  ]


# --- List validation ---


class TestListValidation:

  def test_rejects_empty_providers_list(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    with pytest.raises(ValueError, match='at least one provider'):
      mgr.list(providers=[])

  def test_rejects_unsupported_provider(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    with pytest.raises(ValueError, match='does not support'):
      mgr.list(providers=['deepseek'])

  def test_rejects_no_api_keys(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, [])
    with pytest.raises(ValueError, match='No providers with API keys'):
      mgr.list()


# --- List execution ---


class TestListExecution:

  def test_list_single_provider(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    dispatch = {'gemini': _fake_list_fn}
    with patch.dict(file_helpers.LIST_DISPATCH, dispatch):
      results = mgr.list(providers=['gemini'])
    assert len(results) == 2
    assert all('gemini' in r.provider_file_api_ids for r in results)
    assert results[0].provider_file_api_ids['gemini'] == 'file-1'
    assert results[1].provider_file_api_ids['gemini'] == 'file-2'

  def test_list_multi_provider(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini', 'claude'])
    dispatch = {
        'gemini': _fake_list_fn,
        'claude': _fake_list_single_fn,
    }
    with patch.dict(file_helpers.LIST_DISPATCH, dispatch):
      results = mgr.list(providers=['gemini', 'claude'])
    assert len(results) == 3
    gemini_results = [r for r in results if 'gemini' in r.provider_file_api_ids]
    claude_results = [r for r in results if 'claude' in r.provider_file_api_ids]
    assert len(gemini_results) == 2
    assert len(claude_results) == 1

  def test_list_defaults_to_all_providers_with_keys(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini', 'claude'])
    dispatch = {
        'gemini': _fake_list_single_fn,
        'claude': _fake_list_single_fn,
    }
    with patch.dict(file_helpers.LIST_DISPATCH, dispatch):
      results = mgr.list()
    providers_in_results = set()
    for r in results:
      providers_in_results.update(r.provider_file_api_ids.keys())
    assert 'gemini' in providers_in_results
    assert 'claude' in providers_in_results

  def test_list_result_has_correct_metadata(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    dispatch = {'gemini': _fake_list_single_fn}
    with patch.dict(file_helpers.LIST_DISPATCH, dispatch):
      results = mgr.list(providers=['gemini'])
    mc = results[0]
    assert mc.media_type == 'application/pdf'
    status = mc.provider_file_api_status['gemini']
    assert status.file_id == 'file-single'
    assert status.provider == 'gemini'
    assert status.state == FileUploadState.ACTIVE


# --- Download helpers ---


def _fake_download_fn(file_id, token_map):
  return b'fake-file-content'


# --- Download validation ---


class TestDownloadValidation:

  def test_rejects_no_uploaded_providers(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    with pytest.raises(ValueError, match='No uploaded providers'):
      mgr.download(media=media)

  def test_rejects_provider_not_on_media(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    media.provider_file_api_ids = {'claude': 'file-123'}
    media.provider_file_api_status = {'claude': _fake_upload_metadata('claude')}
    with pytest.raises(ValueError, match='No file_id found'):
      mgr.download(media=media, provider='gemini')

  def test_rejects_unsupported_provider(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    media.provider_file_api_ids = {'deepseek': 'file-123'}
    media.provider_file_api_status = {
        'deepseek': _fake_upload_metadata('deepseek')
    }
    with pytest.raises(ValueError, match='does not support'):
      mgr.download(media=media, provider='deepseek')


# --- Download execution ---


class TestDownloadExecution:

  def _make_uploaded_media(self, providers):
    media = _make_media()
    media.provider_file_api_ids = {}
    media.provider_file_api_status = {}
    for p in providers:
      media.provider_file_api_ids[p] = f'file-{p}-123'
      media.provider_file_api_status[p] = _fake_upload_metadata(p)
    return media

  def test_download_to_memory(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = self._make_uploaded_media(['gemini'])
    dispatch = {'gemini': _fake_download_fn}
    with patch.dict(file_helpers.DOWNLOAD_DISPATCH, dispatch):
      mgr.download(media=media, provider='gemini')
    assert media.data == b'fake-file-content'
    assert media.path is not None  # original path still set

  def test_download_to_path(self, monkeypatch, tmp_path):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = self._make_uploaded_media(['gemini'])
    out_path = str(tmp_path / 'downloaded.pdf')
    dispatch = {'gemini': _fake_download_fn}
    with patch.dict(file_helpers.DOWNLOAD_DISPATCH, dispatch):
      mgr.download(media=media, provider='gemini', path=out_path)
    assert media.path == out_path
    with open(out_path, 'rb') as f:
      assert f.read() == b'fake-file-content'

  def test_download_uses_priority_when_no_provider(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['claude', 'mistral'])
    media = self._make_uploaded_media(['claude', 'mistral'])
    calls = []

    def tracking_download(file_id, token_map):
      calls.append(file_id)
      return b'data'

    dispatch = {'mistral': tracking_download, 'claude': tracking_download}
    with patch.dict(file_helpers.DOWNLOAD_DISPATCH, dispatch):
      mgr.download(media=media)
    assert len(calls) == 1
    assert calls[0] == 'file-mistral-123'

  def test_download_falls_back_to_any_available(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['claude'])
    media = self._make_uploaded_media(['claude'])
    dispatch = {'claude': _fake_download_fn}
    with patch.dict(file_helpers.DOWNLOAD_DISPATCH, dispatch):
      mgr.download(media=media)
    assert media.data == b'fake-file-content'


# --- ProxDash download integration ---


class TestProxDashDownloadIntegration:

  def _make_proxdash_media(self):
    media = _make_media()
    media.proxdash_file_id = 'pd-file-1'
    media.proxdash_file_status = ProxDashFileStatus(
        file_id='pd-file-1', upload_confirmed=True)
    media.provider_file_api_ids = {'mistral': 'file-mistral-123'}
    media.provider_file_api_status = {
        'mistral': _fake_upload_metadata('mistral')
    }
    return media

  def test_proxdash_used_first_when_no_provider_specified(self, monkeypatch):
    mock_pd = MagicMock(spec=proxdash.ProxDashConnection)
    mock_pd.status = types.ProxDashConnectionStatus.CONNECTED
    mock_pd.download_file.return_value = b'proxdash-bytes'
    mgr = _make_files_manager(
        monkeypatch, ['mistral'], proxdash_connection=mock_pd)
    media = self._make_proxdash_media()
    mgr.download(media=media)
    assert media.data == b'proxdash-bytes'
    mock_pd.download_file.assert_called_once_with('pd-file-1')

  def test_falls_back_to_provider_when_proxdash_fails(self, monkeypatch):
    mock_pd = MagicMock(spec=proxdash.ProxDashConnection)
    mock_pd.status = types.ProxDashConnectionStatus.CONNECTED
    mock_pd.download_file.return_value = None
    mgr = _make_files_manager(
        monkeypatch, ['mistral'], proxdash_connection=mock_pd)
    media = self._make_proxdash_media()
    dispatch = {'mistral': _fake_download_fn}
    with patch.dict(file_helpers.DOWNLOAD_DISPATCH, dispatch):
      mgr.download(media=media)
    assert media.data == b'fake-file-content'

  def test_explicit_provider_skips_proxdash(self, monkeypatch):
    mock_pd = MagicMock(spec=proxdash.ProxDashConnection)
    mock_pd.status = types.ProxDashConnectionStatus.CONNECTED
    mgr = _make_files_manager(
        monkeypatch, ['mistral'], proxdash_connection=mock_pd)
    media = self._make_proxdash_media()
    dispatch = {'mistral': _fake_download_fn}
    with patch.dict(file_helpers.DOWNLOAD_DISPATCH, dispatch):
      mgr.download(media=media, provider='mistral')
    assert media.data == b'fake-file-content'
    mock_pd.download_file.assert_not_called()

  def test_no_proxdash_uses_provider(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['mistral'])
    media = self._make_proxdash_media()
    dispatch = {'mistral': _fake_download_fn}
    with patch.dict(file_helpers.DOWNLOAD_DISPATCH, dispatch):
      mgr.download(media=media)
    assert media.data == b'fake-file-content'


# --- Capability checks ---


class TestIsUploadSupported:

  def test_gemini_supports_all(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media(media_type='video/mp4')
    assert mgr.is_upload_supported(media=media, provider='gemini')

  def test_mistral_rejects_audio(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['mistral'])
    media = _make_media(media_type='audio/mpeg')
    assert not mgr.is_upload_supported(media=media, provider='mistral')

  def test_mistral_accepts_pdf(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['mistral'])
    media = _make_media(media_type='application/pdf')
    assert mgr.is_upload_supported(media=media, provider='mistral')

  def test_unsupported_provider(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media(media_type='application/pdf')
    assert not mgr.is_upload_supported(media=media, provider='deepseek')

  def test_no_media_type(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = MessageContent(
        type='document', path='/tmp/test.pdf',
        provider_file_api_ids={'gemini': 'file-123'}
    )
    assert not mgr.is_upload_supported(media=media, provider='gemini')

  def test_openai_rejects_image_reference(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['openai'])
    media = _make_media(media_type='image/jpeg')
    assert not mgr.is_upload_supported(media=media, provider='openai')

  def test_openai_accepts_pdf_reference(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['openai'])
    media = _make_media(media_type='application/pdf')
    assert mgr.is_upload_supported(media=media, provider='openai')

  def test_claude_rejects_audio_reference(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['claude'])
    media = _make_media(media_type='audio/mpeg')
    assert not mgr.is_upload_supported(media=media, provider='claude')

  def test_claude_accepts_image_reference(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['claude'])
    media = _make_media(media_type='image/jpeg')
    assert mgr.is_upload_supported(media=media, provider='claude')

  def test_claude_rejects_markdown_reference(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['claude'])
    media = _make_media(media_type='text/markdown')
    assert not mgr.is_upload_supported(media=media, provider='claude')

  def test_gemini_accepts_video_reference(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media(media_type='video/mp4')
    assert mgr.is_upload_supported(media=media, provider='gemini')


# --- ProxDash integration ---


def _make_mock_proxdash(upload_file_id='proxdash-file-123'):
  """Create a mock ProxDashConnection that's CONNECTED."""
  mock = MagicMock(spec=proxdash.ProxDashConnection)
  mock.status = types.ProxDashConnectionStatus.CONNECTED

  def _upload_file(media):
    media.proxdash_file_id = upload_file_id
    media.proxdash_file_status = ProxDashFileStatus(
        file_id=upload_file_id, upload_confirmed=True
    )
    return upload_file_id

  mock.upload_file.side_effect = _upload_file
  return mock


class TestProxDashUploadIntegration:

  def test_proxdash_upload_runs_with_provider_uploads(self, monkeypatch):
    mock_pd = _make_mock_proxdash()
    mgr = _make_files_manager(
        monkeypatch, ['gemini'], proxdash_connection=mock_pd
    )
    media = _make_media()
    dispatch = {'gemini': _fake_upload_fn}
    with patch.dict(file_helpers.UPLOAD_DISPATCH, dispatch):
      mgr.upload(media=media, providers=['gemini'])
    assert 'gemini' in media.provider_file_api_ids
    assert media.proxdash_file_id == 'proxdash-file-123'
    mock_pd.upload_file.assert_called_once_with(media)
    mock_pd.update_file.assert_called_once()

  def test_proxdash_failure_does_not_break_provider_uploads(self, monkeypatch):
    mock_pd = _make_mock_proxdash()
    mock_pd.upload_file.side_effect = RuntimeError('proxdash down')
    mgr = _make_files_manager(
        monkeypatch, ['gemini'], proxdash_connection=mock_pd
    )
    media = _make_media()
    dispatch = {'gemini': _fake_upload_fn}
    with patch.dict(file_helpers.UPLOAD_DISPATCH, dispatch):
      mgr.upload(media=media, providers=['gemini'])
    assert 'gemini' in media.provider_file_api_ids
    assert media.proxdash_file_id is None

  def test_no_proxdash_connection_skips_silently(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    dispatch = {'gemini': _fake_upload_fn}
    with patch.dict(file_helpers.UPLOAD_DISPATCH, dispatch):
      mgr.upload(media=media, providers=['gemini'])
    assert 'gemini' in media.provider_file_api_ids
    assert media.proxdash_file_id is None


class TestUploadProxDashOnly:

  def test_empty_providers_with_proxdash_uploads_to_proxdash(
      self, monkeypatch
  ):
    mock_pd = _make_mock_proxdash()
    mgr = _make_files_manager(
        monkeypatch, ['gemini'], proxdash_connection=mock_pd
    )
    media = _make_media()
    mgr.upload(media=media, providers=[])
    assert media.proxdash_file_id == 'proxdash-file-123'
    mock_pd.upload_file.assert_called_once()

  def test_empty_providers_without_proxdash_raises(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = _make_media()
    with pytest.raises(ValueError, match='No providers specified'):
      mgr.upload(media=media, providers=[])


class TestIsDownloadSupported:

  def test_mistral_supported(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['mistral'])
    assert mgr.is_download_supported(provider='mistral')

  def test_gemini_not_supported(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    assert not mgr.is_download_supported(provider='gemini')

  def test_claude_not_supported(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['claude'])
    assert not mgr.is_download_supported(provider='claude')

  def test_openai_not_supported(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['openai'])
    assert not mgr.is_download_supported(provider='openai')


# --- ProxDash list integration ---


def _make_proxdash_mc(proxdash_file_id, provider_file_api_ids=None):
  """Create a MessageContent as returned by proxdash.list_files()."""
  return MessageContent(
      media_type='application/pdf',
      proxdash_file_id=proxdash_file_id,
      proxdash_file_status=ProxDashFileStatus(
          file_id=proxdash_file_id, upload_confirmed=True),
      provider_file_api_ids=provider_file_api_ids,
  )


def _make_mock_proxdash_for_list(message_contents):
  """Create a mock ProxDashConnection that returns MessageContent on list."""
  mock = MagicMock(spec=proxdash.ProxDashConnection)
  mock.status = types.ProxDashConnectionStatus.CONNECTED
  mock.list_files.return_value = message_contents
  return mock


class TestProxDashRemoveIntegration:

  def _make_proxdash_media(self, providers):
    media = _make_media()
    media.proxdash_file_id = 'pd-file-1'
    media.proxdash_file_status = ProxDashFileStatus(
        file_id='pd-file-1', upload_confirmed=True)
    media.provider_file_api_ids = {}
    media.provider_file_api_status = {}
    for p in providers:
      media.provider_file_api_ids[p] = f'file-{p}-123'
      media.provider_file_api_status[p] = _fake_upload_metadata(p)
    return media

  def test_proxdash_deleted_alongside_providers(self, monkeypatch):
    mock_pd = MagicMock(spec=proxdash.ProxDashConnection)
    mock_pd.status = types.ProxDashConnectionStatus.CONNECTED
    mock_pd.delete_file.return_value = True
    mgr = _make_files_manager(
        monkeypatch, ['gemini'], proxdash_connection=mock_pd)
    media = self._make_proxdash_media(['gemini'])
    dispatch = {'gemini': _fake_remove_fn}
    with patch.dict(file_helpers.REMOVE_DISPATCH, dispatch):
      mgr.remove(media=media, providers=['gemini'])
    mock_pd.delete_file.assert_called_once_with('pd-file-1')
    assert media.proxdash_file_id is None
    assert media.proxdash_file_status is None
    assert media.provider_file_api_ids == {}

  def test_proxdash_failure_does_not_break_provider_remove(self, monkeypatch):
    mock_pd = MagicMock(spec=proxdash.ProxDashConnection)
    mock_pd.status = types.ProxDashConnectionStatus.CONNECTED
    mock_pd.delete_file.side_effect = RuntimeError('proxdash down')
    mgr = _make_files_manager(
        monkeypatch, ['gemini'], proxdash_connection=mock_pd)
    media = self._make_proxdash_media(['gemini'])
    dispatch = {'gemini': _fake_remove_fn}
    with patch.dict(file_helpers.REMOVE_DISPATCH, dispatch):
      mgr.remove(media=media, providers=['gemini'])
    assert media.provider_file_api_ids == {}
    assert media.proxdash_file_id is None

  def test_no_proxdash_removes_providers_only(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    media = self._make_proxdash_media(['gemini'])
    dispatch = {'gemini': _fake_remove_fn}
    with patch.dict(file_helpers.REMOVE_DISPATCH, dispatch):
      mgr.remove(media=media, providers=['gemini'])
    assert media.provider_file_api_ids == {}
    assert media.proxdash_file_id == 'pd-file-1'  # untouched


class TestProxDashListIntegration:

  def test_proxdash_results_come_first(self, monkeypatch):
    pd_results = [
        _make_proxdash_mc('pd-1', {'gemini': 'files/g-1'})
    ]
    mock_pd = _make_mock_proxdash_for_list(pd_results)
    mgr = _make_files_manager(
        monkeypatch, ['gemini'], proxdash_connection=mock_pd
    )
    dispatch = {'gemini': _fake_list_fn}
    with patch.dict(file_helpers.LIST_DISPATCH, dispatch):
      results = mgr.list(providers=['gemini'])
    assert results[0].proxdash_file_id == 'pd-1'
    assert results[0].provider_file_api_ids == {'gemini': 'files/g-1'}

  def test_provider_files_covered_by_proxdash_are_skipped(self, monkeypatch):
    # ProxDash covers gemini file-1. Provider returns file-1 and file-2.
    pd_results = [
        _make_proxdash_mc('pd-1', {'gemini': 'file-1'})
    ]
    mock_pd = _make_mock_proxdash_for_list(pd_results)
    mgr = _make_files_manager(
        monkeypatch, ['gemini'], proxdash_connection=mock_pd
    )
    dispatch = {'gemini': _fake_list_fn}
    with patch.dict(file_helpers.LIST_DISPATCH, dispatch):
      results = mgr.list(providers=['gemini'])
    proxdash_results = [r for r in results if r.proxdash_file_id]
    provider_only = [r for r in results if not r.proxdash_file_id]
    assert len(proxdash_results) == 1
    assert len(provider_only) == 1
    assert provider_only[0].provider_file_api_ids['gemini'] == 'file-2'

  def test_proxdash_failure_falls_back_to_provider_only(self, monkeypatch):
    mock_pd = _make_mock_proxdash_for_list([])
    mock_pd.list_files.side_effect = RuntimeError('proxdash down')
    mgr = _make_files_manager(
        monkeypatch, ['gemini'], proxdash_connection=mock_pd
    )
    dispatch = {'gemini': _fake_list_fn}
    with patch.dict(file_helpers.LIST_DISPATCH, dispatch):
      results = mgr.list(providers=['gemini'])
    assert len(results) == 2
    assert all(r.proxdash_file_id is None for r in results)

  def test_no_proxdash_behaves_same_as_before(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini'])
    dispatch = {'gemini': _fake_list_fn}
    with patch.dict(file_helpers.LIST_DISPATCH, dispatch):
      results = mgr.list(providers=['gemini'])
    assert len(results) == 2
    assert all(r.proxdash_file_id is None for r in results)
