"""Tests for FilesManager upload and remove orchestration logic."""

from unittest.mock import patch

import pytest

import proxai.connections.api_key_manager as api_key_manager
import proxai.connectors.file_helpers as file_helpers
import proxai.connectors.files as files_module
import proxai.connectors.model_configs as model_configs
import proxai.types as types
from proxai.chat.message_content import (
    FileUploadMetadata,
    FileUploadState,
    MessageContent,
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
      init_from_params=api_key_manager.ApiKeyManagerParams())


def _make_files_manager(monkeypatch, providers, parallel=True):
  akm = _make_api_key_manager(monkeypatch, providers)
  return files_module.FilesManager(
      init_from_params=files_module.FilesManagerParams(
          api_key_manager=akm,
          provider_call_options=types.ProviderCallOptions(
              allow_parallel_file_operations=parallel,
          ),
      ))


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
        media_type='application/pdf')
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
        FileUploadState.ACTIVE)
    assert media.provider_file_api_status['gemini'].provider == 'gemini'

  def test_multi_provider_success(self, monkeypatch):
    mgr = _make_files_manager(monkeypatch, ['gemini', 'claude'])
    media = _make_media()
    dispatch = {'gemini': _fake_upload_fn, 'claude': _fake_upload_fn}
    with patch.dict(file_helpers.UPLOAD_DISPATCH, dispatch):
      mgr.upload(media=media, providers=['gemini', 'claude'])
    assert len(media.provider_file_api_ids) == 2

  def test_partial_failure(self, monkeypatch):
    mgr = _make_files_manager(
        monkeypatch, ['gemini', 'claude'], parallel=False)
    media = _make_media()
    dispatch = {'gemini': _fake_upload_fn, 'claude': _failing_upload_fn}
    with patch.dict(file_helpers.UPLOAD_DISPATCH, dispatch):
      with pytest.raises(files_module.FileUploadError) as exc_info:
        mgr.upload(media=media, providers=['gemini', 'claude'])
    assert 'claude' in exc_info.value.errors
    assert 'gemini' in media.provider_file_api_ids
    assert media.provider_file_api_status['claude'].state == (
        FileUploadState.FAILED)

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
    media.provider_file_api_status = {
        'claude': _fake_upload_metadata('claude')}
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
    mgr = _make_files_manager(
        monkeypatch, ['gemini', 'claude'], parallel=False)
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
          file_id='file-1', filename='doc.pdf',
          mime_type='application/pdf',
          state=FileUploadState.ACTIVE),
      FileUploadMetadata(
          file_id='file-2', filename='img.png',
          mime_type='image/png',
          state=FileUploadState.ACTIVE),
  ]


def _fake_list_single_fn(token_map, limit=100):
  return [
      FileUploadMetadata(
          file_id='file-single', filename='report.pdf',
          mime_type='application/pdf',
          state=FileUploadState.ACTIVE),
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
    gemini_results = [
        r for r in results if 'gemini' in r.provider_file_api_ids]
    claude_results = [
        r for r in results if 'claude' in r.provider_file_api_ids]
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
