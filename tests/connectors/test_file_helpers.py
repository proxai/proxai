"""Tests for SDK-client lifecycle inside file_helpers.

Each upload_to_*, remove_from_*, list_from_*, and download_from_* function
constructs a fresh provider SDK client and must close it before returning,
including when the inner call raises. These tests assert the close()
contract by patching the SDK constructor with a MagicMock and verifying
.close() is invoked on the success and failure paths.
"""

from unittest import mock

import pytest

import proxai.connectors.file_helpers as file_helpers

# Mistral-specific helpers monkeypatch `file_helpers.mistralai.Mistral`, which
# only exists when the optional `mistralai` SDK is installed. The SDK was
# moved to an optional poetry group in proxai 0.3.2 (quarantine on PyPI);
# install it via `poetry install --with mistral-test` to run these locally.
mistralai_required = pytest.mark.skipif(
    not file_helpers._MISTRAL_AVAILABLE,
    reason="mistralai SDK not installed (optional 'mistral-test' poetry group)",
)

# -----------------------------------------------------------------------------
# Helper: build a mock SDK client whose internal call returns a result with
# all attributes the helpers expect (we keep MagicMock auto-spec generous).
# -----------------------------------------------------------------------------


def _mock_token_map():
  return {
      'ANTHROPIC_API_KEY': 'k',
      'OPENAI_API_KEY': 'k',
      'GEMINI_API_KEY': 'k',
      'MISTRAL_API_KEY': 'k',
  }


# -----------------------------------------------------------------------------
# upload_to_*
# -----------------------------------------------------------------------------


class TestUploadCloses:

  def test_upload_to_claude_closes_on_success(self, monkeypatch):
    sdk_client = mock.MagicMock()
    sdk_client.beta.files.upload.return_value = mock.MagicMock(
        id='id1', filename='f.txt', size_bytes=10, mime_type='text/plain',
        created_at=1234,
    )
    monkeypatch.setattr(
        file_helpers.anthropic, 'Anthropic',
        mock.MagicMock(return_value=sdk_client))
    file_helpers.upload_to_claude(
        file_path=None, file_data=b'hi', filename='f.txt',
        mime_type='text/plain', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()

  def test_upload_to_claude_closes_on_exception(self, monkeypatch):
    sdk_client = mock.MagicMock()
    sdk_client.beta.files.upload.side_effect = RuntimeError('boom')
    monkeypatch.setattr(
        file_helpers.anthropic, 'Anthropic',
        mock.MagicMock(return_value=sdk_client))
    with pytest.raises(RuntimeError):
      file_helpers.upload_to_claude(
          file_path=None, file_data=b'hi', filename='f.txt',
          mime_type='text/plain', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()

  def test_upload_to_openai_closes_on_success(self, monkeypatch):
    sdk_client = mock.MagicMock()
    sdk_client.files.create.return_value = mock.MagicMock(
        id='id1', filename='f.txt', bytes=10, created_at=1234,
        expires_at=None,
    )
    monkeypatch.setattr(
        file_helpers.openai, 'OpenAI',
        mock.MagicMock(return_value=sdk_client))
    file_helpers.upload_to_openai(
        file_path=None, file_data=b'hi', filename='f.txt',
        mime_type='text/plain', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()

  @mistralai_required
  def test_upload_to_mistral_closes_on_exception(self, monkeypatch):
    sdk_client = mock.MagicMock()
    sdk_client.files.upload.side_effect = RuntimeError('boom')
    monkeypatch.setattr(
        file_helpers.mistralai, 'Mistral',
        mock.MagicMock(return_value=sdk_client))
    with pytest.raises(RuntimeError):
      file_helpers.upload_to_mistral(
          file_path=None, file_data=b'hi', filename='f.txt',
          mime_type='text/plain', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()


# -----------------------------------------------------------------------------
# remove_from_*
# -----------------------------------------------------------------------------


class TestRemoveCloses:

  def test_remove_from_claude_closes_on_success(self, monkeypatch):
    sdk_client = mock.MagicMock()
    monkeypatch.setattr(
        file_helpers.anthropic, 'Anthropic',
        mock.MagicMock(return_value=sdk_client))
    file_helpers.remove_from_claude(
        file_id='abc', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()

  def test_remove_from_openai_closes_on_exception(self, monkeypatch):
    sdk_client = mock.MagicMock()
    sdk_client.files.delete.side_effect = RuntimeError('boom')
    monkeypatch.setattr(
        file_helpers.openai, 'OpenAI',
        mock.MagicMock(return_value=sdk_client))
    with pytest.raises(RuntimeError):
      file_helpers.remove_from_openai(
          file_id='abc', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()

  def test_remove_from_gemini_closes_on_success(self, monkeypatch):
    sdk_client = mock.MagicMock()
    monkeypatch.setattr(
        file_helpers.genai, 'Client',
        mock.MagicMock(return_value=sdk_client))
    file_helpers.remove_from_gemini(
        file_id='files/abc', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()

  @mistralai_required
  def test_remove_from_mistral_closes_on_success(self, monkeypatch):
    sdk_client = mock.MagicMock()
    monkeypatch.setattr(
        file_helpers.mistralai, 'Mistral',
        mock.MagicMock(return_value=sdk_client))
    file_helpers.remove_from_mistral(
        file_id='abc', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()


# -----------------------------------------------------------------------------
# list_from_*
# -----------------------------------------------------------------------------


class TestListCloses:

  def test_list_from_claude_closes_on_success(self, monkeypatch):
    sdk_client = mock.MagicMock()
    sdk_client.beta.files.list.return_value = mock.MagicMock(data=[])
    monkeypatch.setattr(
        file_helpers.anthropic, 'Anthropic',
        mock.MagicMock(return_value=sdk_client))
    file_helpers.list_from_claude(token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()

  def test_list_from_openai_closes_on_exception(self, monkeypatch):
    sdk_client = mock.MagicMock()
    sdk_client.files.list.side_effect = RuntimeError('boom')
    monkeypatch.setattr(
        file_helpers.openai, 'OpenAI',
        mock.MagicMock(return_value=sdk_client))
    with pytest.raises(RuntimeError):
      file_helpers.list_from_openai(token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()

  @mistralai_required
  def test_list_from_mistral_closes_on_success(self, monkeypatch):
    sdk_client = mock.MagicMock()
    sdk_client.files.list.return_value = mock.MagicMock(data=[])
    monkeypatch.setattr(
        file_helpers.mistralai, 'Mistral',
        mock.MagicMock(return_value=sdk_client))
    file_helpers.list_from_mistral(token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()


# -----------------------------------------------------------------------------
# download_from_*
# -----------------------------------------------------------------------------


class TestDownloadCloses:

  @mistralai_required
  def test_download_from_mistral_closes_on_success(self, monkeypatch):
    response = mock.MagicMock()
    response.content = b'data'
    sdk_client = mock.MagicMock()
    sdk_client.files.download.return_value = response
    monkeypatch.setattr(
        file_helpers.mistralai, 'Mistral',
        mock.MagicMock(return_value=sdk_client))
    file_helpers.download_from_mistral(
        file_id='abc', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()

  @mistralai_required
  def test_download_from_mistral_closes_on_exception(self, monkeypatch):
    sdk_client = mock.MagicMock()
    sdk_client.files.download.side_effect = RuntimeError('boom')
    monkeypatch.setattr(
        file_helpers.mistralai, 'Mistral',
        mock.MagicMock(return_value=sdk_client))
    with pytest.raises(RuntimeError):
      file_helpers.download_from_mistral(
          file_id='abc', token_map=_mock_token_map())
    sdk_client.close.assert_called_once_with()


# -----------------------------------------------------------------------------
# _close_silent helper
# -----------------------------------------------------------------------------


class TestCloseSilentHelper:

  def test_calls_close_when_present(self):
    client = mock.MagicMock()
    file_helpers._close_silent(client)
    client.close.assert_called_once_with()

  def test_swallows_exception(self):
    client = mock.MagicMock()
    client.close.side_effect = RuntimeError('boom')
    file_helpers._close_silent(client)  # must not propagate.

  def test_no_op_when_close_missing(self):
    file_helpers._close_silent(object())  # no .close attr; must not raise.
