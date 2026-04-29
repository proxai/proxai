"""Per-provider file helpers for upload, remove, list, and download."""
from __future__ import annotations

import io
import os
import time
import uuid

import anthropic
import google.genai as genai
import google.genai.types as genai_types
import mistralai
import mistralai.models as mistral_models
import openai

import proxai.chat.message_content as message_content
import proxai.types as types


def upload_to_claude(
    file_path: str | None,
    file_data: bytes | None,
    filename: str,
    mime_type: str,
    token_map: types.ProviderTokenValueMap,
) -> message_content.FileUploadMetadata:
  client = anthropic.Anthropic(api_key=token_map['ANTHROPIC_API_KEY'])
  if file_path:
    with open(file_path, 'rb') as f:
      result = client.beta.files.upload(file=f)
  else:
    result = client.beta.files.upload(
        file=(filename, io.BytesIO(file_data)))
  return message_content.FileUploadMetadata(
      file_id=result.id,
      filename=result.filename,
      size_bytes=result.size_bytes,
      mime_type=result.mime_type,
      created_at=str(result.created_at) if result.created_at else None,
      state=message_content.FileUploadState.ACTIVE,
  )


def upload_to_openai(
    file_path: str | None,
    file_data: bytes | None,
    filename: str,
    mime_type: str,
    token_map: types.ProviderTokenValueMap,
) -> message_content.FileUploadMetadata:
  client = openai.OpenAI(api_key=token_map['OPENAI_API_KEY'])
  if file_path:
    with open(file_path, 'rb') as f:
      result = client.files.create(file=f, purpose="user_data")
  else:
    result = client.files.create(
        file=(filename, io.BytesIO(file_data)), purpose="user_data")
  return message_content.FileUploadMetadata(
      file_id=result.id,
      filename=result.filename,
      size_bytes=result.bytes,
      created_at=str(result.created_at) if result.created_at else None,
      expires_at=(str(result.expires_at)
                  if getattr(result, 'expires_at', None) else None),
      state=message_content.FileUploadState.ACTIVE,
  )


def upload_to_gemini(
    file_path: str | None,
    file_data: bytes | None,
    filename: str,
    mime_type: str,
    token_map: types.ProviderTokenValueMap,
    poll_interval: float = 2.0,
    max_poll_seconds: float = 300.0,
) -> message_content.FileUploadMetadata:
  client = genai.Client(api_key=token_map['GEMINI_API_KEY'])
  config = genai_types.UploadFileConfig(
      display_name=filename, mime_type=mime_type)
  if file_path:
    result = client.files.upload(file=file_path, config=config)
  else:
    result = client.files.upload(
        file=io.BytesIO(file_data), config=config)

  elapsed = 0.0
  while (result.state
         and result.state.name == "PROCESSING"
         and elapsed < max_poll_seconds):
    time.sleep(poll_interval)
    elapsed += poll_interval
    result = client.files.get(name=result.name)

  if result.state and result.state.name == "ACTIVE":
    state = message_content.FileUploadState.ACTIVE
  elif result.state and result.state.name == "PROCESSING":
    state = message_content.FileUploadState.PENDING
  else:
    state = message_content.FileUploadState.FAILED

  return message_content.FileUploadMetadata(
      file_id=result.name,
      filename=result.display_name,
      size_bytes=result.size_bytes,
      mime_type=result.mime_type,
      uri=result.uri,
      state=state,
      expires_at=(str(result.expiration_time)
                  if result.expiration_time else None),
      sha256_hash=result.sha256_hash,
  )


def upload_to_mistral(
    file_path: str | None,
    file_data: bytes | None,
    filename: str,
    mime_type: str,
    token_map: types.ProviderTokenValueMap,
) -> message_content.FileUploadMetadata:
  client = mistralai.Mistral(api_key=token_map['MISTRAL_API_KEY'])
  if file_path:
    with open(file_path, 'rb') as f:
      content = f.read()
  else:
    content = file_data
  result = client.files.upload(
      file=mistral_models.File(
          file_name=filename,
          content=content,
      ),
      purpose='ocr',
  )
  return message_content.FileUploadMetadata(
      file_id=result.id,
      filename=result.filename,
      size_bytes=result.size_bytes,
      mime_type=(result.mimetype
                if getattr(result, 'mimetype', None) else mime_type),
      state=message_content.FileUploadState.ACTIVE,
  )


UPLOAD_DISPATCH = {
    'claude': upload_to_claude,
    'openai': upload_to_openai,
    'gemini': upload_to_gemini,
    'mistral': upload_to_mistral,
}


def remove_from_claude(
    file_id: str,
    token_map: types.ProviderTokenValueMap,
):
  client = anthropic.Anthropic(api_key=token_map['ANTHROPIC_API_KEY'])
  client.beta.files.delete(file_id=file_id)


def remove_from_openai(
    file_id: str,
    token_map: types.ProviderTokenValueMap,
):
  client = openai.OpenAI(api_key=token_map['OPENAI_API_KEY'])
  client.files.delete(file_id=file_id)


def remove_from_gemini(
    file_id: str,
    token_map: types.ProviderTokenValueMap,
):
  client = genai.Client(api_key=token_map['GEMINI_API_KEY'])
  client.files.delete(name=file_id)


def remove_from_mistral(
    file_id: str,
    token_map: types.ProviderTokenValueMap,
):
  client = mistralai.Mistral(api_key=token_map['MISTRAL_API_KEY'])
  client.files.delete(file_id=file_id)


REMOVE_DISPATCH = {
    'claude': remove_from_claude,
    'openai': remove_from_openai,
    'gemini': remove_from_gemini,
    'mistral': remove_from_mistral,
}


def list_from_claude(
    token_map: types.ProviderTokenValueMap,
    limit: int = 100,
) -> list[message_content.FileUploadMetadata]:
  client = anthropic.Anthropic(api_key=token_map['ANTHROPIC_API_KEY'])
  page = client.beta.files.list(limit=limit)
  results = []
  for f in page.data:
    results.append(message_content.FileUploadMetadata(
        file_id=f.id,
        filename=f.filename,
        size_bytes=f.size_bytes,
        mime_type=f.mime_type,
        created_at=str(f.created_at) if f.created_at else None,
        state=message_content.FileUploadState.ACTIVE,
    ))
  return results


def list_from_openai(
    token_map: types.ProviderTokenValueMap,
    limit: int = 100,
) -> list[message_content.FileUploadMetadata]:
  client = openai.OpenAI(api_key=token_map['OPENAI_API_KEY'])
  page = client.files.list(limit=limit, purpose='user_data')
  results = []
  for f in page.data:
    results.append(message_content.FileUploadMetadata(
        file_id=f.id,
        filename=f.filename,
        size_bytes=f.bytes,
        created_at=str(f.created_at) if f.created_at else None,
        expires_at=(str(f.expires_at)
                    if getattr(f, 'expires_at', None) else None),
        state=message_content.FileUploadState.ACTIVE,
    ))
  return results


def list_from_gemini(
    token_map: types.ProviderTokenValueMap,
    limit: int = 100,
) -> list[message_content.FileUploadMetadata]:
  client = genai.Client(api_key=token_map['GEMINI_API_KEY'])
  pager = client.files.list(
      config=genai_types.ListFilesConfig(page_size=limit))
  results = []
  for f in pager:
    if len(results) >= limit:
      break
    state = message_content.FileUploadState.ACTIVE
    if f.state and f.state.name == 'PROCESSING':
      state = message_content.FileUploadState.PENDING
    elif f.state and f.state.name != 'ACTIVE':
      state = message_content.FileUploadState.FAILED
    results.append(message_content.FileUploadMetadata(
        file_id=f.name,
        filename=f.display_name,
        size_bytes=f.size_bytes,
        mime_type=f.mime_type,
        uri=f.uri,
        state=state,
        expires_at=(str(f.expiration_time)
                    if f.expiration_time else None),
        sha256_hash=f.sha256_hash,
    ))
  return results


def list_from_mistral(
    token_map: types.ProviderTokenValueMap,
    limit: int = 100,
) -> list[message_content.FileUploadMetadata]:
  client = mistralai.Mistral(api_key=token_map['MISTRAL_API_KEY'])
  response = client.files.list(page_size=limit, page=0, purpose='ocr')
  results = []
  for f in response.data:
    results.append(message_content.FileUploadMetadata(
        file_id=f.id,
        filename=f.filename,
        size_bytes=f.size_bytes,
        mime_type=(f.mimetype
                   if getattr(f, 'mimetype', None) else None),
        state=message_content.FileUploadState.ACTIVE,
    ))
  return results


LIST_DISPATCH = {
    'claude': list_from_claude,
    'openai': list_from_openai,
    'gemini': list_from_gemini,
    'mistral': list_from_mistral,
}


def download_from_mistral(
    file_id: str,
    token_map: types.ProviderTokenValueMap,
) -> bytes:
  client = mistralai.Mistral(api_key=token_map['MISTRAL_API_KEY'])
  response = client.files.download(file_id=file_id)
  response.read()
  return response.content


DOWNLOAD_DISPATCH = {
    'mistral': download_from_mistral,
}


# --- File API upload support (what the File API accepts) ---
# See docs/development/multimodal_large_file_analysis.md §6

_ALL_MEDIA_TYPES = frozenset({
    'application/pdf',
    'application/vnd.openxmlformats-officedocument'
    '.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument'
    '.spreadsheetml.sheet',
    'text/csv',
    'text/plain',
    'text/markdown',
    'image/png',
    'image/jpeg',
    'image/gif',
    'image/webp',
    'image/heic',
    'image/heif',
    'audio/mpeg',
    'audio/wav',
    'audio/flac',
    'audio/aac',
    'audio/ogg',
    'audio/aiff',
    'video/mp4',
    'video/webm',
    'video/quicktime',
    'video/x-msvideo',
    'video/mpeg',
    'video/x-matroska',
})

_MISTRAL_UPLOAD_SUPPORTED = frozenset({
    'application/pdf',
    'application/vnd.openxmlformats-officedocument'
    '.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument'
    '.spreadsheetml.sheet',
    'text/csv',
    'text/plain',
    'text/markdown',
    'image/png',
    'image/jpeg',
    'image/gif',
    'image/webp',
    'image/heic',
    'image/heif',
})

UPLOAD_SUPPORTED_MEDIA_TYPES = {
    'gemini': _ALL_MEDIA_TYPES,
    'claude': _ALL_MEDIA_TYPES,
    'openai': _ALL_MEDIA_TYPES,
    'mistral': _MISTRAL_UPLOAD_SUPPORTED,
}

# --- File_id/URI reference support in generate endpoints ---
# What the generate endpoint accepts as file_id reference.
# This is stricter than upload support — a provider may accept
# a file upload but reject the file_id in chat messages.
# See docs/development/multimodal_large_file_analysis.md §8

REFERENCE_SUPPORTED_MEDIA_TYPES = {
    'gemini': frozenset({
        'application/pdf',
        'application/vnd.openxmlformats-officedocument'
        '.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument'
        '.spreadsheetml.sheet',
        'text/csv',
        'text/plain',
        'text/markdown',
        'image/png',
        'image/jpeg',
        'image/gif',
        'image/webp',
        'image/heic',
        'image/heif',
        'audio/mpeg',
        'audio/wav',
        'audio/flac',
        'audio/aac',
        'audio/ogg',
        'audio/aiff',
        'video/mp4',
        'video/webm',
        'video/quicktime',
        'video/x-msvideo',
        'video/mpeg',
        'video/x-matroska',
    }),
    'claude': frozenset({
        'application/pdf',
        'text/plain',
        'image/png',
        'image/jpeg',
        'image/gif',
        'image/webp',
        'image/heic',
        'image/heif',
    }),
    'openai': frozenset({
        'application/pdf',
        'application/vnd.openxmlformats-officedocument'
        '.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument'
        '.spreadsheetml.sheet',
        'text/csv',
        'text/plain',
        'text/markdown',
    }),
    'mistral': frozenset({
        'application/pdf',
        'text/plain',
        'text/markdown',
        'text/csv',
        'image/png',
        'image/jpeg',
        'image/gif',
        'image/webp',
        'image/heic',
        'image/heif',
    }),
}

DOWNLOAD_SUPPORTED_PROVIDERS = frozenset({'mistral'})


# --- Mock dispatches for run_type=TEST ---

_MOCK_PROVIDERS = ['gemini', 'claude', 'openai', 'mistral']


def mock_upload(
    file_path, file_data, filename, mime_type, token_map
):
  return message_content.FileUploadMetadata(
      file_id=f'mock-file-{uuid.uuid4().hex[:8]}',
      filename=filename,
      mime_type=mime_type,
      state=message_content.FileUploadState.ACTIVE,
  )


def mock_remove(file_id, token_map):
  pass


def mock_list(token_map, limit=100):
  return []


def mock_download(file_id, token_map):
  return b'mock-file-content'


MOCK_UPLOAD_DISPATCH = {p: mock_upload for p in _MOCK_PROVIDERS}
MOCK_REMOVE_DISPATCH = {p: mock_remove for p in _MOCK_PROVIDERS}
MOCK_LIST_DISPATCH = {p: mock_list for p in _MOCK_PROVIDERS}
MOCK_DOWNLOAD_DISPATCH = {p: mock_download for p in _MOCK_PROVIDERS}
