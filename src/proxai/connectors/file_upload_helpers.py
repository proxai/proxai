"""Per-provider file upload helpers."""

import io
import os
import time

from proxai.chat.message_content import FileUploadMetadata, FileUploadState
from proxai.types import ProviderTokenValueMap


def upload_to_claude(
    file_path: str | None,
    file_data: bytes | None,
    filename: str,
    mime_type: str,
    token_map: ProviderTokenValueMap,
) -> FileUploadMetadata:
  import anthropic
  client = anthropic.Anthropic(api_key=token_map['ANTHROPIC_API_KEY'])
  if file_path:
    with open(file_path, 'rb') as f:
      result = client.beta.files.upload(file=f)
  else:
    result = client.beta.files.upload(
        file=(filename, io.BytesIO(file_data)))
  return FileUploadMetadata(
      file_id=result.id,
      filename=result.filename,
      size_bytes=result.size_bytes,
      mime_type=result.mime_type,
      created_at=str(result.created_at) if result.created_at else None,
      state=FileUploadState.ACTIVE,
  )


def upload_to_openai(
    file_path: str | None,
    file_data: bytes | None,
    filename: str,
    mime_type: str,
    token_map: ProviderTokenValueMap,
) -> FileUploadMetadata:
  from openai import OpenAI
  client = OpenAI(api_key=token_map['OPENAI_API_KEY'])
  if file_path:
    with open(file_path, 'rb') as f:
      result = client.files.create(file=f, purpose="user_data")
  else:
    result = client.files.create(
        file=(filename, io.BytesIO(file_data)), purpose="user_data")
  return FileUploadMetadata(
      file_id=result.id,
      filename=result.filename,
      size_bytes=result.bytes,
      created_at=str(result.created_at) if result.created_at else None,
      expires_at=(str(result.expires_at)
                  if getattr(result, 'expires_at', None) else None),
      state=FileUploadState.ACTIVE,
  )


def upload_to_gemini(
    file_path: str | None,
    file_data: bytes | None,
    filename: str,
    mime_type: str,
    token_map: ProviderTokenValueMap,
    poll_interval: float = 2.0,
    max_poll_seconds: float = 300.0,
) -> FileUploadMetadata:
  from google import genai
  from google.genai import types as genai_types
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
    state = FileUploadState.ACTIVE
  elif result.state and result.state.name == "PROCESSING":
    state = FileUploadState.PENDING
  else:
    state = FileUploadState.FAILED

  return FileUploadMetadata(
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
    token_map: ProviderTokenValueMap,
) -> FileUploadMetadata:
  from mistralai import Mistral
  from mistralai.models import File as MistralFile
  client = Mistral(api_key=token_map['MISTRAL_API_KEY'])
  if file_path:
    with open(file_path, 'rb') as f:
      content = f.read()
  else:
    content = file_data
  result = client.files.upload(
      file=MistralFile(
          file_name=filename,
          content=content,
      ),
      purpose='ocr',
  )
  return FileUploadMetadata(
      file_id=result.id,
      filename=result.filename,
      size_bytes=result.size_bytes,
      mime_type=(result.mimetype
                if getattr(result, 'mimetype', None) else mime_type),
      state=FileUploadState.ACTIVE,
  )


UPLOAD_DISPATCH = {
    'claude': upload_to_claude,
    'openai': upload_to_openai,
    'gemini': upload_to_gemini,
    'mistral': upload_to_mistral,
}
