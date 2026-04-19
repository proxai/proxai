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


def remove_from_claude(
    file_id: str,
    token_map: ProviderTokenValueMap,
):
  import anthropic
  client = anthropic.Anthropic(api_key=token_map['ANTHROPIC_API_KEY'])
  client.beta.files.delete(file_id=file_id)


def remove_from_openai(
    file_id: str,
    token_map: ProviderTokenValueMap,
):
  from openai import OpenAI
  client = OpenAI(api_key=token_map['OPENAI_API_KEY'])
  client.files.delete(file_id=file_id)


def remove_from_gemini(
    file_id: str,
    token_map: ProviderTokenValueMap,
):
  from google import genai
  client = genai.Client(api_key=token_map['GEMINI_API_KEY'])
  client.files.delete(name=file_id)


def remove_from_mistral(
    file_id: str,
    token_map: ProviderTokenValueMap,
):
  from mistralai import Mistral
  client = Mistral(api_key=token_map['MISTRAL_API_KEY'])
  client.files.delete(file_id=file_id)


REMOVE_DISPATCH = {
    'claude': remove_from_claude,
    'openai': remove_from_openai,
    'gemini': remove_from_gemini,
    'mistral': remove_from_mistral,
}


def list_from_claude(
    token_map: ProviderTokenValueMap,
    limit: int = 100,
) -> list[FileUploadMetadata]:
  import anthropic
  client = anthropic.Anthropic(api_key=token_map['ANTHROPIC_API_KEY'])
  page = client.beta.files.list(limit=limit)
  results = []
  for f in page.data:
    results.append(FileUploadMetadata(
        file_id=f.id,
        filename=f.filename,
        size_bytes=f.size_bytes,
        mime_type=f.mime_type,
        created_at=str(f.created_at) if f.created_at else None,
        state=FileUploadState.ACTIVE,
    ))
  return results


def list_from_openai(
    token_map: ProviderTokenValueMap,
    limit: int = 100,
) -> list[FileUploadMetadata]:
  from openai import OpenAI
  client = OpenAI(api_key=token_map['OPENAI_API_KEY'])
  page = client.files.list(limit=limit, purpose='user_data')
  results = []
  for f in page.data:
    results.append(FileUploadMetadata(
        file_id=f.id,
        filename=f.filename,
        size_bytes=f.bytes,
        created_at=str(f.created_at) if f.created_at else None,
        expires_at=(str(f.expires_at)
                    if getattr(f, 'expires_at', None) else None),
        state=FileUploadState.ACTIVE,
    ))
  return results


def list_from_gemini(
    token_map: ProviderTokenValueMap,
    limit: int = 100,
) -> list[FileUploadMetadata]:
  from google import genai
  from google.genai import types as genai_types
  client = genai.Client(api_key=token_map['GEMINI_API_KEY'])
  pager = client.files.list(
      config=genai_types.ListFilesConfig(page_size=limit))
  results = []
  for f in pager:
    if len(results) >= limit:
      break
    state = FileUploadState.ACTIVE
    if f.state and f.state.name == 'PROCESSING':
      state = FileUploadState.PENDING
    elif f.state and f.state.name != 'ACTIVE':
      state = FileUploadState.FAILED
    results.append(FileUploadMetadata(
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
    token_map: ProviderTokenValueMap,
    limit: int = 100,
) -> list[FileUploadMetadata]:
  from mistralai import Mistral
  client = Mistral(api_key=token_map['MISTRAL_API_KEY'])
  response = client.files.list(page_size=limit, page=0, purpose='ocr')
  results = []
  for f in response.data:
    results.append(FileUploadMetadata(
        file_id=f.id,
        filename=f.filename,
        size_bytes=f.size_bytes,
        mime_type=(f.mimetype
                   if getattr(f, 'mimetype', None) else None),
        state=FileUploadState.ACTIVE,
    ))
  return results


LIST_DISPATCH = {
    'claude': list_from_claude,
    'openai': list_from_openai,
    'gemini': list_from_gemini,
    'mistral': list_from_mistral,
}
