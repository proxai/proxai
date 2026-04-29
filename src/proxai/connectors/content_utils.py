"""Shared content conversion utilities for provider connectors."""
from __future__ import annotations

import base64
import io
import os

import pypdf

_TEXT_MIME_TYPES = frozenset({
    'text/plain',
    'text/markdown',
    'text/csv',
})


def _build_header(part_dict: dict) -> str:
  """Build a document header line from filename and MIME type."""
  filename = None
  if 'path' in part_dict:
    filename = os.path.basename(part_dict['path'])
  mime_type = part_dict.get('media_type', '')
  if filename and mime_type:
    return f'[Document: {filename} ({mime_type})]'
  if filename:
    return f'[Document: {filename}]'
  if mime_type:
    return f'[Document: ({mime_type})]'
  return '[Document]'


def read_text_document(part_dict: dict) -> str | None:
  """Read a text-based document content block and return its string content.

  Prepends a header line with filename and MIME type for context.
  Handles data (base64-encoded) and path (local file) fields.
  Returns None if the MIME type is not text-based or no content is available.
  """
  mime_type = part_dict.get('media_type')
  if mime_type not in _TEXT_MIME_TYPES:
    return None
  content = None
  if 'data' in part_dict:
    content = base64.b64decode(part_dict['data']).decode('utf-8')
  elif 'path' in part_dict:
    with open(part_dict['path'], 'r') as f:
      content = f.read()
  if content is None:
    return None
  header = _build_header(part_dict)
  return f'{header}\n{content}'


def read_pdf_document(part_dict: dict) -> str | None:
  """Extract text from a PDF document content block.

  Prepends a header line with filename and MIME type for context.
  Uses pypdf for lightweight text extraction. Works well for
  text-based PDFs; scanned/image-only PDFs will return empty text.
  Handles data (base64-encoded) and path (local file) fields.
  Returns None if the MIME type is not PDF or no content is available.
  """
  if part_dict.get('media_type') != 'application/pdf':
    return None
  if 'data' in part_dict:
    pdf_bytes = base64.b64decode(part_dict['data'])
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
  elif 'path' in part_dict:
    reader = pypdf.PdfReader(part_dict['path'])
  else:
    return None
  pages = [page.extract_text() or '' for page in reader.pages]
  content = '\n'.join(pages).strip()
  if not content:
    return None
  header = _build_header(part_dict)
  return f'{header}\n{content}'
