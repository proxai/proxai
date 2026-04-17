"""Shared content conversion utilities for provider connectors."""

import base64

_TEXT_MIME_TYPES = frozenset({
    'text/plain',
    'text/markdown',
    'text/csv',
})


def read_text_document(part_dict: dict) -> str | None:
  """Read a text-based document content block and return its string content.

  Handles data (base64-encoded), path (local file), and source (URL) fields.
  Returns None if the MIME type is not text-based or no content is available.
  """
  mime_type = part_dict.get('media_type')
  if mime_type not in _TEXT_MIME_TYPES:
    return None
  if 'data' in part_dict:
    return base64.b64decode(part_dict['data']).decode('utf-8')
  if 'path' in part_dict:
    with open(part_dict['path'], 'r') as f:
      return f.read()
  return None
