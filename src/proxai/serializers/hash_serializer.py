"""Cache hash generation for query records.

WARNING: The hash logic in this file MUST stay in sync with the
equality check in type_utils.is_query_record_equal(). Both functions
determine cache identity — the hash finds the cache bucket, the
equality check verifies against hash collisions. If one excludes a
field from comparison, the other must also exclude it, otherwise
cache lookups will silently fail (hash matches but equality doesn't).

Currently excluded from cache identity:
- MessageContent.provider_file_api_ids (transport metadata)
- MessageContent.provider_file_api_status (transport metadata)
- MessageContent.filename (informational label)
- ConnectionOptions fields other than endpoint

See: type_utils._normalize_chat_for_comparison()
See: _content_hash_dict() in this file
"""

from __future__ import annotations

import hashlib
import json
import os

import proxai.chat.chat_session as chat_session
import proxai.chat.message_content as message_content
import proxai.types as types

_SEPARATOR_CHAR = chr(255)
_HASH_LENGTH = 16


def _add_field(hasher: hashlib._Hash, name: str, value: str):
  """Feed a named field into the hasher."""
  hasher.update((name + '=' + value + _SEPARATOR_CHAR).encode())


def _content_hash_dict(
    content_item: message_content.MessageContent,
    provider: str | None,
) -> dict:
  """Build a hash-safe dict from a MessageContent.

  For content with local data (path, data, or source), excludes
  provider_file_api_ids and provider_file_api_status since the
  content identity comes from the local data, not the upload
  metadata. For remote-only content (no local data), includes
  only the current provider's file_id as the content identity.

  WARNING: Any field excluded here must also be excluded in
  type_utils._normalize_chat_for_comparison(). These two
  functions define cache identity together — the hash finds the
  bucket, the equality check verifies the match.
  """
  d = content_item.to_dict()
  d.pop('filename', None)
  d.pop('proxdash_file_id', None)
  d.pop('proxdash_file_status', None)
  has_local_content = (
      'path' in d or 'data' in d or 'source' in d
  )
  if has_local_content:
    d.pop('provider_file_api_ids', None)
    d.pop('provider_file_api_status', None)
  else:
    file_ids = d.pop('provider_file_api_ids', None)
    d.pop('provider_file_api_status', None)
    if file_ids and provider and provider in file_ids:
      d['provider_file_id'] = file_ids[provider]
    elif file_ids:
      d['provider_file_ids'] = file_ids
  return d


def _hash_chat(
    hasher: hashlib._Hash,
    chat: chat_session.Chat,
    provider: str | None = None,
):
  """Hash a Chat into the running hasher.

  Hashes system_prompt (if set), then each message's role and content. For
  MessageContent blocks whose `path` is set, the file's mtime_ns and size
  are folded in so in-place edits invalidate the cache (see MessageContent
  docstring for semantics and limits).

  File API metadata (provider_file_api_ids, provider_file_api_status) is
  excluded from the hash for content with local data, since the content
  identity comes from path/data/source. For remote-only content, the
  current provider's file_id is used as the content identity.
  """
  if chat.system_prompt is not None:
    _add_field(hasher, 'chat.system_prompt', chat.system_prompt)
  for i, msg in enumerate(chat.messages):
    prefix = f'msg[{i}].'
    _add_field(hasher, prefix + 'role', msg.role.value)
    if isinstance(msg.content, str):
      _add_field(hasher, prefix + 'content', msg.content)
      continue
    for j, content_item in enumerate(msg.content):
      content_key = f'{prefix}content[{j}]'
      hash_dict = _content_hash_dict(content_item, provider)
      _add_field(
          hasher, content_key,
          json.dumps(hash_dict, sort_keys=True)
      )
      if content_item.path is not None:
        try:
          stat = os.stat(content_item.path)
        except OSError:
          continue
        _add_field(
            hasher, content_key + '.path_stat',
            f'{stat.st_mtime_ns}:{stat.st_size}'
        )


def _hash_parameters(hasher: hashlib._Hash, parameters: types.ParameterType):
  if parameters.temperature is not None:
    _add_field(hasher, 'temperature', str(parameters.temperature))
  if parameters.max_tokens is not None:
    _add_field(hasher, 'max_tokens', str(parameters.max_tokens))
  if parameters.stop is not None:
    _add_field(hasher, 'stop', str(parameters.stop))
  if parameters.n is not None:
    _add_field(hasher, 'n', str(parameters.n))
  if parameters.thinking is not None:
    _add_field(hasher, 'thinking', parameters.thinking.value)


def _hash_output_format(
    hasher: hashlib._Hash, output_format: types.OutputFormat
):
  if output_format.type is not None:
    _add_field(hasher, 'output_format.type', output_format.type.value)
  # Resolve pydantic metadata: prefer live class, fallback to stored metadata
  pydantic_class_name = None
  pydantic_class_json_schema = None
  if output_format.pydantic_class is not None:
    pydantic_class_name = output_format.pydantic_class.__name__
    pydantic_class_json_schema = (
        output_format.pydantic_class.model_json_schema()
    )
  elif output_format.pydantic_class_name is not None:
    pydantic_class_name = output_format.pydantic_class_name
    pydantic_class_json_schema = output_format.pydantic_class_json_schema
  if pydantic_class_name is not None:
    _add_field(hasher, 'pydantic_class_name', pydantic_class_name)
  if pydantic_class_json_schema is not None:
    _add_field(
        hasher, 'pydantic_class_json_schema',
        json.dumps(pydantic_class_json_schema, sort_keys=True)
    )


def _hash_tools(hasher: hashlib._Hash, tools: list[types.Tools]):
  for i, tool in enumerate(tools):
    _add_field(hasher, f'tool[{i}]', tool.value)


def _hash_connection_options(
    hasher: hashlib._Hash, connection_options: types.ConnectionOptions
):
  if connection_options.endpoint is not None:
    _add_field(hasher, 'endpoint', connection_options.endpoint)


def get_query_record_hash(query_record: types.QueryRecord) -> str:
  """Generate a unique hash for a query record for cache lookup."""
  hasher = hashlib.sha256()
  if query_record.prompt is not None:
    _add_field(hasher, 'prompt', query_record.prompt)
  if query_record.system_prompt is not None:
    _add_field(hasher, 'system_prompt', query_record.system_prompt)
  if query_record.chat is not None:
    provider = None
    if query_record.provider_model is not None:
      provider = query_record.provider_model.provider
    _hash_chat(hasher, query_record.chat, provider=provider)
  if query_record.provider_model is not None:
    pm = query_record.provider_model
    _add_field(hasher, 'provider', pm.provider)
    _add_field(hasher, 'model', pm.model)
    _add_field(
        hasher, 'provider_model_identifier', pm.provider_model_identifier
    )
  if query_record.parameters is not None:
    _hash_parameters(hasher, query_record.parameters)
  if query_record.output_format is not None:
    _hash_output_format(hasher, query_record.output_format)
  if query_record.tools is not None:
    _hash_tools(hasher, query_record.tools)
  if query_record.connection_options is not None:
    _hash_connection_options(hasher, query_record.connection_options)
  return hasher.hexdigest()[:_HASH_LENGTH]
