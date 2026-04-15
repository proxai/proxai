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


def _hash_chat(hasher: hashlib._Hash, chat: chat_session.Chat):
  """Hash a Chat into the running hasher.

  Hashes system_prompt (if set), then each message's role and content. For
  MessageContent blocks whose `path` is set, the file's mtime_ns and size
  are folded in so in-place edits invalidate the cache (see MessageContent
  docstring for semantics and limits).
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
      _add_field(
          hasher, content_key,
          json.dumps(content_item.to_dict(), sort_keys=True)
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


def _hash_response_format(
    hasher: hashlib._Hash, response_format: types.ResponseFormat
):
  if response_format.type is not None:
    _add_field(hasher, 'response_format.type', response_format.type.value)
  # Resolve pydantic metadata: prefer live class, fallback to stored metadata
  pydantic_class_name = None
  pydantic_class_json_schema = None
  if response_format.pydantic_class is not None:
    pydantic_class_name = response_format.pydantic_class.__name__
    pydantic_class_json_schema = (
        response_format.pydantic_class.model_json_schema()
    )
  elif response_format.pydantic_class_name is not None:
    pydantic_class_name = response_format.pydantic_class_name
    pydantic_class_json_schema = response_format.pydantic_class_json_schema
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
    _hash_chat(hasher, query_record.chat)
  if query_record.provider_model is not None:
    pm = query_record.provider_model
    _add_field(hasher, 'provider', pm.provider)
    _add_field(hasher, 'model', pm.model)
    _add_field(
        hasher, 'provider_model_identifier', pm.provider_model_identifier
    )
  if query_record.parameters is not None:
    _hash_parameters(hasher, query_record.parameters)
  if query_record.response_format is not None:
    _hash_response_format(hasher, query_record.response_format)
  if query_record.tools is not None:
    _hash_tools(hasher, query_record.tools)
  if query_record.connection_options is not None:
    _hash_connection_options(hasher, query_record.connection_options)
  return hasher.hexdigest()[:_HASH_LENGTH]
