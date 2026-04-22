import copy
import json
import os
from datetime import datetime
from pprint import pprint

import proxai.serializers.type_serializer as type_serializer
import proxai.types as types

ERROR_LOGGING_FILE_NAME = 'errors.log'
WARNING_LOGGING_FILE_NAME = 'warnings.log'
INFO_LOGGING_FILE_NAME = 'info.log'
MERGED_LOGGING_FILE_NAME = 'merged.log'
PROXDASH_LOGGING_FILE_NAME = 'proxdash.log'
_SENSITIVE_CONTENT_HIDDEN_STRING = '<sensitive content hidden>'


def _hide_sensitive_content_query_record(
    query_record: types.QueryRecord
) -> types.QueryRecord:
  """Replace sensitive fields in a query record with placeholder text."""
  query_record = copy.deepcopy(query_record)
  if query_record.system_prompt:
    query_record.system_prompt = _SENSITIVE_CONTENT_HIDDEN_STRING
  if query_record.prompt:
    query_record.prompt = _SENSITIVE_CONTENT_HIDDEN_STRING
  if query_record.chat is not None:
    if query_record.chat.system_prompt:
      query_record.chat.system_prompt = _SENSITIVE_CONTENT_HIDDEN_STRING
    query_record.chat.messages = [
        types.Message(
            role=types.MessageRoleType.ASSISTANT,
            content=_SENSITIVE_CONTENT_HIDDEN_STRING,
        )
    ]
  return query_record


def _write_log(
    logging_options: types.LoggingOptions, file_name: str, data: dict
):
  """Append a JSON record to a log file."""
  file_path = os.path.join(logging_options.logging_path, file_name)
  with open(file_path, 'a') as f:
    f.write(json.dumps(data) + '\n')
  f.close()


def log_message(
    logging_options: types.LoggingOptions, message: str,
    type: types.LoggingType, query_record: types.QueryRecord | None = None
):
  """Write a message to the appropriate log file based on type."""
  if not logging_options:
    return
  result = {}
  result['logging_type'] = type.value.upper()
  result['message'] = message
  result['timestamp'] = datetime.now().isoformat()
  if query_record:
    if logging_options.hide_sensitive_content:
      query_record = _hide_sensitive_content_query_record(query_record)
    result['query_record'] = type_serializer.encode_query_record(query_record)
  if logging_options.stdout:
    pprint(result)
  if not logging_options.logging_path:
    return
  if type == types.LoggingType.ERROR:
    _write_log(
        logging_options=logging_options, file_name=ERROR_LOGGING_FILE_NAME,
        data=result
    )
  elif type == types.LoggingType.WARNING:
    _write_log(
        logging_options=logging_options, file_name=WARNING_LOGGING_FILE_NAME,
        data=result
    )
  else:
    _write_log(
        logging_options=logging_options, file_name=INFO_LOGGING_FILE_NAME,
        data=result
    )
  _write_log(
      logging_options=logging_options, file_name=MERGED_LOGGING_FILE_NAME,
      data=result
  )


def log_proxdash_message(
    logging_options: types.LoggingOptions,
    proxdash_options: types.ProxDashOptions, message: str,
    type: types.LoggingType, query_record: types.QueryRecord | None = None
):
  """Write a ProxDash-related message to the log files."""
  result = {}
  result['logging_type'] = type.value.upper()
  result['message'] = message
  result['timestamp'] = datetime.now().isoformat()
  if query_record:
    if logging_options.hide_sensitive_content:
      query_record = _hide_sensitive_content_query_record(query_record)
    result['query_record'] = type_serializer.encode_query_record(query_record)
  if proxdash_options.stdout:
    pprint(result)
  if not logging_options.logging_path:
    return
  if type == types.LoggingType.ERROR:
    _write_log(
        logging_options=logging_options, file_name=ERROR_LOGGING_FILE_NAME,
        data=result
    )
  elif type == types.LoggingType.WARNING:
    _write_log(
        logging_options=logging_options, file_name=WARNING_LOGGING_FILE_NAME,
        data=result
    )
  _write_log(
      logging_options=logging_options, file_name=PROXDASH_LOGGING_FILE_NAME,
      data=result
  )
  _write_log(
      logging_options=logging_options, file_name=MERGED_LOGGING_FILE_NAME,
      data=result
  )
