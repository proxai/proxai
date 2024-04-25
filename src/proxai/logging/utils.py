import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional
import proxai.types as types
import proxai.serializers.type_serializer as type_serializer
import json

QUERY_LOGGING_FILE_NAME = 'provider_queries.log'
ERROR_LOGGING_FILE_NAME = 'errors.log'
WARNING_LOGGING_FILE_NAME = 'warnings.log'
INFO_LOGGING_FILE_NAME = 'info.log'
MERGED_LOGGING_FILE_NAME = 'merged.log'


def log_query_record(
    logging_options: types.LoggingOptions,
    logging_record: types.LoggingRecord):
  if not logging_options or not logging_options.path:
    return
  file_path = os.path.join(logging_options.path, QUERY_LOGGING_FILE_NAME)
  with open(file_path, 'a') as f:
    f.write(json.dumps(
        type_serializer.encode_logging_record(logging_record)) + '\n')
  f.close()


def log_message(
    logging_options: types.LoggingOptions,
    message: str,
    type: types.LoggingType,
    query_record: Optional[types.QueryRecord] = None):
  if not logging_options or not logging_options.path:
    return
  result = {}
  result['logging_type'] = type.value.upper()
  result['message'] = message
  result['timestamp'] = datetime.now().isoformat()
  if query_record:
    result['query_record'] = type_serializer.encode_query_record(query_record)
  def _write_log(file_name):
    file_path = os.path.join(logging_options.path, file_name)
    with open(file_path, 'a') as f:
      f.write(json.dumps(result) + '\n')
    f.close()
  if type == types.LoggingType.ERROR:
    _write_log(ERROR_LOGGING_FILE_NAME)
  elif type == types.LoggingType.WARNING:
    _write_log(WARNING_LOGGING_FILE_NAME)
  else:
    _write_log(INFO_LOGGING_FILE_NAME)
  _write_log(MERGED_LOGGING_FILE_NAME)
