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
    query_record: types.QueryRecord,
    response_record: types.QueryResponseRecord,
    from_cache: bool = False):
  if not logging_options or not logging_options.path:
    return
  file_path = os.path.join(logging_options.path, QUERY_LOGGING_FILE_NAME)
  result = {}
  result['from_cache'] = from_cache
  result['query_record'] = type_serializer.encode_query_record(query_record)
  result['response_record'] = type_serializer.encode_query_response_record(
      response_record)
  with open(file_path, 'a') as f:
    f.write(json.dumps(result) + '\n')
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
