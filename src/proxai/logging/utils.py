import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional
import proxai.types as types
import proxai.serializers.type_serializer as type_serializer
import json
from pprint import pprint

QUERY_LOGGING_FILE_NAME = 'provider_queries.log'
ERROR_LOGGING_FILE_NAME = 'errors.log'
WARNING_LOGGING_FILE_NAME = 'warnings.log'
INFO_LOGGING_FILE_NAME = 'info.log'
MERGED_LOGGING_FILE_NAME = 'merged.log'
PROXDASH_LOGGING_FILE_NAME = 'proxdash.log'


def _write_log(
    logging_options: types.LoggingOptions,
    file_name: str,
    data: Dict):
    file_path = os.path.join(logging_options.logging_path, file_name)
    with open(file_path, 'a') as f:
      f.write(json.dumps(data) + '\n')
    f.close()


def log_logging_record(
    logging_options: types.LoggingOptions,
    logging_record: types.LoggingRecord):
  if not logging_options:
    return
  result = type_serializer.encode_logging_record(logging_record)
  if logging_options.stdout:
    pprint(result)
  if not logging_options.logging_path:
    return
  _write_log(
      logging_options=logging_options,
      file_name=QUERY_LOGGING_FILE_NAME,
      data=result)


def log_message(
    logging_options: types.LoggingOptions,
    message: str,
    type: types.LoggingType,
    query_record: Optional[types.QueryRecord] = None):
  if not logging_options:
    return
  result = {}
  result['logging_type'] = type.value.upper()
  result['message'] = message
  result['timestamp'] = datetime.now().isoformat()
  if query_record:
    result['query_record'] = type_serializer.encode_query_record(query_record)
  if logging_options.stdout:
    pprint(result)
  if not logging_options.logging_path:
    return
  if type == types.LoggingType.ERROR:
    _write_log(
        logging_options=logging_options,
        file_name=ERROR_LOGGING_FILE_NAME,
        data=result)
  elif type == types.LoggingType.WARNING:
    _write_log(
        logging_options=logging_options,
        file_name=WARNING_LOGGING_FILE_NAME,
        data=result)
  else:
    _write_log(
        logging_options=logging_options,
        file_name=INFO_LOGGING_FILE_NAME,
        data=result)
  _write_log(
      logging_options=logging_options,
      file_name=MERGED_LOGGING_FILE_NAME,
      data=result)


def log_proxdash_message(
    logging_options: types.LoggingOptions,
    message: str,
    type: types.LoggingType,
    query_record: Optional[types.QueryRecord] = None):
  if not logging_options:
    return
  result = {}
  result['logging_type'] = type.value.upper()
  result['message'] = message
  result['timestamp'] = datetime.now().isoformat()
  if query_record:
    result['query_record'] = type_serializer.encode_query_record(query_record)
  if logging_options.proxdash_stdout:
    pprint(result)
  if not logging_options.logging_path:
    return
  if type == types.LoggingType.ERROR:
    _write_log(
        logging_options=logging_options,
        file_name=ERROR_LOGGING_FILE_NAME,
        data=result)
  elif type == types.LoggingType.WARNING:
    _write_log(
        logging_options=logging_options,
        file_name=WARNING_LOGGING_FILE_NAME,
        data=result)
  _write_log(
      logging_options=logging_options,
      file_name=PROXDASH_LOGGING_FILE_NAME,
      data=result)
