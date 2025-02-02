import os
import copy
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


def _hide_sensitive_content_query_record(
    query_record: types.QueryRecord) -> types.QueryRecord:
  query_record = copy.deepcopy(query_record)
  if query_record.system:
    query_record.system = '<sensitive content hidden>'
  if query_record.prompt:
    query_record.prompt = '<sensitive content hidden>'
  if query_record.messages:
    query_record.messages = [
      {
        'role': 'assistant',
        'content': '<sensitive content hidden>'
      }
    ]
  return query_record


def _hide_sensitive_content_query_response_record(
    query_response_record: types.QueryResponseRecord) -> types.QueryResponseRecord:
  query_response_record = copy.deepcopy(query_response_record)
  if query_response_record.response:
    query_response_record.response = '<sensitive content hidden>'
  return query_response_record


def _hide_sensitive_content_logging_record(
    logging_record: types.LoggingRecord) -> types.LoggingRecord:
  logging_record = copy.deepcopy(logging_record)
  if logging_record.query_record:
    logging_record.query_record = _hide_sensitive_content_query_record(
        logging_record.query_record)
  if logging_record.response_record:
    logging_record.response_record = (
        _hide_sensitive_content_query_response_record(
            logging_record.response_record))
  return logging_record


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
  if logging_options.hide_sensitive_content:
    logging_record = _hide_sensitive_content_logging_record(logging_record)
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
    if logging_options.hide_sensitive_content:
      query_record = _hide_sensitive_content_query_record(query_record)
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
    proxdash_options: types.ProxDashOptions,
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
    if logging_options.hide_sensitive_content:
      query_record = _hide_sensitive_content_query_record(query_record)
    result['query_record'] = type_serializer.encode_query_record(query_record)
  if proxdash_options.stdout:
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
