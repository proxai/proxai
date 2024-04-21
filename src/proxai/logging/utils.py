import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional
import proxai.types as types
import json

QUERY_LOGGING_FILE_NAME = 'provider_queries.log'


def log_query_record(
    logging_options: types.LoggingOptions,
    query_record: types.QueryRecord):
  file_path = os.path.join(logging_options.path, QUERY_LOGGING_FILE_NAME)
  result = {}
  if query_record.call_type:
    result['call_type'] = query_record.call_type
  if query_record.provider:
    result['provider'] = query_record.provider
  if query_record.provider_model:
    result['provider_model'] = query_record.provider_model
  if query_record.max_tokens:
    result['max_tokens'] = query_record.max_tokens
  if query_record.prompt and logging_options.prompt:
    result['prompt'] = query_record.prompt
  if query_record.response and logging_options.response:
    result['response'] = query_record.response
  if query_record.error and logging_options.error:
    result['error'] = query_record.error
  if query_record.start_time and logging_options.time:
    result['start_time'] = query_record.start_time.strftime(
        '%Y-%m-%d %H:%M:%S.%f')
  if query_record.end_time and logging_options.time:
    result['end_time'] = query_record.end_time.strftime(
        '%Y-%m-%d %H:%M:%S.%f')
  if query_record.response_time and logging_options.time:
    result['response_time'] = query_record.response_time.total_seconds()

  with open(file_path, 'a') as f:
    f.write(json.dumps(result) + '\n')
  f.close()
