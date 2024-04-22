import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional
import proxai.types as types
import json

QUERY_LOGGING_FILE_NAME = 'provider_queries.log'


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
  if response_record.response and logging_options.response:
    result['response'] = response_record.response
  if response_record.error and logging_options.error:
    result['error'] = response_record.error
  if response_record.start_time and logging_options.time:
    result['start_time'] = response_record.start_time.strftime(
        '%Y-%m-%d %H:%M:%S.%f')
  if response_record.end_time and logging_options.time:
    result['end_time'] = response_record.end_time.strftime(
        '%Y-%m-%d %H:%M:%S.%f')
  if response_record.response_time and logging_options.time:
    result['response_time'] = response_record.response_time.total_seconds()

  with open(file_path, 'a') as f:
    f.write(json.dumps(result) + '\n')
  f.close()
