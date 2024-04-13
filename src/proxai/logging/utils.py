import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional
import proxai.types as types
import json

QUERY_LOGGING_FILE_NAME = 'provider_queries.log'


def log_generate_text(
    logging_options: types.LoggingOptions,
    provider: str,
    provider_model: str,
    start_time: datetime,
    end_time: datetime,
    params: Dict = None,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    error: Optional[str] = None):
  file_path = os.path.join(logging_options.path, QUERY_LOGGING_FILE_NAME)
  result = {
    'provider': provider,
    'provider_model': provider_model,
    'params': params,
  }
  if logging_options.time:
    result['start_time'] = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    result['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")
  if logging_options.prompt and prompt is not None:
    result['prompt'] = prompt
  if logging_options.response and response is not None:
    result['response'] = response
  if logging_options.error and error is not None:
    result['error'] = error
  with open(file_path, 'a') as f:
    f.write(json.dumps(result) + '\n')
  f.close()
