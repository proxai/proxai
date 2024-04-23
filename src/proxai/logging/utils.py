import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional
import proxai.types as types
import proxai.serializers.type_serializer as type_serializer
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
  result['query_record'] = type_serializer.encode_query_record(query_record)
  result['response_record'] = type_serializer.encode_query_response_record(
      response_record)
  with open(file_path, 'a') as f:
    f.write(json.dumps(result) + '\n')
  f.close()
