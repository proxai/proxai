import datetime
from typing import Any, Dict
import proxai.types as types


def encode_model_type(
    model_type: types.ModelType) -> Dict[str, Any]:
  provider, provider_model = model_type
  record = {}
  if isinstance(provider, types.Provider):
    record['provider'] = provider.value
  elif isinstance(provider, str):
    record['provider'] = provider
  else:
    raise ValueError(
        'Invalid provider type.\n'
        f'{provider=}\n'
        f'{type(provider)=}')
  if isinstance(provider_model, types.ProviderModel):
    record['provider_model'] = provider_model.value
  elif isinstance(provider_model, str):
    record['provider_model'] = provider_model
  else:
    raise ValueError(
        'Invalid provider_model type.\n'
        f'{provider_model=}\n'
        f'{type(provider_model)=}')
  return record


def decode_model_type(
    record: Dict[str, Any]) -> types.ModelType:
  if 'provider' not in record:
    raise ValueError(f'Provider not found in record: {record=}')
  if 'provider_model' not in record:
    raise ValueError(f'Provider model not found in record: {record=}')
  provider = types.Provider(record['provider'])
  provider_model = types.PROVIDER_MODEL_MAP[provider](record['provider_model'])
  return (provider, provider_model)


def encode_query_record(
    query_record: types.QueryRecord) -> Dict[str, Any]:
  record = {}
  if query_record.call_type != None:
    record['call_type'] = query_record.call_type.value
  if query_record.model != None:
    record['model'] = encode_model_type(query_record.model)
  if query_record.prompt != None:
    record['prompt'] = query_record.prompt
  if query_record.system != None:
    record['system'] = query_record.system
  if query_record.messages != None:
    record['messages'] = query_record.messages
  if query_record.max_tokens != None:
    record['max_tokens'] = str(query_record.max_tokens)
  if query_record.temperature != None:
    record['temperature'] = str(query_record.temperature)
  if query_record.stop != None:
    record['stop'] = query_record.stop
  if query_record.hash_value != None:
    record['hash_value'] = query_record.hash_value
  return record


def decode_query_record(
    record: Dict[str, Any]) -> types.QueryRecord:
  query_record = types.QueryRecord()
  if 'call_type' in record:
    query_record.call_type = types.CallType(record['call_type'])
  if 'model' in record:
    query_record.model = decode_model_type(record['model'])
  query_record.prompt = record.get('prompt', None)
  query_record.system = record.get('system', None)
  query_record.messages = record.get('messages', None)
  if 'max_tokens' in record:
    query_record.max_tokens = int(record['max_tokens'])
  if 'temperature' in record:
    query_record.temperature = float(record['temperature'])
  query_record.stop = record.get('stop', None)
  query_record.hash_value = record.get('hash_value', None)
  return query_record


def encode_query_response_record(
    query_response_record: types.QueryResponseRecord
) -> Dict[str, Any]:
  record = {}
  if query_response_record.response != None:
    record['response'] = query_response_record.response
  if query_response_record.error != None:
    record['error'] = query_response_record.error
  if query_response_record.start_time != None:
    record['start_time'] = query_response_record.start_time.isoformat()
  if query_response_record.end_time != None:
    record['end_time'] = query_response_record.end_time.isoformat()
  if query_response_record.response_time != None:
    record['response_time'] = (
        query_response_record.response_time.total_seconds())
  return record


def decode_query_response_record(
    record: Dict[str, Any]) -> types.QueryResponseRecord:
  query_response_record = types.QueryResponseRecord()
  query_response_record.response = record.get('response', None)
  query_response_record.error = record.get('error', None)
  if 'start_time' in record:
    query_response_record.start_time = datetime.datetime.fromisoformat(
        record['start_time'])
  if 'end_time' in record:
    query_response_record.end_time = datetime.datetime.fromisoformat(
        record['end_time'])
  if 'response_time' in record:
    query_response_record.response_time = datetime.timedelta(
        seconds=record['response_time'])
  return query_response_record


def encode_cache_record(
    cache_record: types.CacheRecord) -> Dict[str, Any]:
  record = {}
  if cache_record.query_record != None:
    record['query_record'] = encode_query_record(
        cache_record.query_record)
  if cache_record.query_responses != None:
    record['query_responses'] = []
    for query_response_record in cache_record.query_responses:
      record['query_responses'].append(
          encode_query_response_record(query_response_record))
  if cache_record.shard_id != None:
    try:
      record['shard_id'] = int(cache_record.shard_id)
    except ValueError:
      record['shard_id'] = cache_record.shard_id
  if cache_record.last_access_time != None:
    record['last_access_time'] = cache_record.last_access_time.isoformat()
  if cache_record.call_count != None:
    record['call_count'] = cache_record.call_count
  return record


def decode_cache_record(
    record: Dict[str, Any]) -> types.CacheRecord:
  cache_record = types.CacheRecord()
  if 'query_record' in record:
    cache_record.query_record = decode_query_record(
        record['query_record'])
  if 'query_responses' in record:
    cache_record.query_responses = []
    for query_response_record in record['query_responses']:
      cache_record.query_responses.append(
          decode_query_response_record(query_response_record))
  if 'shard_id' in record:
    try:
      cache_record.shard_id = int(record['shard_id'])
    except ValueError:
      cache_record.shard_id = record['shard_id']
  if 'last_access_time' in record:
    cache_record.last_access_time = datetime.datetime.fromisoformat(
        record['last_access_time'])
  if 'call_count' in record:
    cache_record.call_count = int(record['call_count'])
  return cache_record


def encode_light_cache_record(
    light_cache_record: types.LightCacheRecord) -> Dict[str, Any]:
  record = {}
  if light_cache_record.query_record_hash != None:
    record['query_record_hash'] = light_cache_record.query_record_hash
  if light_cache_record.query_response_count != None:
    record['query_response_count'] = light_cache_record.query_response_count
  if light_cache_record.shard_id != None:
    try:
      record['shard_id'] = int(light_cache_record.shard_id)
    except ValueError:
      record['shard_id'] = light_cache_record.shard_id
  if light_cache_record.last_access_time != None:
    record['last_access_time'] = (
        light_cache_record.last_access_time.isoformat())
  if light_cache_record.call_count != None:
    record['call_count'] = light_cache_record.call_count
  return record


def decode_light_cache_record(
    record: Dict[str, Any]) -> types.LightCacheRecord:
  light_cache_record = types.LightCacheRecord()
  light_cache_record.query_record_hash = record.get('query_record_hash', None)
  if 'query_response_count' in record:
    light_cache_record.query_response_count = int(
        record['query_response_count'])
  if 'shard_id' in record:
    try:
      light_cache_record.shard_id = int(record['shard_id'])
    except ValueError:
      light_cache_record.shard_id = record['shard_id']
  if 'last_access_time' in record:
    light_cache_record.last_access_time = datetime.datetime.fromisoformat(
        record['last_access_time'])
  if 'call_count' in record:
    light_cache_record.call_count = int(record['call_count'])
  return light_cache_record


def encode_logging_record(
    logging_record: types.LoggingRecord) -> Dict[str, Any]:
  record = {}
  if logging_record.query_record != None:
    record['query_record'] = encode_query_record(
        logging_record.query_record)
  if logging_record.response_record != None:
    record['response_record'] = encode_query_response_record(
        logging_record.response_record)
  if logging_record.response_source != None:
    record['response_source'] = logging_record.response_source.value
  return record


def decode_logging_record(
    record: Dict[str, Any]) -> types.LoggingRecord:
  logging_record = types.LoggingRecord()
  if 'query_record' in record:
    logging_record.query_record = decode_query_record(
        record['query_record'])
  if 'response_record' in record:
    logging_record.response_record = decode_query_response_record(
        record['response_record'])
  if 'response_source' in record:
    logging_record.response_source = (
        types.ResponseSource(record['response_source']))
  return logging_record


def encode_model_status(
    model_status: types.ModelStatus) -> Dict[str, Any]:
  record = {}
  if model_status.unprocessed_models != None:
    record['unprocessed_models'] = []
    for model_type in model_status.unprocessed_models:
      record['unprocessed_models'].append(encode_model_type(model_type))
  if model_status.working_models != None:
    record['working_models'] = []
    for model_type in model_status.working_models:
      record['working_models'].append(encode_model_type(model_type))
  if model_status.failed_models != None:
    record['failed_models'] = []
    for model_type in model_status.failed_models:
      record['failed_models'].append(encode_model_type(model_type))
  if model_status.filtered_models != None:
    record['filtered_models'] = []
    for model_type in model_status.filtered_models:
      record['filtered_models'].append(encode_model_type(model_type))
  if model_status.provider_queries:
    record['provider_queries'] = []
    for provider_query in model_status.provider_queries:
      record['provider_queries'].append(encode_logging_record(provider_query))
  return record


def decode_model_status(
    record: Dict[str, Any]) -> types.ModelStatus:
  model_status = types.ModelStatus()
  if 'unprocessed_models' in record:
    for model_type_record in record['unprocessed_models']:
      model_status.unprocessed_models.add(decode_model_type(model_type_record))
  if 'working_models' in record:
    for model_type_record in record['working_models']:
      model_status.working_models.add(decode_model_type(model_type_record))
  if 'failed_models' in record:
    for model_type_record in record['failed_models']:
      model_status.failed_models.add(decode_model_type(model_type_record))
  if 'filtered_models' in record:
    for model_type_record in record['filtered_models']:
      model_status.filtered_models.add(decode_model_type(model_type_record))
  if 'provider_queries' in record:
    for provider_query_record in record['provider_queries']:
      model_status.provider_queries.append(
          decode_logging_record(provider_query_record))
  return model_status
