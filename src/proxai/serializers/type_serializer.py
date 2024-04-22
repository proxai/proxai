import datetime
from typing import Any, Dict
import proxai.types as types


def encode_query_record(
    query_record: types.QueryRecord) -> Dict[str, Any]:
  record = {}
  if query_record.call_type != None:
    record['call_type'] = query_record.call_type.value
  if query_record.provider != None:
    if isinstance(query_record.provider, types.Provider):
      record['provider'] = query_record.provider.value
    elif isinstance(query_record.provider, str):
      record['provider'] = query_record.provider
    else:
      raise ValueError(
          'Invalid provider type.\n'
          f'{query_record.provider=}\n'
          f'{type(query_record.provider)=}')
  if query_record.provider_model != None:
    if isinstance(query_record.provider_model, types.ProviderModel):
      record['provider_model'] = query_record.provider_model.value
    elif isinstance(query_record.provider_model, str):
      record['provider_model'] = query_record.provider_model
    else:
      raise ValueError(
          'Invalid provider_model type.\n'
          f'{query_record.provider_model=}\n'
          f'{type(query_record.provider_model)=}')
  if query_record.max_tokens != None:
    record['max_tokens'] = str(query_record.max_tokens)
  if query_record.prompt != None:
    record['prompt'] = query_record.prompt
  if query_record.hash_value != None:
    record['hash_value'] = query_record.hash_value
  return record


def decode_query_record(
    record: Dict[str, Any]) -> types.QueryRecord:
  if 'provider_model' in record and 'provider' not in record:
    raise ValueError('provider_model without provider')
  query_record = types.QueryRecord()
  if 'call_type' in record:
    query_record.call_type = types.CallType(record['call_type'])
  if 'provider' in record:
    query_record.provider = types.Provider(record['provider'])
  if 'provider_model' in record:
    query_record.provider_model = types.PROVIDER_MODEL_MAP[
        query_record.provider](record['provider_model'])
  if 'max_tokens' in record:
    query_record.max_tokens = int(record['max_tokens'])
  query_record.prompt = record.get('prompt', None)
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
