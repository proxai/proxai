import os
import datetime
import json
from typing import Any, Dict, Optional
import proxai.types as types

CACHE_DIR = 'query_cache'
LIGHT_CACHE_RECORDS_PATH = 'light_cache_records.json'


class BaseQueryCache:
  _cache_options: types.CacheOptions

  def __init__(self, cache_options: types.CacheOptions):
    self._cache_options = cache_options

  @property
  def _cache_dir(self) -> str:
    if not self._cache_options.path:
      return None
    return os.path.join(self._cache_options.path, CACHE_DIR)

  @staticmethod
  def _get_query_record_hash(query_record: types.QueryRecord) -> str:
    _PRIME_1 = 1000000007
    _PRIME_2 = 1000000009
    signature_str = ''
    if query_record.call_type is not None:
      signature_str += query_record.call_type + chr(255)
    if query_record.provider is not None:
      signature_str += query_record.provider + chr(255)
    if query_record.provider_model is not None:
      signature_str += query_record.provider_model + chr(255)
    if query_record.max_tokens is not None:
      signature_str += str(query_record.max_tokens) + chr(255)
    if query_record.prompt is not None:
      signature_str += query_record.prompt + chr(255)
    hash_val = 0
    for char in signature_str:
      hash_val = (hash_val * _PRIME_1 + ord(char)) % _PRIME_2
    return str(hash_val)

  def _encode_query_record(
      self, query_record: types.QueryRecord) -> Dict[str, Any]:
    record = {}
    if query_record.call_type != None:
      record['call_type'] = query_record.call_type.value
    if query_record.provider != None:
      record['provider'] = query_record.provider.value
    if query_record.provider_model != None:
      record['provider_model'] = query_record.provider_model.value
    if query_record.max_tokens != None:
      record['max_tokens'] = str(query_record.max_tokens)
    if query_record.prompt != None:
      record['prompt'] = query_record.prompt
    if query_record.hash_value != None:
      record['hash_value'] = query_record.hash_value
    else:
      record['hash_value'] = self._get_query_record_hash(query_record)
    return record

  def _decode_query_record(
      self, record: Dict[str, Any]) -> types.QueryRecord:
    query_record = types.QueryRecord()
    if 'call_type' in record:
      query_record.call_type = types.CallType[record['call_type']]
    if 'provider' in record:
      query_record.provider = types.Provider[record['provider']]
    if 'provider_model' in record:
      query_record.provider_model = types.ProviderModel[record['provider_model']]
    query_record.max_tokens = int(record.get('max_tokens', 0))
    query_record.prompt = record.get('prompt', None)
    query_record.hash_value = record.get('hash_value', None)
    return query_record

  def _encode_query_response_record(
      self, query_response_record: types.QueryResponseRecord
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

  def _decode_query_response_record(
      self, record: Dict[str, Any]) -> types.QueryResponseRecord:
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

  def _encode_cache_record(
      self, cache_record: types.CacheRecord) -> Dict[str, Any]:
    record = {}
    if cache_record.query_record != None:
      record['query_record'] = self._encode_query_record(
          cache_record.query_record)
    if cache_record.query_responses != None:
      record['query_responses'] = []
      for query_response_record in cache_record.query_responses:
        record['query_responses'].append(
            self._encode_query_response_record(query_response_record))
    if cache_record.shard_id != None:
      record['shard_id'] = cache_record.shard_id
    if cache_record.last_access_time != None:
      record['last_access_time'] = cache_record.last_access_time.isoformat()
    return record

  def _decode_cache_record(
      self, record: Dict[str, Any]) -> types.CacheRecord:
    cache_record = types.CacheRecord()
    if 'query_record' in record:
      cache_record.query_record = self._decode_query_record(
          record['query_record'])
    if 'query_responses' in record:
      cache_record.query_responses = []
      for query_response_record in record['query_responses']:
        cache_record.query_responses.append(
            self._decode_query_response_record(query_response_record))
    cache_record.shard_id = int(record.get('shard_id', 0))
    if 'last_access_time' in record:
      cache_record.last_access_time = datetime.datetime.fromisoformat(
          record['last_access_time'])
    return cache_record

  def _encode_light_cache_record(
      self, light_cache_record: types.LightCacheRecord) -> Dict[str, Any]:
    record = {}
    if light_cache_record.query_record_hash != None:
      record['query_record_hash'] = light_cache_record.query_record_hash
    if light_cache_record.shard_id != None:
      record['shard_id'] = str(light_cache_record.shard_id)
    if light_cache_record.last_access_time != None:
      record['last_access_time'] = (
          light_cache_record.last_access_time.isoformat())
    return record

  def _decode_light_cache_record(
      self, record: Dict[str, Any]) -> types.LightCacheRecord:
    light_cache_record = types.LightCacheRecord()
    light_cache_record.query_record_hash = record.get('query_record_hash', None)
    light_cache_record.shard_id = int(record.get('shard_id', 0))
    if 'last_access_time' in record:
      light_cache_record.last_access_time = datetime.datetime.fromisoformat(
          record['last_access_time'])
    return light_cache_record

  def _to_light_cache_record(self, cache_record: types.CacheRecord):
    return types.LightCacheRecord(
        query_record_hash=cache_record.query_record.hash_value,
        shard_id=cache_record.shard_id,
        last_access_time=cache_record.last_access_time,
        call_count=cache_record.call_count)


class ShardNameProvider:
  _path: str
  _shard_count: int
  _shard_names: Dict[int, str]
  _backlog_shard_name: str

  def __init__(self, path: str, shard_count: int):
    self._shard_names = {}
    self._path = path
    self._shard_count = shard_count

  @property
  def shard_names(self):
    if self._shard_names:
      return self._shard_names
    self._shard_names = {}
    for i in range(self._shard_count):
      name = f'shard_{i:010000}-of-{self._shard_count:010000}.jsonl'
      self._shard_names[i] = name
    return self._shard_names

  @property
  def backlog_shard_name(self):
    if self._backlog_shard_name:
      return self._backlog_shard_name
    self._backlog_shard_name = os.path.join(
        self._path, f'shard_backlog_{self._shard_count:010000}.jsonl')
    return self._backlog_shard_name


class QueryCache(BaseQueryCache):
  _loaded_cache_records: Dict[str, types.CacheRecord]
  _light_cache_records: Dict[str, types.LightCacheRecord]
  _shard_count: int
  _response_per_file: int
  _total_response: int
  _shard_name_provider: ShardNameProvider

  def __init__(
      self,
      cache_options: types.CacheOptions,
      shard_count: int = 800,
      response_per_file: int = 100,
      total_response: int = 20000):
    super().__init__(cache_options)
    self._loaded_cache_records = {}
    self._light_cache_records = {}
    self._shard_count = shard_count
    self._response_per_file = response_per_file
    self._total_response = total_response
    self._shard_name_provider = ShardNameProvider(
        path=self._cache_dir,
        shard_count=shard_count)

    if self._cache_dir:
      self._load_light_cache_records()

  @property
  def _light_cache_records_path(self) -> str:
    if not self._cache_dir:
      return None
    return os.path.join(self._cache_dir, LIGHT_CACHE_RECORDS_PATH)

  def _load_light_cache_records(self):
    if not os.path.exists(self._light_cache_records_path):
      return
    with open(self._light_cache_records_path, 'r') as f:
      self._light_cache_records: Dict[str, Any] = json.load(f)
    for query_record_hash, record in self._light_cache_records.items():
      light_cache_record = self._decode_light_cache_record(record)
      if query_record_hash != light_cache_record.query_record_hash:
        continue
      self._light_cache_records[query_record_hash] = light_cache_record

  def _save_light_cache_records(self):
    data = {}
    for query_record_hash, light_cache_record in (
        self._light_cache_records.items()):
      data[query_record_hash] = self._encode_light_cache_record(
          light_cache_record)
    with open(self._light_cache_records_path, 'w') as f:
      json.dump(data, f)

  def _load_shard(self, shard_id: str):
    shard_name = self._shard_name_provider.shard_names[shard_id]
    if not os.path.exists(shard_name):
      return
    with open(shard_name, 'r') as f:
      for line in f:
        record = json.loads(line)
        cache_record = self._decode_cache_record(record)
        self._loaded_cache_records[
            cache_record.query_record.hash_value] = cache_record

  def look(
      self,
      query_record: types.QueryRecord,
      update: bool = True
  ) -> Optional[types.QueryResponseRecord]:
    query_record_hash = self._get_query_record_hash(query_record)
    if query_record_hash not in self._light_cache_records:
      return None

    if query_record_hash not in self._loaded_cache_records:
      light_cache_record = self._light_cache_records[query_record_hash]
      self._load_shard(light_cache_record.shard_id)

    if query_record_hash not in self._loaded_cache_records:
      return None

    cache_record = self._loaded_cache_records[query_record_hash]
    if update:
      cache_record.last_access_time = datetime.datetime.now()
      cache_record.call_count += 1
      self._loaded_cache_records[query_record_hash] = cache_record
      self._light_cache_records[query_record_hash] = (
          self._to_light_cache_record(cache_record))
      raise ValueError('Not implemented')
    return cache_record.query_responses[
        cache_record.call_count % len(cache_record.query_responses)]

  def cache(
      self,
      query_record: types.QueryRecord,
      response_record: types.QueryResponseRecord):
    current_time = datetime.datetime.now()
    query_record_hash = self._get_query_record_hash(query_record)
    if (query_record_hash in self._light_cache_records
        and query_record_hash not in self._loaded_cache_records):
      light_cache_record = self._light_cache_records[query_record_hash]
      self._load_shard(light_cache_record.shard_id)

    if query_record_hash in self._loaded_cache_records:
      cache_record = self._loaded_cache_records[query_record_hash]
      cache_record.query_responses.append(response_record)
      cache_record.last_access_time = current_time
      cache_record.call_count += 1
      self._loaded_cache_records[query_record_hash] = cache_record
      self._light_cache_records[query_record_hash] = (
          self._to_light_cache_record(cache_record))
      raise ValueError('Not implemented')
      return

    cache_record = types.CacheRecord(
        query_record=query_record,
        query_responses=[response_record],
        shard_id=None,
        last_access_time=current_time,
        call_count=0)
    self._loaded_cache_records[query_record_hash] = cache_record
    self._light_cache_records[query_record_hash] = (
        self._to_light_cache_record(cache_record))
    raise ValueError('Not implemented')
    return

