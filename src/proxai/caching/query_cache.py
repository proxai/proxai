import copy
import os
import collections
import datetime
import json
import heapq
from typing import Any, Dict, Optional, Union, List, Tuple, Set
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

  @staticmethod
  def _encode_query_record(
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

  @staticmethod
  def _decode_query_record(
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

  @staticmethod
  def _encode_query_response_record(
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

  @staticmethod
  def _decode_query_response_record(
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

  @staticmethod
  def _encode_cache_record(
      cache_record: types.CacheRecord) -> Dict[str, Any]:
    record = {}
    if cache_record.query_record != None:
      record['query_record'] = BaseQueryCache._encode_query_record(
          cache_record.query_record)
    if cache_record.query_responses != None:
      record['query_responses'] = []
      for query_response_record in cache_record.query_responses:
        record['query_responses'].append(
            BaseQueryCache._encode_query_response_record(query_response_record))
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

  @staticmethod
  def _decode_cache_record(
      record: Dict[str, Any]) -> types.CacheRecord:
    cache_record = types.CacheRecord()
    if 'query_record' in record:
      cache_record.query_record = BaseQueryCache._decode_query_record(
          record['query_record'])
    if 'query_responses' in record:
      cache_record.query_responses = []
      for query_response_record in record['query_responses']:
        cache_record.query_responses.append(
            BaseQueryCache._decode_query_response_record(query_response_record))
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

  @staticmethod
  def _encode_light_cache_record(
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

  @staticmethod
  def _decode_light_cache_record(
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

  @staticmethod
  def _to_light_cache_record(cache_record: types.CacheRecord):
    return types.LightCacheRecord(
        query_record_hash=cache_record.query_record.hash_value,
        query_response_count=len(cache_record.query_responses),
        shard_id=cache_record.shard_id,
        last_access_time=cache_record.last_access_time,
        call_count=cache_record.call_count)

  @staticmethod
  def _get_cache_size(
      cache_record: Union[types.CacheRecord, types.LightCacheRecord]) -> int:
    if isinstance(cache_record, types.LightCacheRecord):
      return cache_record.query_response_count + 1
    return len(cache_record.query_responses) + 1

  @staticmethod
  def _get_hash_value(
      cache_record: Union[
          str,
          types.CacheRecord,
          types.LightCacheRecord,
          types.QueryRecord]
  ) -> str:
    if isinstance(cache_record, str):
      return cache_record
    if isinstance(cache_record, types.CacheRecord):
      if cache_record.query_record.hash_value:
        return cache_record.query_record.hash_value
      else:
        cache_record.query_record.hash_value = (
            BaseQueryCache._get_query_record_hash(
                cache_record.query_record))
        return cache_record.query_record.hash_value
    if isinstance(cache_record, types.LightCacheRecord):
      if cache_record.query_record_hash:
        return cache_record.query_record_hash
      else:
        raise ValueError('LightCacheRecord doen\'t have query_record_hash')
    if isinstance(cache_record, types.QueryRecord):
      query_record = cache_record
      if query_record.hash_value:
        return query_record.hash_value
      else:
        query_record.hash_value = BaseQueryCache._get_query_record_hash(
            query_record)
        return query_record.hash_value


class HeapManager:
  _heap: List[Tuple[int, Union[int, str]]]
  _active_values: Dict[Union[int, str], int]
  _record_size_map: Dict[Union[int, str], int]
  _with_size: bool
  _total_size: int

  def __init__(self, with_size: bool = False):
    self._heap = []
    self._active_values = {}
    self._record_size_map = {}
    self._with_size = with_size
    if with_size:
      self._total_size = 0

  def push(
      self,
      key: Union[int, str],
      value: int,
      record_size: int = None):
    if not self._with_size and record_size:
      raise ValueError('Cannot push record size without with_size=True')
    if self._with_size and not record_size:
      raise ValueError('Cannot push without record size with with_size=False')
    if key in self._active_values and self._with_size:
      self._total_size -= self._record_size_map[key]
    self._active_values[key] = value
    if self._with_size:
      self._record_size_map[key] = record_size
      self._total_size += record_size
    heapq.heappush(self._heap, (value, key))

  def pop(self) -> Optional[Tuple[int, Union[int, str]]]:
    while self._heap:
      value, key = heapq.heappop(self._heap)
      if key in self._active_values and self._active_values[key] == value:
        del self._active_values[key]
        if self._with_size:
          self._total_size -= self._record_size_map[key]
          del self._record_size_map[key]
        return value, key
    return None, None

  def top(self) -> Optional[Tuple[int, Union[int, str]]]:
    while self._heap:
      value, key = self._heap[0]
      if key in self._active_values and self._active_values[key] == value:
        return value, key
      heapq.heappop(self._heap)
    return None, None

  def __len__(self):
    if self._with_size:
      return self._total_size
    return len(self._active_values)


class ShardManager:
  _path: str
  _shard_count: int
  _response_per_file: int
  _shard_paths: Dict[Union[int, str], str]
  _backlog_shard_path: str
  _light_cache_records_path: str
  _shard_active_count: Dict[Union[int, str], int]
  _shard_heap: HeapManager
  _loaded_cache_records: Dict[str, types.CacheRecord]
  _light_cache_records: Dict[str, types.LightCacheRecord]
  _map_shard_to_cache: Dict[Union[str, int], Set[str]]

  def __init__(
      self,
      path: str,
      shard_count: int,
      response_per_file: int):
    if shard_count < 1 or shard_count > 99999:
      raise ValueError('shard_count should be between 1 and 99999')
    if response_per_file < 1:
      raise ValueError('response_per_file should be greater than 0')

    self._path = path
    self._shard_count = shard_count
    self._response_per_file = response_per_file
    self._shard_paths = {}
    self._backlog_shard_path = None
    self._light_cache_records_path = None
    self._load_light_cache_records()

  @property
  def backlog_shard_path(self):
    if self._backlog_shard_path:
      return self._backlog_shard_path
    self._backlog_shard_path = os.path.join(
        self._path, f'shard_backlog_{self._shard_count:05}.jsonl')
    return self._backlog_shard_path

  @property
  def shard_paths(self):
    if self._shard_paths:
      return self._shard_paths
    self._shard_paths = {}
    for i in range(self._shard_count):
      self._shard_paths[i] = os.path.join(
          self._path, f'shard_{i:05}-of-{self._shard_count:05}.jsonl')
    self._shard_paths['backlog'] = self.backlog_shard_path
    return self._shard_paths

  @property
  def light_cache_records_path(self) -> str:
    if self._light_cache_records_path:
      return self._light_cache_records_path
    self._light_cache_records_path = os.path.join(
        self._path, f'light_cache_records_{self._shard_count:05}.json')
    return self._light_cache_records_path

  def _check_shard_id(self, shard_id: Union[int, str]):
    if shard_id not in self.shard_paths:
      raise ValueError('Invalid shard_id')

  def _update_cache_record(
      self,
      cache_record: Union[types.CacheRecord, types.LightCacheRecord],
      delete_only: bool = False,
      write_to_file: bool = True):
    hash_value = BaseQueryCache._get_hash_value(cache_record)
    shard_id = cache_record.shard_id
    if isinstance(cache_record, types.CacheRecord):
      light_cache_record = BaseQueryCache._to_light_cache_record(cache_record)
    else:
      light_cache_record = cache_record

    # Clean previous values
    if hash_value in self._light_cache_records:
      old_shard_id = self._light_cache_records[hash_value].shard_id
      self._map_shard_to_cache[old_shard_id].remove(hash_value)
      self._shard_active_count[old_shard_id] -= BaseQueryCache._get_cache_size(
          self._light_cache_records[hash_value])
      if old_shard_id != 'backlog':
        self._shard_heap.push(
            key=old_shard_id, value=self._shard_active_count[old_shard_id])
      if hash_value in self._loaded_cache_records:
        del self._loaded_cache_records[hash_value]
      del self._light_cache_records[hash_value]
      if write_to_file:
        with open(self._light_cache_records_path, 'a') as f:
          f.write(json.dumps({hash_value: {}}))
          f.write('\n')
    if delete_only:
      return

    # Insert new values
    self._light_cache_records[hash_value] = light_cache_record
    self._map_shard_to_cache[shard_id].add(hash_value)
    self._shard_active_count[shard_id] += BaseQueryCache._get_cache_size(
        light_cache_record)
    if shard_id != 'backlog':
      self._shard_heap.push(
          key=shard_id, value=self._shard_active_count[shard_id])
    if isinstance(cache_record, types.CacheRecord):
      self._loaded_cache_records[hash_value] = cache_record
    if write_to_file:
      with open(self._light_cache_records_path, 'a') as f:
        f.write(json.dumps(
            {hash_value: BaseQueryCache._encode_light_cache_record(
              light_cache_record)}))
        f.write('\n')

  def _save_light_cache_records(self):
    data = {}
    for query_record_hash, light_cache_record in (
        self._light_cache_records.items()):
      data[query_record_hash] = BaseQueryCache._encode_light_cache_record(
          light_cache_record)
    try:
      os.rename(
          self._light_cache_records_path,
          self._light_cache_records_path + '_backup')
    except OSError:
      pass

    with open(self._light_cache_records_path, 'w') as f:
      f.write(json.dumps(data))
      f.write('\n')

  def _load_light_cache_records(self):
    # Reset all values
    self._shard_active_count = {}
    self._shard_heap = HeapManager()
    self._loaded_cache_records = {}
    self._light_cache_records = {}
    self._map_shard_to_cache = collections.defaultdict(set)
    for shard_id in self.shard_paths:
      self._shard_active_count[shard_id] = 0
      if shard_id != 'backlog':
        self._shard_heap.push(key=shard_id, value=0)
    data = {}

    # Load light cache records from backup if primary file is corrupted
    def load_data(file_path: str):
      data = {}
      with open(file_path, 'r') as f:
        for line in f:
          try:
            record_data = json.loads(line)
          except Exception:
            continue
          for hash_value, record in record_data.items():
            data[hash_value] = record
      return data
    try:
      data = load_data(self.light_cache_records_path)
    except Exception as e1:
      try:
        data = load_data(self.light_cache_records_path + '_backup')
      except Exception as e2:
        return

    # Load light cache records from data
    for query_record_hash, record in data.items():
      if record == {}:
        continue
      try:
        light_cache_record = BaseQueryCache._decode_light_cache_record(record)
      except Exception:
        continue
      if query_record_hash != light_cache_record.query_record_hash:
        continue
      if (isinstance(light_cache_record.shard_id, str)
          and light_cache_record.shard_id != 'backlog'):
        continue
      if (isinstance(light_cache_record.shard_id, int)
          and (light_cache_record.shard_id < 0
            or light_cache_record.shard_id >= self._shard_count)):
        continue
      self._update_cache_record(light_cache_record, write_to_file=False)
    self._save_light_cache_records()

  def _check_cache_record_is_up_to_date(
      self, cache_record: types.CacheRecord) -> bool:
    hash_value = BaseQueryCache._get_hash_value(cache_record)
    if hash_value not in self._light_cache_records:
      return False
    light_cache_record = self._light_cache_records[hash_value]
    comparison_light_cache_record = BaseQueryCache._to_light_cache_record(
        cache_record)
    if light_cache_record != comparison_light_cache_record:
      return False
    return True

  def _load_shard(
      self, shard_id: Union[int, str]) ->  List[str]:
    result = []
    try:
      with open(self.shard_paths[shard_id], 'r') as f:
        for line in f:
          try:
            cache_record = BaseQueryCache._decode_cache_record(json.loads(line))
          except Exception:
            continue
          if not self._check_cache_record_is_up_to_date(cache_record):
            continue
          self._update_cache_record(cache_record)
          result.append(BaseQueryCache._get_hash_value(cache_record))
    except Exception:
      pass
    for hash_value in list(self._map_shard_to_cache[shard_id]):
      if hash_value not in result:
        light_cache_value = copy.deepcopy(
            self._light_cache_records[hash_value])
        self._update_cache_record(light_cache_value, delete_only=True)
    return result

  def _move_backlog_to_shard(self, shard_id: Union[int, str]):
    self._check_shard_id(shard_id)
    if shard_id == 'backlog':
      raise ValueError('Cannot move backlog to backlog')
    shard_hash_values = self._load_shard(shard_id=shard_id)
    backlog_hash_values = self._load_shard(shard_id='backlog')
    for hash_value in backlog_hash_values:
      cache_record = copy.deepcopy(self._loaded_cache_records[hash_value])
      cache_record.shard_id = shard_id
      self._update_cache_record(cache_record)

    with open(self.shard_paths[shard_id] + '_backup', 'w') as f:
      for hash_value in shard_hash_values + backlog_hash_values:
        try:
          cache_record = self._loaded_cache_records[hash_value]
          f.write(json.dumps(BaseQueryCache._encode_cache_record(cache_record)))
          f.write('\n')
        except Exception:
          continue
    os.rename(
        self.shard_paths[shard_id] + '_backup',
        self.shard_paths[shard_id])
    try:
      os.remove(self.shard_paths['backlog'])
    except OSError:
      pass

  def _add_to_backlog(self, cache_record: types.CacheRecord):
    cache_record = copy.deepcopy(cache_record)
    cache_record.shard_id = 'backlog'
    self._update_cache_record(cache_record)
    with open(self.shard_paths['backlog'], 'a') as f:
      f.write(json.dumps(BaseQueryCache._encode_cache_record(cache_record)))
      f.write('\n')

  def get_cache_record(
      self, query_record: Union[types.QueryRecord, str]) -> Optional[types.CacheRecord]:
    hash_value = BaseQueryCache._get_hash_value(query_record)
    if hash_value not in self._light_cache_records:
      return None
    light_cache_record = self._light_cache_records[hash_value]
    if light_cache_record.shard_id not in self.shard_paths:
      return None
    self._load_shard(shard_id=light_cache_record.shard_id)
    if hash_value not in self._loaded_cache_records:
      return None
    return self._loaded_cache_records[hash_value]

  def delete_record(
      self,
      cache_record: Union[
          str,
          types.CacheRecord,
          types.LightCacheRecord,
          types.QueryRecord]):
    hash_value = BaseQueryCache._get_hash_value(cache_record)
    if hash_value not in self._light_cache_records:
      return
    light_cache_records = copy.deepcopy(
        self._light_cache_records[hash_value])
    self._update_cache_record(light_cache_records, delete_only=True)

  def save_record(self, cache_record: types.CacheRecord):
    hash_value = BaseQueryCache._get_hash_value(cache_record)
    self.delete_record(hash_value)

    backlog_size = self._shard_active_count['backlog']
    record_size = BaseQueryCache._get_cache_size(cache_record)
    lowest_shard_value, lowest_shard_key = self._shard_heap.top()
    if (backlog_size + record_size
        > self._response_per_file - lowest_shard_value):
      self._move_backlog_to_shard(shard_id=lowest_shard_key)
      self._add_to_backlog(cache_record)
      self._save_light_cache_records()
    else:
      self._add_to_backlog(cache_record)


class QueryCacheManager(BaseQueryCache):
  _shard_count: int
  _response_per_file: int
  _cache_response_size: int
  _shard_manager: ShardManager
  _record_heap: HeapManager

  def __init__(
      self,
      cache_options: types.CacheOptions,
      shard_count: int = 800,
      response_per_file: int = 200,
      cache_response_size: int = 40000):
    super().__init__(cache_options)

    self._shard_count = shard_count
    self._response_per_file = response_per_file
    self._cache_response_size = cache_response_size
    os.makedirs(self._cache_dir, exist_ok=True)

    self._shard_manager = ShardManager(
        path=self._cache_dir,
        shard_count=shard_count,
        response_per_file=response_per_file)
    self._record_heap = HeapManager(with_size=True)
    for record in self._shard_manager._light_cache_records.values():
      self._push_record_heap(record)

  def _push_record_heap(
      self, cache_record: Union[types.CacheRecord, types.LightCacheRecord]):
    hash_value = BaseQueryCache._get_hash_value(cache_record)
    last_access_time = cache_record.last_access_time.timestamp()
    self._record_heap.push(
        key=hash_value,
        value=last_access_time,
        record_size=BaseQueryCache._get_cache_size(cache_record))
    while len(self._record_heap) > self._cache_response_size:
      _, hash_value = self._record_heap.pop()
      self._shard_manager.delete_record(hash_value)

  def look(
      self,
      query_record: types.QueryRecord,
      update: bool = True
  ) -> Optional[types.QueryResponseRecord]:
    if not isinstance(query_record, types.QueryRecord):
      raise ValueError('query_record should be of type QueryRecord')
    cache_record = self._shard_manager.get_cache_record(query_record)
    if cache_record is None:
      return None
    if cache_record.query_record != query_record:
      return None
    if (len(cache_record.query_responses)
        < self._cache_options.unique_response_limit):
      return None
    if update:
      cache_record.last_access_time = datetime.datetime.now()
      cache_record.call_count += 1
      self._shard_manager.save_record(cache_record=cache_record)
      self._push_record_heap(cache_record)
    return cache_record.query_responses[
        (cache_record.call_count - 1) % len(cache_record.query_responses)]

  def cache(
      self,
      query_record: types.QueryRecord,
      response_record: types.QueryResponseRecord):
    current_time = datetime.datetime.now()
    cache_record = self._shard_manager.get_cache_record(query_record)
    if cache_record:
      cache_record.query_responses.append(response_record)
      cache_record.last_access_time = current_time
      cache_record.call_count += 1
    else:
      query_record.hash_value = self._get_query_record_hash(query_record)
      cache_record = types.CacheRecord(
          query_record=query_record,
          query_responses=[response_record],
          last_access_time=current_time,
          call_count=1)
    self._shard_manager.save_record(cache_record=cache_record)
    self._push_record_heap(cache_record)
