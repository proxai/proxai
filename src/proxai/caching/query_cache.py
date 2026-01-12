import collections
import contextlib
import copy
import dataclasses
import datetime
import heapq
import json
import os
import shutil
from collections.abc import Callable

import proxai.serializers.hash_serializer as hash_serializer
import proxai.serializers.type_serializer as type_serializer
import proxai.state_controllers.state_controller as state_controller
import proxai.type_utils as type_utils
import proxai.types as types

CACHE_DIR = 'query_cache'
LIGHT_CACHE_RECORDS_PATH = 'light_cache_records.json'
_QUERY_CACHE_MANAGER_STATE_PROPERTY = '_query_cache_manager_state'


def _to_light_cache_record(cache_record: types.CacheRecord):
  return types.LightCacheRecord(
      query_record_hash=cache_record.query_record.hash_value,
      query_response_count=len(cache_record.query_responses),
      shard_id=cache_record.shard_id,
      last_access_time=cache_record.last_access_time,
      call_count=cache_record.call_count)


def _get_cache_size(
    cache_record: types.CacheRecord | types.LightCacheRecord) -> int:
  if isinstance(cache_record, types.LightCacheRecord):
    return cache_record.query_response_count + 1
  return len(cache_record.query_responses) + 1


def _get_hash_value(
    cache_record: str | types.CacheRecord | types.LightCacheRecord | types.QueryRecord
) -> str:
  if isinstance(cache_record, str):
    return cache_record
  if isinstance(cache_record, types.CacheRecord):
    if cache_record.query_record.hash_value:
      return cache_record.query_record.hash_value
    else:
      cache_record.query_record.hash_value = (
          hash_serializer.get_query_record_hash(
              cache_record.query_record))
      return cache_record.query_record.hash_value
  if isinstance(cache_record, types.LightCacheRecord):
    if cache_record.query_record_hash:
      return cache_record.query_record_hash
    else:
      raise ValueError('LightCacheRecord doesn\'t have query_record_hash')
  if isinstance(cache_record, types.QueryRecord):
    query_record = cache_record
    if query_record.hash_value:
      return query_record.hash_value
    else:
      query_record.hash_value = hash_serializer.get_query_record_hash(
          query_record)
      return query_record.hash_value


class HeapManager:
  _heap: list[tuple[int, int | str]]
  _active_values: dict[int | str, int]
  _record_size_map: dict[int | str, int]
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
      key: int | str,
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

  def pop(self) -> tuple[int, int | str] | None:
    while self._heap:
      value, key = heapq.heappop(self._heap)
      if key in self._active_values and self._active_values[key] == value:
        del self._active_values[key]
        if self._with_size:
          self._total_size -= self._record_size_map[key]
          del self._record_size_map[key]
        return value, key
    return None, None

  def top(self) -> tuple[int, int | str] | None:
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
  _shard_paths: dict[int | str, str]
  _backlog_shard_path: str
  _light_cache_records_path: str
  _shard_active_count: dict[int | str, int]
  _shard_heap: HeapManager
  _loaded_cache_records: dict[str, types.CacheRecord]
  _light_cache_records: dict[str, types.LightCacheRecord]
  _map_shard_to_cache: dict[str | int, set[str]]

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

  def _check_shard_id(self, shard_id: int | str):
    if shard_id not in self.shard_paths:
      raise ValueError('Invalid shard_id')

  def _update_cache_record(
      self,
      cache_record: types.CacheRecord | types.LightCacheRecord,
      delete_only: bool = False,
      write_to_file: bool = True):
    hash_value = _get_hash_value(cache_record)
    shard_id = cache_record.shard_id
    if isinstance(cache_record, types.CacheRecord):
      light_cache_record = _to_light_cache_record(cache_record)
    else:
      light_cache_record = cache_record

    # Clean previous values
    if hash_value in self._light_cache_records:
      old_shard_id = self._light_cache_records[hash_value].shard_id
      self._map_shard_to_cache[old_shard_id].remove(hash_value)
      self._shard_active_count[old_shard_id] -= _get_cache_size(
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
    self._shard_active_count[shard_id] += _get_cache_size(
        light_cache_record)
    if shard_id != 'backlog':
      self._shard_heap.push(
          key=shard_id, value=self._shard_active_count[shard_id])
    if isinstance(cache_record, types.CacheRecord):
      self._loaded_cache_records[hash_value] = cache_record
    if write_to_file:
      with open(self._light_cache_records_path, 'a') as f:
        f.write(json.dumps(
            {hash_value: type_serializer.encode_light_cache_record(
              light_cache_record)}))
        f.write('\n')

  def _save_light_cache_records(self):
    data = {}
    for query_record_hash, light_cache_record in (
        self._light_cache_records.items()):
      data[query_record_hash] = type_serializer.encode_light_cache_record(
          light_cache_record)
    with contextlib.suppress(OSError):
      os.rename(
          self._light_cache_records_path,
          self._light_cache_records_path + '_backup')

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
      with open(file_path) as f:
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
    except Exception:
      try:
        data = load_data(self.light_cache_records_path + '_backup')
      except Exception:
        return

    # Load light cache records from data
    for query_record_hash, record in data.items():
      if record == {}:
        continue
      try:
        light_cache_record = type_serializer.decode_light_cache_record(record)
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
      light_cache_record.call_count = 0
      self._update_cache_record(light_cache_record, write_to_file=False)
    self._save_light_cache_records()

  def _check_cache_record_is_up_to_date(
      self, cache_record: types.CacheRecord) -> bool:
    hash_value = _get_hash_value(cache_record)
    if hash_value not in self._light_cache_records:
      return False
    light_cache_record = copy.deepcopy(
        self._light_cache_records[hash_value])
    comparison_light_cache_record = copy.deepcopy(
        _to_light_cache_record(
            cache_record))
    light_cache_record.call_count = None
    comparison_light_cache_record.call_count = None
    return light_cache_record == comparison_light_cache_record

  def _load_shard(
      self, shard_id: int | str) ->  list[str]:
    result = []
    try:
      with open(self.shard_paths[shard_id]) as f:
        for line in f:
          try:
            cache_record = type_serializer.decode_cache_record(json.loads(line))
          except Exception:
            continue
          if not self._check_cache_record_is_up_to_date(cache_record):
            continue
          cache_record.call_count = self._light_cache_records[
              cache_record.query_record.hash_value].call_count
          self._update_cache_record(cache_record)
          result.append(_get_hash_value(cache_record))
    except Exception:
      pass
    for hash_value in list(self._map_shard_to_cache[shard_id]):
      if hash_value not in result:
        light_cache_value = copy.deepcopy(
            self._light_cache_records[hash_value])
        self._update_cache_record(light_cache_value, delete_only=True)
    return result

  def _move_backlog_to_shard(self, shard_id: int | str):
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
          f.write(json.dumps(type_serializer.encode_cache_record(cache_record)))
          f.write('\n')
        except Exception:
          continue
    os.rename(
        self.shard_paths[shard_id] + '_backup',
        self.shard_paths[shard_id])
    with contextlib.suppress(OSError):
      os.remove(self.shard_paths['backlog'])

  def _add_to_backlog(self, cache_record: types.CacheRecord):
    cache_record = copy.deepcopy(cache_record)
    cache_record.shard_id = 'backlog'
    self._update_cache_record(cache_record)
    with open(self.shard_paths['backlog'], 'a') as f:
      f.write(json.dumps(type_serializer.encode_cache_record(cache_record)))
      f.write('\n')

  def get_cache_record(
      self, query_record: types.QueryRecord | str) -> types.CacheRecord | None:
    hash_value = _get_hash_value(query_record)
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
      cache_record: str | types.CacheRecord | types.LightCacheRecord | types.QueryRecord):
    hash_value = _get_hash_value(cache_record)
    if hash_value not in self._light_cache_records:
      return
    light_cache_records = copy.deepcopy(
        self._light_cache_records[hash_value])
    self._update_cache_record(light_cache_records, delete_only=True)

  def save_record(self, cache_record: types.CacheRecord):
    hash_value = _get_hash_value(cache_record)
    self.delete_record(hash_value)

    backlog_size = self._shard_active_count['backlog']
    record_size = _get_cache_size(cache_record)
    lowest_shard_value, lowest_shard_key = self._shard_heap.top()
    if (backlog_size + record_size
        > self._response_per_file - lowest_shard_value):
      self._move_backlog_to_shard(shard_id=lowest_shard_key)
      self._add_to_backlog(cache_record)
      self._save_light_cache_records()
    else:
      self._add_to_backlog(cache_record)


@dataclasses.dataclass
class QueryCacheManagerParams:
  cache_options: types.CacheOptions | None = None
  get_cache_options: Callable[[], types.CacheOptions] | None = None
  shard_count: int | None = None
  response_per_file: int | None = None
  cache_response_size: int | None = None


class QueryCacheManager(state_controller.StateControlled):
  _cache_options: types.CacheOptions
  _shard_count: int
  _response_per_file: int
  _cache_response_size: int
  _query_cache_manager_state: types.QueryCacheManagerState
  _shard_manager: ShardManager
  _record_heap: HeapManager

  def __init__(
      self,
      init_from_params: QueryCacheManagerParams | None = None,
      init_from_state: types.QueryCacheManagerState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params,
        init_from_state=init_from_state)

    self.set_property_value(
        'status', types.QueryCacheManagerStatus.INITIALIZING)

    if init_from_state:
      self.load_state(init_from_state)
      self._init_dir()
      self._init_managers()
    elif init_from_params:
      self._get_cache_options = init_from_params.get_cache_options
      self.cache_options = init_from_params.cache_options
      if init_from_params.response_per_file is not None:
        self.response_per_file = init_from_params.response_per_file
      if init_from_params.shard_count is not None:
        self.shard_count = init_from_params.shard_count
      if init_from_params.cache_response_size is not None:
        self.cache_response_size = init_from_params.cache_response_size
    self.init_status()

  def get_internal_state_property_name(self):
    return _QUERY_CACHE_MANAGER_STATE_PROPERTY

  def get_internal_state_type(self):
    return types.QueryCacheManagerState

  def init_status(
      self):
    if self.cache_options is None:
      self.status = types.QueryCacheManagerStatus.CACHE_OPTIONS_NOT_FOUND
      return

    if self.cache_options.cache_path is None:
      self.status = types.QueryCacheManagerStatus.CACHE_PATH_NOT_FOUND
      return

    if not os.access(self.cache_options.cache_path, os.W_OK):
      self.status = types.QueryCacheManagerStatus.CACHE_PATH_NOT_WRITABLE
      return

    self._init_dir()
    self._init_managers()
    self.status = types.QueryCacheManagerStatus.WORKING

  def clear_cache(self):
    if (
        self.status == types.QueryCacheManagerStatus.INITIALIZING or
        self.status == types.QueryCacheManagerStatus.CACHE_OPTIONS_NOT_FOUND or
        self.status == types.QueryCacheManagerStatus.CACHE_PATH_NOT_FOUND or
        self.status == types.QueryCacheManagerStatus.CACHE_PATH_NOT_WRITABLE):
      raise ValueError(f'QueryCacheManager status is {self.status}')
    cache_dir = self._get_cache_dir(self.cache_options.cache_path)
    if os.path.exists(cache_dir):
      shutil.rmtree(cache_dir)
    self._init_dir()
    self._init_managers()

  def _init_dir(self):
    if self.cache_options.cache_path is None:
      return
    os.makedirs(
        self._get_cache_dir(self.cache_options.cache_path),
        exist_ok=True)

  def _init_managers(self):
    if self.cache_options.cache_path is None:
      return
    self._shard_manager = ShardManager(
        path=self._get_cache_dir(self.cache_options.cache_path),
        shard_count=self.shard_count,
        response_per_file=self.response_per_file)
    self._record_heap = HeapManager(with_size=True)
    for record in self._shard_manager._light_cache_records.values():
      self._push_record_heap(record)

  @staticmethod
  def _get_cache_dir(cache_path: str) -> str:
    return os.path.join(cache_path, CACHE_DIR)

  @property
  def status(self) -> types.QueryCacheManagerStatus:
    return self.get_property_value('status')

  @status.setter
  def status(self, value: types.QueryCacheManagerStatus):
    self.set_property_value('status', value)

  @property
  def cache_options(self) -> types.CacheOptions:
    return self.get_property_value('cache_options')

  @cache_options.setter
  def cache_options(self, value: types.CacheOptions):
    self.set_property_value('cache_options', value)

  @property
  def shard_count(self) -> int:
    return self.get_property_value('shard_count')

  @shard_count.setter
  def shard_count(self, value: int):
    self.set_property_value('shard_count', value)

  @property
  def response_per_file(self) -> int:
    return self.get_property_value('response_per_file')

  @response_per_file.setter
  def response_per_file(self, value: int):
    self.set_property_value('response_per_file', value)

  @property
  def cache_response_size(self) -> int:
    return self.get_property_value('cache_response_size')

  @cache_response_size.setter
  def cache_response_size(self, value: int):
    self.set_property_value('cache_response_size', value)

  def _push_record_heap(
      self, cache_record: types.CacheRecord | types.LightCacheRecord):
    hash_value = _get_hash_value(cache_record)
    last_access_time = cache_record.last_access_time.timestamp()
    self._record_heap.push(
        key=hash_value,
        value=last_access_time,
        record_size=_get_cache_size(cache_record))
    while len(self._record_heap) > self.cache_response_size:
      _, hash_value = self._record_heap.pop()
      self._shard_manager.delete_record(hash_value)

  def look(
      self,
      query_record: types.QueryRecord,
      update: bool = True,
      unique_response_limit: int | None = None,
  ) -> types.CacheLookResult:
    if self.status != types.QueryCacheManagerStatus.WORKING:
      raise ValueError(f'QueryCacheManager status is {self.status}')

    if not isinstance(query_record, types.QueryRecord):
      raise ValueError('query_record should be of type QueryRecord')
    cache_record = self._shard_manager.get_cache_record(query_record)
    if cache_record is None:
      return types.CacheLookResult(
          look_fail_reason=types.CacheLookFailReason.CACHE_NOT_FOUND)
    if not type_utils.is_query_record_equal(cache_record.query_record, query_record):
      return types.CacheLookResult(
          look_fail_reason=types.CacheLookFailReason.CACHE_NOT_MATCHED)
    if unique_response_limit is None:
      unique_response_limit = self.cache_options.unique_response_limit
    if len(cache_record.query_responses) < unique_response_limit:
      return types.CacheLookResult(
          look_fail_reason=
          types.CacheLookFailReason.UNIQUE_RESPONSE_LIMIT_NOT_REACHED)
    query_response: types.QueryResponseRecord = cache_record.query_responses[
        cache_record.call_count % len(cache_record.query_responses)]
    if (query_response.error
        and self.cache_options.retry_if_error_cached
        and cache_record.call_count < len(cache_record.query_responses)):
      cache_record.last_access_time = datetime.datetime.now()
      cache_record.call_count += 1
      self._shard_manager.save_record(cache_record=cache_record)
      self._push_record_heap(cache_record)
      return types.CacheLookResult(
          look_fail_reason=types.CacheLookFailReason.PROVIDER_ERROR_CACHED)
    if update:
      cache_record.last_access_time = datetime.datetime.now()
      cache_record.call_count += 1
      self._shard_manager.save_record(cache_record=cache_record)
      self._push_record_heap(cache_record)
    return types.CacheLookResult(query_response=query_response)

  def cache(
      self,
      query_record: types.QueryRecord,
      response_record: types.QueryResponseRecord,
      unique_response_limit: int | None = None):
    if self.status != types.QueryCacheManagerStatus.WORKING:
      raise ValueError(f'QueryCacheManager status is {self.status}')

    current_time = datetime.datetime.now()
    cache_record = self._shard_manager.get_cache_record(query_record)
    if not cache_record:
      query_record.hash_value = hash_serializer.get_query_record_hash(
          query_record)
      cache_record = types.CacheRecord(
          query_record=query_record,
          query_responses=[response_record],
          last_access_time=current_time,
          call_count=0)
      self._shard_manager.save_record(cache_record=cache_record)
      self._push_record_heap(cache_record)
      return
    if unique_response_limit is None:
      unique_response_limit = self.cache_options.unique_response_limit
    if len(cache_record.query_responses) < unique_response_limit:
      cache_record.query_responses.append(response_record)
      cache_record.last_access_time = current_time
      self._shard_manager.save_record(cache_record=cache_record)
      self._push_record_heap(cache_record)
      return
    if (self.cache_options.retry_if_error_cached
        and response_record.error is None):
      for idx, previous_response in enumerate(cache_record.query_responses):
        if previous_response.error:
          cache_record.query_responses[idx] = response_record
          cache_record.last_access_time = current_time
          self._shard_manager.save_record(cache_record=cache_record)
          self._push_record_heap(cache_record)
          return
