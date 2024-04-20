import copy
import datetime
import os
from typing import Dict
import json
import proxai.types as types
import proxai.caching.query_cache as query_cache
import pytest
import functools
import tempfile

_SHARD_COUNT = 3
_RESPONSE_PER_FILE = 4


def _get_query_record_options():
  return [
      {'call_type': types.CallType.GENERATE_TEXT},
      {'provider': types.Provider.OPENAI,},
      {'provider_model': types.OpenAIModel.GPT_4,},
      {'max_tokens': 100},
      {'prompt': 'Hello, world!'},
      {'call_type': types.CallType.GENERATE_TEXT,
       'provider': types.Provider.OPENAI,
       'provider_model': types.OpenAIModel.GPT_4,
       'max_tokens': 100,
       'prompt': 'Hello, world!'},]


def _get_query_response_record_options():
  return [
      {'response': 'Hello, world!'},
      {'error': 'Error message'},
      {'start_time': datetime.datetime.now()},
      {'end_time': datetime.datetime.now()},
      {'response_time': datetime.timedelta(seconds=1)},
      {'response': 'Hello, world!',
       'error': 'Error message',
       'start_time': datetime.datetime.now(),
       'end_time': datetime.datetime.now(),
       'response_time': datetime.timedelta(seconds=1)},]


def _get_cache_record_options():
  return [
      {'query_record': types.QueryRecord(
          call_type=types.CallType.GENERATE_TEXT)},
      {'query_responses': [types.QueryResponseRecord(
          response='Hello, world!')]},
      {'shard_id': 0},
      {'shard_id': 'backlog'},
      {'last_access_time': datetime.datetime.now()},
      {'call_count': 1},
      {'query_record': types.QueryRecord(
          call_type=types.CallType.GENERATE_TEXT),
       'query_responses': [types.QueryResponseRecord(
          response='Hello, world!')],
       'shard_id': 0,
       'last_access_time': datetime.datetime.now(),
       'call_count': 1},]


def _get_light_cache_record_options():
  return [
      {'query_record_hash': 'hash_value'},
      {'query_response_count': 1},
      {'shard_id': 0},
      {'last_access_time': datetime.datetime.now()},
      {'call_count': 1},
      {'query_record_hash': 'hash_value',
       'query_response_count': 1,
       'shard_id': 0,
       'last_access_time': datetime.datetime.now(),
       'call_count': 1},]


def _save_shard_manager(path, records=None):
    save_shard_manager = query_cache.ShardManager(
        path=path, shard_count=_SHARD_COUNT,
        response_per_file=_RESPONSE_PER_FILE)
    if records:
      save_shard_manager._light_cache_records = records
    save_shard_manager._save_light_cache_records()
    return save_shard_manager


def _create_cache_record(
    prompt,
    responses,
    shard_id,
    call_count,
    all_values=False):
  query_record = types.QueryRecord(prompt=prompt)
  query_record.hash_value = (
      query_cache.BaseQueryCache._get_query_record_hash(
          query_record=query_record))
  cache_record = types.CacheRecord(
      query_record=query_record,
      query_responses=[
          types.QueryResponseRecord(response=response)
          for response in responses],
      shard_id=shard_id,
      last_access_time=datetime.datetime.now(),
      call_count=call_count)
  if not all_values:
    return cache_record
  light_record = query_cache.BaseQueryCache._to_light_cache_record(cache_record)
  return (
      query_record.hash_value,
      cache_record,
      light_record,
      query_cache.BaseQueryCache._encode_cache_record(cache_record),
      query_cache.BaseQueryCache._encode_light_cache_record(light_record))


def _get_example_records(shard_dir=None, shard_count=None):
  if shard_dir:
      shard_manager = query_cache.ShardManager(
          path=shard_dir, shard_count=shard_count,
          response_per_file=_RESPONSE_PER_FILE)

  def _generate_vals(records, write_to_shard=False):
    cache_records = {}
    light_cache_records = {}
    enc_cache_records = {}
    enc_light_cache_records = {}
    for record in records:
      if isinstance(record, types.CacheRecord):
        hash_value = record.query_record.hash_value
      else:
        record, new_hash_value = record
        hash_value = record.query_record.hash_value
        record.query_record.hash_value = new_hash_value
      cache_records[hash_value] = record
      light_cache_records[hash_value] = (
          query_cache.BaseQueryCache._to_light_cache_record(record))
      enc_cache_records[hash_value] = (
          query_cache.BaseQueryCache._encode_cache_record(record))
      enc_light_cache_records[hash_value] = (
          query_cache.BaseQueryCache._encode_light_cache_record(
              light_cache_record=light_cache_records[hash_value]))
      if write_to_shard:
        with open(shard_manager.shard_paths[record.shard_id], 'a') as f:
          f.write(json.dumps(enc_cache_records[hash_value]))
          f.write('\n')
    return (cache_records, light_cache_records, enc_cache_records,
            enc_light_cache_records)

  records = [
      _create_cache_record(
          prompt='p1', responses=[], shard_id=0, call_count=0),
      _create_cache_record(
          prompt='p2', responses=['r1'], shard_id=1, call_count=1),
      _create_cache_record(
          prompt='p3', responses=['r2', 'r3'], shard_id=2, call_count=2),
      _create_cache_record(
          prompt='p4', responses=['r4'], shard_id='backlog', call_count=1)]
  (cache_records, light_cache_records, enc_cache_records,
    enc_light_cache_records) = _generate_vals(
      records, write_to_shard=(shard_dir is not None))

  records = [
      _create_cache_record(
          prompt='p5', responses=['r5'], shard_id='corrupted', call_count=1),
      _create_cache_record(
          prompt='p6', responses=['r6'], shard_id=-1, call_count=1),
      (_create_cache_record(
          prompt='p7', responses=['r7'], shard_id=0, call_count=1),
        'corrupted_hash_value')]
  (cor_cache_records, cor_light_cache_records, enc_cor_cache_records,
    enc_cor_light_cache_records) = _generate_vals(records)

  all_light_cache_records = copy.deepcopy(light_cache_records)
  enc_all_light_cache_records = copy.deepcopy(enc_light_cache_records)
  for k, v in cor_light_cache_records.items():
    all_light_cache_records[k] = v
    enc_all_light_cache_records[k] = enc_cor_light_cache_records[k]

  if shard_dir:
    with open(shard_manager.light_cache_records_path, 'w') as f:
      f.write(json.dumps(enc_all_light_cache_records))

  return {
      'cache_records': cache_records,
      'light_cache_records': light_cache_records,
      'enc_cache_records': enc_cache_records,
      'enc_light_cache_records': enc_light_cache_records,
      'cor_cache_records': cor_cache_records,
      'cor_light_cache_records': cor_light_cache_records,
      'enc_cor_cache_records': enc_cor_cache_records,
      'enc_cor_light_cache_records': enc_cor_light_cache_records,
      'all_light_cache_records': all_light_cache_records,
      'enc_all_light_cache_records': enc_all_light_cache_records}


def _check_file_json(filepath, expected):
  with open(filepath, 'r') as f:
    for line, expected_data in zip(f, expected):
      assert json.loads(line) == expected_data


def _check_light_cache_records_file(
    filepath, expected):
  data = {}
  with open(filepath, 'r') as f:
    for line in f:
      record_data = json.loads(line)
      for hash_value, record in record_data.items():
        data[hash_value] = record
  for k in list(data.keys()):
    if data[k] == {}:
      del data[k]
  assert data == expected


def _check_shard_heap(shard_manager, expected):
  current_heap = copy.deepcopy(shard_manager._shard_heap)
  current_heap_list = []
  while len(current_heap) > 0:
    current_heap_list.append(current_heap.pop())
  assert sorted(current_heap_list) == expected


def _check_record_heap(query_cache_manager, expected):
  current_heap: query_cache.HeapManager = (
      copy.deepcopy(query_cache_manager._record_heap))
  current_heap_list = []
  while len(current_heap) > 0:
    current_heap_list.append(current_heap.pop())
  current_heap_list = [v for k, v in current_heap_list]
  assert current_heap_list == expected


def _check_shard_manager_state(
    shard_manager: query_cache.ShardManager,
    light_cache_records=None,
    enc_light_cache_records=None,
    enc_light_cache_records_backup=None,
    map_shard_to_cache=None,
    shard_active_count=None,
    shard_heap=None,
    loaded_cache_records=None,
):
  if light_cache_records:
    assert shard_manager._light_cache_records == light_cache_records
  if enc_light_cache_records:
    _check_light_cache_records_file(
        filepath=shard_manager.light_cache_records_path,
        expected=enc_light_cache_records)
  if enc_light_cache_records_backup:
    _check_light_cache_records_file(
        filepath=shard_manager.light_cache_records_path + '_backup',
        expected=enc_light_cache_records_backup)
  if map_shard_to_cache:
    for k, v in shard_manager._map_shard_to_cache.items():
      if (len(v) == 0 and k in map_shard_to_cache):
        assert shard_manager._map_shard_to_cache[k] == map_shard_to_cache[k]
      if len(v) != 0:
        assert shard_manager._map_shard_to_cache[k] == map_shard_to_cache[k]
  if shard_active_count:
    assert shard_manager._shard_active_count == shard_active_count
  if shard_heap:
    _check_shard_heap(shard_manager, shard_heap)
  if loaded_cache_records:
    assert shard_manager._loaded_cache_records == loaded_cache_records


def _get_cache_record_from_prompt(prompts, records):
  cache_records = []
  enc_cache_records = []
  light_cache_records = []
  enc_light_cache_records = []
  for prompt in prompts:
    for hash, cache_record in records['cache_records'].items():
      if cache_record.query_record.prompt == prompt:
        cache_records.append(copy.deepcopy(cache_record))
        enc_cache_records.append(copy.deepcopy(
            records['enc_cache_records'][hash]))
        light_cache_records.append(copy.deepcopy(
            records['light_cache_records'][hash]))
        enc_light_cache_records.append(copy.deepcopy(
            records['enc_light_cache_records'][hash]))
  return {
      'cache_records': cache_records,
      'enc_cache_records': enc_cache_records,
      'light_cache_records': light_cache_records,
      'enc_light_cache_records': enc_light_cache_records}


def _get_hash_from_prompt(prompt, records):
  for hash, cache_record in records['cache_records'].items():
    if cache_record.query_record.prompt == prompt:
      return hash


def _unpack_records(records):
  hash_1 = copy.deepcopy(
      _get_hash_from_prompt(prompt='p1', records=records))
  light_1 = copy.deepcopy(
      records['light_cache_records'][hash_1])
  enc_light_1 = copy.deepcopy(
      records['enc_light_cache_records'][hash_1])
  hash_2 = copy.deepcopy(
      _get_hash_from_prompt(prompt='p2', records=records))
  light_2 = copy.deepcopy(
      records['light_cache_records'][hash_2])
  enc_light_2 = copy.deepcopy(
      records['enc_light_cache_records'][hash_2])
  hash_3 = copy.deepcopy(
      _get_hash_from_prompt(prompt='p3', records=records))
  light_3 = copy.deepcopy(
      records['light_cache_records'][hash_3])
  enc_light_3 = copy.deepcopy(
      records['enc_light_cache_records'][hash_3])
  hash_4 = copy.deepcopy(
      _get_hash_from_prompt(prompt='p4', records=records))
  light_4 = copy.deepcopy(
      records['light_cache_records'][hash_4])
  enc_light_4 = copy.deepcopy(
      records['enc_light_cache_records'][hash_4])
  return (
      hash_1, light_1, enc_light_1,
      hash_2, light_2, enc_light_2,
      hash_3, light_3, enc_light_3,
      hash_4, light_4, enc_light_4)

def _print_state(
    shard_manager: query_cache.ShardManager,
    records):
  names = {}
  for i in range(1, 10):
    val = types.QueryRecord(prompt=f'p{i}')
    names[query_cache.BaseQueryCache._get_query_record_hash(
        query_record=val)] = f'p{i}'
  from pprint import pprint
  print('---- _light_cache_records')
  for k in shard_manager._light_cache_records:
    print(f'{k}: {names[k]}')
  print('---- _loaded_cache_records')
  for k in shard_manager._loaded_cache_records:
    print(f'{k}: {names[k]}')
  print('---- _shard_active_count')
  pprint(shard_manager._shard_active_count)
  for shard_id in shard_manager.shard_paths:
    print(f'---- Shard {shard_id}')
    with open(shard_manager.shard_paths[shard_id], 'r') as f:
      for line in f:
        pprint(json.loads(line))


class TestBaseQueryCache:
  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_get_query_record_hash(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    query_hash_value = query_cache.BaseQueryCache._get_query_record_hash(
        query_record=query_record)

    query_record_options['max_tokens'] = 222
    query_record_2 = types.QueryRecord(**query_record_options)
    query_hash_value_2 = query_cache.BaseQueryCache._get_query_record_hash(
        query_record=query_record_2)

    assert query_hash_value != query_hash_value_2
    assert query_hash_value == (
        query_cache.BaseQueryCache._get_query_record_hash(
            query_record=query_record))

  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_encode_decode_query_record(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    encoded_query_record = query_cache.BaseQueryCache._encode_query_record(
        query_record=query_record)
    if ('provider' not in query_record_options
        and 'provider_model' in query_record_options):
      with pytest.raises(ValueError):
        _ = query_cache.BaseQueryCache._decode_query_record(
            record=encoded_query_record)
    else:
      decoded_query_record = query_cache.BaseQueryCache._decode_query_record(
          record=encoded_query_record)
      assert query_record == decoded_query_record

  @pytest.mark.parametrize(
      'query_response_record_options', _get_query_response_record_options())
  def test_encode_decode_query_response_record(
      self, query_response_record_options):
    query_response_record = types.QueryResponseRecord(
        **query_response_record_options)
    encoded_query_response_record = (
        query_cache.BaseQueryCache._encode_query_response_record(
            query_response_record=query_response_record))
    decoded_query_response_record = (
        query_cache.BaseQueryCache._decode_query_response_record(
            record=encoded_query_response_record))
    assert query_response_record == decoded_query_response_record

  @pytest.mark.parametrize('cache_record_options', _get_cache_record_options())
  def test_encode_decode_cache_record(self, cache_record_options):
    cache_record = types.CacheRecord(**cache_record_options)
    encoded_cache_record = query_cache.BaseQueryCache._encode_cache_record(
        cache_record=cache_record)
    decoded_cache_record = query_cache.BaseQueryCache._decode_cache_record(
        record=encoded_cache_record)
    assert cache_record == decoded_cache_record

  @pytest.mark.parametrize(
      'light_cache_record_options', _get_light_cache_record_options())
  def test_encode_decode_light_cache_record(self, light_cache_record_options):
    light_cache_record = types.LightCacheRecord(**light_cache_record_options)
    encoded_light_cache_record = (
        query_cache.BaseQueryCache._encode_light_cache_record(
            light_cache_record=light_cache_record))
    decoded_light_cache_record = (
        query_cache.BaseQueryCache._decode_light_cache_record(
            record=encoded_light_cache_record))
    assert light_cache_record == decoded_light_cache_record

  def test_to_light_cache_record(self):
    query_record = types.QueryRecord(call_type=types.CallType.GENERATE_TEXT)
    query_record.hash_value = (
        query_cache.BaseQueryCache._get_query_record_hash(
            query_record=query_record))
    cache_record = types.CacheRecord(
        query_record=query_record,
        query_responses=[
          types.QueryResponseRecord(response='Hello, world! - 1'),
          types.QueryResponseRecord(response='Hello, world! - 2'),],
        shard_id=5,
        last_access_time=datetime.datetime.now(),
        call_count=7)

    light_cache_record = query_cache.BaseQueryCache._to_light_cache_record(
        cache_record=cache_record)
    assert light_cache_record.query_record_hash == (
        query_cache.BaseQueryCache._get_query_record_hash(
            query_record=cache_record.query_record))
    assert light_cache_record.query_response_count == 2
    assert light_cache_record.shard_id == 5
    assert light_cache_record.last_access_time == cache_record.last_access_time
    assert light_cache_record.call_count == 7

  def test_get_cache_size(self):
    cache_record = types.CacheRecord(
        query_record=types.QueryRecord(call_type=types.CallType.GENERATE_TEXT))
    light_cache_record = query_cache.BaseQueryCache._to_light_cache_record(
        cache_record=cache_record)
    cache_record_size = query_cache.BaseQueryCache._get_cache_size(cache_record)
    light_cache_record_size = query_cache.BaseQueryCache._get_cache_size(
        light_cache_record)
    assert cache_record_size == light_cache_record_size == 1

    cache_record = types.CacheRecord(
        query_record=types.QueryRecord(call_type=types.CallType.GENERATE_TEXT),
        query_responses=[
          types.QueryResponseRecord(response='Hello, world! - 1'),
          types.QueryResponseRecord(response='Hello, world! - 2'),])
    light_cache_record = query_cache.BaseQueryCache._to_light_cache_record(
        cache_record=cache_record)
    cache_record_size = query_cache.BaseQueryCache._get_cache_size(cache_record)
    light_cache_record_size = query_cache.BaseQueryCache._get_cache_size(
        light_cache_record)
    assert cache_record_size == light_cache_record_size == 3


class TestHeapManager:
  def test_simple_heap(self):
    heap_manager = query_cache.HeapManager()
    assert len(heap_manager) == 0
    assert heap_manager.top() == (None, None)
    assert heap_manager.pop() == (None, None)

    heap_manager.push(key='b', value=10)
    assert len(heap_manager) == 1

    heap_manager.push(key='a', value=1)
    assert len(heap_manager) == 2

    heap_manager.push(key='c', value=7)
    assert len(heap_manager) == 3
    assert heap_manager.top() == (1, 'a')
    assert len(heap_manager) == 3

    assert heap_manager.pop() == (1, 'a')
    assert heap_manager.top() == (7, 'c')
    assert len(heap_manager) == 2

    heap_manager.push(key='a', value=2)
    assert heap_manager.top() == (2, 'a')
    assert len(heap_manager) == 3

    heap_manager.push(key='a', value=12)
    assert heap_manager.top() == (7, 'c')
    assert len(heap_manager) == 3

    assert heap_manager.pop() == (7, 'c')
    assert heap_manager.pop() == (10, 'b')
    assert heap_manager.pop() == (12, 'a')
    assert len(heap_manager) == 0
    assert heap_manager.top() == (None, None)
    assert heap_manager.pop() == (None, None)

    heap_manager.push(key='a', value=3)
    assert len(heap_manager) == 1
    assert heap_manager.top() == (3, 'a')

  def test_heap_with_size(self):
    heap_manager = query_cache.HeapManager(with_size=True)
    assert len(heap_manager) == 0
    assert heap_manager.top() == (None, None)
    assert heap_manager.pop() == (None, None)

    heap_manager.push(key='b', value=10, record_size=5)
    assert len(heap_manager) == 5

    heap_manager.push(key='a', value=1, record_size=10)
    assert len(heap_manager) == 15

    heap_manager.push(key='c', value=7, record_size=20)
    assert len(heap_manager) == 35
    assert heap_manager.top() == (1, 'a')
    assert len(heap_manager) == 35

    assert heap_manager.pop() == (1, 'a')
    assert heap_manager.top() == (7, 'c')
    assert len(heap_manager) == 25

    heap_manager.push(key='a', value=2, record_size=100)
    assert heap_manager.top() == (2, 'a')
    assert len(heap_manager) == 125

    heap_manager.push(key='a', value=12, record_size=1000)
    assert heap_manager.top() == (7, 'c')
    assert len(heap_manager) == 1025

    assert heap_manager.pop() == (7, 'c')
    assert heap_manager.pop() == (10, 'b')
    assert heap_manager.pop() == (12, 'a')
    assert len(heap_manager) == 0
    assert heap_manager.top() == (None, None)
    assert heap_manager.pop() == (None, None)

    heap_manager.push(key='a', value=3, record_size=10)
    assert len(heap_manager) == 10
    assert heap_manager.top() == (3, 'a')


class TestShardManager:
  def test_shard_names(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      assert shard_manager.shard_paths == {
          0: temp_dir + '/shard_00000-of-00003.jsonl',
          1: temp_dir + '/shard_00001-of-00003.jsonl',
          2: temp_dir + '/shard_00002-of-00003.jsonl',
          'backlog': temp_dir + '/shard_backlog_00003.jsonl'}
      assert (shard_manager.backlog_shard_path
              == temp_dir + '/shard_backlog_00003.jsonl')
      assert (shard_manager.light_cache_records_path
              == temp_dir + '/light_cache_records_00003.json')

      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=1, response_per_file=1)
      assert shard_manager.shard_paths == {
          0: temp_dir + '/shard_00000-of-00001.jsonl',
          'backlog': temp_dir + '/shard_backlog_00001.jsonl'}
      assert (shard_manager.backlog_shard_path
              == temp_dir + '/shard_backlog_00001.jsonl')
      assert (shard_manager.light_cache_records_path
              == temp_dir + '/light_cache_records_00001.json')

  def test_invalid_shard_count(self):
    temp_shard_manager = functools.partial(
        query_cache.ShardManager, path='temp_path',
        response_per_file=_RESPONSE_PER_FILE)
    with pytest.raises(ValueError):
      _ = temp_shard_manager(shard_count=0)
    with pytest.raises(ValueError):
      _ = temp_shard_manager(shard_count=-1)
    with pytest.raises(ValueError):
      _ = temp_shard_manager(shard_count=100000)

  def test_invalid_response_per_file(self):
    temp_shard_manager = functools.partial(
        query_cache.ShardManager, path='temp_path', shard_count=_SHARD_COUNT)
    with pytest.raises(ValueError):
      _ = temp_shard_manager(response_per_file=0)
    with pytest.raises(ValueError):
      _ = temp_shard_manager(response_per_file=-1)

  def test_check_shard_id(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      shard_manager._check_shard_id(shard_id=0)
      shard_manager._check_shard_id(shard_id=1)
      shard_manager._check_shard_id(shard_id=2)
      shard_manager._check_shard_id(shard_id='backlog')
      with pytest.raises(ValueError):
        shard_manager._check_shard_id(shard_id=-1)
      with pytest.raises(ValueError):
        shard_manager._check_shard_id(shard_id=4)

  def test_update_cache_record(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)

      hash_1, record_1, light_1, _, enc_light_1 = _create_cache_record(
          prompt='p1', responses=[], shard_id=0, call_count=1,
          all_values=True)
      shard_manager._update_cache_record(record_1)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={hash_1: light_1},
          enc_light_cache_records={hash_1: enc_light_1},
          map_shard_to_cache={0: {hash_1}},
          shard_active_count={0: 1, 1: 0, 2: 0, 'backlog': 0},
          shard_heap=[(0, 1), (0, 2), (1, 0)])

      hash_2, record_2, light_2, _, enc_light_2 = _create_cache_record(
          prompt='p2', responses=['r1'], shard_id=1, call_count=1,
          all_values=True)
      shard_manager._update_cache_record(record_2)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={hash_1: light_1, hash_2: light_2},
          enc_light_cache_records={hash_1: enc_light_1, hash_2: enc_light_2},
          map_shard_to_cache={0: {hash_1}, 1: {hash_2}},
          shard_active_count={0: 1, 1: 2, 2: 0, 'backlog': 0},
          shard_heap=[(0, 2), (1, 0), (2, 1)])

      hash_3, record_3, light_3, _, enc_light_3 = _create_cache_record(
          prompt='p3', responses=['r2', 'r3'], shard_id='backlog', call_count=0,
          all_values=True)
      shard_manager._update_cache_record(record_3)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_2: light_2, hash_3: light_3},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_2: enc_light_2, hash_3: enc_light_3},
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_2}, 'backlog': {hash_3}},
          shard_active_count={0: 1, 1: 2, 2: 0, 'backlog': 3},
          shard_heap=[(0, 2), (1, 0), (2, 1)])

      hash_1, record_1, light_1, _, enc_light_1 = _create_cache_record(
          prompt='p1', responses=[], shard_id=2, call_count=1,
          all_values=True)
      shard_manager._update_cache_record(record_1)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_2: light_2, hash_3: light_3},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_2: enc_light_2, hash_3: enc_light_3},
          map_shard_to_cache={
              1: {hash_2}, 2:{hash_1}, 'backlog': {hash_3}},
          shard_active_count={0: 0, 1: 2, 2: 1, 'backlog': 3},
          shard_heap=[(0, 0), (1, 2), (2, 1)])

      hash_3, record_3, light_3, _, enc_light_3 = _create_cache_record(
          prompt='p3', responses=['r2', 'r3', 'r4'], shard_id=1, call_count=3,
          all_values=True)
      shard_manager._update_cache_record(record_3)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_2: light_2, hash_3: light_3},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_2: enc_light_2, hash_3: enc_light_3},
          map_shard_to_cache={1: {hash_2, hash_3}, 2:{hash_1}},
          shard_active_count={0: 0, 1: 6, 2: 1, 'backlog': 0},
          shard_heap=[(0, 0), (1, 2), (6, 1)])

      hash_2, record_2, light_2, _, enc_light_2 = _create_cache_record(
          prompt='p2', responses=['r2', 'r3', 'r4'], shard_id='not_important',
          call_count=3, all_values=True)
      shard_manager._update_cache_record(record_2, delete_only=True)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={hash_1: light_1, hash_3: light_3},
          enc_light_cache_records={hash_1: enc_light_1, hash_3: enc_light_3},
          map_shard_to_cache={1: {hash_3}, 2:{hash_1}},
          shard_active_count={0: 0, 1: 4, 2: 1, 'backlog': 0},
          shard_heap=[(0, 0), (1, 2), (4, 1)])

      hash_2, record_2, light_2, _, enc_light_2 = _create_cache_record(
          prompt='p2', responses=['r2', 'r3', 'r4'], shard_id=0,
          call_count=3, all_values=True)
      shard_manager._update_cache_record(record_2)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_2: light_2, hash_3: light_3},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_2: enc_light_2, hash_3: enc_light_3},
          map_shard_to_cache={0: {hash_2}, 1: {hash_3}, 2:{hash_1}},
          shard_active_count={0: 4, 1: 4, 2: 1, 'backlog': 0},
          shard_heap=[(1, 2), (4, 0), (4, 1)])


  def test_save_load_light_cache_records(self):
    records = _get_example_records()
    with tempfile.TemporaryDirectory() as temp_dir:
      _save_shard_manager(
          path=temp_dir, records=records['all_light_cache_records'])
      (hash_1, light_1, enc_light_1,
       hash_2, light_2, enc_light_2,
       hash_3, light_3, enc_light_3,
       hash_4, light_4, enc_light_4) = _unpack_records(records=records)

      # Regular load check
      load_shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      _check_shard_manager_state(
          load_shard_manager,
          light_cache_records=records['light_cache_records'],
          enc_light_cache_records=records['enc_light_cache_records'],
          enc_light_cache_records_backup=records['enc_all_light_cache_records'],
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_2}, 2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 2},
          shard_heap=[(1, 0), (2, 1), (3, 2)])

      # Backup file check
      load_shard_manager._save_light_cache_records()
      _check_shard_manager_state(
          load_shard_manager,
          light_cache_records=records['light_cache_records'],
          enc_light_cache_records=records['enc_light_cache_records'],
          enc_light_cache_records_backup=records['enc_light_cache_records'],
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_2}, 2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 2},
          shard_heap=[(1, 0), (2, 1), (3, 2)])

      # Recovery from backup file check
      os.remove(load_shard_manager.light_cache_records_path)
      load_shard_manager_2 = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      load_shard_manager_2._load_light_cache_records()
      _check_shard_manager_state(
          load_shard_manager_2,
          light_cache_records=records['light_cache_records'],
          enc_light_cache_records=records['enc_light_cache_records'],
          enc_light_cache_records_backup=records['enc_light_cache_records'],
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_2}, 2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 2},
          shard_heap=[(1, 0), (2, 1), (3, 2)])

      # No light_cache_records file
      os.remove(load_shard_manager.light_cache_records_path)
      os.remove(load_shard_manager.light_cache_records_path + '_backup')
      load_shard_manager_3 = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      _check_shard_manager_state(
          load_shard_manager_3,
          light_cache_records={},
          enc_light_cache_records={},
          enc_light_cache_records_backup={},
          map_shard_to_cache={},
          shard_active_count={0: 0, 1: 0, 2: 0, 'backlog': 0},
          shard_heap=[(0, 0), (0, 1), (0, 2)])

  def test_move_backlog_to_invalid_shard(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      with pytest.raises(ValueError):
        shard_manager._move_backlog_to_shard(shard_id=-1)

  def test_move_backlog_to_shard(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      records = _get_example_records(
          shard_dir=temp_dir, shard_count=_SHARD_COUNT)
      _save_shard_manager(
          path=temp_dir, records=records['all_light_cache_records'])
      (hash_1, light_1, enc_light_1,
       hash_2, light_2, enc_light_2,
       hash_3, light_3, enc_light_3,
       hash_4, light_4, enc_light_4) = _unpack_records(records=records)

      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      shard_manager._move_backlog_to_shard(shard_id=1)
      light_4.shard_id = 1
      enc_light_4 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_4)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_2: light_2,
              hash_3: light_3, hash_4: light_4},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_2: enc_light_2,
              hash_3: enc_light_3, hash_4: enc_light_4},
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_2, hash_4}, 2: {hash_3}},
          shard_active_count={0: 1, 1: 4, 2: 3, 'backlog': 0},
          shard_heap=[(1, 0), (3, 2), (4, 1)])

      prompt_records = _get_cache_record_from_prompt(
          prompts=['p2', 'p4'], records=records)
      prompt_records['enc_cache_records'][1]['shard_id'] = 1
      _check_file_json(
          filepath=shard_manager.shard_paths[1],
          expected=prompt_records['enc_cache_records'])

  def test_move_backlog_to_not_existed_shard(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      records = _get_example_records(
          shard_dir=temp_dir, shard_count=_SHARD_COUNT)
      save_shard_manager = _save_shard_manager(
          path=temp_dir, records=records['all_light_cache_records'])
      os.remove(save_shard_manager.shard_paths[1])
      (hash_1, light_1, enc_light_1,
       hash_2, light_2, enc_light_2,
       hash_3, light_3, enc_light_3,
       hash_4, light_4, enc_light_4) = _unpack_records(records=records)

      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      shard_manager._move_backlog_to_shard(shard_id=1)
      light_4.shard_id = 1
      enc_light_4 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_4)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_3: light_3, hash_4: light_4},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_3: enc_light_3, hash_4: enc_light_4},
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_4}, 2: {hash_3}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 0},
          shard_heap=[(1, 0), (2, 1), (3, 2)])

      prompt_records = _get_cache_record_from_prompt(
          prompts=['p4'], records=records)
      prompt_records['enc_cache_records'][0]['shard_id'] = 1
      _check_file_json(
          filepath=shard_manager.shard_paths[1],
          expected=prompt_records['enc_cache_records'])

  def test_move_not_existed_backlog_to_shard(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      records = _get_example_records(
          shard_dir=temp_dir, shard_count=_SHARD_COUNT)
      save_shard_manager = _save_shard_manager(
          path=temp_dir, records=records['all_light_cache_records'])
      os.remove(save_shard_manager.backlog_shard_path)
      (hash_1, light_1, enc_light_1,
       hash_2, light_2, enc_light_2,
       hash_3, light_3, enc_light_3,
       hash_4, light_4, enc_light_4) = _unpack_records(records=records)

      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      shard_manager._move_backlog_to_shard(shard_id=1)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_2: light_2, hash_3: light_3},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_2: enc_light_2, hash_3: enc_light_3},
          map_shard_to_cache={0: {hash_1}, 1: {hash_2}, 2: {hash_3}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 0},
          shard_heap=[(1, 0), (2, 1), (3, 2)])

      prompt_records = _get_cache_record_from_prompt(
          prompts=['p2'], records=records)
      _check_file_json(
          filepath=shard_manager.shard_paths[1],
          expected=prompt_records['enc_cache_records'])

  def test_add_to_backlog(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      records = _get_example_records(
          shard_dir=temp_dir, shard_count=_SHARD_COUNT)
      _save_shard_manager(
          path=temp_dir, records=records['all_light_cache_records'])
      (hash_1, light_1, enc_light_1,
       hash_2, light_2, enc_light_2,
       hash_3, light_3, enc_light_3,
       hash_4, light_4, enc_light_4) = _unpack_records(records=records)

      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      hash_5, record_5, light_5, _, enc_light_5 = _create_cache_record(
          prompt='p8', responses=['r8'], shard_id='not_important', call_count=1,
          all_values=True)
      shard_manager._add_to_backlog(copy.deepcopy(record_5))
      light_5.shard_id = 'backlog'
      enc_light_5 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_5)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_2: light_2, hash_3: light_3,
              hash_4: light_4, hash_5: light_5},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_2: enc_light_2, hash_3: enc_light_3,
              hash_4: enc_light_4, hash_5: enc_light_5},
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_2}, 2:{hash_3},
              'backlog': {hash_4, hash_5}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 4},
          shard_heap=[(1, 0), (2, 1), (3, 2)])

  def test_get_cache_record(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      records = _get_example_records(
          shard_dir=temp_dir, shard_count=_SHARD_COUNT)
      _save_shard_manager(
          path=temp_dir, records=records['all_light_cache_records'])

      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)

      # Test with hash value
      hash_value = _get_hash_from_prompt(prompt='p1', records=records)
      assert shard_manager.get_cache_record(
          hash_value) == records['cache_records'][hash_value]

      # Test no hash value
      query_record = types.QueryRecord(prompt='p1')
      assert shard_manager.get_cache_record(
          query_record) == records['cache_records'][
              _get_hash_from_prompt(prompt='p1', records=records)]

      # Test with hash value
      query_record = records['cache_records'][
          _get_hash_from_prompt(prompt='p2', records=records)].query_record
      assert shard_manager.get_cache_record(
          query_record) == records['cache_records'][
              _get_hash_from_prompt(prompt='p2', records=records)]

      # Test from backlog
      query_record = _get_cache_record_from_prompt(
          prompts=['p4'], records=records)['cache_records'][0].query_record
      assert shard_manager.get_cache_record(
          query_record) == records['cache_records'][
              _get_hash_from_prompt(prompt='p4', records=records)]

      # Test not existed
      query_record = types.QueryRecord(prompt='p8')
      assert shard_manager.get_cache_record(query_record) is None

  def test_delete_record(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      records = _get_example_records(
          shard_dir=temp_dir, shard_count=_SHARD_COUNT)
      _save_shard_manager(
          path=temp_dir, records=records['all_light_cache_records'])
      (hash_1, light_1, enc_light_1,
       hash_2, light_2, enc_light_2,
       hash_3, light_3, enc_light_3,
       hash_4, light_4, enc_light_4) = _unpack_records(records=records)

      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)

      # Test simple remove via hash value
      assert shard_manager.get_cache_record(hash_1) is not None
      shard_manager.delete_record(hash_1)
      assert shard_manager.get_cache_record(hash_1) is None
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_2: light_2, hash_3: light_3, hash_4: light_4},
          enc_light_cache_records={
              hash_2: enc_light_2, hash_3: enc_light_3, hash_4: enc_light_4},
          map_shard_to_cache={1: {hash_2}, 2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 0, 1: 2, 2: 3, 'backlog': 2},
          shard_heap=[(0, 0), (2, 1), (3, 2)])

      # Test simple remove via query record
      cache_record = types.CacheRecord(
          query_record=types.QueryRecord(prompt='p2'))
      assert shard_manager.get_cache_record(
          cache_record.query_record) is not None
      shard_manager.delete_record(cache_record)
      assert shard_manager.get_cache_record(
          cache_record.query_record) is None
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={hash_3: light_3, hash_4: light_4},
          enc_light_cache_records={hash_3: enc_light_3, hash_4: enc_light_4},
          map_shard_to_cache={2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 0, 1: 0, 2: 3, 'backlog': 2},
          shard_heap=[(0, 0), (0, 1), (3, 2)])

      # Test simple remove via query record hash value
      assert shard_manager.get_cache_record(hash_3) is not None
      shard_manager.delete_record(light_3)
      assert shard_manager.get_cache_record(hash_3) is None
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={hash_4: light_4},
          enc_light_cache_records={hash_4: enc_light_4},
          map_shard_to_cache={'backlog': {hash_4}},
          shard_active_count={0: 0, 1: 0, 2: 0, 'backlog': 2},
          shard_heap=[(0, 0), (0, 1), (0, 2)])

      # Test remove from backlog
      assert shard_manager.get_cache_record(hash_4) is not None
      shard_manager.delete_record(hash_4)
      assert shard_manager.get_cache_record(hash_4) is None
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={},
          enc_light_cache_records={},
          map_shard_to_cache={},
          shard_active_count={0: 0, 1: 0, 2: 0, 'backlog': 0},
          shard_heap=[(0, 0), (0, 1), (0, 2)])

      # Test delete not existed
      shard_manager.delete_record(cache_record='invalid_hash_value')
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={},
          enc_light_cache_records={},
          map_shard_to_cache={},
          shard_active_count={0: 0, 1: 0, 2: 0, 'backlog': 0},
          shard_heap=[(0, 0), (0, 1), (0, 2)])

  def test_save_record(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      records = _get_example_records(
          shard_dir=temp_dir, shard_count=_SHARD_COUNT)
      _save_shard_manager(
          path=temp_dir, records=records['all_light_cache_records'])
      (hash_1, light_1, enc_light_1,
       hash_2, light_2, enc_light_2,
       hash_3, light_3, enc_light_3,
       hash_4, light_4, enc_light_4) = _unpack_records(records=records)

      # Delete some records to open space for new records
      shard_manager = query_cache.ShardManager(
          path=temp_dir, shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE)
      shard_manager.delete_record(hash_2)
      shard_manager.delete_record(hash_3)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={hash_1: light_1, hash_4: light_4},
          enc_light_cache_records={hash_1: enc_light_1,  hash_4: enc_light_4},
          map_shard_to_cache={0: {hash_1}, 'backlog': {hash_4}},
          shard_active_count={0: 1, 1: 0, 2: 0, 'backlog': 2},
          shard_heap=[(0, 1), (0, 2), (1, 0)])

      # Test new record to backlog
      hash_5, record_5, light_5, _, enc_light_5 = _create_cache_record(
          prompt='p8', responses=[], shard_id='not_important',
          call_count=0, all_values=True)
      shard_manager.save_record(record_5)
      light_5.shard_id = 'backlog'
      enc_light_5 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_5)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_4: light_4, hash_5: light_5},
          enc_light_cache_records={
              hash_1: enc_light_1,  hash_4: enc_light_4, hash_5: enc_light_5},
          map_shard_to_cache={
              0: {hash_1}, 'backlog': {hash_4, hash_5}},
          shard_active_count={0: 1, 1: 0, 2: 0, 'backlog': 3},
          shard_heap=[(0, 1), (0, 2), (1, 0)])

      # Test backlog overload
      hash_6, record_6, light_6, _, enc_light_6 = _create_cache_record(
          prompt='p9', responses=['r9', 'r10'], shard_id='not_important',
          call_count=0, all_values=True)
      shard_manager.save_record(record_6)
      light_4.shard_id = 1
      enc_light_4 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_4)
      light_5.shard_id = 1
      enc_light_5 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_5)
      light_6.shard_id = 'backlog'
      enc_light_6 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_6)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_4: light_4,
              hash_5: light_5, hash_6: light_6},
          enc_light_cache_records={
              hash_1: enc_light_1,  hash_4: enc_light_4,
              hash_5: enc_light_5, hash_6: enc_light_6},
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_4, hash_5}, 'backlog': {hash_6}},
          shard_active_count={0: 1, 1: 3, 2: 0, 'backlog': 3},
          shard_heap=[(0, 2), (1, 0), (3, 1)])

      # Test override record
      hash_1, record_1, light_1, _, enc_light_1 = _create_cache_record(
          prompt='p1', responses=['new_r1'], shard_id='not_important',
          call_count=1, all_values=True)
      shard_manager.save_record(record_1)
      light_1.shard_id = 'backlog'
      enc_light_1 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_1)
      light_6.shard_id = 0
      enc_light_6 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_6)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_4: light_4,
              hash_5: light_5, hash_6: light_6},
          enc_light_cache_records={
              hash_1: enc_light_1,  hash_4: enc_light_4,
              hash_5: enc_light_5, hash_6: enc_light_6},
          map_shard_to_cache={
              0: {hash_6}, 1: {hash_4, hash_5}, 'backlog': {hash_1}},
          shard_active_count={0: 3, 1: 3, 2: 0, 'backlog': 2},
          shard_heap=[(0, 2), (3, 0), (3, 1)])

      # Test override backlog record
      hash_1, record_1, light_1, _, enc_light_1 = _create_cache_record(
          prompt='p1', responses=['new_r1', 'new_r2', 'new_r3'],
          shard_id='not_important', call_count=3, all_values=True)
      shard_manager.save_record(record_1)
      light_1.shard_id = 'backlog'
      enc_light_1 = query_cache.BaseQueryCache._encode_light_cache_record(
          light_1)
      _check_shard_manager_state(
          shard_manager,
          light_cache_records={
              hash_1: light_1, hash_4: light_4,
              hash_5: light_5, hash_6: light_6},
          enc_light_cache_records={
              hash_1: enc_light_1,  hash_4: enc_light_4,
              hash_5: enc_light_5, hash_6: enc_light_6},
          map_shard_to_cache={
              0: {hash_6}, 1: {hash_4, hash_5}, 'backlog': {hash_1}},
          shard_active_count={0: 3, 1: 3, 2: 0, 'backlog': 4},
          shard_heap=[(0, 2), (3, 0), (3, 1)])


class TestQueryCache:
  def test_push_record_heap_empty_dir(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      query_cache_manager = query_cache.QueryCacheManager(
          cache_options=types.CacheOptions(path=temp_dir),
          shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE,
          cache_response_size=10)
      _check_record_heap(query_cache_manager, [])

      # First record
      record_1 = _create_cache_record(
          prompt='p1', responses=[], shard_id='not_important', call_count=0)
      hash_1 = query_cache.BaseQueryCache._get_query_record_hash(
          query_record=record_1.query_record)
      query_cache_manager._push_record_heap(cache_record=record_1)
      _check_record_heap(query_cache_manager, [hash_1])

      # Second record
      query_cache_manager._push_record_heap(
          cache_record=_create_cache_record(
              prompt='p2', responses=[], shard_id=0, call_count=0))
      _check_record_heap(query_cache_manager, [
          query_cache.BaseQueryCache._get_query_record_hash(
              query_record=types.QueryRecord(prompt='p1')),
          query_cache.BaseQueryCache._get_query_record_hash(
              query_record=types.QueryRecord(prompt='p2')),])

      # Third record
      query_cache_manager._push_record_heap(
          cache_record=_create_cache_record(
              prompt='p3', responses=[], shard_id=0, call_count=0))
      _check_record_heap(query_cache_manager, [
          query_cache.BaseQueryCache._get_query_record_hash(
              query_record=types.QueryRecord(prompt='p1')),
          query_cache.BaseQueryCache._get_query_record_hash(
              query_record=types.QueryRecord(prompt='p2')),
          query_cache.BaseQueryCache._get_query_record_hash(
              query_record=types.QueryRecord(prompt='p3')),])

      # Override first record
      query_cache_manager._push_record_heap(
          cache_record=_create_cache_record(
              prompt='p1', responses=['r1'], shard_id=0, call_count=0))
      _check_record_heap(query_cache_manager, [
          query_cache.BaseQueryCache._get_query_record_hash(
              query_record=types.QueryRecord(prompt='p2')),
          query_cache.BaseQueryCache._get_query_record_hash(
              query_record=types.QueryRecord(prompt='p3')),
          query_cache.BaseQueryCache._get_query_record_hash(
              query_record=types.QueryRecord(prompt='p1')),])

  def test_push_record_heap_existing_path(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      cache_path = os.path.join(temp_dir, query_cache.CACHE_DIR)
      os.makedirs(cache_path)
      records = _get_example_records(
          shard_dir=cache_path,
          shard_count=_SHARD_COUNT)
      _save_shard_manager(
          path=cache_path, records=records['all_light_cache_records'])

      query_cache_manager = query_cache.QueryCacheManager(
          cache_options=types.CacheOptions(path=temp_dir),
          shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE,
          cache_response_size=10)
      _check_record_heap(
          query_cache_manager, [
              _get_hash_from_prompt(prompt='p1', records=records),
              _get_hash_from_prompt(prompt='p2', records=records),
              _get_hash_from_prompt(prompt='p3', records=records),
              _get_hash_from_prompt(prompt='p4', records=records)])

      # New value
      query_cache_manager._push_record_heap(
          cache_record=_create_cache_record(
              prompt='p5', responses=['r1'], shard_id=0, call_count=0))
      _check_record_heap(
          query_cache_manager, [
              _get_hash_from_prompt(prompt='p1', records=records),
              _get_hash_from_prompt(prompt='p2', records=records),
              _get_hash_from_prompt(prompt='p3', records=records),
              _get_hash_from_prompt(prompt='p4', records=records),
              query_cache.BaseQueryCache._get_query_record_hash(
                  query_record=types.QueryRecord(prompt='p5'))])

      # Override value
      query_cache_manager._push_record_heap(
          cache_record=_create_cache_record(
              prompt='p3', responses=['r1'], shard_id=0, call_count=0))
      _check_record_heap(
          query_cache_manager, [
              _get_hash_from_prompt(prompt='p1', records=records),
              _get_hash_from_prompt(prompt='p2', records=records),
              _get_hash_from_prompt(prompt='p4', records=records),
              query_cache.BaseQueryCache._get_query_record_hash(
                  query_record=types.QueryRecord(prompt='p5')),
              _get_hash_from_prompt(prompt='p3', records=records),])

  def test_push_record_heap_overflow(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      os.makedirs(os.path.join(temp_dir, query_cache.CACHE_DIR))
      query_cache_manager = query_cache.QueryCacheManager(
          cache_options=types.CacheOptions(path=temp_dir),
          shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE,
          cache_response_size=10)

      # record_1: size = 4, heap = [record_1]
      record_1 = _create_cache_record(
              prompt='p1', responses=['r1', 'r2', 'r3'],
              shard_id=0, call_count=0)
      hash_1 = query_cache.BaseQueryCache._get_query_record_hash(
          query_record=record_1.query_record)
      query_cache_manager._push_record_heap(cache_record=record_1)
      query_cache_manager._shard_manager.save_record(record_1)
      _check_record_heap(query_cache_manager, [hash_1])

      # record_2: size = 4, heap = [record_1, record_2]
      record_2 = _create_cache_record(
              prompt='p2', responses=['r1', 'r2', 'r3'],
              shard_id=0, call_count=0)
      hash_2 = query_cache.BaseQueryCache._get_query_record_hash(
          query_record=record_2.query_record)
      query_cache_manager._push_record_heap(cache_record=record_2)
      query_cache_manager._shard_manager.save_record(record_2)
      _check_record_heap(query_cache_manager, [hash_1, hash_2])

      # record_3: size = 4, heap = [record_2, record_3]
      record_3 = _create_cache_record(
              prompt='p3', responses=['r1', 'r2', 'r3'],
              shard_id=0, call_count=0)
      hash_3 = query_cache.BaseQueryCache._get_query_record_hash(
          query_record=record_3.query_record)
      query_cache_manager._push_record_heap(cache_record=record_3)
      query_cache_manager._shard_manager.save_record(record_3)
      _check_record_heap(query_cache_manager, [hash_2, hash_3])
      assert (
          set(query_cache_manager._shard_manager._light_cache_records.keys())
          == set([hash_2, hash_3]))

      # record_4: size = 2, heap = [record_2, record_3, record_4]
      record_4 = _create_cache_record(
              prompt='p4', responses=['r1'],
              shard_id=0, call_count=0)
      hash_4 = query_cache.BaseQueryCache._get_query_record_hash(
          query_record=record_4.query_record)
      query_cache_manager._push_record_heap(cache_record=record_4)
      query_cache_manager._shard_manager.save_record(record_4)
      _check_record_heap(query_cache_manager, [hash_2, hash_3, hash_4])
      assert (
          set(query_cache_manager._shard_manager._light_cache_records.keys())
          == set([hash_2, hash_3, hash_4]))

      # Override and overflow
      # record_2: size = 10, heap = [record_2]
      record_2 = _create_cache_record(
              prompt='p2', responses=[
                  'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9'],
              shard_id=0, call_count=0)
      query_cache_manager._push_record_heap(cache_record=record_2)
      query_cache_manager._shard_manager.save_record(record_2)
      _check_record_heap(query_cache_manager, [hash_2])
      assert (
          set(query_cache_manager._shard_manager._light_cache_records.keys())
          == set([hash_2]))

  def test_look(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      cache_dir = os.path.join(temp_dir, query_cache.CACHE_DIR)
      os.makedirs(cache_dir)
      records = _get_example_records(
          shard_dir=cache_dir, shard_count=_SHARD_COUNT)
      _save_shard_manager(
          path=cache_dir, records=records['all_light_cache_records'])

      query_cache_manager = query_cache.QueryCacheManager(
          cache_options=types.CacheOptions(path=temp_dir),
          shard_count=_SHARD_COUNT,
          response_per_file=_RESPONSE_PER_FILE,
          cache_response_size=10)

      (hash_1, light_1, enc_light_1,
       hash_2, light_2, enc_light_2,
       hash_3, light_3, enc_light_3,
       hash_4, light_4, enc_light_4) = _unpack_records(records=records)
      record_1 = records['cache_records'][hash_1]
      record_2 = records['cache_records'][hash_2]
      record_3 = records['cache_records'][hash_3]
      record_4 = records['cache_records'][hash_4]

      assert query_cache_manager.look(record_1) is None
      _check_record_heap(query_cache_manager, [
          hash_1, hash_2, hash_3, hash_4])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          light_cache_records={
              hash_1: light_1, hash_2: light_2,
              hash_3: light_3, hash_4: light_4},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_2: enc_light_2,
              hash_3: enc_light_3, hash_4: enc_light_4},
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_2}, 2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 2},
          shard_heap=[(1, 0), (2, 1), (3, 2)])

      assert query_cache_manager.look(record_1) is None
      _check_record_heap(query_cache_manager, [
          hash_1, hash_2, hash_3, hash_4])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          light_cache_records={
              hash_1: light_1, hash_2: light_2,
              hash_3: light_3, hash_4: light_4},
          enc_light_cache_records={
              hash_1: enc_light_1, hash_2: enc_light_2,
              hash_3: enc_light_3, hash_4: enc_light_4},
          map_shard_to_cache={
              0: {hash_1}, 1: {hash_2}, 2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 2},
          shard_heap=[(1, 0), (2, 1), (3, 2)])

      assert query_cache_manager.look(record_2).response == 'r1'
      _check_record_heap(query_cache_manager, [
          hash_1, hash_3, hash_4, hash_2])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          map_shard_to_cache={
              0: {hash_1}, 2: {hash_3}, 'backlog': {hash_2, hash_4}},
          shard_active_count={0: 1, 1: 0, 2: 3, 'backlog': 4},
          shard_heap=[(0, 1), (1, 0), (3, 2)])

      assert query_cache_manager.look(record_2).response == 'r1'
      _check_record_heap(query_cache_manager, [
          hash_1, hash_3, hash_4, hash_2])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          map_shard_to_cache={
              0: {hash_1}, 2: {hash_3}, 'backlog': {hash_2, hash_4}},
          shard_active_count={0: 1, 1: 0, 2: 3, 'backlog': 4},
          shard_heap=[(0, 1), (1, 0), (3, 2)])

      assert query_cache_manager.look(record_3).response == 'r2'
      _check_record_heap(query_cache_manager, [
          hash_1, hash_4, hash_2, hash_3])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          map_shard_to_cache={
              0: {hash_1}, 1:{hash_2, hash_4}, 'backlog': {hash_3}},
          shard_active_count={0: 1, 1: 4, 2: 0, 'backlog': 3},
          shard_heap=[(0, 2), (1, 0), (4, 1)])

      assert query_cache_manager.look(record_3).response == 'r3'
      _check_record_heap(query_cache_manager, [
          hash_1, hash_4, hash_2, hash_3])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          map_shard_to_cache={
              0: {hash_1}, 1:{hash_2, hash_4}, 'backlog': {hash_3}},
          shard_active_count={0: 1, 1: 4, 2: 0, 'backlog': 3},
          shard_heap=[(0, 2), (1, 0), (4, 1)])

      assert query_cache_manager.look(record_3).response == 'r2'
      _check_record_heap(query_cache_manager, [
          hash_1, hash_4, hash_2, hash_3])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          map_shard_to_cache={
              0: {hash_1}, 1:{hash_2, hash_4}, 'backlog': {hash_3}},
          shard_active_count={0: 1, 1: 4, 2: 0, 'backlog': 3},
          shard_heap=[(0, 2), (1, 0), (4, 1)])

      assert query_cache_manager.look(record_3).response == 'r3'
      _check_record_heap(query_cache_manager, [
          hash_1, hash_4, hash_2, hash_3])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          map_shard_to_cache={
              0: {hash_1}, 1:{hash_2, hash_4}, 'backlog': {hash_3}},
          shard_active_count={0: 1, 1: 4, 2: 0, 'backlog': 3},
          shard_heap=[(0, 2), (1, 0), (4, 1)])

      assert query_cache_manager.look(record_4).response == 'r4'
      _check_record_heap(query_cache_manager, [
          hash_1, hash_2, hash_3, hash_4])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          map_shard_to_cache={
              0: {hash_1}, 1:{hash_2}, 2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 2},
          shard_heap=[(1, 0), (2, 1), (3, 2)])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          map_shard_to_cache={
              0: {hash_1}, 1:{hash_2}, 2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 2},
          shard_heap=[(1, 0), (2, 1), (3, 2)])

      assert query_cache_manager.look(record_4).response == 'r4'
      _check_record_heap(query_cache_manager, [
          hash_1, hash_2, hash_3, hash_4])
      _check_shard_manager_state(
          query_cache_manager._shard_manager,
          map_shard_to_cache={
              0: {hash_1}, 1:{hash_2}, 2: {hash_3}, 'backlog': {hash_4}},
          shard_active_count={0: 1, 1: 2, 2: 3, 'backlog': 2},
          shard_heap=[(1, 0), (2, 1), (3, 2)])


# ShardManager: new methods tests

# QueryCacheManager: add new response to existing record
      # ? -> when return when return from cache(?) like there is 3 responses
      # should we add 4? Who will check that?

# Test for save record
#   -> corner cases like shard count is 1, file count is 1/0/-1, etc.



# Different cache, same hash bug
# If hits to the same hash??!!!
