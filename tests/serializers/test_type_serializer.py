import datetime
import proxai.types as types
import proxai.stat_types as stat_types
import proxai.serializers.type_serializer as type_serializer
import proxai.serializers.hash_serializer as hash_serializer
import pytest


def _get_model_type_options():
  return [
      {'provider': types.Provider.OPENAI,
       'provider_model': types.OpenAIModel.GPT_4},
      {'provider': types.Provider.OPENAI,
       'provider_model': 'gpt-4'},
      {'provider': 'openai',
       'provider_model': types.OpenAIModel.GPT_4},
      {'provider': 'openai',
       'provider_model': 'gpt-4'},]


def _get_query_record_options():
  return [
      {'call_type': types.CallType.GENERATE_TEXT},
      {'model': ('openai', types.OpenAIModel.GPT_4),},
      {'model': (types.Provider.OPENAI, 'gpt-4'),},
      {'prompt': 'Hello, world!'},
      {'system': 'Hello, system!'},
      {'messages': [{'role': 'user', 'content': 'Hello, user!'}]},
      {'max_tokens': 100},
      {'temperature': 0.5},
      {'stop': ['stop']},
      {'call_type': types.CallType.GENERATE_TEXT,
       'model': (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
       'prompt': 'Hello, world!',
       'system': 'Hello, system!',
       'messages': [{'role': 'user', 'content': 'Hello, user!'}],
       'max_tokens': 100,
       'temperature': 0.5,
       'stop': ['stop']},]


def _get_query_response_record_options():
  return [
      {'response': 'Hello, world!'},
      {'error': 'Error message'},
      {'start_utc_date': datetime.datetime.now(datetime.timezone.utc)},
      {'end_utc_date': datetime.datetime.now(datetime.timezone.utc)},
      {'local_time_offset_minute': (
          datetime.datetime.now().astimezone().utcoffset().total_seconds()
          // 60) * -1},
      {'response_time': datetime.timedelta(seconds=1)},
      {'estimated_cost': 1},
      {'response': 'Hello, world!',
       'error': 'Error message',
       'start_utc_date': datetime.datetime.now(datetime.timezone.utc),
       'end_utc_date': datetime.datetime.now(datetime.timezone.utc),
       'local_time_offset_minute': (
          datetime.datetime.now().astimezone().utcoffset().total_seconds()
          // 60) * -1,
       'response_time': datetime.timedelta(seconds=1),
       'estimated_cost': 1},]


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


def _get_logging_record_options():
  return [
      {'query_record': types.QueryRecord(
          call_type=types.CallType.GENERATE_TEXT)},
      {'response_record': types.QueryResponseRecord(
          response='Hello, world!')},
      {'response_source': types.ResponseSource.CACHE},
      {'look_fail_reason': types.CacheLookFailReason.CACHE_NOT_FOUND},
      {'query_record': types.QueryRecord(
          call_type=types.CallType.GENERATE_TEXT),
       'response_record': types.QueryResponseRecord(
          response='Hello, world!'),
       'response_source': types.ResponseSource.CACHE,
       'look_fail_reason': types.CacheLookFailReason.CACHE_NOT_FOUND},]


def _get_model_status_options():
  model_1 = (types.Provider.OPENAI, types.OpenAIModel.GPT_4)
  model_2 = (types.Provider.OPENAI,
                  types.OpenAIModel.GPT_3_5_TURBO_INSTRUCT)
  model_3 = (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_OPUS)
  model_4 = (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_SONNET)
  return [
      {},
      {'unprocessed_models': {model_1}},
      {'working_models': {model_1, model_2}},
      {'failed_models': {model_1, model_2, model_3}},
      {'filtered_models': {model_1, model_2, model_3, model_4}},
      {'provider_queries': [
          types.LoggingRecord(
            query_record=types.QueryRecord(
                call_type=types.CallType.GENERATE_TEXT),
            response_record=types.QueryResponseRecord(
                response='Hello, world!'))]},
      {'unprocessed_models': {model_1},
       'working_models': {model_2},
       'failed_models': {model_3},
       'filtered_models': {model_4}},
       {'unprocessed_models': {model_1, model_2},
       'working_models': {model_2, model_3},
       'failed_models': {model_3, model_4},
       'filtered_models': {model_4, model_1},
       'provider_queries': [
          types.LoggingRecord(
            query_record=types.QueryRecord(
                call_type=types.CallType.GENERATE_TEXT),
            response_record=types.QueryResponseRecord(
                response='Hello, world!'))]}]


def _get_base_provider_stats_options():
  return [
    {'total_queries': 1},
    {'total_successes': 2},
    {'total_fails': 3},
    {'total_token_count': 4},
    {'total_query_token_count': 5},
    {'total_response_token_count': 6},
    {'total_successes': 1,
     'total_response_time': 7.0},
    {'estimated_cost': 8.0},
    {'total_cache_look_fail_reasons': {
        types.CacheLookFailReason.CACHE_NOT_FOUND: 9}},
    {'total_queries': 1,
     'total_successes': 2,
     'total_fails': 3,
     'total_token_count': 4,
     'total_query_token_count': 5,
     'total_response_token_count': 6,
     'total_successes': 1,
     'total_response_time': 7.0,
     'estimated_cost': 8.0,
     'total_cache_look_fail_reasons': {
        types.CacheLookFailReason.CACHE_NOT_FOUND: 9}}]


def _get_base_cache_stats_options():
  return [
    {'total_cache_hit': 1},
    {'total_success_return': 2},
    {'total_fail_return': 3},
    {'saved_token_count': 4},
    {'saved_query_token_count': 5},
    {'saved_response_token_count': 6},
    {'total_success_return': 1,
     'saved_total_response_time': 7.0},
    {'saved_estimated_cost': 8.0},
    {'total_cache_hit': 1,
     'total_success_return': 2,
     'total_fail_return': 3,
     'saved_token_count': 4,
     'saved_query_token_count': 5,
     'saved_response_token_count': 6,
     'saved_total_response_time': 7.0,
     'saved_estimated_cost': 8.0}]


def _get_model_stats_options():
  return [
    {'model': (types.Provider.OPENAI, types.OpenAIModel.GPT_4)},
    {'provider_stats': stat_types.BaseProviderStats(total_queries=1)},
    {'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)},
    {'model': (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
     'provider_stats': stat_types.BaseProviderStats(total_queries=1),
     'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)}]


def _get_provider_stats_options():
  return [
    {'provider': types.Provider.OPENAI},
    {'provider_stats': stat_types.BaseProviderStats(total_queries=1)},
    {'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)},
    {'models': {
        (types.Provider.OPENAI,
         types.OpenAIModel.GPT_4): stat_types.ModelStats()}},
    {'provider': types.Provider.OPENAI,
     'provider_stats': stat_types.BaseProviderStats(total_queries=1),
     'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1),
     'models': {
        (types.Provider.OPENAI,
          types.OpenAIModel.GPT_4): stat_types.ModelStats()}}]


def _get_run_stats_options():
  return [
    {'provider_stats': stat_types.BaseProviderStats(total_queries=1)},
    {'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)},
    {'provider_stats': stat_types.BaseProviderStats(total_queries=1),
     'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)}]


class TestTypeSerializer:
  @pytest.mark.parametrize('model_type_options', _get_model_type_options())
  def test_encode_decode_model_type(self, model_type_options):
    model_type = (model_type_options['provider'],
                  model_type_options['provider_model'])
    encoded_model_type = type_serializer.encode_model_type(model_type=model_type)
    decoded_model_type = type_serializer.decode_model_type(record=encoded_model_type)
    assert model_type == decoded_model_type

  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_get_query_record_hash(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    query_hash_value = hash_serializer.get_query_record_hash(
        query_record=query_record)

    query_record_options['max_tokens'] = 222
    query_record_2 = types.QueryRecord(**query_record_options)
    query_hash_value_2 = hash_serializer.get_query_record_hash(
        query_record=query_record_2)

    assert query_hash_value != query_hash_value_2
    assert query_hash_value == (
        hash_serializer.get_query_record_hash(
            query_record=query_record))

  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_encode_decode_query_record(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    encoded_query_record = type_serializer.encode_query_record(
        query_record=query_record)
    if ('provider' not in query_record_options
        and 'provider_model' in query_record_options):
      with pytest.raises(ValueError):
        _ = type_serializer.decode_query_record(
            record=encoded_query_record)
    else:
      decoded_query_record = type_serializer.decode_query_record(
          record=encoded_query_record)
      assert query_record == decoded_query_record

  @pytest.mark.parametrize(
      'query_response_record_options', _get_query_response_record_options())
  def test_encode_decode_query_response_record(
      self, query_response_record_options):
    query_response_record = types.QueryResponseRecord(
        **query_response_record_options)
    encoded_query_response_record = (
        type_serializer.encode_query_response_record(
            query_response_record=query_response_record))
    decoded_query_response_record = (
        type_serializer.decode_query_response_record(
            record=encoded_query_response_record))
    assert query_response_record == decoded_query_response_record

  @pytest.mark.parametrize('cache_record_options', _get_cache_record_options())
  def test_encode_decode_cache_record(self, cache_record_options):
    cache_record = types.CacheRecord(**cache_record_options)
    encoded_cache_record = type_serializer.encode_cache_record(
        cache_record=cache_record)
    decoded_cache_record = type_serializer.decode_cache_record(
        record=encoded_cache_record)
    assert cache_record == decoded_cache_record

  @pytest.mark.parametrize(
      'light_cache_record_options', _get_light_cache_record_options())
  def test_encode_decode_light_cache_record(self, light_cache_record_options):
    light_cache_record = types.LightCacheRecord(**light_cache_record_options)
    encoded_light_cache_record = (
        type_serializer.encode_light_cache_record(
            light_cache_record=light_cache_record))
    decoded_light_cache_record = (
        type_serializer.decode_light_cache_record(
            record=encoded_light_cache_record))
    assert light_cache_record == decoded_light_cache_record

  @pytest.mark.parametrize(
      'logging_record_options', _get_logging_record_options())
  def test_encode_decode_logging_record(self, logging_record_options):
    logging_record = types.LoggingRecord(**logging_record_options)
    encoded_logging_record = type_serializer.encode_logging_record(
        logging_record=logging_record)
    decoded_logging_record = type_serializer.decode_logging_record(
        record=encoded_logging_record)
    assert logging_record == decoded_logging_record

  @pytest.mark.parametrize('model_status_options', _get_model_status_options())
  def test_encode_decode_model_status(self, model_status_options):
    model_status = types.ModelStatus(**model_status_options)
    encoded_model_status = type_serializer.encode_model_status(
        model_status=model_status)
    decoded_model_status = type_serializer.decode_model_status(
        record=encoded_model_status)
    assert model_status == decoded_model_status

  @pytest.mark.parametrize(
      'base_provider_stats_options', _get_base_provider_stats_options())
  def test_encode_decode_base_provider_stats(
      self, base_provider_stats_options):
    provider_stats = stat_types.BaseProviderStats(**base_provider_stats_options)
    encoded_provider_stats = type_serializer.encode_base_provider_stats(
        base_provider_stats=provider_stats)
    decoded_provider_stats = type_serializer.decode_base_provider_stats(
        record=encoded_provider_stats)
    assert provider_stats == decoded_provider_stats

  @pytest.mark.parametrize(
      'base_cache_stats_options', _get_base_cache_stats_options())
  def test_encode_decode_base_cache_stats(self, base_cache_stats_options):
    cache_stats = stat_types.BaseCacheStats(**base_cache_stats_options)
    encoded_cache_stats = type_serializer.encode_base_cache_stats(
        base_cache_stats=cache_stats)
    decoded_cache_stats = type_serializer.decode_base_cache_stats(
        record=encoded_cache_stats)
    assert cache_stats == decoded_cache_stats

  @pytest.mark.parametrize('model_stats_options', _get_model_stats_options())
  def test_encode_decode_model_stats(self, model_stats_options):
    model_stats = stat_types.ModelStats(**model_stats_options)
    encoded_model_stats = type_serializer.encode_model_stats(
        model_stats=model_stats)
    decoded_model_stats = type_serializer.decode_model_stats(
        record=encoded_model_stats)
    assert model_stats == decoded_model_stats

  @pytest.mark.parametrize(
      'provider_stats_options', _get_provider_stats_options())
  def test_encode_decode_provider_stats(self, provider_stats_options):
    provider_stats = stat_types.ProviderStats(**provider_stats_options)
    encoded_provider_stats = type_serializer.encode_provider_stats(
        provider_stats=provider_stats)
    decoded_provider_stats = type_serializer.decode_provider_stats(
        record=encoded_provider_stats)
    assert provider_stats == decoded_provider_stats

  @pytest.mark.parametrize('run_stats_options', _get_run_stats_options())
  def test_encode_decode_run_stats(self, run_stats_options):
    run_stats = stat_types.RunStats(**run_stats_options)
    encoded_run_stats = type_serializer.encode_run_stats(
        run_stats=run_stats)
    decoded_run_stats = type_serializer.decode_run_stats(
        record=encoded_run_stats)
    assert run_stats == decoded_run_stats
