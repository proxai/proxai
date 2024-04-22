import datetime
import proxai.types as types
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
      {'max_tokens': 100},
      {'prompt': 'Hello, world!'},
      {'call_type': types.CallType.GENERATE_TEXT,
       'model': (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
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
      {'provider_queries': [(
          types.QueryRecord(
              call_type=types.CallType.GENERATE_TEXT),
          types.QueryResponseRecord(
              response='Hello, world!'))]},
      {'unprocessed_models': {model_1},
       'working_models': {model_2},
       'failed_models': {model_3},
       'filtered_models': {model_4}},
       {'unprocessed_models': {model_1, model_2},
       'working_models': {model_2, model_3},
       'failed_models': {model_3, model_4},
       'filtered_models': {model_4, model_1},
       'provider_queries': [(
          types.QueryRecord(
              call_type=types.CallType.GENERATE_TEXT),
          types.QueryResponseRecord(
              response='Hello, world!'))]}]


class TestTypeSerializer:
  @pytest.mark.parametrize('model_type_options', _get_model_type_options())
  def test_encode_decode_model_type(self, model_type_options):
    model_type = (model_type_options['provider'],
                  model_type_options['provider_model'])
    encoded_model_type = type_serializer.encode_model_type(model_type=model_type)
    decoded_model_type = type_serializer.decode_model_type(record=encoded_model_type)
    assert model_type == decoded_model_type

  @pytest.mark.parametrize('model_status_options', _get_model_status_options())
  def test_encode_decode_model_status(self, model_status_options):
    model_status = types.ModelStatus(**model_status_options)
    encoded_model_status = type_serializer.encode_model_status(
        model_status=model_status)
    decoded_model_status = type_serializer.decode_model_status(
        record=encoded_model_status)
    assert model_status == decoded_model_status

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
