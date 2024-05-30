import proxai.types as types
import proxai.serializers.hash_serializer as hash_serializer
import pytest


def _get_query_record_options():
  return [
      {'call_type': types.CallType.GENERATE_TEXT},
      {'model': (types.Provider.OPENAI, types.OpenAIModel.GPT_4)},
      {'prompt': 'Hello, world!'},
      {'system': 'Hello, system!'},
      {'messages': [{'role': 'user', 'content': 'Hello, user!'}]},
      {'max_tokens': 100},
      {'temperature': 0.5},
      {'stop': ['.', '?', '!']},
      {'call_type': types.CallType.GENERATE_TEXT,
       'model': (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
       'prompt': 'Hello, world!',
       'system': 'Hello, system!',
       'messages': [{'role': 'user', 'content': 'Hello, user!'}],
       'max_tokens': 100,
       'temperature': 0.5,
        'stop': ['.', '?', '!']},]


class TestBaseQueryCache:
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
