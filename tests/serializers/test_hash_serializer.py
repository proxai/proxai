import proxai.types as types
import proxai.serializers.hash_serializer as hash_serializer
import pytest


def _get_query_record_options():
  return [
      {'call_type': types.CallType.GENERATE_TEXT},
      {'provider': types.Provider.OPENAI,},
      {'provider': 'openai',},
      {'provider_model': types.OpenAIModel.GPT_4,},
      {'provider_model': 'gpt-4',},
      {'max_tokens': 100},
      {'prompt': 'Hello, world!'},
      {'call_type': types.CallType.GENERATE_TEXT,
       'provider': types.Provider.OPENAI,
       'provider_model': types.OpenAIModel.GPT_4,
       'max_tokens': 100,
       'prompt': 'Hello, world!'},]


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
