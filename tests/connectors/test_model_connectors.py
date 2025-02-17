import pytest
import datetime
import tempfile
from typing import Dict, Optional
import proxai.types as types
from proxai.connectors.model_connector import ModelConnector
import proxai.caching.query_cache as query_cache
import proxai.stat_types as stats_type
import proxai.connections.proxdash as proxdash


class MockModelConnector(ModelConnector):
  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return query_record

  def _get_token_count(self, logging_record: types.LoggingRecord):
    return 100

  def _get_query_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def _get_response_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def _get_estimated_cost(self, logging_record: types.LoggingRecord):
    return 0.002

  def generate_text_proc(self, query_record: types.QueryRecord):
    return "mock response"


class TestModelConnector:
  @pytest.fixture
  def model_connector(self):
    return MockModelConnector(
        model=(types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
        run_type=types.RunType.TEST)

  def test_initialization(self, model_connector):
    assert model_connector.model == (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)
    assert model_connector.provider == types.Provider.OPENAI
    assert model_connector.provider_model == types.OpenAIModel.GPT_3_5_TURBO
    assert model_connector.run_type == types.RunType.TEST
    assert model_connector.strict_feature_test == False

  def test_initialization_with_invalid_params(self):
    with pytest.raises(ValueError):
        MockModelConnector(
            model=(types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
            run_type=types.RunType.TEST,
            query_cache_manager=query_cache.QueryCacheManager(
                cache_options=types.CacheOptions(cache_path="test_path")),
            get_query_cache_manager=lambda: query_cache.QueryCacheManager(
                cache_options=types.CacheOptions(cache_path="test_path")))

  def test_api_property(self, model_connector):
    assert model_connector.api is None  # Based on our mock implementation

    # Test caching behavior
    model_connector._api = "test_api"
    assert model_connector.api == "test_api"

  def test_feature_fail_strict(self):
    connector = MockModelConnector(
        model=(types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
        run_type=types.RunType.TEST,
        strict_feature_test=True)

    with pytest.raises(Exception, match="Test error"):
      connector.feature_fail("Test error")

  def test_feature_fail_non_strict(self, model_connector):
    # Should not raise an exception in non-strict mode
    model_connector.feature_fail("Test error")

  def test_generate_text(self, model_connector):
    result = model_connector.generate_text(
        prompt="Hello",
        max_tokens=100)

    assert isinstance(result, types.LoggingRecord)
    assert result.response_record.response == "mock response"
    assert result.query_record.prompt == "Hello"
    assert result.query_record.max_tokens == 100
    assert result.response_source == types.ResponseSource.PROVIDER

  def test_generate_text_with_cache(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      cache_manager = query_cache.QueryCacheManager(
          cache_options=types.CacheOptions(cache_path=temp_dir))

      connector = MockModelConnector(
          model=(types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
          run_type=types.RunType.TEST,
          query_cache_manager=cache_manager)

      # First call - should hit the provider
      result1 = connector.generate_text(prompt="Hello")
      assert result1.response_source == types.ResponseSource.PROVIDER

      # Second call - should hit the cache
      result2 = connector.generate_text(prompt="Hello")
      assert result2.response_source == types.ResponseSource.CACHE

  def test_stats_update(self):
    model = (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)
    stats = {
        stats_type.GlobalStatType.RUN_TIME: stats_type.ModelStats(model=model),
        stats_type.GlobalStatType.SINCE_CONNECT: stats_type.ModelStats(model=model)
    }

    connector = MockModelConnector(
        model=model,
        run_type=types.RunType.TEST,
        stats=stats)

    result = connector.generate_text(prompt="Hello")

    # Verify stats were updated
    run_time_stats = stats[stats_type.GlobalStatType.RUN_TIME]
    assert run_time_stats.provider_stats.total_queries == 1
    assert run_time_stats.provider_stats.total_successes == 1
    assert run_time_stats.provider_stats.total_token_count == 100

  def test_invalid_model_combination(self, model_connector):
    with pytest.raises(ValueError, match="Model provider does not match the connector provider"):
      model_connector.generate_text(
          prompt="Hello",
          model=(types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_OPUS))

  def test_mutually_exclusive_params(self, model_connector):
    with pytest.raises(ValueError, match="prompt and messages cannot be set at the same time"):
      model_connector.generate_text(
          prompt="Hello",
          messages=[{"role": "user", "content": "Hello"}])
