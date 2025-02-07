import datetime
import proxai.types as types
import proxai.caching.model_cache as model_cache
import pytest
import tempfile


class TestModelCache:
  def test_save_and_load(self):
    with tempfile.TemporaryDirectory() as cache_dir:
      save_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(cache_path=cache_dir))
      data = types.ModelStatus()
      data.working_models.add(('openai', 'gpt-4'))
      data.working_models.add(('claude', types.ClaudeModel.CLAUDE_3_OPUS))
      data.failed_models.add(('gemini', types.GeminiModel.GEMINI_PRO))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  model=('openai', 'gpt-4')),
              response_record=types.QueryResponseRecord(
                  response='response1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  model=('claude', types.ClaudeModel.CLAUDE_3_OPUS)),
              response_record=types.QueryResponseRecord(
                  response='response2',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      save_cache.update(data, types.CallType.GENERATE_TEXT)

      load_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(cache_path=cache_dir))
      loaded_data = load_cache.get(types.CallType.GENERATE_TEXT)
      assert loaded_data == data

  def test_filter_duration(self):
    with tempfile.TemporaryDirectory() as cache_dir:
      save_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(cache_path=cache_dir))

      data = types.ModelStatus()
      data.working_models.add(('openai', 'gpt-4'))
      data.failed_models.add(('claude', types.ClaudeModel.CLAUDE_3_OPUS))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  model=('openai', 'gpt-4')),
              response_record=types.QueryResponseRecord(
                  response='response1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  model=('claude', types.ClaudeModel.CLAUDE_3_OPUS)),
              response_record=types.QueryResponseRecord(
                  error='error1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      save_cache.update(
          model_status=data, call_type=types.CallType.GENERATE_TEXT)

      data = types.ModelStatus()
      data.working_models.add(('gemini', types.GeminiModel.GEMINI_PRO))
      data.failed_models.add(('cohere', types.CohereModel.COMMAND_R))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  model=('openai', 'gpt-4')),
              response_record=types.QueryResponseRecord(
                  response='response1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  model=('claude', types.ClaudeModel.CLAUDE_3_OPUS)),
              response_record=types.QueryResponseRecord(
                  error='error1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      save_cache.update(
          model_status=data, call_type=types.CallType.GENERATE_TEXT)

      load_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(
              cache_path=cache_dir,
              duration=10))
      loaded_data = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
      assert loaded_data.working_models == set(
          [('gemini', types.GeminiModel.GEMINI_PRO)])
      assert loaded_data.failed_models == set(
          [('cohere', types.CohereModel.COMMAND_R)])

  def test_clear_cache(self):
    with tempfile.TemporaryDirectory() as cache_dir:
      # Create initial cache with some data
      data = types.ModelStatus()
      data.working_models.add(('openai', 'gpt-4'))
      data.working_models.add(('claude', types.ClaudeModel.CLAUDE_3_OPUS))
      data.failed_models.add(('gemini', types.GeminiModel.GEMINI_PRO))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  model=('openai', 'gpt-4')),
              response_record=types.QueryResponseRecord(
                  response='response1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))

      # First cache - populate cache
      save_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(
              cache_path=cache_dir,
              clear_model_cache_on_connect=False))
      save_cache.update(data, types.CallType.GENERATE_TEXT)

      # Verify data exists
      loaded_data = save_cache.get(types.CallType.GENERATE_TEXT)
      assert loaded_data == data

      # Create new cache with clear_cache_on_connect=False
      load_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(
              cache_path=cache_dir,
              clear_model_cache_on_connect=False))

      # Verify data still exists
      loaded_data = load_cache.get(types.CallType.GENERATE_TEXT)
      assert loaded_data == data

      # Create new cache with clear_cache_on_connect=True
      clear_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(
              cache_path=cache_dir,
              clear_model_cache_on_connect=True))

      # Verify data is cleared
      loaded_data = clear_cache.get(types.CallType.GENERATE_TEXT)
      assert loaded_data.working_models == set()
      assert loaded_data.failed_models == set()
      assert loaded_data.provider_queries == []
