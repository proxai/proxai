import datetime
import proxai.types as types
import proxai.caching.model_cache as model_cache
import pytest
import tempfile
import proxai.connectors.model_configs as model_configs


class TestModelCacheManager:
  def test_save_and_load(self):
    with tempfile.TemporaryDirectory() as cache_dir:
      save_cache = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(cache_path=cache_dir))
      data = types.ModelStatus()
      data.working_models.add(model_configs.ALL_MODELS['openai']['gpt-4'])
      data.working_models.add(
          model_configs.ALL_MODELS['claude']['claude-3-opus'])
      data.failed_models.add(model_configs.ALL_MODELS['gemini']['gemini-pro'])
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  provider_model=model_configs.ALL_MODELS['openai']['gpt-4']),
              response_record=types.QueryResponseRecord(
                  response='response1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  provider_model=model_configs.ALL_MODELS[
                      'claude']['claude-3-opus']),
              response_record=types.QueryResponseRecord(
                  response='response2',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      save_cache.update(data, types.CallType.GENERATE_TEXT)

      load_cache = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(cache_path=cache_dir))
      loaded_data = load_cache.get(types.CallType.GENERATE_TEXT)
      assert loaded_data == data

  def test_filter_duration(self):
    with tempfile.TemporaryDirectory() as cache_dir:
      save_cache = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(cache_path=cache_dir))

      data = types.ModelStatus()
      data.working_models.add(model_configs.ALL_MODELS['openai']['gpt-4'])
      data.failed_models.add(
          model_configs.ALL_MODELS['claude']['claude-3-opus'])
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  provider_model=model_configs.ALL_MODELS['openai']['gpt-4']),
              response_record=types.QueryResponseRecord(
                  response='response1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  provider_model=model_configs.ALL_MODELS['claude'][
                      'claude-3-opus']),
              response_record=types.QueryResponseRecord(
                  error='error1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      save_cache.update(
          model_status=data, call_type=types.CallType.GENERATE_TEXT)

      data = types.ModelStatus()
      data.working_models.add(model_configs.ALL_MODELS['gemini']['gemini-pro'])
      data.failed_models.add(model_configs.ALL_MODELS['cohere']['command-r'])
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  provider_model=model_configs.ALL_MODELS['openai']['gpt-4']),
              response_record=types.QueryResponseRecord(
                  response='response1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  provider_model=model_configs.ALL_MODELS[
                      'claude']['claude-3-opus']),
              response_record=types.QueryResponseRecord(
                  error='error1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))
      save_cache.update(
          model_status=data, call_type=types.CallType.GENERATE_TEXT)

      load_cache = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(
              cache_path=cache_dir,
              duration=10))
      loaded_data = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
      assert loaded_data.working_models == set(
          [model_configs.ALL_MODELS['gemini']['gemini-pro']])
      assert loaded_data.failed_models == set(
          [model_configs.ALL_MODELS['cohere']['command-r']])

  def test_clear_cache(self):
    with tempfile.TemporaryDirectory() as cache_dir:
      # Create initial cache with some data
      data = types.ModelStatus()
      data.working_models.add(model_configs.ALL_MODELS['openai']['gpt-4'])
      data.working_models.add(
          model_configs.ALL_MODELS['claude']['claude-3-opus'])
      data.failed_models.add(
          model_configs.ALL_MODELS['gemini']['gemini-pro'])
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  provider_model=model_configs.ALL_MODELS['openai']['gpt-4']),
              response_record=types.QueryResponseRecord(
                  response='response1',
                  end_utc_date=datetime.datetime.now(
                      datetime.timezone.utc) - datetime.timedelta(days=1))
          ))

      # First cache - populate cache
      save_cache = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(
              cache_path=cache_dir,
              clear_model_cache_on_connect=False))
      save_cache.update(data, types.CallType.GENERATE_TEXT)

      # Verify data exists
      loaded_data = save_cache.get(types.CallType.GENERATE_TEXT)
      assert loaded_data == data

      # Create new cache with clear_cache_on_connect=False
      load_cache = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(
              cache_path=cache_dir,
              clear_model_cache_on_connect=False))

      # Verify data still exists
      loaded_data = load_cache.get(types.CallType.GENERATE_TEXT)
      assert loaded_data == data

      # Create new cache with clear_cache_on_connect=True
      clear_cache = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(
              cache_path=cache_dir,
              clear_model_cache_on_connect=True))

      # Verify data is cleared
      loaded_data = clear_cache.get(types.CallType.GENERATE_TEXT)
      assert loaded_data.working_models == set()
      assert loaded_data.failed_models == set()
      assert loaded_data.provider_queries == []
