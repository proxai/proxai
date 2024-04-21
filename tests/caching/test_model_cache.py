import datetime
import proxai.types as types
import proxai.caching.model_cache as model_cache
import pytest
import tempfile


class TestModelCache:
  def test_save_and_load(self):
    with tempfile.TemporaryDirectory() as cache_dir:
      save_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(path=cache_dir))
      data = types.ModelStatus()
      data.working_models.add(('provider1', 'provider_model1'))
      data.working_models.add(('provider2', 'provider_model2'))
      data.failed_models.add(('provider3', 'provider_model3'))
      save_cache.update(data, types.CallType.GENERATE_TEXT)

      load_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(path=cache_dir))
      loaded_data = load_cache.get(types.CallType.GENERATE_TEXT)
      assert loaded_data == data

  def test_filter_duration(self):
    update_time = datetime.datetime.now() - datetime.timedelta(seconds=200)
    with tempfile.TemporaryDirectory() as cache_dir:
      save_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(
              path=cache_dir))

      data = types.ModelStatus()
      data.working_models.add(('provider1', 'provider_model1'))
      data.failed_models.add(('provider2', 'provider_model2'))
      save_cache.update(
          models=data,
          call_type=types.CallType.GENERATE_TEXT,
          update_time=update_time)

      data = types.ModelStatus()
      data.working_models.add(('provider3', 'provider_model3'))
      data.failed_models.add(('provider4', 'provider_model4'))
      save_cache.update(
          models=data,
          call_type=types.CallType.GENERATE_TEXT)

      load_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(
              path=cache_dir,
              duration=10))
      loaded_data = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
      assert loaded_data.working_models == set(
          [('provider3', 'provider_model3')])
      assert loaded_data.failed_models == set(
          [('provider4', 'provider_model4')])
