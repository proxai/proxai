import datetime
import os
import re
import proxai.types as types
import proxai.caching.model_cache as model_cache
import pytest
import tempfile
import proxai.connectors.model_configs as model_configs
from typing import Optional


def _get_path_dir(temp_path: str):
  temp_dir = tempfile.TemporaryDirectory()
  path = os.path.join(temp_dir.name, temp_path)
  os.makedirs(path, exist_ok=True)
  return path, temp_dir


def _get_example_logging_record(
    model: types.ProviderModelType,
    response: Optional[str] = None,
    error: Optional[str] = None,
    long: bool = False):
  end_utc_date = datetime.datetime.now(datetime.timezone.utc)
  if long:
    end_utc_date = end_utc_date - datetime.timedelta(days=1)
  return types.LoggingRecord(
      query_record=types.QueryRecord(
          call_type=types.CallType.GENERATE_TEXT,
          prompt='hello',
          provider_model=model),
      response_record=types.QueryResponseRecord(
          response=response,
          error=error,
          end_utc_date=end_utc_date))


def _get_example_model_status():
  data = types.ModelStatus()
  model_configs_instance = model_configs.ModelConfigs()
  models = [
      model_configs_instance.get_provider_model(('openai', 'gpt-3.5-turbo')),
      model_configs_instance.get_provider_model(('claude', 'haiku-3.5')),
      model_configs_instance.get_provider_model(('openai', 'gpt-4')),
      model_configs_instance.get_provider_model(('claude', 'opus-4')),
      model_configs_instance.get_provider_model(('openai', 'gpt-4.1-mini')),
      model_configs_instance.get_provider_model(('claude', 'sonnet-4')),
      model_configs_instance.get_provider_model(('gemini', 'gemini-1.5-pro')),
      model_configs_instance.get_provider_model(('cohere', 'command-r'))
  ]

  data.unprocessed_models.add(models[0])
  data.unprocessed_models.add(models[1])
  data.working_models.add(models[2])
  data.working_models.add(models[3])
  data.failed_models.add(models[4])
  data.failed_models.add(models[5])
  data.filtered_models.add(models[6])
  data.filtered_models.add(models[7])

  data.provider_queries = {
      models[2]: _get_example_logging_record(
          models[2],
          response='model_2 response'),
      models[3]: _get_example_logging_record(
          models[3],
          response='model_3 response',
          long=True),
      models[4]: _get_example_logging_record(
          models[4],
          error='model_4 error'),
      models[5]: _get_example_logging_record(
          models[5],
          error='model_5 error',
          long=True),
  }

  return data, models


class TestModelCacheManagerGettersSetters:
  def test_cache_path(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    assert cache_manager.cache_path == os.path.join(
        cache_path, model_cache.AVAILABLE_MODELS_PATH)

    with pytest.raises(ValueError):
      cache_manager.cache_path = 'invalid_set_value'

  def test_status(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    assert cache_manager.status == types.ModelCacheManagerStatus.WORKING

    cache_manager.status = types.ModelCacheManagerStatus.INITIALIZING
    assert cache_manager.status == types.ModelCacheManagerStatus.INITIALIZING
    assert (
        cache_manager._model_cache_manager_state.status ==
        types.ModelCacheManagerStatus.INITIALIZING)

    cache_manager.status = types.ModelCacheManagerStatus.DISABLED
    assert cache_manager.status == types.ModelCacheManagerStatus.DISABLED
    assert (
        cache_manager._model_cache_manager_state.status ==
        types.ModelCacheManagerStatus.DISABLED)

  def test_cache_options(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    assert cache_manager.cache_options == types.CacheOptions(
        cache_path=cache_path)

    cache_manager.cache_options = types.CacheOptions(
        cache_path=cache_path,
        model_cache_duration=20)
    assert cache_manager.cache_options == types.CacheOptions(
        cache_path=cache_path,
        model_cache_duration=20)
    assert (
        cache_manager._model_cache_manager_state.cache_options ==
        types.CacheOptions(
            cache_path=cache_path,
            model_cache_duration=20))

  def test_cache_options_function(self):
    cache_path, _ = _get_path_dir('test_cache')
    dynamic_cache_options = types.CacheOptions(
        cache_path=cache_path)
    cache_manager = model_cache.ModelCacheManager(
        get_cache_options=lambda: dynamic_cache_options)
    assert cache_manager.cache_options == types.CacheOptions(
        cache_path=cache_path)
    assert (
        cache_manager._model_cache_manager_state.cache_options ==
        types.CacheOptions(
            cache_path=cache_path))

    dynamic_cache_options.model_cache_duration = 20
    assert (
        cache_manager._model_cache_manager_state.cache_options ==
        types.CacheOptions(
            cache_path=cache_path))

    assert cache_manager.cache_options == types.CacheOptions(
        cache_path=cache_path,
        model_cache_duration=20)
    assert (
        cache_manager._model_cache_manager_state.cache_options ==
        types.CacheOptions(
            cache_path=cache_path,
            model_cache_duration=20))

  def test_model_status_by_call_type(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    assert cache_manager.model_status_by_call_type == {}

    cache_manager.model_status_by_call_type = {
        types.CallType.GENERATE_TEXT: types.ModelStatus()
    }
    assert cache_manager.model_status_by_call_type == {
        types.CallType.GENERATE_TEXT: types.ModelStatus()
    }

    cache_manager.model_status_by_call_type = None
    assert cache_manager.model_status_by_call_type == {}


class TestModelCacheManagerInit:
  def test_init(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    assert cache_manager.cache_options == types.CacheOptions(
        cache_path=cache_path)
    assert cache_manager.model_status_by_call_type == {}

  def test_init_with_all_options(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(
            cache_path=cache_path,
            disable_model_cache=False,
            clear_model_cache_on_connect=True,
            model_cache_duration=20))
    assert cache_manager.cache_options == types.CacheOptions(
        cache_path=cache_path,
        disable_model_cache=False,
        clear_model_cache_on_connect=True,
        model_cache_duration=20)
    assert cache_manager.model_status_by_call_type == {}

  def test_init_invalid_combinations(self):
    cache_path, _ = _get_path_dir('test_cache')
    with pytest.raises(
        ValueError,
        match=(
            'Only one of cache_options or get_cache_options should be set '
            'while initializing the StateControlled object.')):
      model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(cache_path=cache_path),
          get_cache_options=lambda: types.CacheOptions(cache_path=cache_path))

    with pytest.raises(
        ValueError,
        match=(
            'init_state and other parameters cannot be set at the same time.')):
      model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(cache_path=cache_path),
          init_state=types.ModelCacheManagerState(
              status=types.ModelCacheManagerStatus.WORKING))

  def test_init_none_cache_options(self):
    cache_manager = model_cache.ModelCacheManager()
    assert cache_manager.cache_options is None
    assert (
        cache_manager.status ==
        types.ModelCacheManagerStatus.CACHE_OPTIONS_NOT_FOUND)
    assert cache_manager.cache_path is None

  def test_init_none_cache_path(self):
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions())
    assert cache_manager.cache_options == types.CacheOptions()
    assert (
        cache_manager.status ==
        types.ModelCacheManagerStatus.CACHE_PATH_NOT_FOUND)
    assert cache_manager.cache_path is None

  def test_init_disabled_model_cache(self):
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(disable_model_cache=True))
    assert cache_manager.cache_options == types.CacheOptions(
        disable_model_cache=True)
    assert cache_manager.status == types.ModelCacheManagerStatus.DISABLED

  def test_init_state(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_manager_state = types.ModelCacheManagerState(
        status=types.ModelCacheManagerStatus.WORKING,
        cache_options=types.CacheOptions(
            cache_path=cache_path,
            model_cache_duration=20),)
    cache_manager = model_cache.ModelCacheManager(
        init_state=cache_manager_state)
    assert cache_manager.cache_options == types.CacheOptions(
        cache_path=cache_path,
        model_cache_duration=20)
    assert cache_manager.status == types.ModelCacheManagerStatus.WORKING
    assert cache_manager.model_status_by_call_type == {}

  def test_init_corrupted_cache_file(self):
    cache_path, _ = _get_path_dir('test_cache')
    with open(os.path.join(
        cache_path, model_cache.AVAILABLE_MODELS_PATH), 'w') as f:
      f.write('invalid_json')
    with pytest.raises(
        ValueError,
        match='_load_from_cache_path failed because of the parsing error.*'):
      model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))


class TestModelCacheManager:
  def test_save_and_load(self):
    cache_path, temp_dir = _get_path_dir('test_cache')
    save_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    data, _ = _get_example_model_status()
    save_cache.update(data, types.CallType.GENERATE_TEXT)

    load_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    loaded_data = load_cache.get(types.CallType.GENERATE_TEXT)
    assert loaded_data == data

  def test_duration_filter(self):
    cache_path, temp_dir = _get_path_dir('test_cache')
    save_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    data, _ = _get_example_model_status()
    save_cache.update(data, types.CallType.GENERATE_TEXT)

    load_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(
            cache_path=cache_path,
            model_cache_duration=10000))
    loaded_data = load_cache.get(types.CallType.GENERATE_TEXT)

    new_data, models = _get_example_model_status()
    new_data.working_models.remove(models[3])
    new_data.failed_models.remove(models[5])
    new_data.provider_queries.pop(models[3])
    new_data.provider_queries.pop(models[5])

    assert loaded_data.unprocessed_models == new_data.unprocessed_models
    assert loaded_data.working_models == new_data.working_models
    assert loaded_data.failed_models == new_data.failed_models
    assert loaded_data.filtered_models == new_data.filtered_models
    assert set(loaded_data.provider_queries.keys()) == set(
        new_data.provider_queries.keys())

  def test_clear_cache(self):
    cache_path, temp_dir = _get_path_dir('test_cache')
    save_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    data, _ = _get_example_model_status()
    save_cache.update(data, types.CallType.GENERATE_TEXT)

    load_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    assert load_cache.get(types.CallType.GENERATE_TEXT) == data

    load_cache.clear_cache()
    assert load_cache.get(types.CallType.GENERATE_TEXT) == types.ModelStatus()

    load_cache_2 = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    assert load_cache_2.get(types.CallType.GENERATE_TEXT) == types.ModelStatus()

  def test_update(self):
    cache_path, temp_dir = _get_path_dir('test_cache')
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    data, models = _get_example_model_status()
    cache_manager.update(data, types.CallType.GENERATE_TEXT)

    # Round Robin update for even models
    updates = types.ModelStatus()
    updates.unprocessed_models.add(models[6])
    updates.working_models.add(models[0])
    updates.failed_models.add(models[2])
    updates.filtered_models.add(models[4])
    updates.provider_queries[models[0]] = _get_example_logging_record(
        models[0],
        response='model_0 response')
    updates.provider_queries[models[2]] = _get_example_logging_record(
        models[2],
        error='model_2 error')

    cache_manager.update(updates, types.CallType.GENERATE_TEXT)
    result_data = types.ModelStatus(
        unprocessed_models={models[1], models[6]},
        working_models={models[0], models[3]},
        failed_models={models[2], models[5]},
        filtered_models={models[4], models[7]},
        provider_queries={
            models[0]: _get_example_logging_record(
                models[0],
                response='model_0 response'),
            models[2]: _get_example_logging_record(
                models[2],
                error='model_2 error'),
            models[3]: _get_example_logging_record(
                models[3],
                response='model_4 response',
                long=True),
            models[5]: _get_example_logging_record(
                models[5],
                error='model_6 error',
                long=True),
        })
    updated_data = cache_manager.get(types.CallType.GENERATE_TEXT)
    assert updated_data.unprocessed_models == result_data.unprocessed_models
    assert updated_data.working_models == result_data.working_models
    assert updated_data.failed_models == result_data.failed_models
    assert updated_data.filtered_models == result_data.filtered_models
    assert set(updated_data.provider_queries.keys()) == set(
        result_data.provider_queries.keys())
    # Check model 2 provider query is updated
    assert updated_data.provider_queries[
        models[2]].response_record.error == 'model_2 error'

  def test_update_invalid_provider_query(self):
    model_configs_instance = model_configs.ModelConfigs()
    cache_path, temp_dir = _get_path_dir('test_cache')
    cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    data, models = _get_example_model_status()
    cache_manager.update(data, types.CallType.GENERATE_TEXT)

    with pytest.raises(
        ValueError,
        match=re.escape(
            'Model (mistral, open-mistral-7b) is not in any of the '
            'unprocessed, working, failed, or filtered models. Please provide '
            'the provider model in one of the sets when updating '
            'provider_queries.')):
      cache_manager.update(
        types.ModelStatus(
            provider_queries={
                model_configs_instance.get_provider_model(('mistral', 'open-mistral-7b')):
                    _get_example_logging_record(
                        model_configs_instance.get_provider_model(('mistral', 'open-mistral-7b')),
                        response='model_1 response')
            }),
        types.CallType.GENERATE_TEXT)

  def test_handle_changes(self):
    cache_path, temp_dir = _get_path_dir('test_cache')
    save_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    data, models = _get_example_model_status()
    save_cache.update(data, types.CallType.GENERATE_TEXT)

    cache_manager = model_cache.ModelCacheManager()
    cache_manager.apply_state_changes(
      types.ModelCacheManagerState(
        status=types.ModelCacheManagerStatus.WORKING,
        cache_options=types.CacheOptions(cache_path=cache_path),
      )
    )
    assert cache_manager.model_status_by_call_type[
        types.CallType.GENERATE_TEXT] == data
    assert cache_manager.status == types.ModelCacheManagerStatus.WORKING
    assert (
        cache_manager._model_cache_manager_state.status ==
        types.ModelCacheManagerStatus.WORKING)
    assert cache_manager.cache_options == types.CacheOptions(
        cache_path=cache_path)
    assert (
        cache_manager._model_cache_manager_state.cache_options ==
        types.CacheOptions(cache_path=cache_path))

  def test_handle_changes_cache_options_not_found(self):
    cache_manager = model_cache.ModelCacheManager()
    cache_manager.apply_state_changes(
      types.ModelCacheManagerState(
        status=types.ModelCacheManagerStatus.WORKING,
      )
    )
    assert (
        cache_manager.status ==
        types.ModelCacheManagerStatus.CACHE_OPTIONS_NOT_FOUND)
    assert (
        cache_manager._model_cache_manager_state.status ==
        types.ModelCacheManagerStatus.CACHE_OPTIONS_NOT_FOUND)
    assert cache_manager.cache_options is None

  def test_handle_changes_same_state(self):
    cache_path, temp_dir = _get_path_dir('test_cache')
    data, models = _get_example_model_status()

    model_cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    assert model_cache_manager.model_status_by_call_type == {}

    save_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=cache_path))
    save_cache.update(data, types.CallType.GENERATE_TEXT)

    model_cache_manager.apply_state_changes(
      types.ModelCacheManagerState(
        status=types.ModelCacheManagerStatus.WORKING,
        cache_options=types.CacheOptions(cache_path=cache_path),
      )
    )
    # No changes because nothing changed in the state and load from cache path
    # is not called
    assert model_cache_manager.model_status_by_call_type == {}

    model_cache_manager.apply_state_changes(
      types.ModelCacheManagerState(
        status=types.ModelCacheManagerStatus.WORKING,
        cache_options=types.CacheOptions(
            cache_path=cache_path,
            model_cache_duration=10000000),
      )
    )
    # Load from cache path called because of model_cache_duration change
    assert model_cache_manager.model_status_by_call_type == {
        types.CallType.GENERATE_TEXT: data
    }
