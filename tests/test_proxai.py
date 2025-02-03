import os
import proxai.types as types
from proxai import proxai
import pytest
import tempfile
import proxai.caching.model_cache as model_cache

class MockFailingConnector:
    def __init__(self, *args, **kwargs):
      pass

    def generate_text(self, prompt):
      raise ValueError('Temp Error')


class TestRunType:
    def test_setup_run_type(self):
      proxai._set_run_type(types.RunType.TEST)
      assert proxai._RUN_TYPE == types.RunType.TEST


class TestInitExperimentPath:
  def test_valid_path(self):
    assert proxai._init_experiment_path('test_experiment') == 'test_experiment'

  def test_empty_string(self):
    assert proxai._init_experiment_path() is None

  def test_invalid_path(self):
    with pytest.raises(ValueError):
      proxai._init_experiment_path('////invalid_path')

  def test_invalid_type(self):
    with pytest.raises(TypeError):
      proxai._init_experiment_path(123)

  def test_global_init(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._init_experiment_path('test_experiment', global_init=True)
    assert proxai._EXPERIMENT_PATH == 'test_experiment'

  def test_global_init_none(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._init_experiment_path(None, global_init=True)
    assert proxai._EXPERIMENT_PATH is None

  def test_global_init_multiple(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._init_experiment_path('test_1', global_init=True)
    proxai._init_experiment_path('test_2', global_init=True)
    assert proxai._EXPERIMENT_PATH == 'test_2'


class TestInitLoggingPath:

  def test_valid_path(self):
    with tempfile.TemporaryDirectory() as logging_path:
      logging_options, root_logging_path = proxai._init_logging_options(
          logging_path=logging_path)
      assert logging_options.logging_path == logging_path
      assert root_logging_path == logging_path

  def test_experiment_path(self):
    with tempfile.TemporaryDirectory() as logging_path:
      logging_options, root_logging_path = proxai._init_logging_options(
          logging_path=logging_path, experiment_path='test_experiment')
      assert logging_options.logging_path == os.path.join(
          logging_path, 'test_experiment')
      assert root_logging_path == logging_path

  def test_logging_options(self):
    with tempfile.TemporaryDirectory() as logging_path:
      logging_options = types.LoggingOptions(logging_path=logging_path)
      logging_options, root_logging_path = proxai._init_logging_options(
          logging_options=logging_options)
      assert logging_options.logging_path == logging_path
      assert root_logging_path == logging_path

  def test_both_logging_path_and_logging_options(self):
    with tempfile.TemporaryDirectory() as logging_path:
      with pytest.raises(ValueError):
        proxai._init_logging_options(
            logging_path=logging_path,
            logging_options=types.LoggingOptions(logging_path=logging_path))

  def test_not_exist_logging_path(self):
    with pytest.raises(ValueError):
      proxai._init_logging_options(logging_path='not_exist_logging_path')

  def test_not_exist_logging_options(self):
    with pytest.raises(ValueError):
      logging_options = types.LoggingOptions(
          logging_path='not_exist_logging_path')
      proxai._init_logging_options(logging_options=logging_options)

  def test_global_init(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    with tempfile.TemporaryDirectory() as logging_path:
      proxai._init_logging_options(
          experiment_path='test_experiment',
          logging_path=logging_path,
          global_init=True)
      assert proxai._LOGGING_OPTIONS.logging_path == os.path.join(
          logging_path, 'test_experiment')
      assert proxai._ROOT_LOGGING_PATH == logging_path

  def test_inherit_options(self):
    with tempfile.TemporaryDirectory() as logging_path:
      base_options = types.LoggingOptions(
          logging_path=logging_path,
          stdout=True,
          hide_sensitive_content=True
      )
      result_options, _ = proxai._init_logging_options(
          logging_options=base_options)
      assert result_options.stdout == True
      assert result_options.hide_sensitive_content == True

  def test_creates_experiment_subdirectory(self):
    with tempfile.TemporaryDirectory() as root_path:
      logging_options, _ = proxai._init_logging_options(
          logging_path=root_path,
          experiment_path='new_subdir/new_subdir2'
      )
      expected_path_1 = os.path.join(root_path, 'new_subdir')
      expected_path_2 = os.path.join(expected_path_1, 'new_subdir2')
      assert os.path.exists(expected_path_1)
      assert os.path.exists(expected_path_2)
      assert logging_options.logging_path == expected_path_2

  def test_all_none(self):
    logging_options, root_logging_path = proxai._init_logging_options(
        experiment_path=None,
        logging_path=None,
        logging_options=None
    )
    assert logging_options.logging_path is None
    assert root_logging_path is None

  def test_global_cleanup(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    with tempfile.TemporaryDirectory() as logging_path:
      proxai._init_logging_options(
          logging_path=logging_path,
          global_init=True
      )
      assert proxai._ROOT_LOGGING_PATH == logging_path

    proxai._init_logging_options(
        logging_path=None,
        global_init=True
    )
    assert proxai._ROOT_LOGGING_PATH is None

  def test_default_options(self):
    logging_options, root_logging_path = proxai._init_logging_options()
    assert logging_options.logging_path is None
    assert root_logging_path is None


class TestInitProxdashOptions:
  def test_default_options(self):
    proxdash_options = proxai._init_proxdash_options()
    assert proxdash_options.stdout == False
    assert proxdash_options.hide_sensitive_content == False
    assert proxdash_options.disable_proxdash == False

  def test_custom_options(self):
    base_options = types.ProxDashOptions(
        stdout=True,
        hide_sensitive_content=True,
        disable_proxdash=True
    )
    proxdash_options = proxai._init_proxdash_options(
        proxdash_options=base_options)
    assert proxdash_options.stdout == True
    assert proxdash_options.hide_sensitive_content == True
    assert proxdash_options.disable_proxdash == True

  def test_global_init(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    base_options = types.ProxDashOptions(
        stdout=True,
        hide_sensitive_content=True,
        disable_proxdash=True
    )
    proxai._init_proxdash_options(
        proxdash_options=base_options,
        global_init=True)
    assert proxai._PROXDASH_OPTIONS.stdout == True
    assert proxai._PROXDASH_OPTIONS.hide_sensitive_content == True
    assert proxai._PROXDASH_OPTIONS.disable_proxdash == True

  def test_global_init_multiple(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    options_1 = types.ProxDashOptions(stdout=True)
    options_2 = types.ProxDashOptions(hide_sensitive_content=True)

    proxai._init_proxdash_options(proxdash_options=options_1, global_init=True)
    proxai._init_proxdash_options(proxdash_options=options_2, global_init=True)

    assert proxai._PROXDASH_OPTIONS.stdout == False
    assert proxai._PROXDASH_OPTIONS.hide_sensitive_content == True
    assert proxai._PROXDASH_OPTIONS.disable_proxdash == False

  def test_inherit_options(self):
    base_options = types.ProxDashOptions(
        stdout=True,
        hide_sensitive_content=True,
        disable_proxdash=True
    )
    result_options = proxai._init_proxdash_options(proxdash_options=base_options)
    assert result_options.stdout == True
    assert result_options.hide_sensitive_content == True
    assert result_options.disable_proxdash == True


class TestInitAllowMultiprocessing:
  def test_default_value(self):
    assert proxai._init_allow_multiprocessing() is None

  def test_valid_value(self):
    assert proxai._init_allow_multiprocessing(True) == True
    assert proxai._init_allow_multiprocessing(False) == False

  def test_global_init(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._init_allow_multiprocessing(True, global_init=True)
    assert proxai._ALLOW_MULTIPROCESSING == True

  def test_global_init_multiple(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._init_allow_multiprocessing(True, global_init=True)
    proxai._init_allow_multiprocessing(False, global_init=True)
    assert proxai._ALLOW_MULTIPROCESSING == False

  def test_no_global_init(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    original_value = proxai._ALLOW_MULTIPROCESSING
    proxai._init_allow_multiprocessing(True, global_init=False)
    assert proxai._ALLOW_MULTIPROCESSING == original_value


class TestInitStrictFeatureTest:
  def test_default_value(self):
    assert proxai._init_strict_feature_test() is None

  def test_valid_value(self):
    assert proxai._init_strict_feature_test(True) == True
    assert proxai._init_strict_feature_test(False) == False

  def test_global_init(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._init_strict_feature_test(True, global_init=True)
    assert proxai._STRICT_FEATURE_TEST == True

  def test_global_init_multiple(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._init_strict_feature_test(True, global_init=True)
    proxai._init_strict_feature_test(False, global_init=True)
    assert proxai._STRICT_FEATURE_TEST == False

  def test_no_global_init(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.connect()
    original_value = proxai._STRICT_FEATURE_TEST
    proxai._init_strict_feature_test(True, global_init=False)
    assert proxai._STRICT_FEATURE_TEST == original_value


class TestRegisterModel:
  def test_not_supported_provider(self):
    with pytest.raises(ValueError):
      proxai.set_model(
          generate_text=('not_supported_provider', 'not_supported_model'))

  def test_not_supported_model(self):
    with pytest.raises(ValueError):
      proxai.set_model(generate_text=('openai', 'not_supported_model'))

  def test_successful_register_model(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.set_model(generate_text=('openai', 'gpt-3.5-turbo'))
    assert proxai._REGISTERED_VALUES[types.CallType.GENERATE_TEXT] == (
        'openai', 'gpt-3.5-turbo')


class TestGenerateText:
  def _test_generate_text(self, model: types.ModelType):
    proxai._set_run_type(types.RunType.TEST)
    proxai.set_model(generate_text=model)
    print(proxai._REGISTERED_VALUES)
    assert proxai._REGISTERED_VALUES[types.CallType.GENERATE_TEXT] == model

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'
    assert model in proxai._INITIALIZED_MODEL_CONNECTORS
    assert proxai._INITIALIZED_MODEL_CONNECTORS[model] is not None

  def test_openai(self):
    self._test_generate_text(('openai', 'gpt-3.5-turbo'))

  def test_claude(self):
    self._test_generate_text(('claude', 'claude-3-opus-20240229'))

  def test_gemini(self):
    self._test_generate_text(('gemini', 'models/gemini-1.0-pro'))

  def test_cohere(self):
    self._test_generate_text(('cohere', 'command-r'))

  def test_databricks(self):
    self._test_generate_text(('databricks', 'databricks-dbrx-instruct'))

  def test_mistral(self):
    self._test_generate_text(('mistral', 'open-mistral-7b'))

  def test_hugging_face(self):
    self._test_generate_text(('hugging_face', 'google/gemma-7b-it'))


class TestAvailableModels:
  def test_filter_by_key(self):
    proxai._set_run_type(types.RunType.TEST)
    with tempfile.TemporaryDirectory() as cache_dir:
      proxai.connect(cache_path=cache_dir)
      available_models = proxai.get_available_models()
      available_models._providers_with_key = [
          types.Provider.OPENAI, types.Provider.CLAUDE]
      models = types.ModelStatus()
      available_models._get_all_models(
          models, call_type=types.CallType.GENERATE_TEXT)
      available_models._filter_by_provider_key(models)
      assert models.unprocessed_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW),
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
          (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_HAIKU),
          (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_OPUS),
          (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_SONNET)])

  def test_filter_by_cache(self):
    proxai._set_run_type(types.RunType.TEST)
    with tempfile.TemporaryDirectory() as cache_dir:
      proxai.connect(cache_path=cache_dir)
      save_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(cache_path=cache_dir))
      data = types.ModelStatus()
      data.working_models.add(
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4))
      data.failed_models.add(
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  model=(types.Provider.OPENAI, types.OpenAIModel.GPT_4)),
              response_record=types.QueryResponseRecord(response='response1')))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  model=(types.Provider.OPENAI,
                          types.OpenAIModel.GPT_4_TURBO_PREVIEW)),
              response_record=types.QueryResponseRecord(error='error1')))
      save_cache.update(
          model_status=data, call_type=types.CallType.GENERATE_TEXT)

      available_models = proxai.get_available_models()
      available_models._providers_with_key = [types.Provider.OPENAI]
      models = types.ModelStatus()
      available_models._get_all_models(
          models, call_type=types.CallType.GENERATE_TEXT)
      available_models._filter_by_provider_key(models)
      available_models._filter_by_cache(
          models, call_type=types.CallType.GENERATE_TEXT)
      assert models.unprocessed_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)])
      assert models.working_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4)])
      assert models.failed_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)])

  @pytest.mark.parametrize('allow_multiprocessing', [True, False])
  def test_test_models(self, allow_multiprocessing):
    proxai._set_run_type(types.RunType.TEST)
    with tempfile.TemporaryDirectory() as cache_dir:
      proxai.connect(
          cache_path=cache_dir,
          allow_multiprocessing=allow_multiprocessing)
      available_models = proxai.get_available_models()
      available_models._providers_with_key = [types.Provider.OPENAI]
      models = types.ModelStatus()
      available_models._get_all_models(
          models, call_type=types.CallType.GENERATE_TEXT)
      available_models._filter_by_provider_key(models)
      # Fail for GPT_3_5_TURBO model. Other openai models should work.
      proxai._INITIALIZED_MODEL_CONNECTORS[
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)] = (
              MockFailingConnector())
      available_models._test_models(
          models, call_type=types.CallType.GENERATE_TEXT)
      assert models.unprocessed_models == set()
      assert models.working_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)])
      assert models.failed_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)])

      load_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(cache_path=cache_dir))
      loaded_data = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
      assert loaded_data.working_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)])
      assert loaded_data.failed_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)])

      del proxai._INITIALIZED_MODEL_CONNECTORS[
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)]

  def test_generate_text(self):
    proxai._set_run_type(types.RunType.TEST)
    with tempfile.TemporaryDirectory() as cache_dir:
      proxai.connect(cache_path=cache_dir)
      available_models = proxai.get_available_models()
      available_models._providers_with_key = [types.Provider.OPENAI]
      models = available_models.generate_text()
      assert models == [
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)]

  def test_generate_text_filters(self):
    proxai._set_run_type(types.RunType.TEST)
    with tempfile.TemporaryDirectory() as cache_dir:
      proxai.connect(cache_path=cache_dir)

      # _filter_by_cache filter
      save_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(cache_path=cache_dir))
      data = types.ModelStatus()
      data.failed_models.add(
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW))
      data.provider_queries.append(
          types.LoggingRecord(
              query_record=types.QueryRecord(
                  model=(types.Provider.OPENAI,
                         types.OpenAIModel.GPT_4_TURBO_PREVIEW)),
              response_record=types.QueryResponseRecord(error='error1')))
      save_cache.update(
          model_status=data, call_type=types.CallType.GENERATE_TEXT)

      # _test_models filter
      proxai._INITIALIZED_MODEL_CONNECTORS[
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)] = (
              MockFailingConnector())

      available_models = proxai.get_available_models()

      # _filter_by_provider_key filter
      available_models._providers_with_key = [types.Provider.OPENAI]

      # Check that the failed model was filtered out
      models = available_models.generate_text()
      assert models == [
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4)]

      # Check cache memory values
      models = available_models._model_cache.get(
          call_type=types.CallType.GENERATE_TEXT)
      assert models.working_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4)])
      assert models.failed_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW),
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)])

      # Check cache file values
      load_cache = model_cache.ModelCache(
          cache_options=types.CacheOptions(cache_path=cache_dir))
      models = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
      assert models.working_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4)])
      assert models.failed_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW),
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)])

      del proxai._INITIALIZED_MODEL_CONNECTORS[
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)]
