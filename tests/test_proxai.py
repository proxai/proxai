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
          cache_options=types.CacheOptions(path=cache_dir))
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
          cache_options=types.CacheOptions(
              path=cache_dir))
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
          cache_options=types.CacheOptions(
              path=cache_dir))
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
          cache_options=types.CacheOptions(
              path=cache_dir))
      models = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
      assert models.working_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4)])
      assert models.failed_models == set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW),
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)])

      del proxai._INITIALIZED_MODEL_CONNECTORS[
          (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)]
