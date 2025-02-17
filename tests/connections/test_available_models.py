from typing import Dict, Optional
import proxai.types as types
from proxai import proxai
import pytest
import tempfile
import proxai.connections.available_models as available_models
import proxai.caching.model_cache as model_cache
import proxai.connectors.model_registry as model_registry
import proxai.connectors.model_connector as model_connector
import proxai.connectors.mock_model_connector as mock_model_connector

class TestAvailableModels:
  cache_dir: Optional[
      tempfile.TemporaryDirectory] = None
  initialized_model_connectors: Optional[
      Dict[types.ModelType, model_connector.ModelConnector]] = None
  model_cache_manager: Optional[
      model_cache.ModelCacheManager] = None

  def _init_test_variables(self):
    if self.cache_dir is None:
      self.cache_dir = tempfile.TemporaryDirectory()
    if self.initialized_model_connectors is None:
      self.initialized_model_connectors = {}

  def _init_model_connector(self, model: types.ModelType):
    if model in self.initialized_model_connectors:
      return self.initialized_model_connectors[model]
    connector = model_registry.get_model_connector(model)
    self.initialized_model_connectors[model] = connector(
        run_type=types.RunType.TEST)
    return self.initialized_model_connectors[model]

  def _get_initialized_model_connectors(self):
    return self.initialized_model_connectors

  def _get_available_models(
        self,
        allow_multiprocessing: bool = True):
    self._init_test_variables()
    self.model_cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
    available_models_manager = available_models.AvailableModels(
        run_type=types.RunType.TEST,
        allow_multiprocessing=allow_multiprocessing,
        model_cache_manager=self.model_cache_manager,
        get_initialized_model_connectors=self._get_initialized_model_connectors,
        init_model_connector=self._init_model_connector,
    )
    return available_models_manager

  def _save_temp_cache_state(self):
    self._init_test_variables()
    save_cache = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
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

  def test_filter_by_key(self):
    available_models_manager = self._get_available_models()
    available_models_manager._providers_with_key = [
        types.Provider.OPENAI,
        types.Provider.CLAUDE]
    models = types.ModelStatus()
    available_models_manager._get_all_models(
        models, call_type=types.CallType.GENERATE_TEXT)
    available_models_manager._filter_by_provider_key(models)
    assert models.unprocessed_models == set([
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
        (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_HAIKU),
        (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_OPUS),
        (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_SONNET)])

  def test_filter_by_cache(self):
    self._save_temp_cache_state()
    available_models_manager = self._get_available_models()
    available_models_manager._providers_with_key = [
        types.Provider.OPENAI]
    models = types.ModelStatus()
    available_models_manager._get_all_models(
        models, call_type=types.CallType.GENERATE_TEXT)
    available_models_manager._filter_by_provider_key(models)
    available_models_manager._filter_by_cache(
        models, call_type=types.CallType.GENERATE_TEXT)
    assert models.unprocessed_models == set([
        (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)])
    assert models.working_models == set([
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4)])
    assert models.failed_models == set([
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)])

  @pytest.mark.parametrize('allow_multiprocessing', [True, False])
  def test_test_models(self, allow_multiprocessing):
    available_models_manager = self._get_available_models(
        allow_multiprocessing=allow_multiprocessing)
    available_models_manager._providers_with_key = [
        types.Provider.OPENAI,
        types.Provider.MOCK_FAILING_PROVIDER]
    models = types.ModelStatus()
    available_models_manager._get_all_models(
        models, call_type=types.CallType.GENERATE_TEXT)
    available_models_manager._filter_by_provider_key(models)

    available_models_manager._test_models(
        models, call_type=types.CallType.GENERATE_TEXT)

    assert models.unprocessed_models == set()
    assert models.working_models == set([
        (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)])
    assert models.failed_models == set([
        (types.Provider.MOCK_FAILING_PROVIDER,
         types.MockFailingModel.MOCK_FAILING_MODEL)])

    load_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
    loaded_data = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
    assert loaded_data.working_models == set([
        (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)])
    assert loaded_data.failed_models == set([
        (types.Provider.MOCK_FAILING_PROVIDER,
         types.MockFailingModel.MOCK_FAILING_MODEL)])

  def test_generate_text(self):
    available_models_manager = self._get_available_models()
    available_models_manager._providers_with_key = [
        types.Provider.OPENAI]
    models = available_models_manager.generate_text()
    assert models == [
        (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)]

  def test_generate_text_filters(self):
    self._save_temp_cache_state()
    available_models_manager = self._get_available_models()
    available_models_manager._providers_with_key = [
        types.Provider.OPENAI,
        types.Provider.MOCK_PROVIDER,
        types.Provider.MOCK_FAILING_PROVIDER]

    # Check that the failed model was filtered out
    models = available_models_manager.generate_text()
    assert models == [
        (types.Provider.MOCK_PROVIDER, types.MockModel.MOCK_MODEL),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4)]

    # Check cache memory values
    models = available_models_manager._model_cache_manager.get(
        call_type=types.CallType.GENERATE_TEXT)
    assert models.working_models == set([
        (types.Provider.MOCK_PROVIDER, types.MockModel.MOCK_MODEL),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4)])
    assert models.failed_models == set([
        (types.Provider.MOCK_FAILING_PROVIDER,
         types.MockFailingModel.MOCK_FAILING_MODEL),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)])

    # Check cache file values
    load_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
    models = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
    assert models.working_models == set([
        (types.Provider.MOCK_PROVIDER, types.MockModel.MOCK_MODEL),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4)])
    assert models.failed_models == set([
        (types.Provider.MOCK_FAILING_PROVIDER,
         types.MockFailingModel.MOCK_FAILING_MODEL),
        (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW)])
