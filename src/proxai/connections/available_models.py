import datetime
import multiprocessing
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import proxai.types as types
import proxai.caching.model_cache as model_cache
from proxai.connectors.model_connector import ModelConnector


class AvailableModels:
  _model_cache: Optional[model_cache.ModelCache] = None
  _generate_text: Dict[types.ModelType, Any] = {}
  _get_cache_options: Callable[[], types.CacheOptions]
  _get_initialized_model_connectors: Callable[
      [], Dict[types.ModelType, ModelConnector]]
  _providers_with_key: Set[types.Provider] = set()

  def __init__(
      self,
      get_cache_options: Callable[[], types.CacheOptions],
      get_initialized_model_connectors: Callable[
          [], Dict[types.ModelType, ModelConnector]],
      init_model_connector: Callable[[types.ModelType], ModelConnector]):
    self._get_cache_options = get_cache_options
    self._get_initialized_model_connectors = get_initialized_model_connectors
    self._init_model_connector= init_model_connector
    self._load_provider_keys()

  def _load_provider_keys(self):
    for provider, provider_key_name in types.PROVIDER_KEY_MAP.items():
      provider_flag = True
      for key_name in provider_key_name:
        if key_name not in os.environ:
          provider_flag = False
          break
      if provider_flag:
        self._providers_with_key.add(provider)

  def generate_text(
      self,
      only_largest_models: bool = False,
      verbose: bool = False,
      failed_models: bool = False
  ) -> List[types.ModelType]:
    start_time = datetime.datetime.now()
    models = types.ModelStatus()
    self._get_all_models(models, call_type=types.CallType.GENERATE_TEXT)
    self._filter_by_provider_key(models)
    self._filter_by_cache(models, call_type=types.CallType.GENERATE_TEXT)

    # TODO: This very experimental and require proper design. One alternative is
    # registering models according to their sizes in px.types. Then, find
    # working largest model for each provider.
    if only_largest_models:
      _allowed_models = set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW),
          (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_OPUS),
          (types.Provider.GEMINI, types.GeminiModel.GEMINI_1_5_PRO_LATEST),
          (types.Provider.COHERE, types.CohereModel.COMMAND_R_PLUS),
          (types.Provider.DATABRICKS, types.DatabricksModel.DBRX_INSTRUCT),
          (types.Provider.DATABRICKS,
           types.DatabricksModel.LLAMA_3_70B_INSTRUCT),
          (types.Provider.MISTRAL, types.MistralModel.MISTRAL_LARGE_LATEST),
      ])
      for model in list(models.unprocessed_models):
        if model not in _allowed_models:
          models.unprocessed_models.remove(model)
          models.filtered_models.add(model)
      for model in list(models.working_models):
        if model not in _allowed_models:
          models.working_models.remove(model)
          models.filtered_models.add(model)

    print_flag = bool(verbose and models.unprocessed_models)
    if print_flag:
      print(f'From cache;\n'
            f'  {len(models.working_models)} models are working.\n'
            f'  {len(models.failed_models)} models are failed.')
      print(f'Running test for {len(models.unprocessed_models)} models.')
    self._test_models(models, call_type=types.CallType.GENERATE_TEXT)
    end_time = datetime.datetime.now()
    if print_flag:
      print(f'After test;\n'
            f'  {len(models.working_models)} models are working.\n'
            f'  {len(models.failed_models)} models are failed.')
      duration = (end_time - start_time).total_seconds()
      print(f'Test duration: {duration} seconds.')

    if failed_models:
      return self._format_set(models.failed_models)
    return self._format_set(models.working_models)

  def _get_all_models(self, models: types.ModelStatus, call_type: str):
    if call_type == types.CallType.GENERATE_TEXT:
      for provider, provider_models in types.GENERATE_TEXT_MODELS.items():
        for provider_model in provider_models:
          if provider_model not in (
              models.working_models
              | models.failed_models
              | models.filtered_models):
            models.unprocessed_models.add((provider, provider_model))

  def _filter_by_provider_key(self, models: types.ModelStatus):
    def _filter_set(
        model_set: Set[types.ModelType]
    ) -> Tuple[Set[types.ModelType], Set[types.ModelType]]:
      not_filtered = set()
      for model in model_set:
        provider, _ = model
        if provider in self._providers_with_key:
          not_filtered.add(model)
        else:
          models.filtered_models.add(model)
      return not_filtered
    models.unprocessed_models = _filter_set(models.unprocessed_models)
    models.working_models = _filter_set(models.working_models)
    models.failed_models = _filter_set(models.failed_models)

  def _filter_by_cache(
      self,
      models: types.ModelStatus,
      call_type: str):
    cache_options = self._get_cache_options()
    if not cache_options.path:
      return
    if not self._model_cache:
      self._model_cache = model_cache.ModelCache(cache_options)
    cache_result = self._model_cache.get(call_type=call_type)

    def _remove_model(model: types.ModelType):
      if model in models.unprocessed_models:
        models.unprocessed_models.remove(model)
      if model in models.working_models:
        models.working_models.remove(model)
      elif model in models.failed_models:
        models.failed_models.remove(model)

    for model in cache_result.working_models:
      if model not in models.filtered_models:
        _remove_model(model)
        models.working_models.add(model)
    for model in cache_result.failed_models:
      if model not in models.filtered_models:
        _remove_model(model)
        models.failed_models.add(model)

  @staticmethod
  def _test_generate_text(
      model: types.ModelType,
      model_connector: ModelConnector
  ) -> List[types.LoggingRecord]:
    start_time = datetime.datetime.now()
    prompt = 'Hello model!?'
    max_tokens = 100
    try:
      logging_record = model_connector.generate_text(
          model=model,
          prompt=prompt,
          max_tokens=max_tokens,
          use_cache=False)
      return logging_record
    except Exception as e:
      return types.LoggingRecord(
          query_record=types.QueryRecord(
              call_type=types.CallType.GENERATE_TEXT,
              model=model,
              prompt=prompt,
              max_tokens=max_tokens),
          response_record=types.QueryResponseRecord(
              error=str(e),
              error_traceback=traceback.format_exc(),
              start_time=start_time,
              end_time=datetime.datetime.now(),
              response_time=datetime.datetime.now() - start_time),
          response_source=types.ResponseSource.PROVIDER)

  def _test_models(self, models: types.ModelStatus, call_type: str):
    if not models.unprocessed_models:
      return
    initialized_model_connectors = self._get_initialized_model_connectors()
    for model in models.unprocessed_models:
      if model not in initialized_model_connectors:
        initialized_model_connectors[
            model] = self._init_model_connector(model)

    test_models = list(models.unprocessed_models)
    test_func = None
    if call_type == types.CallType.GENERATE_TEXT:
      test_func = self._test_generate_text
    else:
      raise ValueError(f'Call type not supported: {call_type}')
    pool = multiprocessing.Pool(processes=len(test_models))
    test_results = []
    for test_model in test_models:
      result = pool.apply_async(
          test_func,
          args=(test_model, initialized_model_connectors[test_model]))
      test_results.append(result)
    pool.close()
    pool.join()
    test_results: List[types.LoggingRecord] = [
        result.get() for result in test_results]
    update_models = types.ModelStatus()
    for logging_record in test_results:
      models.unprocessed_models.remove(logging_record.query_record.model)
      if logging_record.response_record.response != None:
        models.working_models.add(logging_record.query_record.model)
        update_models.working_models.add(logging_record.query_record.model)
      else:
        models.failed_models.add(logging_record.query_record.model)
        update_models.failed_models.add(logging_record.query_record.model)
      update_models.provider_queries.append(logging_record)
    cache_options = self._get_cache_options()
    if not cache_options.path:
      return
    if not self._model_cache:
      self._model_cache = model_cache.ModelCache(cache_options)
    self._model_cache.update(model_status=update_models, call_type=call_type)

  def _format_set(
      self,
      model_set: Set[types.ModelType]
  ) -> List[types.ModelType]:
    return sorted(list(model_set))
