import copy
import dataclasses
import datetime
import json
import os

import proxai.serializers.type_serializer as type_serializer
import proxai.state_controllers.state_controller as state_controller
import proxai.types as types

AVAILABLE_MODELS_PATH = 'available_models.json'
_MODEL_CACHE_MANAGER_STATE_PROPERTY = '_model_cache_manager_state'


@dataclasses.dataclass
class ModelCacheManagerParams:
  """Initialization parameters for ModelCacheManager."""

  cache_options: types.CacheOptions | None = None


class ModelCacheManager(state_controller.StateControlled):
  """Manages caching of model availability status by call type."""

  _cache_options: types.CacheOptions
  _model_status_by_output_format_type: types.ModelStatusByOutputFormatType
  _model_cache_manager_state: types.ModelCacheManagerState

  def __init__(
      self, init_from_params: ModelCacheManagerParams | None = None,
      init_from_state: types.ModelCacheManagerState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

    self.set_property_value(
        'status', types.ModelCacheManagerStatus.INITIALIZING
    )

    if init_from_state:
      self.load_state(init_from_state)
    elif init_from_params:
      self.cache_options = init_from_params.cache_options
    self.init_status()

  def get_internal_state_property_name(self):
    """Return the name of the internal state property."""
    return _MODEL_CACHE_MANAGER_STATE_PROPERTY

  def get_internal_state_type(self):
    """Return the dataclass type used for state storage."""
    return types.ModelCacheManagerState

  def init_status(self):
    """Initialize the cache manager status based on configuration."""
    if self.cache_options is None:
      self.status = types.ModelCacheManagerStatus.CACHE_OPTIONS_NOT_FOUND
      self.model_status_by_output_format_type = None
      return

    if self.cache_options.disable_model_cache:
      self.status = types.ModelCacheManagerStatus.DISABLED
      self.model_status_by_output_format_type = None
      return

    if self.cache_options.cache_path is None:
      self.status = types.ModelCacheManagerStatus.CACHE_PATH_NOT_FOUND
      self.model_status_by_output_format_type = None
      return

    if not os.access(self.cache_options.cache_path, os.W_OK):
      self.status = types.ModelCacheManagerStatus.CACHE_PATH_NOT_WRITABLE
      self.model_status_by_output_format_type = None
      return

    self._load_from_cache_path()
    self.status = types.ModelCacheManagerStatus.WORKING

  @property
  def cache_path(self) -> str:
    if self.cache_options is None or self.cache_options.cache_path is None:
      return None
    return os.path.join(self.cache_options.cache_path, AVAILABLE_MODELS_PATH)

  @cache_path.setter
  def cache_path(self, value: str):
    raise ValueError('cache_path needs to be set through cache_options.')

  @property
  def status(self) -> types.ModelCacheManagerStatus:
    return self.get_property_value('status')

  @status.setter
  def status(self, value: types.ModelCacheManagerStatus):
    self.set_property_value('status', value)

  @property
  def cache_options(self) -> types.CacheOptions:
    return self.get_property_value('cache_options')

  @cache_options.setter
  def cache_options(self, value: types.CacheOptions):
    self.set_property_value('cache_options', value)

  @property
  def model_status_by_output_format_type(self) -> types.ModelStatusByOutputFormatType:
    if getattr(self, '_model_status_by_output_format_type', None) is None:
      self._model_status_by_output_format_type = {}
    return self._model_status_by_output_format_type

  @model_status_by_output_format_type.setter
  def model_status_by_output_format_type(self, value: types.ModelStatusByOutputFormatType):
    self._model_status_by_output_format_type = value

  def clear_cache(self):
    """Remove all cached model status data."""
    self.model_status_by_output_format_type = None
    if self.cache_path is None:
      return
    if os.path.exists(self.cache_path):
      os.remove(self.cache_path)

  def _save_to_cache_path(self):
    if self.cache_path is None:
      return
    data = copy.deepcopy(self.model_status_by_output_format_type)
    for call_value in data:
      data[call_value] = type_serializer.encode_model_status(data[call_value])
    with open(self.cache_path, 'w') as f:
      json.dump(data, f)

  def _load_from_cache_path(self):
    self.model_status_by_output_format_type = {}
    if self.cache_path is None:
      return
    if not os.path.exists(self.cache_path):
      return
    data = {}
    with open(self.cache_path) as f:
      try:
        data = json.load(f)
        for call_value in data:
          self.model_status_by_output_format_type[call_value] = (
              type_serializer.decode_model_status(data[call_value])
          )
      except Exception:
        error_message = (
            '_load_from_cache_path failed because of the parsing error.\n'
            '* Please check cache path is correct.\n'
            f'    > Cache path: {self.cache_path}\n'
            '* Try to clean the model cache on px.connect:\n'
            '    > px.connect(\n'
            '    >    cache_options=px.CacheOptions(\n'
            '    >        clear_model_cache_on_connect=True))\n'
            '* If the problem persists, delete the cache file and try again:\n'
            f'    > rm {self.cache_path};\n'
            '* Open bug report at https://github.com/proxai/proxai/issues'
        )
        raise ValueError(error_message) from None

  def _clean_model_from_tested_models(
      self, output_format_type: types.OutputFormatType,
      model: types.ProviderModelType
  ):
    """Cleans a model from tested models: working_models and failed_models."""
    if output_format_type not in self.model_status_by_output_format_type:
      return
    model_status = self.model_status_by_output_format_type[output_format_type]
    if model in model_status.working_models:
      model_status.working_models.remove(model)
    if model in model_status.failed_models:
      model_status.failed_models.remove(model)
    model_status.provider_queries.pop(model, None)

  def _clean_model_from_model_status(
      self, output_format_type: types.OutputFormatType,
      model: types.ProviderModelType
  ):
    if output_format_type not in self.model_status_by_output_format_type:
      return
    model_status = self.model_status_by_output_format_type[output_format_type]
    model_status.unprocessed_models.discard(model)
    model_status.working_models.discard(model)
    model_status.failed_models.discard(model)
    model_status.filtered_models.discard(model)
    model_status.provider_queries.pop(model, None)

  def get(self, output_format_type: types.OutputFormatType) -> types.ModelStatus:
    """Retrieve cached model status for a response type."""
    result = types.ModelStatus()
    if output_format_type not in self.model_status_by_output_format_type:
      return result

    if self.cache_options.model_cache_duration is None:
      return self.model_status_by_output_format_type[output_format_type]

    model_status = self.model_status_by_output_format_type[output_format_type]
    for model in list(model_status.provider_queries.keys()):
      if model not in model_status.provider_queries:
        continue
      provider_query = model_status.provider_queries[model]
      time_since_response = (
          datetime.datetime.now(datetime.timezone.utc) -
          provider_query.result.timestamp.end_utc_date
      ).total_seconds()
      if time_since_response > self.cache_options.model_cache_duration:
        self._clean_model_from_tested_models(
            output_format_type=output_format_type, model=model)

    self._save_to_cache_path()
    return model_status

  def update(
      self, model_status_updates: types.ModelStatus,
      output_format_type: types.OutputFormatType
  ):
    """Apply incremental updates to cached model status."""
    if output_format_type not in self.model_status_by_output_format_type:
      self.model_status_by_output_format_type[output_format_type] = types.ModelStatus()
    all_updated_models = (
        model_status_updates.unprocessed_models |
        model_status_updates.working_models |
        model_status_updates.failed_models |
        model_status_updates.filtered_models
    )
    for provider_query_model in model_status_updates.provider_queries:
      if provider_query_model not in all_updated_models:
        raise ValueError(
            f'Model {provider_query_model} is not in any of the unprocessed, '
            'working, failed, or filtered models. Please provide the provider '
            'model in one of the sets when updating provider_queries.'
        )

    for model in all_updated_models:
      self._clean_model_from_model_status(
          output_format_type=output_format_type, model=model)

    model_status = self.model_status_by_output_format_type[output_format_type]
    for model in model_status_updates.unprocessed_models:
      model_status.unprocessed_models.add(model)
    for model in model_status_updates.working_models:
      model_status.working_models.add(model)
    for model in model_status_updates.failed_models:
      model_status.failed_models.add(model)
    for model in model_status_updates.filtered_models:
      model_status.filtered_models.add(model)
    for provider_model, provider_query in (
        model_status_updates.provider_queries.items()
    ):
      model_status.provider_queries[provider_model] = provider_query
    self._save_to_cache_path()

  def save(
      self, model_status: types.ModelStatus,
      output_format_type: types.OutputFormatType
  ):
    """Replace cached model status for a response type."""
    self.model_status_by_output_format_type[output_format_type] = copy.deepcopy(
        model_status)
    self._save_to_cache_path()
