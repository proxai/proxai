import os
import copy
import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set
import proxai.types as types
import proxai.type_utils as type_utils
import json

AVAILABLE_MODELS_PATH = 'available_models.json'
LAST_TESTED = 'last_tested'
STATUS = 'status'

ModelCacheType = Dict[types.CallType, Dict[types.ModelType, Dict[str, Any]]]


@dataclass
class CacheOptions:
  path: Optional[str] = None
  duration: Optional[int] = 24 * 60 * 60


class ModelCache:
  _cache_options: CacheOptions
  _data: ModelCacheType = {}

  def __init__(self, cache_options: CacheOptions):
    self._data = {}
    self._cache_options = cache_options
    if self._cache_path:
      self._load_from_cache()

  @property
  def _cache_path(self) -> str:
    if not self._cache_options.path:
      return None
    return os.path.join(self._cache_options.path, AVAILABLE_MODELS_PATH)

  def _save_to_cache(self):
    if not self._cache_path:
      return
    data = copy.deepcopy(self._data)
    for call_value in data.values():
      for provider_value in call_value.values():
        for provider_model_value in provider_value.values():
          provider_model_value[LAST_TESTED] = type_utils.encode_datetime(
              provider_model_value[LAST_TESTED])
    with open(self._cache_path, 'w') as f:
      json.dump(data, f)

  def _load_from_cache(self):
    if not os.path.exists(self._cache_path):
      return
    with open(self._cache_path, 'r') as f:
      self._data: ModelCacheType = json.load(f)
    for call_value in self._data.values():
      for provider_value in call_value.values():
        for provider_model_value in provider_value.values():
          provider_model_value[LAST_TESTED] = type_utils.decode_datetime(
              provider_model_value[LAST_TESTED])

  def get(self, call_type: str) -> types.ModelStatus:
    result = types.ModelStatus()
    if call_type not in self._data:
      return result
    current_time = datetime.datetime.now()
    for provider, provider_value in self._data[call_type].items():
      for provider_model, provider_model_value in provider_value.items():
        model = (provider, provider_model)
        if self._cache_options.duration:
          if ((current_time - provider_model_value[LAST_TESTED]).total_seconds()
              > self._cache_options.duration):
            continue
        if provider_model_value[STATUS]:
          result.working_models.add(model)
        else:
          result.failed_models.add(model)
    return result

  def update(
        self,
        models: types.ModelStatus,
        call_type: str,
        update_time: Optional[datetime.datetime] = None):
    if not update_time:
      update_time = datetime.datetime.now()
    if call_type not in self._data:
      self._data[call_type] = {}

    def _update_model(model: types.ModelType, status: bool):
      provider, provider_model = model
      if provider not in self._data[call_type]:
        self._data[call_type][provider] = {}
      if provider_model not in self._data[call_type][provider]:
        self._data[call_type][provider][provider_model] = {}
      self._data[call_type][provider][provider_model][LAST_TESTED] = (
          update_time)
      self._data[call_type][provider][provider_model][STATUS] = status

    for model in models.working_models:
      _update_model(model, True)
    for model in models.failed_models:
      _update_model(model, False)
    self._save_to_cache()
