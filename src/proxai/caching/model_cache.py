import os
import copy
import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set
import proxai.types as types
import proxai.serializers.type_serializer as type_serializer
import json

AVAILABLE_MODELS_PATH = 'available_models.json'


class ModelCacheManager:
  _cache_options: types.CacheOptions
  _data: Dict[types.CallType, types.ModelStatus] = {}

  def __init__(self, cache_options: types.CacheOptions):
    self._data = {}
    self._cache_options = cache_options
    if self._cache_options.clear_model_cache_on_connect:
      self._clear_cache()
    if self._cache_path:
      self._load_from_cache()

  @property
  def _cache_path(self) -> str:
    if not self._cache_options.cache_path:
      return None
    return os.path.join(self._cache_options.cache_path, AVAILABLE_MODELS_PATH)

  def _clear_cache(self):
    if not self._cache_path:
      return
    if os.path.exists(self._cache_path):
      os.remove(self._cache_path)

  def _save_to_cache(self):
    if not self._cache_path:
      return
    data = copy.deepcopy(self._data)
    for call_value in data.keys():
      data[call_value] = type_serializer.encode_model_status(data[call_value])
    with open(self._cache_path, 'w') as f:
      json.dump(data, f)

  def _load_from_cache(self):
    if not os.path.exists(self._cache_path):
      return
    with open(self._cache_path, 'r') as f:
      self._data: Dict[str, Any] = json.load(f)
    for call_value in self._data.keys():
      self._data[call_value] = type_serializer.decode_model_status(
          self._data[call_value])

  def _clean_model(
      self,
      call_type: types.CallType,
      model: types.ModelType):
    if model in self._data[call_type].working_models:
      self._data[call_type].working_models.remove(model)
    if model in self._data[call_type].failed_models:
      self._data[call_type].failed_models.remove(model)
    for idx, provider_query in enumerate(
        self._data[call_type].provider_queries):
      if provider_query.query_record.model == model:
        del self._data[call_type].provider_queries[idx]
        break

  def get(self, call_type: str) -> types.ModelStatus:
    result = types.ModelStatus()
    if call_type not in self._data:
      return result
    for provider_query in copy.deepcopy(self._data[call_type].provider_queries):
      if self._cache_options.duration == None:
        continue
      passing_time = (datetime.datetime.now(datetime.timezone.utc)
                      - provider_query.response_record.end_utc_date
                      ).total_seconds()
      if passing_time > self._cache_options.duration:
        self._clean_model(call_type=call_type,
                          model=provider_query.query_record.model)
    self._save_to_cache()
    return self._data[call_type]

  def update(
        self,
        model_status: types.ModelStatus,
        call_type: str):
    if call_type not in self._data:
      self._data[call_type] = types.ModelStatus()
    for model in model_status.working_models:
      self._clean_model(call_type=call_type, model=model)
    for model in model_status.failed_models:
      self._clean_model(call_type=call_type, model=model)
    for model in model_status.working_models:
      self._data[call_type].working_models.add(model)
    for model in model_status.failed_models:
      self._data[call_type].failed_models.add(model)
    for provider_query in model_status.provider_queries:
      self._data[call_type].provider_queries.append(provider_query)
    self._save_to_cache()
