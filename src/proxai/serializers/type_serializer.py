import datetime
import json
from typing import Any, Dict
import proxai.types as types
import proxai.stat_types as stat_types


def encode_provider_model_type(
    provider_model_type: types.ProviderModelType) -> Dict[str, Any]:
  record = {}
  record['provider'] = provider_model_type.provider
  record['model'] = provider_model_type.model
  record['provider_model_identifier'] = (
      provider_model_type.provider_model_identifier)
  return record


def decode_provider_model_type(
    record: Dict[str, Any]) -> types.ProviderModelType:
  if 'provider' not in record:
    raise ValueError(f'Provider not found in record: {record=}')
  if 'model' not in record:
    raise ValueError(f'Model not found in record: {record=}')
  if 'provider_model_identifier' not in record:
    raise ValueError(
        f'Provider model identifier not found in record: {record=}')
  provider_model = types.ProviderModelType(
      provider=record['provider'],
      model=record['model'],
      provider_model_identifier=record['provider_model_identifier'])
  return provider_model


def encode_provider_model_identifier(
    provider_model_identifier: types.ProviderModelIdentifierType
) -> Dict[str, Any]:
  if isinstance(provider_model_identifier, types.ProviderModelType):
    return encode_provider_model_type(provider_model_identifier)
  else:
    # ProviderModelTupleType
    return {
        'provider': provider_model_identifier[0],
        'model': provider_model_identifier[1]
    }


def decode_provider_model_identifier(
    record: Dict[str, Any]) -> types.ProviderModelIdentifierType:
  if 'provider_model_identifier' in record:
    # Full ProviderModelType
    return decode_provider_model_type(record)
  else:
    # ProviderModelTupleType
    if 'provider' not in record:
      raise ValueError(f'Provider not found in record: {record=}')
    if 'model' not in record:
      raise ValueError(f'Model not found in record: {record=}')
    return (record['provider'], record['model'])


def encode_provider_model_pricing_type(
    provider_model_pricing_type: types.ProviderModelPricingType
) -> Dict[str, Any]:
  record = {}
  record['per_response_token_cost'] = (
      provider_model_pricing_type.per_response_token_cost)
  record['per_query_token_cost'] = (
      provider_model_pricing_type.per_query_token_cost)
  return record


def decode_provider_model_pricing_type(
    record: Dict[str, Any]) -> types.ProviderModelPricingType:
  if 'per_response_token_cost' not in record:
    raise ValueError(f'per_response_token_cost not found in record: {record=}')
  if 'per_query_token_cost' not in record:
    raise ValueError(f'per_query_token_cost not found in record: {record=}')
  return types.ProviderModelPricingType(
      per_response_token_cost=record['per_response_token_cost'],
      per_query_token_cost=record['per_query_token_cost'])


def encode_provider_model_feature_type(
    provider_model_feature_type: types.ProviderModelFeatureType
) -> Dict[str, Any]:
  record = {}
  if provider_model_feature_type.supported != None:
    record['supported'] = provider_model_feature_type.supported
  if provider_model_feature_type.best_effort != None:
    record['best_effort'] = provider_model_feature_type.best_effort
  if provider_model_feature_type.not_supported != None:
    record['not_supported'] = provider_model_feature_type.not_supported
  return record


def decode_provider_model_feature_type(
    record: Dict[str, Any]) -> types.ProviderModelFeatureType:
  provider_model_feature_type = types.ProviderModelFeatureType()
  if 'supported' in record:
    provider_model_feature_type.supported = record['supported']
  if 'best_effort' in record:
    provider_model_feature_type.best_effort = record['best_effort']
  if 'not_supported' in record:
    provider_model_feature_type.not_supported = record['not_supported']
  return provider_model_feature_type


def encode_provider_model_metadata_type(
    provider_model_metadata_type: types.ProviderModelMetadataType
) -> Dict[str, Any]:
  record = {}
  if provider_model_metadata_type.call_type != None:
    record['call_type'] = provider_model_metadata_type.call_type.value
  if provider_model_metadata_type.is_featured != None:
    record['is_featured'] = provider_model_metadata_type.is_featured
  if provider_model_metadata_type.model_size_tags != None:
    record['model_size_tags'] = [
        model_size_tag.value
        for model_size_tag in provider_model_metadata_type.model_size_tags]
  if provider_model_metadata_type.is_default_candidate != None:
    record['is_default_candidate'] = (
        provider_model_metadata_type.is_default_candidate)
  if provider_model_metadata_type.default_candidate_priority != None:
    record['default_candidate_priority'] = (
        provider_model_metadata_type.default_candidate_priority)
  if provider_model_metadata_type.tags != None:
    record['tags'] = provider_model_metadata_type.tags
  return record


def decode_provider_model_metadata_type(
    record: Dict[str, Any]) -> types.ProviderModelMetadataType:
  provider_model_metadata_type = types.ProviderModelMetadataType()
  if 'call_type' in record and record['call_type'] is not None:
    provider_model_metadata_type.call_type = types.CallType(record['call_type'])
  if 'is_featured' in record:
    provider_model_metadata_type.is_featured = record['is_featured']
  if 'model_size_tags' in record and record['model_size_tags'] is not None:
    provider_model_metadata_type.model_size_tags = [
        types.ModelSizeType(model_size_tag)
        for model_size_tag in record['model_size_tags']]
  if 'is_default_candidate' in record:
    provider_model_metadata_type.is_default_candidate = (
        record['is_default_candidate'])
  if 'default_candidate_priority' in record:
    provider_model_metadata_type.default_candidate_priority = (
        record['default_candidate_priority'])
  if 'tags' in record:
    provider_model_metadata_type.tags = record['tags']
  return provider_model_metadata_type


def encode_provider_model_config_type(
    provider_model_config_type: types.ProviderModelConfigType
) -> Dict[str, Any]:
  record = {}
  if provider_model_config_type.provider_model != None:
    record['provider_model'] = encode_provider_model_type(
        provider_model_config_type.provider_model)
  if provider_model_config_type.pricing != None:
    record['pricing'] = encode_provider_model_pricing_type(
        provider_model_config_type.pricing)
  if provider_model_config_type.features != None:
    record['features'] = encode_provider_model_feature_type(
        provider_model_config_type.features)
  if provider_model_config_type.metadata != None:
    record['metadata'] = encode_provider_model_metadata_type(
        provider_model_config_type.metadata)
  return record


def decode_provider_model_config_type(
    record: Dict[str, Any]) -> types.ProviderModelConfigType:
  provider_model_config_type = types.ProviderModelConfigType()
  if 'provider_model' in record:
    provider_model_config_type.provider_model = decode_provider_model_type(
        record['provider_model'])
  if 'pricing' in record:
    provider_model_config_type.pricing = decode_provider_model_pricing_type(
        record['pricing'])
  if 'features' in record:
    provider_model_config_type.features = decode_provider_model_feature_type(
        record['features'])
  if 'metadata' in record:
    provider_model_config_type.metadata = decode_provider_model_metadata_type(
        record['metadata'])
  return provider_model_config_type


def encode_provider_model_configs_type(
    provider_model_configs: types.ProviderModelConfigsType
) -> Dict[str, Any]:
  record = {}
  for provider, model_configs_dict in provider_model_configs.items():
    record[provider] = {}
    for model, provider_model_config in model_configs_dict.items():
      record[provider][model] = encode_provider_model_config_type(
          provider_model_config)
  return record


def decode_provider_model_configs_type(
    record: Dict[str, Any]) -> types.ProviderModelConfigsType:
  provider_model_configs = {}
  for provider, model_configs_dict_record in record.items():
    provider_model_configs[provider] = {}
    for model, provider_model_config_record in (
        model_configs_dict_record.items()):
      provider_model_configs[provider][model] = (
          decode_provider_model_config_type(provider_model_config_record))
  return provider_model_configs


def encode_featured_models_type(
    featured_models: types.FeaturedModelsType) -> Dict[str, Any]:
  record = {}
  for provider, provider_model_identifiers in featured_models.items():
    record[provider] = []
    for provider_model_identifier in provider_model_identifiers:
      record[provider].append(
          encode_provider_model_identifier(provider_model_identifier))
  return record


def decode_featured_models_type(
    record: Dict[str, Any]) -> types.FeaturedModelsType:
  featured_models = {}
  for provider, provider_model_identifier_records in record.items():
    provider_model_identifiers = []
    for provider_model_identifier_record in (
        provider_model_identifier_records):
      provider_model_identifiers.append(
          decode_provider_model_identifier(provider_model_identifier_record))
    featured_models[provider] = tuple(provider_model_identifiers)
  return featured_models


def encode_models_by_call_type_type(
    models_by_call_type: types.ModelsByCallTypeType) -> Dict[str, Any]:
  record = {}
  for call_type, provider_dict in models_by_call_type.items():
    record[call_type.value] = {}
    for provider, provider_model_identifiers in provider_dict.items():
      record[call_type.value][provider] = []
      for provider_model_identifier in provider_model_identifiers:
        record[call_type.value][provider].append(
            encode_provider_model_identifier(provider_model_identifier))
  return record


def decode_models_by_call_type_type(
    record: Dict[str, Any]) -> types.ModelsByCallTypeType:
  models_by_call_type = {}
  for call_type_str, provider_dict_record in record.items():
    call_type = types.CallType(call_type_str)
    provider_dict = {}
    for provider, provider_model_identifier_records in (
        provider_dict_record.items()):
      provider_model_identifiers = []
      for provider_model_identifier_record in (
          provider_model_identifier_records):
        provider_model_identifiers.append(
            decode_provider_model_identifier(provider_model_identifier_record))
      provider_dict[provider] = tuple(provider_model_identifiers)
    models_by_call_type[call_type] = provider_dict
  return models_by_call_type


def encode_models_by_size_type(
    models_by_size: types.ModelsBySizeType) -> Dict[str, Any]:
  record = {}
  for model_size, provider_model_identifiers in models_by_size.items():
    record[model_size.value] = []
    for provider_model_identifier in provider_model_identifiers:
      record[model_size.value].append(
          encode_provider_model_identifier(provider_model_identifier))
  return record


def decode_models_by_size_type(
    record: Dict[str, Any]) -> types.ModelsBySizeType:
  models_by_size = {}
  for model_size_str, provider_model_identifier_records in record.items():
    model_size = types.ModelSizeType(model_size_str)
    provider_model_identifiers = []
    for provider_model_identifier_record in (
        provider_model_identifier_records):
      provider_model_identifiers.append(
          decode_provider_model_identifier(provider_model_identifier_record))
    models_by_size[model_size] = tuple(provider_model_identifiers)
  return models_by_size


def encode_default_model_priority_list_type(
    default_model_priority_list: types.DefaultModelPriorityListType
) -> Dict[str, Any]:
  record = []
  for provider_model_identifier in default_model_priority_list:
    record.append(encode_provider_model_identifier(provider_model_identifier))
  return record


def decode_default_model_priority_list_type(
    record: Dict[str, Any]) -> types.DefaultModelPriorityListType:
  default_model_priority_list = []
  for provider_model_identifier_record in record:
    default_model_priority_list.append(
        decode_provider_model_identifier(provider_model_identifier_record))
  return tuple(default_model_priority_list)


def encode_model_configs_schema_metadata_type(
    model_configs_schema_metadata_type: types.ModelConfigsSchemaMetadataType
) -> Dict[str, Any]:
  record = {}
  if model_configs_schema_metadata_type.version != None:
    record['version'] = model_configs_schema_metadata_type.version
  if model_configs_schema_metadata_type.released_at != None:
    record['released_at'] = (
        model_configs_schema_metadata_type.released_at.isoformat())
  if model_configs_schema_metadata_type.min_proxai_version != None:
    record['min_proxai_version'] = (
        model_configs_schema_metadata_type.min_proxai_version)
  if model_configs_schema_metadata_type.config_origin != None:
    record['config_origin'] = (
        model_configs_schema_metadata_type.config_origin.value)
  if model_configs_schema_metadata_type.release_notes != None:
    record['release_notes'] = model_configs_schema_metadata_type.release_notes
  return record


def decode_model_configs_schema_metadata_type(
    record: Dict[str, Any]) -> types.ModelConfigsSchemaMetadataType:
  model_configs_schema_metadata_type = types.ModelConfigsSchemaMetadataType()
  if 'version' in record:
    model_configs_schema_metadata_type.version = record['version']
  if 'released_at' in record:
    model_configs_schema_metadata_type.released_at = (
        datetime.datetime.fromisoformat(record['released_at']))
  if 'min_proxai_version' in record:
    model_configs_schema_metadata_type.min_proxai_version = (
        record['min_proxai_version'])
  if 'config_origin' in record:
    model_configs_schema_metadata_type.config_origin = (
        types.ConfigOriginType(record['config_origin']))
  if 'release_notes' in record:
    model_configs_schema_metadata_type.release_notes = record['release_notes']
  return model_configs_schema_metadata_type


def encode_model_configs_schema_version_config_type(
    model_configs_schema_version_config_type: (
        types.ModelConfigsSchemaVersionConfigType)
) -> Dict[str, Any]:
  record = {}
  if model_configs_schema_version_config_type.provider_model_configs != None:
    record['provider_model_configs'] = encode_provider_model_configs_type(
        model_configs_schema_version_config_type.provider_model_configs)
  if model_configs_schema_version_config_type.featured_models != None:
    record['featured_models'] = encode_featured_models_type(
        model_configs_schema_version_config_type.featured_models)
  if model_configs_schema_version_config_type.models_by_call_type != None:
    record['models_by_call_type'] = encode_models_by_call_type_type(
        model_configs_schema_version_config_type.models_by_call_type)
  if model_configs_schema_version_config_type.models_by_size != None:
    record['models_by_size'] = encode_models_by_size_type(
        model_configs_schema_version_config_type.models_by_size)
  if model_configs_schema_version_config_type.default_model_priority_list != None:
    record['default_model_priority_list'] = (
        encode_default_model_priority_list_type(
            model_configs_schema_version_config_type.default_model_priority_list))
  return record


def decode_model_configs_schema_version_config_type(
    record: Dict[str, Any]) -> types.ModelConfigsSchemaVersionConfigType:
  model_configs_schema_version_config_type = (
      types.ModelConfigsSchemaVersionConfigType())
  if 'provider_model_configs' in record:
    model_configs_schema_version_config_type.provider_model_configs = (
        decode_provider_model_configs_type(record['provider_model_configs']))
  if 'featured_models' in record:
    model_configs_schema_version_config_type.featured_models = (
        decode_featured_models_type(record['featured_models']))
  if 'models_by_call_type' in record:
    model_configs_schema_version_config_type.models_by_call_type = (
        decode_models_by_call_type_type(record['models_by_call_type']))
  if 'models_by_size' in record:
    model_configs_schema_version_config_type.models_by_size = (
        decode_models_by_size_type(record['models_by_size']))
  if 'default_model_priority_list' in record:
    model_configs_schema_version_config_type.default_model_priority_list = (
        decode_default_model_priority_list_type(
            record['default_model_priority_list']))
  return model_configs_schema_version_config_type


def encode_model_configs_schema_type(
    model_configs_schema_type: types.ModelConfigsSchemaType
) -> Dict[str, Any]:
  record = {}
  if model_configs_schema_type.metadata != None:
    record['metadata'] = encode_model_configs_schema_metadata_type(
        model_configs_schema_type.metadata)
  if model_configs_schema_type.version_config != None:
    record['version_config'] = encode_model_configs_schema_version_config_type(
        model_configs_schema_type.version_config)
  return record


def decode_model_configs_schema_type(
    record: Dict[str, Any]) -> types.ModelConfigsSchemaType:
  model_configs_schema_type = types.ModelConfigsSchemaType()
  if 'metadata' in record:
    model_configs_schema_type.metadata = (
        decode_model_configs_schema_metadata_type(record['metadata']))
  if 'version_config' in record:
    model_configs_schema_type.version_config = (
        decode_model_configs_schema_version_config_type(
            record['version_config']))
  return model_configs_schema_type


def encode_response_format_pydantic_value(
    pydantic_value: types.ResponseFormatPydanticValue) -> Dict[str, Any]:
  record = {}
  if (pydantic_value.class_json_schema_value != None and
      pydantic_value.class_value != None):
    raise ValueError(
        'ResponseFormatPydanticValue cannot have both '
        'class_json_schema_value and class_value set.')
  json_schema = None
  if pydantic_value.class_value != None:
    json_schema = pydantic_value.class_value.model_json_schema()
  elif pydantic_value.class_json_schema_value != None:
    json_schema = pydantic_value.class_json_schema_value
  if json_schema != None:
    record['class_json_schema_value'] = json.dumps(
        json_schema,
        sort_keys=True)
  if pydantic_value.class_name != None:
    record['class_name'] = pydantic_value.class_name
  return record


def decode_response_format_pydantic_value(
    record: Dict[str, Any]) -> types.ResponseFormatPydanticValue:
  pydantic_value = types.ResponseFormatPydanticValue()
  pydantic_value.class_name = record.get('class_name', None)
  if 'class_json_schema_value' in record:
    pydantic_value.class_json_schema_value = json.loads(
        record['class_json_schema_value'])
  return pydantic_value


def encode_response_format(
    response_format: types.ResponseFormat) -> Dict[str, Any]:
  record = {}
  if response_format.type != None:
    record['type'] = response_format.type.value
  if response_format.value != None:
    if response_format.type == types.ResponseFormatType.TEXT:
      pass
    elif response_format.type == types.ResponseFormatType.JSON:
      pass
    elif response_format.type == types.ResponseFormatType.JSON_SCHEMA:
      record['value'] = json.dumps(
          response_format.value,
          sort_keys=True)
    elif response_format.type == types.ResponseFormatType.PYDANTIC:
      record.update(encode_response_format_pydantic_value(response_format.value))
  return record


def decode_response_format(
    record: Dict[str, Any]) -> types.ResponseFormat:
  response_format = types.ResponseFormat()
  if 'type' in record:
    response_format.type = types.ResponseFormatType(record['type'])
  if response_format.type == types.ResponseFormatType.TEXT:
    pass
  elif response_format.type == types.ResponseFormatType.JSON:
    pass
  elif response_format.type == types.ResponseFormatType.JSON_SCHEMA:
    if 'value' in record:
      response_format.value = json.loads(record['value'])
  elif response_format.type == types.ResponseFormatType.PYDANTIC:
    response_format.value = decode_response_format_pydantic_value(record)
  return response_format


def encode_response_pydantic_value(
    pydantic_value: types.ResponsePydanticValue) -> Dict[str, Any]:
  record = {}
  if (pydantic_value.instance_json_value != None and
      pydantic_value.instance_value != None):
    raise ValueError(
        'ResponsePydanticValue cannot have both '
        'instance_json_value and instance_value set.')
  instance_json = None
  if pydantic_value.instance_value != None:
    instance_json = pydantic_value.instance_value.model_dump()
  elif pydantic_value.instance_json_value != None:
    instance_json = pydantic_value.instance_json_value
  if instance_json != None:
    record['instance_json_value'] = json.dumps(
        instance_json,
        sort_keys=True)
  if pydantic_value.class_name != None:
    record['class_name'] = pydantic_value.class_name
  return record


def decode_response_pydantic_value(
    record: Dict[str, Any]) -> types.ResponsePydanticValue:
  pydantic_value = types.ResponsePydanticValue()
  pydantic_value.class_name = record.get('class_name', None)
  if 'instance_json_value' in record:
    pydantic_value.instance_json_value = json.loads(
        record['instance_json_value'])
  return pydantic_value


def encode_response(
    response: types.Response) -> Dict[str, Any]:
  record = {}
  if response.type != None:
    record['type'] = response.type.value
  if response.value != None:
    if response.type == types.ResponseType.TEXT:
      record['value'] = response.value
    elif response.type == types.ResponseType.JSON:
      record['value'] = json.dumps(
          response.value,
          sort_keys=True)
    elif response.type == types.ResponseType.PYDANTIC:
      record.update(encode_response_pydantic_value(response.value))
  return record


def decode_response(
    record: Dict[str, Any]) -> types.Response:
  response = types.Response()
  if 'type' in record:
    response.type = types.ResponseType(record['type'])
  if response.type == types.ResponseType.TEXT:
    response.value = record.get('value', None)
  elif response.type == types.ResponseType.JSON:
    if 'value' in record:
      response.value = json.loads(record['value'])
  elif response.type == types.ResponseType.PYDANTIC:
    response.value = decode_response_pydantic_value(record)
  return response


def encode_query_record(
    query_record: types.QueryRecord) -> Dict[str, Any]:
  record = {}
  if query_record.call_type != None:
    record['call_type'] = query_record.call_type.value
  if query_record.provider_model != None:
    record['provider_model'] = encode_provider_model_type(
        query_record.provider_model)
  if query_record.prompt != None:
    record['prompt'] = query_record.prompt
  if query_record.system != None:
    record['system'] = query_record.system
  if query_record.messages != None:
    record['messages'] = query_record.messages
  if query_record.max_tokens != None:
    record['max_tokens'] = str(query_record.max_tokens)
  if query_record.temperature != None:
    record['temperature'] = str(query_record.temperature)
  if query_record.stop != None:
    record['stop'] = query_record.stop
  if query_record.token_count != None:
    record['token_count'] = str(query_record.token_count)
  if query_record.response_format != None:
    record['response_format'] = encode_response_format(
        query_record.response_format)
  if query_record.web_search != None:
    record['web_search'] = query_record.web_search
  if query_record.hash_value != None:
    record['hash_value'] = query_record.hash_value
  return record


def decode_query_record(
    record: Dict[str, Any]) -> types.QueryRecord:
  query_record = types.QueryRecord()
  if 'call_type' in record:
    query_record.call_type = types.CallType(record['call_type'])
  if 'provider_model' in record:
    query_record.provider_model = decode_provider_model_type(
        record['provider_model'])
  query_record.prompt = record.get('prompt', None)
  query_record.system = record.get('system', None)
  query_record.messages = record.get('messages', None)
  if 'max_tokens' in record:
    query_record.max_tokens = int(record['max_tokens'])
  if 'temperature' in record:
    query_record.temperature = float(record['temperature'])
  query_record.stop = record.get('stop', None)
  if 'token_count' in record:
    query_record.token_count = int(record['token_count'])
  if 'response_format' in record:
    query_record.response_format = decode_response_format(
        record['response_format'])
  if 'web_search' in record:
    query_record.web_search = bool(record['web_search'])
  query_record.hash_value = record.get('hash_value', None)
  return query_record


def encode_query_response_record(
    query_response_record: types.QueryResponseRecord
) -> Dict[str, Any]:
  record = {}
  if query_response_record.response != None:
    record['response'] = encode_response(query_response_record.response)
  if query_response_record.error != None:
    record['error'] = query_response_record.error
  if query_response_record.start_utc_date != None:
    record['start_utc_date'] = query_response_record.start_utc_date.isoformat()
  if query_response_record.end_utc_date != None:
    record['end_utc_date'] = query_response_record.end_utc_date.isoformat()
  if query_response_record.local_time_offset_minute != None:
    record['local_time_offset_minute'] = (
        query_response_record.local_time_offset_minute)
  if query_response_record.response_time != None:
    record['response_time'] = (
        query_response_record.response_time.total_seconds())
  if query_response_record.estimated_cost != None:
    record['estimated_cost'] = query_response_record.estimated_cost
  if query_response_record.token_count != None:
    record['token_count'] = str(query_response_record.token_count)
  return record


def decode_query_response_record(
    record: Dict[str, Any]) -> types.QueryResponseRecord:
  query_response_record = types.QueryResponseRecord()
  if 'response' in record:
    query_response_record.response = decode_response(record['response'])
  query_response_record.error = record.get('error', None)
  if 'start_utc_date' in record:
    query_response_record.start_utc_date = datetime.datetime.fromisoformat(
        record['start_utc_date'])
  if 'end_utc_date' in record:
    query_response_record.end_utc_date = datetime.datetime.fromisoformat(
        record['end_utc_date'])
  if 'local_time_offset_minute' in record:
    query_response_record.local_time_offset_minute = (
        record['local_time_offset_minute'])
  if 'response_time' in record:
    query_response_record.response_time = datetime.timedelta(
        seconds=record['response_time'])
  if 'estimated_cost' in record:
    query_response_record.estimated_cost = record['estimated_cost']
  if 'token_count' in record:
    query_response_record.token_count = int(record['token_count'])
  return query_response_record


def encode_cache_record(
    cache_record: types.CacheRecord) -> Dict[str, Any]:
  record = {}
  if cache_record.query_record != None:
    record['query_record'] = encode_query_record(
        cache_record.query_record)
  if cache_record.query_responses != None:
    record['query_responses'] = []
    for query_response_record in cache_record.query_responses:
      record['query_responses'].append(
          encode_query_response_record(query_response_record))
  if cache_record.shard_id != None:
    try:
      record['shard_id'] = int(cache_record.shard_id)
    except ValueError:
      record['shard_id'] = cache_record.shard_id
  if cache_record.last_access_time != None:
    record['last_access_time'] = cache_record.last_access_time.isoformat()
  if cache_record.call_count != None:
    record['call_count'] = cache_record.call_count
  return record


def decode_cache_record(
    record: Dict[str, Any]) -> types.CacheRecord:
  cache_record = types.CacheRecord()
  if 'query_record' in record:
    cache_record.query_record = decode_query_record(
        record['query_record'])
  if 'query_responses' in record:
    cache_record.query_responses = []
    for query_response_record in record['query_responses']:
      cache_record.query_responses.append(
          decode_query_response_record(query_response_record))
  if 'shard_id' in record:
    try:
      cache_record.shard_id = int(record['shard_id'])
    except ValueError:
      cache_record.shard_id = record['shard_id']
  if 'last_access_time' in record:
    cache_record.last_access_time = datetime.datetime.fromisoformat(
        record['last_access_time'])
  if 'call_count' in record:
    cache_record.call_count = int(record['call_count'])
  return cache_record


def encode_light_cache_record(
    light_cache_record: types.LightCacheRecord) -> Dict[str, Any]:
  record = {}
  if light_cache_record.query_record_hash != None:
    record['query_record_hash'] = light_cache_record.query_record_hash
  if light_cache_record.query_response_count != None:
    record['query_response_count'] = light_cache_record.query_response_count
  if light_cache_record.shard_id != None:
    try:
      record['shard_id'] = int(light_cache_record.shard_id)
    except ValueError:
      record['shard_id'] = light_cache_record.shard_id
  if light_cache_record.last_access_time != None:
    record['last_access_time'] = (
        light_cache_record.last_access_time.isoformat())
  if light_cache_record.call_count != None:
    record['call_count'] = light_cache_record.call_count
  return record


def decode_light_cache_record(
    record: Dict[str, Any]) -> types.LightCacheRecord:
  light_cache_record = types.LightCacheRecord()
  light_cache_record.query_record_hash = record.get('query_record_hash', None)
  if 'query_response_count' in record:
    light_cache_record.query_response_count = int(
        record['query_response_count'])
  if 'shard_id' in record:
    try:
      light_cache_record.shard_id = int(record['shard_id'])
    except ValueError:
      light_cache_record.shard_id = record['shard_id']
  if 'last_access_time' in record:
    light_cache_record.last_access_time = datetime.datetime.fromisoformat(
        record['last_access_time'])
  if 'call_count' in record:
    light_cache_record.call_count = int(record['call_count'])
  return light_cache_record


def encode_logging_record(
    logging_record: types.LoggingRecord) -> Dict[str, Any]:
  record = {}
  if logging_record.query_record != None:
    record['query_record'] = encode_query_record(
        logging_record.query_record)
  if logging_record.response_record != None:
    record['response_record'] = encode_query_response_record(
        logging_record.response_record)
  if logging_record.response_source != None:
    record['response_source'] = logging_record.response_source.value
  if logging_record.look_fail_reason != None:
    record['look_fail_reason'] = logging_record.look_fail_reason.value
  return record


def decode_logging_record(
    record: Dict[str, Any]) -> types.LoggingRecord:
  logging_record = types.LoggingRecord()
  if 'query_record' in record:
    logging_record.query_record = decode_query_record(
        record['query_record'])
  if 'response_record' in record:
    logging_record.response_record = decode_query_response_record(
        record['response_record'])
  if 'response_source' in record:
    logging_record.response_source = (
        types.ResponseSource(record['response_source']))
  if 'look_fail_reason' in record:
    logging_record.look_fail_reason = (
        types.CacheLookFailReason(record['look_fail_reason']))
  return logging_record


def encode_model_status(
    model_status: types.ModelStatus) -> Dict[str, Any]:
  record = {}
  if model_status.unprocessed_models != None:
    record['unprocessed_models'] = []
    for provider_model in model_status.unprocessed_models:
      record['unprocessed_models'].append(
          encode_provider_model_type(provider_model))
  if model_status.working_models != None:
    record['working_models'] = []
    for provider_model in model_status.working_models:
      record['working_models'].append(
          encode_provider_model_type(provider_model))
  if model_status.failed_models != None:
    record['failed_models'] = []
    for provider_model in model_status.failed_models:
      record['failed_models'].append(encode_provider_model_type(provider_model))
  if model_status.filtered_models != None:
    record['filtered_models'] = []
    for provider_model in model_status.filtered_models:
      record['filtered_models'].append(
          encode_provider_model_type(provider_model))
  if model_status.provider_queries != None:
    record['provider_queries'] = {}
    for provider_model, provider_query in model_status.provider_queries.items():
      provider_model = json.dumps(encode_provider_model_type(provider_model))
      record['provider_queries'][provider_model] = (
          encode_logging_record(provider_query))
  return record


def decode_model_status(
    record: Dict[str, Any]) -> types.ModelStatus:
  model_status = types.ModelStatus()
  if 'unprocessed_models' in record:
    for provider_model_record in record['unprocessed_models']:
      model_status.unprocessed_models.add(
          decode_provider_model_type(provider_model_record))
  if 'working_models' in record:
    for provider_model_record in record['working_models']:
      model_status.working_models.add(
          decode_provider_model_type(provider_model_record))
  if 'failed_models' in record:
    for provider_model_record in record['failed_models']:
      model_status.failed_models.add(
          decode_provider_model_type(provider_model_record))
  if 'filtered_models' in record:
    for provider_model_record in record['filtered_models']:
      model_status.filtered_models.add(
          decode_provider_model_type(provider_model_record))
  if 'provider_queries' in record:
    for provider_model, provider_query_record in record[
        'provider_queries'].items():
      provider_model = json.loads(provider_model)
      provider_model = decode_provider_model_type(provider_model)
      model_status.provider_queries[provider_model] = (
          decode_logging_record(provider_query_record))
  return model_status


def encode_logging_options(
    logging_options: types.LoggingOptions) -> Dict[str, Any]:
  record = {}
  if logging_options.logging_path != None:
    record['logging_path'] = logging_options.logging_path
  if logging_options.stdout != None:
    record['stdout'] = logging_options.stdout
  if logging_options.hide_sensitive_content != None:
    record['hide_sensitive_content'] = logging_options.hide_sensitive_content
  return record


def encode_cache_options(
    cache_options: types.CacheOptions) -> Dict[str, Any]:
  record = {}
  if cache_options.cache_path != None:
    record['cache_path'] = cache_options.cache_path
  if cache_options.unique_response_limit != None:
    record['unique_response_limit'] = cache_options.unique_response_limit
  if cache_options.retry_if_error_cached != None:
    record['retry_if_error_cached'] = cache_options.retry_if_error_cached
  if cache_options.clear_query_cache_on_connect != None:
    record['clear_query_cache_on_connect'] = (
        cache_options.clear_query_cache_on_connect)
  if cache_options.clear_model_cache_on_connect != None:
    record['clear_model_cache_on_connect'] = (
        cache_options.clear_model_cache_on_connect)
  return record


def encode_proxdash_options(
    proxdash_options: types.ProxDashOptions) -> Dict[str, Any]:
  record = {}
  if proxdash_options.stdout != None:
    record['stdout'] = proxdash_options.stdout
  if proxdash_options.hide_sensitive_content != None:
    record['hide_sensitive_content'] = proxdash_options.hide_sensitive_content
  if proxdash_options.disable_proxdash != None:
    record['disable_proxdash'] = proxdash_options.disable_proxdash
  return record


def encode_run_options(
    run_options: types.RunOptions) -> Dict[str, Any]:
  record = {}
  if run_options.run_type != None:
    record['run_type'] = run_options.run_type.value
  if run_options.hidden_run_key != None:
    record['hidden_run_key'] = run_options.hidden_run_key
  if run_options.experiment_path != None:
    record['experiment_path'] = run_options.experiment_path
  if run_options.root_logging_path != None:
    record['root_logging_path'] = run_options.root_logging_path
  if run_options.default_model_cache_path != None:
    record['default_model_cache_path'] = run_options.default_model_cache_path
  if run_options.logging_options != None:
    record['logging_options'] = encode_logging_options(
        run_options.logging_options)
  if run_options.cache_options != None:
    record['cache_options'] = encode_cache_options(
        run_options.cache_options)
  if run_options.proxdash_options != None:
    record['proxdash_options'] = encode_proxdash_options(
        run_options.proxdash_options)
  if run_options.allow_multiprocessing != None:
    record['allow_multiprocessing'] = run_options.allow_multiprocessing
  if run_options.model_test_timeout != None:
    record['model_test_timeout'] = run_options.model_test_timeout
  if run_options.feature_mapping_strategy != None:
    record['feature_mapping_strategy'] = run_options.feature_mapping_strategy
  if run_options.suppress_provider_errors != None:
    record['suppress_provider_errors'] = run_options.suppress_provider_errors
  return record


def decode_logging_options(
    record: Dict[str, Any]) -> types.LoggingOptions:
  logging_options = types.LoggingOptions()
  if 'logging_path' in record:
    logging_options.logging_path = record['logging_path']
  if 'stdout' in record:
    logging_options.stdout = record['stdout']
  if 'hide_sensitive_content' in record:
    logging_options.hide_sensitive_content = record['hide_sensitive_content']
  return logging_options


def decode_cache_options(
    record: Dict[str, Any]) -> types.CacheOptions:
  cache_options = types.CacheOptions()
  if 'cache_path' in record:
    cache_options.cache_path = record['cache_path']
  if 'unique_response_limit' in record:
    cache_options.unique_response_limit = record['unique_response_limit']
  if 'retry_if_error_cached' in record:
    cache_options.retry_if_error_cached = record['retry_if_error_cached']
  if 'clear_query_cache_on_connect' in record:
    cache_options.clear_query_cache_on_connect = (
        record['clear_query_cache_on_connect'])
  if 'clear_model_cache_on_connect' in record:
    cache_options.clear_model_cache_on_connect = (
        record['clear_model_cache_on_connect'])
  return cache_options


def decode_proxdash_options(
    record: Dict[str, Any]) -> types.ProxDashOptions:
  proxdash_options = types.ProxDashOptions()
  if 'stdout' in record:
    proxdash_options.stdout = record['stdout']
  if 'hide_sensitive_content' in record:
    proxdash_options.hide_sensitive_content = record['hide_sensitive_content']
  if 'disable_proxdash' in record:
    proxdash_options.disable_proxdash = record['disable_proxdash']
  return proxdash_options


def decode_run_options(
    record: Dict[str, Any]) -> types.RunOptions:
  run_options = types.RunOptions()
  if 'run_type' in record:
    run_options.run_type = types.RunType(record['run_type'])
  if 'hidden_run_key' in record:
    run_options.hidden_run_key = record['hidden_run_key']
  if 'experiment_path' in record:
    run_options.experiment_path = record['experiment_path']
  if 'root_logging_path' in record:
    run_options.root_logging_path = record['root_logging_path']
  if 'default_model_cache_path' in record:
    run_options.default_model_cache_path = record['default_model_cache_path']
  if 'logging_options' in record:
    run_options.logging_options = decode_logging_options(
        record['logging_options'])
  if 'cache_options' in record:
    run_options.cache_options = decode_cache_options(
        record['cache_options'])
  if 'proxdash_options' in record:
    run_options.proxdash_options = decode_proxdash_options(
        record['proxdash_options'])
  if 'allow_multiprocessing' in record:
    run_options.allow_multiprocessing = record['allow_multiprocessing']
  if 'model_test_timeout' in record:
    run_options.model_test_timeout = record['model_test_timeout']
  if 'feature_mapping_strategy' in record:
    run_options.feature_mapping_strategy = types.FeatureMappingStrategy(
        record['feature_mapping_strategy'])
  if 'suppress_provider_errors' in record:
    run_options.suppress_provider_errors = record['suppress_provider_errors']
  return run_options


def encode_base_provider_stats(
    base_provider_stats: stat_types.BaseProviderStats) -> Dict[str, Any]:
  record = {}
  if base_provider_stats.total_queries:
    record['total_queries'] = base_provider_stats.total_queries
  if base_provider_stats.total_successes:
    record['total_successes'] = base_provider_stats.total_successes
  if base_provider_stats.total_fails:
    record['total_fails'] = base_provider_stats.total_fails
  if base_provider_stats.total_token_count:
    record['total_token_count'] = base_provider_stats.total_token_count
  if base_provider_stats.total_query_token_count:
    record['total_query_token_count'] = (
        base_provider_stats.total_query_token_count)
  if base_provider_stats.total_response_token_count:
    record['total_response_token_count'] = (
        base_provider_stats.total_response_token_count)
  if base_provider_stats.total_response_time:
    record['total_response_time'] = base_provider_stats.total_response_time
  if base_provider_stats.avr_response_time:
    record['avr_response_time'] = base_provider_stats.avr_response_time
  if base_provider_stats.estimated_cost:
    record['estimated_cost'] = base_provider_stats.estimated_cost
  if base_provider_stats.total_cache_look_fail_reasons:
    record['total_cache_look_fail_reasons'] = {}
    for k, v in base_provider_stats.total_cache_look_fail_reasons.items():
      record['total_cache_look_fail_reasons'][k.value] = v
  return record


def decode_base_provider_stats(
    record: Dict[str, Any]) -> stat_types.BaseProviderStats:
  base_provider_stats = stat_types.BaseProviderStats()
  if 'total_queries' in record:
    base_provider_stats.total_queries = record['total_queries']
  if 'total_successes' in record:
    base_provider_stats.total_successes = record['total_successes']
  if 'total_fails' in record:
    base_provider_stats.total_fails = record['total_fails']
  if 'total_token_count' in record:
    base_provider_stats.total_token_count = record['total_token_count']
  if 'total_query_token_count' in record:
    base_provider_stats.total_query_token_count = (
        record['total_query_token_count'])
  if 'total_response_token_count' in record:
    base_provider_stats.total_response_token_count = (
        record['total_response_token_count'])
  if 'total_response_time' in record:
    base_provider_stats.total_response_time = record['total_response_time']
  if 'estimated_cost' in record:
    base_provider_stats.estimated_cost = record['estimated_cost']
  if 'total_cache_look_fail_reasons' in record:
    base_provider_stats.total_cache_look_fail_reasons = {}
    for k, v in record['total_cache_look_fail_reasons'].items():
      base_provider_stats.total_cache_look_fail_reasons[
          types.CacheLookFailReason(k)] = v
  return base_provider_stats


def encode_base_cache_stats(
    base_cache_stats: stat_types.BaseCacheStats) -> Dict[str, Any]:
  record = {}
  if base_cache_stats.total_cache_hit:
    record['total_cache_hit'] = base_cache_stats.total_cache_hit
  if base_cache_stats.total_success_return:
    record['total_success_return'] = base_cache_stats.total_success_return
  if base_cache_stats.total_fail_return:
    record['total_fail_return'] = base_cache_stats.total_fail_return
  if base_cache_stats.saved_token_count:
    record['saved_token_count'] = base_cache_stats.saved_token_count
  if base_cache_stats.saved_query_token_count:
    record['saved_query_token_count'] = base_cache_stats.saved_query_token_count
  if base_cache_stats.saved_response_token_count:
    record['saved_response_token_count'] = (
        base_cache_stats.saved_response_token_count)
  if base_cache_stats.saved_total_response_time:
    record['saved_total_response_time'] = (
        base_cache_stats.saved_total_response_time)
  if base_cache_stats.saved_avr_response_time:
    record['saved_avr_response_time'] = base_cache_stats.saved_avr_response_time
  if base_cache_stats.saved_estimated_cost:
    record['saved_estimated_cost'] = base_cache_stats.saved_estimated_cost
  return record


def decode_base_cache_stats(record) -> stat_types.BaseCacheStats:
  base_cache_stats = stat_types.BaseCacheStats()
  if 'total_cache_hit' in record:
    base_cache_stats.total_cache_hit = record['total_cache_hit']
  if 'total_success_return' in record:
    base_cache_stats.total_success_return = record['total_success_return']
  if 'total_fail_return' in record:
    base_cache_stats.total_fail_return = record['total_fail_return']
  if 'saved_token_count' in record:
    base_cache_stats.saved_token_count = record['saved_token_count']
  if 'saved_query_token_count' in record:
    base_cache_stats.saved_query_token_count = record['saved_query_token_count']
  if 'saved_response_token_count' in record:
    base_cache_stats.saved_response_token_count = (
        record['saved_response_token_count'])
  if 'saved_total_response_time' in record:
    base_cache_stats.saved_total_response_time = (
        record['saved_total_response_time'])
  if 'saved_estimated_cost' in record:
    base_cache_stats.saved_estimated_cost = record['saved_estimated_cost']
  return base_cache_stats


def encode_provider_model_stats(
    provider_model_stats: stat_types.ProviderModelStats) -> Dict[str, Any]:
  record = {}
  if provider_model_stats.provider_model:
    record['provider_model'] = encode_provider_model_type(
        provider_model_stats.provider_model)
  if provider_model_stats.provider_stats:
    record['provider_stats'] = encode_base_provider_stats(
        provider_model_stats.provider_stats)
  if provider_model_stats.cache_stats:
    record['cache_stats'] = encode_base_cache_stats(
        provider_model_stats.cache_stats)
  return record


def decode_provider_model_stats(
    record: Dict[str, Any]) -> stat_types.ProviderModelStats:
  provider_model_stats = stat_types.ProviderModelStats()
  if 'provider_model' in record:
    provider_model_stats.provider_model = decode_provider_model_type(
        record['provider_model'])
  if 'provider_stats' in record:
    provider_model_stats.provider_stats = decode_base_provider_stats(
        record['provider_stats'])
  if 'cache_stats' in record:
    provider_model_stats.cache_stats = decode_base_cache_stats(
        record['cache_stats'])
  return provider_model_stats


def encode_provider_stats(
    provider_stats: stat_types.ProviderStats) -> Dict[str, Any]:
  record = {}
  if provider_stats.provider:
    record['provider'] = provider_stats.provider
  if provider_stats.provider_stats:
    record['provider_stats'] = encode_base_provider_stats(
        provider_stats.provider_stats)
  if provider_stats.cache_stats:
    record['cache_stats'] = encode_base_cache_stats(provider_stats.cache_stats)
  if provider_stats.provider_models:
    record['provider_models'] = []
    for k, v in provider_stats.provider_models.items():
      value = encode_provider_model_type(k)
      value['provider_model_stats'] = encode_provider_model_stats(v)
      record['provider_models'].append(value)
  return record


def decode_provider_stats(record: Dict[str, Any]) -> stat_types.ProviderStats:
  provider_stats = stat_types.ProviderStats()
  if 'provider' in record:
    provider_stats.provider = record['provider']
  if 'provider_stats' in record:
    provider_stats.provider_stats = decode_base_provider_stats(
        record['provider_stats'])
  if 'cache_stats' in record:
    provider_stats.cache_stats = decode_base_cache_stats(
        record['cache_stats'])
  if 'provider_models' in record:
    provider_stats.provider_models = {}
    for provider_model_record in record['provider_models']:
      provider_model_type = decode_provider_model_type({
          'provider': provider_model_record['provider'],
          'model': provider_model_record['model'],
          'provider_model_identifier': provider_model_record[
              'provider_model_identifier']
      })
      provider_stats.provider_models[provider_model_type] = (
          decode_provider_model_stats(
              provider_model_record['provider_model_stats']))
  return provider_stats


def encode_run_stats(
    run_stats: stat_types.RunStats) -> Dict[str, Any]:
  record = {}
  if run_stats.provider_stats:
    record['provider_stats'] = encode_base_provider_stats(
        run_stats.provider_stats)
  if run_stats.cache_stats:
    record['cache_stats'] = encode_base_cache_stats(run_stats.cache_stats)
  if run_stats.providers:
    record['providers'] = {}
    for k, v in run_stats.providers.items():
      record['providers'][k] = encode_provider_stats(v)
  return record


def decode_run_stats(
    record: Dict[str, Any]) -> stat_types.RunStats:
  run_stats = stat_types.RunStats()
  if 'provider_stats' in record:
    run_stats.provider_stats = decode_base_provider_stats(
        record['provider_stats'])
  if 'cache_stats' in record:
    run_stats.cache_stats = decode_base_cache_stats(record['cache_stats'])
  if 'providers' in record:
    run_stats.providers = {}
    for k, v in record['providers'].items():
      run_stats.providers[k] = decode_provider_stats(v)
  return run_stats
