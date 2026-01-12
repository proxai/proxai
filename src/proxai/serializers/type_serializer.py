import datetime
import json
from typing import Any

import proxai.types as types


def encode_provider_model_type(
    provider_model_type: types.ProviderModelType) -> dict[str, Any]:
  record = {}
  record['provider'] = provider_model_type.provider
  record['model'] = provider_model_type.model
  record['provider_model_identifier'] = (
      provider_model_type.provider_model_identifier)
  return record


def decode_provider_model_type(
    record: dict[str, Any]) -> types.ProviderModelType:
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
) -> dict[str, Any]:
  if isinstance(provider_model_identifier, types.ProviderModelType):
    return encode_provider_model_type(provider_model_identifier)
  else:
    # ProviderModelTupleType
    return {
        'provider': provider_model_identifier[0],
        'model': provider_model_identifier[1]
    }


def decode_provider_model_identifier(
    record: dict[str, Any]) -> types.ProviderModelIdentifierType:
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
) -> dict[str, Any]:
  record = {}
  if provider_model_pricing_type.per_response_token_cost is not None:
    record['per_response_token_cost'] = (
        provider_model_pricing_type.per_response_token_cost)
  if provider_model_pricing_type.per_query_token_cost is not None:
    record['per_query_token_cost'] = (
        provider_model_pricing_type.per_query_token_cost)
  return record


def decode_provider_model_pricing_type(
    record: dict[str, Any]) -> types.ProviderModelPricingType:
  provider_model_pricing_type = types.ProviderModelPricingType()
  if 'per_response_token_cost' in record:
    provider_model_pricing_type.per_response_token_cost = (
        float(record['per_response_token_cost']))
  if 'per_query_token_cost' in record:
    provider_model_pricing_type.per_query_token_cost = (
        float(record['per_query_token_cost']))
  return provider_model_pricing_type


def encode_endpoint_feature_info_type(
    endpoint_feature_info_type: types.EndpointFeatureInfoType
) -> dict[str, Any]:
  record = {}
  if endpoint_feature_info_type.supported is not None:
    record['supported'] = endpoint_feature_info_type.supported
  if endpoint_feature_info_type.best_effort is not None:
    record['best_effort'] = endpoint_feature_info_type.best_effort
  if endpoint_feature_info_type.not_supported is not None:
    record['not_supported'] = endpoint_feature_info_type.not_supported
  return record


def decode_endpoint_feature_info_type(
    record: dict[str, Any]) -> types.EndpointFeatureInfoType:
  endpoint_feature_info_type = types.EndpointFeatureInfoType()
  if 'supported' in record:
    endpoint_feature_info_type.supported = record['supported']
  if 'best_effort' in record:
    endpoint_feature_info_type.best_effort = record['best_effort']
  if 'not_supported' in record:
    endpoint_feature_info_type.not_supported = record['not_supported']
  return endpoint_feature_info_type


def encode_feature_mapping_type(
    feature_mapping: types.FeatureMappingType
) -> dict[str, Any]:
  record = {}
  for feature_name, endpoint_feature_info in feature_mapping.items():
    record[feature_name.value] = encode_endpoint_feature_info_type(
        endpoint_feature_info)
  return record


def decode_feature_mapping_type(
    record: dict[str, Any]) -> types.FeatureMappingType:
  feature_mapping = {}
  for feature_name_str, endpoint_feature_info_record in record.items():
    feature_name = types.FeatureNameType(feature_name_str)
    feature_mapping[feature_name] = decode_endpoint_feature_info_type(
        endpoint_feature_info_record)
  return feature_mapping


def encode_provider_model_metadata_type(
    provider_model_metadata_type: types.ProviderModelMetadataType
) -> dict[str, Any]:
  record = {}
  if provider_model_metadata_type.call_type is not None:
    record['call_type'] = provider_model_metadata_type.call_type.value
  if provider_model_metadata_type.is_featured is not None:
    record['is_featured'] = provider_model_metadata_type.is_featured
  if provider_model_metadata_type.model_size_tags is not None:
    record['model_size_tags'] = [
        model_size_tag.value
        for model_size_tag in provider_model_metadata_type.model_size_tags]
  if provider_model_metadata_type.is_default_candidate is not None:
    record['is_default_candidate'] = (
        provider_model_metadata_type.is_default_candidate)
  if provider_model_metadata_type.default_candidate_priority is not None:
    record['default_candidate_priority'] = (
        provider_model_metadata_type.default_candidate_priority)
  if provider_model_metadata_type.tags is not None:
    record['tags'] = provider_model_metadata_type.tags
  return record


def decode_provider_model_metadata_type(
    record: dict[str, Any]) -> types.ProviderModelMetadataType:
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
) -> dict[str, Any]:
  record = {}
  if provider_model_config_type.provider_model is not None:
    record['provider_model'] = encode_provider_model_type(
        provider_model_config_type.provider_model)
  if provider_model_config_type.pricing is not None:
    record['pricing'] = encode_provider_model_pricing_type(
        provider_model_config_type.pricing)
  if provider_model_config_type.features is not None:
    record['features'] = encode_feature_mapping_type(
        provider_model_config_type.features)
  if provider_model_config_type.metadata is not None:
    record['metadata'] = encode_provider_model_metadata_type(
        provider_model_config_type.metadata)
  return record


def decode_provider_model_config_type(
    record: dict[str, Any]) -> types.ProviderModelConfigType:
  provider_model_config_type = types.ProviderModelConfigType()
  if 'provider_model' in record:
    provider_model_config_type.provider_model = decode_provider_model_type(
        record['provider_model'])
  if 'pricing' in record:
    provider_model_config_type.pricing = decode_provider_model_pricing_type(
        record['pricing'])
  if 'features' in record:
    provider_model_config_type.features = decode_feature_mapping_type(
        record['features'])
  if 'metadata' in record:
    provider_model_config_type.metadata = decode_provider_model_metadata_type(
        record['metadata'])
  return provider_model_config_type


def encode_provider_model_configs_type(
    provider_model_configs: types.ProviderModelConfigsType
) -> dict[str, Any]:
  record = {}
  for provider, model_configs_dict in provider_model_configs.items():
    record[provider] = {}
    for model, provider_model_config in model_configs_dict.items():
      record[provider][model] = encode_provider_model_config_type(
          provider_model_config)
  return record


def decode_provider_model_configs_type(
    record: dict[str, Any]) -> types.ProviderModelConfigsType:
  provider_model_configs = {}
  for provider, model_configs_dict_record in record.items():
    provider_model_configs[provider] = {}
    for model, provider_model_config_record in (
        model_configs_dict_record.items()):
      provider_model_configs[provider][model] = (
          decode_provider_model_config_type(provider_model_config_record))
  return provider_model_configs


def encode_featured_models_type(
    featured_models: types.FeaturedModelsType) -> dict[str, Any]:
  record = {}
  for provider, provider_model_identifiers in featured_models.items():
    record[provider] = []
    for provider_model_identifier in provider_model_identifiers:
      record[provider].append(
          encode_provider_model_identifier(provider_model_identifier))
  return record


def decode_featured_models_type(
    record: dict[str, Any]) -> types.FeaturedModelsType:
  featured_models = {}
  for provider, provider_model_identifier_records in record.items():
    provider_model_identifiers = []
    for provider_model_identifier_record in (
        provider_model_identifier_records):
      provider_model_identifiers.append(
          decode_provider_model_identifier(provider_model_identifier_record))
    featured_models[provider] = provider_model_identifiers
  return featured_models


def encode_models_by_call_type_type(
    models_by_call_type: types.ModelsByCallTypeType) -> dict[str, Any]:
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
    record: dict[str, Any]) -> types.ModelsByCallTypeType:
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
      provider_dict[provider] = provider_model_identifiers
    models_by_call_type[call_type] = provider_dict
  return models_by_call_type


def encode_models_by_size_type(
    models_by_size: types.ModelsBySizeType) -> dict[str, Any]:
  record = {}
  for model_size, provider_model_identifiers in models_by_size.items():
    record[model_size.value] = []
    for provider_model_identifier in provider_model_identifiers:
      record[model_size.value].append(
          encode_provider_model_identifier(provider_model_identifier))
  return record


def decode_models_by_size_type(
    record: dict[str, Any]) -> types.ModelsBySizeType:
  models_by_size = {}
  for model_size_str, provider_model_identifier_records in record.items():
    model_size = types.ModelSizeType(model_size_str)
    provider_model_identifiers = []
    for provider_model_identifier_record in (
        provider_model_identifier_records):
      provider_model_identifiers.append(
          decode_provider_model_identifier(provider_model_identifier_record))
    models_by_size[model_size] = provider_model_identifiers
  return models_by_size


def encode_default_model_priority_list_type(
    default_model_priority_list: types.DefaultModelPriorityListType
) -> dict[str, Any]:
  record = []
  for provider_model_identifier in default_model_priority_list:
    record.append(encode_provider_model_identifier(provider_model_identifier))
  return record


def decode_default_model_priority_list_type(
    record: dict[str, Any]) -> types.DefaultModelPriorityListType:
  default_model_priority_list = []
  for provider_model_identifier_record in record:
    default_model_priority_list.append(
        decode_provider_model_identifier(provider_model_identifier_record))
  return default_model_priority_list


def encode_model_configs_schema_metadata_type(
    model_configs_schema_metadata_type: types.ModelConfigsSchemaMetadataType
) -> dict[str, Any]:
  record = {}
  if model_configs_schema_metadata_type.version is not None:
    record['version'] = model_configs_schema_metadata_type.version
  if model_configs_schema_metadata_type.released_at is not None:
    record['released_at'] = (
        model_configs_schema_metadata_type.released_at.isoformat())
  if model_configs_schema_metadata_type.min_proxai_version is not None:
    record['min_proxai_version'] = (
        model_configs_schema_metadata_type.min_proxai_version)
  if model_configs_schema_metadata_type.config_origin is not None:
    record['config_origin'] = (
        model_configs_schema_metadata_type.config_origin.value)
  if model_configs_schema_metadata_type.release_notes is not None:
    record['release_notes'] = model_configs_schema_metadata_type.release_notes
  return record


def decode_model_configs_schema_metadata_type(
    record: dict[str, Any]) -> types.ModelConfigsSchemaMetadataType:
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
) -> dict[str, Any]:
  record = {}
  if model_configs_schema_version_config_type.provider_model_configs is not None:
    record['provider_model_configs'] = encode_provider_model_configs_type(
        model_configs_schema_version_config_type.provider_model_configs)
  if model_configs_schema_version_config_type.featured_models is not None:
    record['featured_models'] = encode_featured_models_type(
        model_configs_schema_version_config_type.featured_models)
  if model_configs_schema_version_config_type.models_by_call_type is not None:
    record['models_by_call_type'] = encode_models_by_call_type_type(
        model_configs_schema_version_config_type.models_by_call_type)
  if model_configs_schema_version_config_type.models_by_size is not None:
    record['models_by_size'] = encode_models_by_size_type(
        model_configs_schema_version_config_type.models_by_size)
  if model_configs_schema_version_config_type.default_model_priority_list is not None:
    record['default_model_priority_list'] = (
        encode_default_model_priority_list_type(
            model_configs_schema_version_config_type.default_model_priority_list))
  return record


def decode_model_configs_schema_version_config_type(
    record: dict[str, Any]) -> types.ModelConfigsSchemaVersionConfigType:
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
) -> dict[str, Any]:
  record = {}
  if model_configs_schema_type.metadata is not None:
    record['metadata'] = encode_model_configs_schema_metadata_type(
        model_configs_schema_type.metadata)
  if model_configs_schema_type.version_config is not None:
    record['version_config'] = encode_model_configs_schema_version_config_type(
        model_configs_schema_type.version_config)
  return record


def decode_model_configs_schema_type(
    record: dict[str, Any]) -> types.ModelConfigsSchemaType:
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
    pydantic_value: types.ResponseFormatPydanticValue) -> dict[str, Any]:
  record = {}
  if (pydantic_value.class_json_schema_value is not None and
      pydantic_value.class_value is not None):
    raise ValueError(
        'ResponseFormatPydanticValue cannot have both '
        'class_json_schema_value and class_value set.')
  json_schema = None
  if pydantic_value.class_value is not None:
    json_schema = pydantic_value.class_value.model_json_schema()
  elif pydantic_value.class_json_schema_value is not None:
    json_schema = pydantic_value.class_json_schema_value
  if json_schema is not None:
    record['class_json_schema_value'] = json.dumps(
        json_schema,
        sort_keys=True)
  if pydantic_value.class_name is not None:
    record['class_name'] = pydantic_value.class_name
  return record


def decode_response_format_pydantic_value(
    record: dict[str, Any]) -> types.ResponseFormatPydanticValue:
  pydantic_value = types.ResponseFormatPydanticValue()
  pydantic_value.class_name = record.get('class_name')
  if 'class_json_schema_value' in record:
    pydantic_value.class_json_schema_value = json.loads(
        record['class_json_schema_value'])
  return pydantic_value


def encode_response_format(
    response_format: types.ResponseFormat) -> dict[str, Any]:
  record = {}
  if response_format.type is not None:
    record['type'] = response_format.type.value
  if response_format.value is not None:
    if response_format.type == types.ResponseFormatType.TEXT or response_format.type == types.ResponseFormatType.JSON:
      pass
    elif response_format.type == types.ResponseFormatType.JSON_SCHEMA:
      record['value'] = json.dumps(
          response_format.value,
          sort_keys=True)
    elif response_format.type == types.ResponseFormatType.PYDANTIC:
      record.update(encode_response_format_pydantic_value(response_format.value))
  return record


def decode_response_format(
    record: dict[str, Any]) -> types.ResponseFormat:
  response_format = types.ResponseFormat()
  if 'type' in record:
    response_format.type = types.ResponseFormatType(record['type'])
  if response_format.type == types.ResponseFormatType.TEXT or response_format.type == types.ResponseFormatType.JSON:
    pass
  elif response_format.type == types.ResponseFormatType.JSON_SCHEMA:
    if 'value' in record:
      response_format.value = json.loads(record['value'])
  elif response_format.type == types.ResponseFormatType.PYDANTIC:
    response_format.value = decode_response_format_pydantic_value(record)
  return response_format


def encode_pydantic_metadata(
    pydantic_metadata: types.PydanticMetadataType) -> dict[str, Any]:
  record = {}
  if pydantic_metadata.class_name is not None:
    record['class_name'] = pydantic_metadata.class_name
  if pydantic_metadata.instance_json_value is not None:
    record['instance_json_value'] = json.dumps(
        pydantic_metadata.instance_json_value,
        sort_keys=True)
  return record


def decode_pydantic_metadata(
    record: dict[str, Any]) -> types.PydanticMetadataType:
  pydantic_metadata = types.PydanticMetadataType()
  pydantic_metadata.class_name = record.get('class_name')
  if 'instance_json_value' in record:
    pydantic_metadata.instance_json_value = json.loads(
        record['instance_json_value'])
  return pydantic_metadata


def encode_response(
    response: types.Response) -> dict[str, Any]:
  record = {}
  if response.type is not None:
    record['type'] = response.type.value
  if response.type == types.ResponseType.TEXT:
    if response.value is not None:
      record['value'] = response.value
  elif response.type == types.ResponseType.JSON:
    if response.value is not None:
      record['value'] = json.dumps(
          response.value,
          sort_keys=True)
  elif response.type == types.ResponseType.PYDANTIC:
    # For PYDANTIC: convert value (instance) to instance_json_value if needed
    pydantic_metadata = response.pydantic_metadata
    if pydantic_metadata is None:
      pydantic_metadata = types.PydanticMetadataType()
    # If value exists (live instance), convert to JSON for serialization
    if response.value is not None and pydantic_metadata.instance_json_value is None:
      pydantic_metadata.instance_json_value = response.value.model_dump()
    record['pydantic_metadata'] = encode_pydantic_metadata(pydantic_metadata)
  return record


def decode_response(
    record: dict[str, Any]) -> types.Response:
  response = types.Response()
  if 'type' in record:
    response.type = types.ResponseType(record['type'])
  if response.type == types.ResponseType.TEXT:
    response.value = record.get('value')
  elif response.type == types.ResponseType.JSON:
    if 'value' in record:
      response.value = json.loads(record['value'])
  elif response.type == types.ResponseType.PYDANTIC:
    # For PYDANTIC: restore pydantic_metadata, value stays None until runtime
    if 'pydantic_metadata' in record:
      response.pydantic_metadata = decode_pydantic_metadata(
          record['pydantic_metadata'])
  return response


def encode_query_record(
    query_record: types.QueryRecord) -> dict[str, Any]:
  record = {}
  if query_record.call_type is not None:
    record['call_type'] = query_record.call_type.value
  if query_record.provider_model is not None:
    record['provider_model'] = encode_provider_model_type(
        query_record.provider_model)
  if query_record.prompt is not None:
    record['prompt'] = query_record.prompt
  if query_record.system is not None:
    record['system'] = query_record.system
  if query_record.messages is not None:
    record['messages'] = query_record.messages
  if query_record.max_tokens is not None:
    record['max_tokens'] = str(query_record.max_tokens)
  if query_record.temperature is not None:
    record['temperature'] = str(query_record.temperature)
  if query_record.stop is not None:
    record['stop'] = query_record.stop
  if query_record.token_count is not None:
    record['token_count'] = str(query_record.token_count)
  if query_record.response_format is not None:
    record['response_format'] = encode_response_format(
        query_record.response_format)
  if query_record.web_search is not None:
    record['web_search'] = query_record.web_search
  if query_record.feature_mapping_strategy is not None:
    record['feature_mapping_strategy'] = (
        query_record.feature_mapping_strategy.value)
  if query_record.hash_value is not None:
    record['hash_value'] = query_record.hash_value
  if query_record.chosen_endpoint is not None:
    record['chosen_endpoint'] = query_record.chosen_endpoint
  return record


def decode_query_record(
    record: dict[str, Any]) -> types.QueryRecord:
  query_record = types.QueryRecord()
  if 'call_type' in record:
    query_record.call_type = types.CallType(record['call_type'])
  if 'provider_model' in record:
    query_record.provider_model = decode_provider_model_type(
        record['provider_model'])
  query_record.prompt = record.get('prompt')
  query_record.system = record.get('system')
  query_record.messages = record.get('messages')
  if 'max_tokens' in record:
    query_record.max_tokens = int(record['max_tokens'])
  if 'temperature' in record:
    query_record.temperature = float(record['temperature'])
  query_record.stop = record.get('stop')
  if 'token_count' in record:
    query_record.token_count = int(record['token_count'])
  if 'response_format' in record:
    query_record.response_format = decode_response_format(
        record['response_format'])
  if 'web_search' in record:
    query_record.web_search = bool(record['web_search'])
  if 'feature_mapping_strategy' in record:
    query_record.feature_mapping_strategy = (
        types.FeatureMappingStrategy(record['feature_mapping_strategy']))
  if 'chosen_endpoint' in record:
    query_record.chosen_endpoint = record['chosen_endpoint']
  query_record.hash_value = record.get('hash_value')
  return query_record


def encode_query_response_record(
    query_response_record: types.QueryResponseRecord
) -> dict[str, Any]:
  record = {}
  if query_response_record.response is not None:
    record['response'] = encode_response(query_response_record.response)
  if query_response_record.error is not None:
    record['error'] = query_response_record.error
  if query_response_record.error_traceback is not None:
    record['error_traceback'] = query_response_record.error_traceback
  if query_response_record.start_utc_date is not None:
    record['start_utc_date'] = query_response_record.start_utc_date.isoformat()
  if query_response_record.end_utc_date is not None:
    record['end_utc_date'] = query_response_record.end_utc_date.isoformat()
  if query_response_record.local_time_offset_minute is not None:
    record['local_time_offset_minute'] = (
        query_response_record.local_time_offset_minute)
  if query_response_record.response_time is not None:
    record['response_time'] = (
        query_response_record.response_time.total_seconds())
  if query_response_record.estimated_cost is not None:
    record['estimated_cost'] = query_response_record.estimated_cost
  if query_response_record.token_count is not None:
    record['token_count'] = str(query_response_record.token_count)
  return record


def decode_query_response_record(
    record: dict[str, Any]) -> types.QueryResponseRecord:
  query_response_record = types.QueryResponseRecord()
  if 'response' in record:
    query_response_record.response = decode_response(record['response'])
  query_response_record.error = record.get('error')
  query_response_record.error_traceback = record.get('error_traceback')
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
    cache_record: types.CacheRecord) -> dict[str, Any]:
  record = {}
  if cache_record.query_record is not None:
    record['query_record'] = encode_query_record(
        cache_record.query_record)
  if cache_record.query_responses is not None:
    record['query_responses'] = []
    for query_response_record in cache_record.query_responses:
      record['query_responses'].append(
          encode_query_response_record(query_response_record))
  if cache_record.shard_id is not None:
    try:
      record['shard_id'] = int(cache_record.shard_id)
    except ValueError:
      record['shard_id'] = cache_record.shard_id
  if cache_record.last_access_time is not None:
    record['last_access_time'] = cache_record.last_access_time.isoformat()
  if cache_record.call_count is not None:
    record['call_count'] = cache_record.call_count
  return record


def decode_cache_record(
    record: dict[str, Any]) -> types.CacheRecord:
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
    light_cache_record: types.LightCacheRecord) -> dict[str, Any]:
  record = {}
  if light_cache_record.query_record_hash is not None:
    record['query_record_hash'] = light_cache_record.query_record_hash
  if light_cache_record.query_response_count is not None:
    record['query_response_count'] = light_cache_record.query_response_count
  if light_cache_record.shard_id is not None:
    try:
      record['shard_id'] = int(light_cache_record.shard_id)
    except ValueError:
      record['shard_id'] = light_cache_record.shard_id
  if light_cache_record.last_access_time is not None:
    record['last_access_time'] = (
        light_cache_record.last_access_time.isoformat())
  if light_cache_record.call_count is not None:
    record['call_count'] = light_cache_record.call_count
  return record


def decode_light_cache_record(
    record: dict[str, Any]) -> types.LightCacheRecord:
  light_cache_record = types.LightCacheRecord()
  light_cache_record.query_record_hash = record.get('query_record_hash')
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
    logging_record: types.LoggingRecord) -> dict[str, Any]:
  record = {}
  if logging_record.query_record is not None:
    record['query_record'] = encode_query_record(
        logging_record.query_record)
  if logging_record.response_record is not None:
    record['response_record'] = encode_query_response_record(
        logging_record.response_record)
  if logging_record.response_source is not None:
    record['response_source'] = logging_record.response_source.value
  if logging_record.look_fail_reason is not None:
    record['look_fail_reason'] = logging_record.look_fail_reason.value
  return record


def decode_logging_record(
    record: dict[str, Any]) -> types.LoggingRecord:
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
    model_status: types.ModelStatus) -> dict[str, Any]:
  record = {}
  if model_status.unprocessed_models is not None:
    record['unprocessed_models'] = []
    for provider_model in model_status.unprocessed_models:
      record['unprocessed_models'].append(
          encode_provider_model_type(provider_model))
  if model_status.working_models is not None:
    record['working_models'] = []
    for provider_model in model_status.working_models:
      record['working_models'].append(
          encode_provider_model_type(provider_model))
  if model_status.failed_models is not None:
    record['failed_models'] = []
    for provider_model in model_status.failed_models:
      record['failed_models'].append(encode_provider_model_type(provider_model))
  if model_status.filtered_models is not None:
    record['filtered_models'] = []
    for provider_model in model_status.filtered_models:
      record['filtered_models'].append(
          encode_provider_model_type(provider_model))
  if model_status.provider_queries is not None:
    record['provider_queries'] = {}
    for provider_model, provider_query in model_status.provider_queries.items():
      provider_model = json.dumps(encode_provider_model_type(provider_model))
      record['provider_queries'][provider_model] = (
          encode_logging_record(provider_query))
  return record


def decode_model_status(
    record: dict[str, Any]) -> types.ModelStatus:
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
    logging_options: types.LoggingOptions) -> dict[str, Any]:
  record = {}
  if logging_options.logging_path is not None:
    record['logging_path'] = logging_options.logging_path
  if logging_options.stdout is not None:
    record['stdout'] = logging_options.stdout
  if logging_options.hide_sensitive_content is not None:
    record['hide_sensitive_content'] = logging_options.hide_sensitive_content
  return record


def encode_cache_options(
    cache_options: types.CacheOptions) -> dict[str, Any]:
  record = {}
  if cache_options.cache_path is not None:
    record['cache_path'] = cache_options.cache_path
  if cache_options.unique_response_limit is not None:
    record['unique_response_limit'] = cache_options.unique_response_limit
  if cache_options.retry_if_error_cached is not None:
    record['retry_if_error_cached'] = cache_options.retry_if_error_cached
  if cache_options.clear_query_cache_on_connect is not None:
    record['clear_query_cache_on_connect'] = (
        cache_options.clear_query_cache_on_connect)
  if cache_options.clear_model_cache_on_connect is not None:
    record['clear_model_cache_on_connect'] = (
        cache_options.clear_model_cache_on_connect)
  if cache_options.disable_model_cache is not None:
    record['disable_model_cache'] = cache_options.disable_model_cache
  if cache_options.model_cache_duration is not None:
    record['model_cache_duration'] = cache_options.model_cache_duration
  return record


def encode_proxdash_options(
    proxdash_options: types.ProxDashOptions) -> dict[str, Any]:
  record = {}
  if proxdash_options.stdout is not None:
    record['stdout'] = proxdash_options.stdout
  if proxdash_options.hide_sensitive_content is not None:
    record['hide_sensitive_content'] = proxdash_options.hide_sensitive_content
  if proxdash_options.disable_proxdash is not None:
    record['disable_proxdash'] = proxdash_options.disable_proxdash
  if proxdash_options.api_key is not None:
    record['api_key'] = proxdash_options.api_key
  if proxdash_options.base_url is not None:
    record['base_url'] = proxdash_options.base_url
  return record


def encode_run_options(
    run_options: types.RunOptions) -> dict[str, Any]:
  record = {}
  if run_options.run_type is not None:
    record['run_type'] = run_options.run_type.value
  if run_options.hidden_run_key is not None:
    record['hidden_run_key'] = run_options.hidden_run_key
  if run_options.experiment_path is not None:
    record['experiment_path'] = run_options.experiment_path
  if run_options.root_logging_path is not None:
    record['root_logging_path'] = run_options.root_logging_path
  if run_options.default_model_cache_path is not None:
    record['default_model_cache_path'] = run_options.default_model_cache_path
  if run_options.logging_options is not None:
    record['logging_options'] = encode_logging_options(
        run_options.logging_options)
  if run_options.cache_options is not None:
    record['cache_options'] = encode_cache_options(
        run_options.cache_options)
  if run_options.proxdash_options is not None:
    record['proxdash_options'] = encode_proxdash_options(
        run_options.proxdash_options)
  if run_options.allow_multiprocessing is not None:
    record['allow_multiprocessing'] = run_options.allow_multiprocessing
  if run_options.model_test_timeout is not None:
    record['model_test_timeout'] = run_options.model_test_timeout
  if run_options.feature_mapping_strategy is not None:
    record['feature_mapping_strategy'] = run_options.feature_mapping_strategy
  if run_options.suppress_provider_errors is not None:
    record['suppress_provider_errors'] = run_options.suppress_provider_errors
  return record


def decode_logging_options(
    record: dict[str, Any]) -> types.LoggingOptions:
  logging_options = types.LoggingOptions()
  if 'logging_path' in record:
    logging_options.logging_path = record['logging_path']
  if 'stdout' in record:
    logging_options.stdout = record['stdout']
  if 'hide_sensitive_content' in record:
    logging_options.hide_sensitive_content = record['hide_sensitive_content']
  return logging_options


def decode_cache_options(
    record: dict[str, Any]) -> types.CacheOptions:
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
  if 'disable_model_cache' in record:
    cache_options.disable_model_cache = record['disable_model_cache']
  if 'model_cache_duration' in record:
    cache_options.model_cache_duration = record['model_cache_duration']
  return cache_options


def decode_proxdash_options(
    record: dict[str, Any]) -> types.ProxDashOptions:
  proxdash_options = types.ProxDashOptions()
  if 'stdout' in record:
    proxdash_options.stdout = record['stdout']
  if 'hide_sensitive_content' in record:
    proxdash_options.hide_sensitive_content = record['hide_sensitive_content']
  if 'disable_proxdash' in record:
    proxdash_options.disable_proxdash = record['disable_proxdash']
  if 'api_key' in record:
    proxdash_options.api_key = record['api_key']
  if 'base_url' in record:
    proxdash_options.base_url = record['base_url']
  return proxdash_options


def decode_run_options(
    record: dict[str, Any]) -> types.RunOptions:
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
