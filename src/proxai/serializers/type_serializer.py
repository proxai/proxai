import base64
import datetime
import json
from typing import Any

import proxai.chat.chat_session as chat_session
import proxai.chat.message_content as message_content
import proxai.types as types


def encode_provider_model_type(
    provider_model_type: types.ProviderModelType
) -> dict[str, Any]:
  """Serialize ProviderModelType to a dictionary."""
  record = {}
  record['provider'] = provider_model_type.provider
  record['model'] = provider_model_type.model
  record['provider_model_identifier'] = (
      provider_model_type.provider_model_identifier
  )
  return record


def decode_provider_model_type(
    record: dict[str, Any]
) -> types.ProviderModelType:
  """Deserialize ProviderModelType from a dictionary."""
  if 'provider' not in record:
    raise ValueError(f'Provider not found in record: {record=}')
  if 'model' not in record:
    raise ValueError(f'Model not found in record: {record=}')
  if 'provider_model_identifier' not in record:
    raise ValueError(
        f'Provider model identifier not found in record: {record=}'
    )
  provider_model = types.ProviderModelType(
      provider=record['provider'], model=record['model'],
      provider_model_identifier=record['provider_model_identifier']
  )
  return provider_model



def encode_provider_model_pricing_type(
    provider_model_pricing_type: types.ProviderModelPricingType
) -> dict[str, Any]:
  """Serialize ProviderModelPricingType to a dictionary."""
  record = {}
  if provider_model_pricing_type.input_token_cost is not None:
    record['input_token_cost'] = (
        provider_model_pricing_type.input_token_cost
    )
  if provider_model_pricing_type.output_token_cost is not None:
    record['output_token_cost'] = (
        provider_model_pricing_type.output_token_cost
    )
  return record


def decode_provider_model_pricing_type(
    record: dict[str, Any]
) -> types.ProviderModelPricingType:
  """Deserialize ProviderModelPricingType from a dictionary."""
  provider_model_pricing_type = types.ProviderModelPricingType()
  if 'input_token_cost' in record:
    provider_model_pricing_type.input_token_cost = (
        float(record['input_token_cost'])
    )
  if 'output_token_cost' in record:
    provider_model_pricing_type.output_token_cost = (
        float(record['output_token_cost'])
    )
  return provider_model_pricing_type


def encode_feature_support_type(
    feature_support_type: types.FeatureSupportType
) -> str:
  """Serialize FeatureSupportType to a string."""
  return feature_support_type.value


def decode_feature_support_type(
    value: str
) -> types.FeatureSupportType:
  """Deserialize FeatureSupportType from a string."""
  return types.FeatureSupportType(value)


def encode_parameter_config_type(
    parameter_config_type: types.ParameterConfigType
) -> dict[str, Any]:
  """Serialize ParameterConfigType to a dictionary."""
  record = {}
  if parameter_config_type.temperature is not None:
    record['temperature'] = encode_feature_support_type(
        parameter_config_type.temperature)
  if parameter_config_type.max_tokens is not None:
    record['max_tokens'] = encode_feature_support_type(
        parameter_config_type.max_tokens)
  if parameter_config_type.stop is not None:
    record['stop'] = encode_feature_support_type(
        parameter_config_type.stop)
  if parameter_config_type.n is not None:
    record['n'] = encode_feature_support_type(
        parameter_config_type.n)
  if parameter_config_type.thinking is not None:
    record['thinking'] = encode_feature_support_type(
        parameter_config_type.thinking)
  return record


def decode_parameter_config_type(
    record: dict[str, Any]
) -> types.ParameterConfigType:
  """Deserialize ParameterConfigType from a dictionary."""
  parameter_config_type = types.ParameterConfigType()
  if 'temperature' in record:
    parameter_config_type.temperature = decode_feature_support_type(
        record['temperature'])
  if 'max_tokens' in record:
    parameter_config_type.max_tokens = decode_feature_support_type(
        record['max_tokens'])
  if 'stop' in record:
    parameter_config_type.stop = decode_feature_support_type(
        record['stop'])
  if 'n' in record:
    parameter_config_type.n = decode_feature_support_type(
        record['n'])
  if 'thinking' in record:
    parameter_config_type.thinking = decode_feature_support_type(
        record['thinking'])
  return parameter_config_type


def encode_tool_config_type(
    tool_config_type: types.ToolConfigType
) -> dict[str, Any]:
  """Serialize ToolConfigType to a dictionary."""
  record = {}
  if tool_config_type.web_search is not None:
    record['web_search'] = encode_feature_support_type(
        tool_config_type.web_search)
  return record


def decode_tool_config_type(
    record: dict[str, Any]
) -> types.ToolConfigType:
  """Deserialize ToolConfigType from a dictionary."""
  tool_config_type = types.ToolConfigType()
  if 'web_search' in record:
    tool_config_type.web_search = decode_feature_support_type(
        record['web_search'])
  return tool_config_type


def encode_output_format_config_type(
    output_format_config_type: types.OutputFormatConfigType
) -> dict[str, Any]:
  """Serialize OutputFormatConfigType to a dictionary."""
  record = {}
  if output_format_config_type.text is not None:
    record['text'] = encode_feature_support_type(
        output_format_config_type.text)
  if output_format_config_type.image is not None:
    record['image'] = encode_feature_support_type(
        output_format_config_type.image)
  if output_format_config_type.audio is not None:
    record['audio'] = encode_feature_support_type(
        output_format_config_type.audio)
  if output_format_config_type.video is not None:
    record['video'] = encode_feature_support_type(
        output_format_config_type.video)
  if output_format_config_type.json is not None:
    record['json'] = encode_feature_support_type(
        output_format_config_type.json)
  if output_format_config_type.pydantic is not None:
    record['pydantic'] = encode_feature_support_type(
        output_format_config_type.pydantic)
  if output_format_config_type.multi_modal is not None:
    record['multi_modal'] = encode_feature_support_type(
        output_format_config_type.multi_modal)
  return record


def decode_output_format_config_type(
    record: dict[str, Any]
) -> types.OutputFormatConfigType:
  """Deserialize OutputFormatConfigType from a dictionary."""
  output_format_config_type = types.OutputFormatConfigType()
  if 'text' in record:
    output_format_config_type.text = decode_feature_support_type(
        record['text'])
  if 'image' in record:
    output_format_config_type.image = decode_feature_support_type(
        record['image'])
  if 'audio' in record:
    output_format_config_type.audio = decode_feature_support_type(
        record['audio'])
  if 'video' in record:
    output_format_config_type.video = decode_feature_support_type(
        record['video'])
  if 'json' in record:
    output_format_config_type.json = decode_feature_support_type(
        record['json'])
  if 'pydantic' in record:
    output_format_config_type.pydantic = decode_feature_support_type(
        record['pydantic'])
  if 'multi_modal' in record:
    output_format_config_type.multi_modal = decode_feature_support_type(
        record['multi_modal'])
  return output_format_config_type


def encode_input_format_config_type(
    input_format_config_type: types.InputFormatConfigType
) -> dict[str, Any]:
  """Serialize InputFormatConfigType to a dictionary."""
  record = {}
  if input_format_config_type.text is not None:
    record['text'] = encode_feature_support_type(
        input_format_config_type.text)
  if input_format_config_type.image is not None:
    record['image'] = encode_feature_support_type(
        input_format_config_type.image)
  if input_format_config_type.document is not None:
    record['document'] = encode_feature_support_type(
        input_format_config_type.document)
  if input_format_config_type.audio is not None:
    record['audio'] = encode_feature_support_type(
        input_format_config_type.audio)
  if input_format_config_type.video is not None:
    record['video'] = encode_feature_support_type(
        input_format_config_type.video)
  return record


def decode_input_format_config_type(
    record: dict[str, Any]
) -> types.InputFormatConfigType:
  """Deserialize InputFormatConfigType from a dictionary."""
  input_format_config_type = types.InputFormatConfigType()
  if 'text' in record:
    input_format_config_type.text = decode_feature_support_type(
        record['text'])
  if 'image' in record:
    input_format_config_type.image = decode_feature_support_type(
        record['image'])
  if 'document' in record:
    input_format_config_type.document = decode_feature_support_type(
        record['document'])
  if 'audio' in record:
    input_format_config_type.audio = decode_feature_support_type(
        record['audio'])
  if 'video' in record:
    input_format_config_type.video = decode_feature_support_type(
        record['video'])
  return input_format_config_type


def encode_feature_config_type(
    feature_config_type: types.FeatureConfigType
) -> dict[str, Any]:
  """Serialize FeatureConfigType to a dictionary."""
  record = {}
  if feature_config_type.prompt is not None:
    record['prompt'] = encode_feature_support_type(
        feature_config_type.prompt)
  if feature_config_type.messages is not None:
    record['messages'] = encode_feature_support_type(
        feature_config_type.messages)
  if feature_config_type.system_prompt is not None:
    record['system_prompt'] = encode_feature_support_type(
        feature_config_type.system_prompt)
  if feature_config_type.add_system_to_messages is not None:
    record['add_system_to_messages'] = (
        feature_config_type.add_system_to_messages)
  if feature_config_type.parameters is not None:
    record['parameters'] = encode_parameter_config_type(
        feature_config_type.parameters)
  if feature_config_type.tools is not None:
    record['tools'] = encode_tool_config_type(
        feature_config_type.tools)
  if feature_config_type.output_format is not None:
    record['output_format'] = encode_output_format_config_type(
        feature_config_type.output_format)
  if feature_config_type.input_format is not None:
    record['input_format'] = encode_input_format_config_type(
        feature_config_type.input_format)
  return record


def decode_feature_config_type(
    record: dict[str, Any]
) -> types.FeatureConfigType:
  """Deserialize FeatureConfigType from a dictionary."""
  feature_config_type = types.FeatureConfigType()
  if 'prompt' in record:
    feature_config_type.prompt = decode_feature_support_type(
        record['prompt'])
  if 'messages' in record:
    feature_config_type.messages = decode_feature_support_type(
        record['messages'])
  if 'system_prompt' in record:
    feature_config_type.system_prompt = decode_feature_support_type(
        record['system_prompt'])
  if 'add_system_to_messages' in record:
    feature_config_type.add_system_to_messages = (
        record['add_system_to_messages'])
  if 'parameters' in record:
    feature_config_type.parameters = decode_parameter_config_type(
        record['parameters'])
  if 'tools' in record:
    feature_config_type.tools = decode_tool_config_type(
        record['tools'])
  if 'output_format' in record:
    feature_config_type.output_format = decode_output_format_config_type(
        record['output_format'])
  elif 'response_format' in record:
    feature_config_type.output_format = decode_output_format_config_type(
        record['response_format'])
  if 'input_format' in record:
    feature_config_type.input_format = decode_input_format_config_type(
        record['input_format'])
  return feature_config_type


def encode_provider_model_metadata_type(
    provider_model_metadata_type: types.ProviderModelMetadataType
) -> dict[str, Any]:
  """Serialize ProviderModelMetadataType to a dictionary."""
  record = {}
  if provider_model_metadata_type.is_recommended is not None:
    record['is_recommended'] = provider_model_metadata_type.is_recommended
  if provider_model_metadata_type.model_size_tags is not None:
    record['model_size_tags'] = [
        model_size_tag.value
        for model_size_tag in provider_model_metadata_type.model_size_tags
    ]
  if provider_model_metadata_type.tags is not None:
    record['tags'] = provider_model_metadata_type.tags
  return record


def decode_provider_model_metadata_type(
    record: dict[str, Any]
) -> types.ProviderModelMetadataType:
  """Deserialize ProviderModelMetadataType from a dictionary."""
  provider_model_metadata_type = types.ProviderModelMetadataType()
  # Gracefully ignore old keys: call_type, response_type, input_type
  if 'is_recommended' in record:
    provider_model_metadata_type.is_recommended = record['is_recommended']
  if 'model_size_tags' in record and record['model_size_tags'] is not None:
    provider_model_metadata_type.model_size_tags = [
        types.ModelSizeType(model_size_tag)
        for model_size_tag in record['model_size_tags']
    ]
  if 'tags' in record:
    provider_model_metadata_type.tags = record['tags']
  return provider_model_metadata_type


def encode_provider_model_config(
    provider_model_config: types.ProviderModelConfig
) -> dict[str, Any]:
  """Serialize ProviderModelConfig to a dictionary."""
  record = {}
  if provider_model_config.provider_model is not None:
    record['provider_model'] = encode_provider_model_type(
        provider_model_config.provider_model
    )
  if provider_model_config.pricing is not None:
    record['pricing'] = encode_provider_model_pricing_type(
        provider_model_config.pricing
    )
  if provider_model_config.features is not None:
    record['features'] = encode_feature_config_type(
        provider_model_config.features
    )
  if provider_model_config.metadata is not None:
    record['metadata'] = encode_provider_model_metadata_type(
        provider_model_config.metadata
    )
  return record


def decode_provider_model_config(
    record: dict[str, Any]
) -> types.ProviderModelConfig:
  """Deserialize ProviderModelConfig from a dictionary."""
  provider_model = None
  pricing = None
  features = None
  metadata = None
  if 'provider_model' in record:
    provider_model = decode_provider_model_type(
        record['provider_model']
    )
  if 'pricing' in record:
    pricing = decode_provider_model_pricing_type(
        record['pricing']
    )
  if 'features' in record:
    features = decode_feature_config_type(
        record['features']
    )
  if 'metadata' in record:
    metadata = decode_provider_model_metadata_type(
        record['metadata']
    )
  return types.ProviderModelConfig(
      provider_model=provider_model,
      pricing=pricing,
      features=features,
      metadata=metadata,
  )


def encode_provider_model_configs_mapping_type(
    provider_model_configs: types.ProviderModelConfigsMappingType
) -> dict[str, Any]:
  """Serialize ProviderModelConfigsMappingType to a dictionary."""
  record = {}
  for provider, model_configs_dict in provider_model_configs.items():
    record[provider] = {}
    for model, config in model_configs_dict.items():
      record[provider][model] = encode_provider_model_config(config)
  return record


def decode_provider_model_configs_mapping_type(
    record: dict[str, Any]
) -> types.ProviderModelConfigsMappingType:
  """Deserialize ProviderModelConfigsMappingType from a dictionary."""
  provider_model_configs = {}
  for provider, model_configs_dict_record in record.items():
    provider_model_configs[provider] = {}
    for model, config_record in model_configs_dict_record.items():
      provider_model_configs[provider][model] = (
          decode_provider_model_config(config_record)
      )
  return provider_model_configs


def encode_recommended_models_mapping_type(
    recommended_models: types.RecommendedModelsMappingType
) -> dict[str, Any]:
  """Serialize RecommendedModelsMappingType to a dictionary."""
  record = {}
  for provider, provider_model_identifiers in recommended_models.items():
    record[provider] = []
    for provider_model_identifier in provider_model_identifiers:
      record[provider].append(
          encode_provider_model_type(provider_model_identifier)
      )
  return record


def decode_recommended_models_mapping_type(
    record: dict[str, Any]
) -> types.RecommendedModelsMappingType:
  """Deserialize RecommendedModelsMappingType from a dictionary."""
  recommended_models = {}
  for provider, provider_model_identifier_records in record.items():
    provider_model_identifiers = []
    for provider_model_identifier_record in (provider_model_identifier_records):
      provider_model_identifiers.append(
          decode_provider_model_type(provider_model_identifier_record)
      )
    recommended_models[provider] = provider_model_identifiers
  return recommended_models


def encode_output_format_type_mapping_type(
    output_format_type_mapping: types.OutputFormatTypeMappingType
) -> dict[str, Any]:
  """Serialize OutputFormatTypeMappingType to a dictionary."""
  record = {}
  for output_format_type, provider_models in (
      output_format_type_mapping.items()
  ):
    record[output_format_type.value] = []
    for provider_model in provider_models:
      record[output_format_type.value].append(
          encode_provider_model_type(provider_model)
      )
  return record


def decode_output_format_type_mapping_type(
    record: dict[str, Any]
) -> types.OutputFormatTypeMappingType:
  """Deserialize OutputFormatTypeMappingType from a dictionary."""
  output_format_type_mapping = {}
  for output_format_type_str, provider_model_records in record.items():
    output_format_type = types.OutputFormatType(output_format_type_str)
    provider_models = []
    for provider_model_record in provider_model_records:
      provider_models.append(
          decode_provider_model_type(provider_model_record)
      )
    output_format_type_mapping[output_format_type] = provider_models
  return output_format_type_mapping


def encode_model_size_mapping_type(
    model_size_mapping: types.ModelSizeMappingType
) -> dict[str, Any]:
  """Serialize ModelSizeMappingType to a dictionary."""
  record = {}
  for model_size, provider_models in model_size_mapping.items():
    record[model_size.value] = []
    for provider_model in provider_models:
      record[model_size.value].append(
          encode_provider_model_type(provider_model)
      )
  return record


def decode_model_size_mapping_type(
    record: dict[str, Any]
) -> types.ModelSizeMappingType:
  """Deserialize ModelSizeMappingType from a dictionary."""
  model_size_mapping = {}
  for model_size_str, provider_model_records in record.items():
    model_size = types.ModelSizeType(model_size_str)
    provider_models = []
    for provider_model_record in provider_model_records:
      provider_models.append(
          decode_provider_model_type(provider_model_record)
      )
    model_size_mapping[model_size] = provider_models
  return model_size_mapping


def encode_default_model_priority_list(
    default_model_priority_list: list[types.ProviderModelType]
) -> list[dict[str, Any]]:
  """Serialize default model priority list to a list."""
  record = []
  for provider_model in default_model_priority_list:
    record.append(encode_provider_model_type(provider_model))
  return record


def decode_default_model_priority_list(
    record: list[dict[str, Any]]
) -> list[types.ProviderModelType]:
  """Deserialize default model priority list from a list."""
  default_model_priority_list = []
  for provider_model_record in record:
    default_model_priority_list.append(
        decode_provider_model_type(provider_model_record)
    )
  return default_model_priority_list


def encode_model_configs_schema_metadata_type(
    model_configs_schema_metadata_type: types.ModelConfigsSchemaMetadataType
) -> dict[str, Any]:
  """Serialize ModelConfigsSchemaMetadataType to a dictionary."""
  record = {}
  if model_configs_schema_metadata_type.version is not None:
    record['version'] = model_configs_schema_metadata_type.version
  if model_configs_schema_metadata_type.released_at is not None:
    record['released_at'] = (
        model_configs_schema_metadata_type.released_at.isoformat()
    )
  if model_configs_schema_metadata_type.min_proxai_version is not None:
    record['min_proxai_version'] = (
        model_configs_schema_metadata_type.min_proxai_version
    )
  if model_configs_schema_metadata_type.config_origin is not None:
    record['config_origin'] = (
        model_configs_schema_metadata_type.config_origin.value
    )
  if model_configs_schema_metadata_type.release_notes is not None:
    record['release_notes'] = model_configs_schema_metadata_type.release_notes
  return record


def decode_model_configs_schema_metadata_type(
    record: dict[str, Any]
) -> types.ModelConfigsSchemaMetadataType:
  """Deserialize ModelConfigsSchemaMetadataType from a dictionary."""
  model_configs_schema_metadata_type = types.ModelConfigsSchemaMetadataType()
  if 'version' in record:
    model_configs_schema_metadata_type.version = record['version']
  if 'released_at' in record:
    model_configs_schema_metadata_type.released_at = (
        datetime.datetime.fromisoformat(record['released_at'])
    )
  if 'min_proxai_version' in record:
    model_configs_schema_metadata_type.min_proxai_version = (
        record['min_proxai_version']
    )
  if 'config_origin' in record:
    model_configs_schema_metadata_type.config_origin = (
        types.ConfigOriginType(record['config_origin'])
    )
  if 'release_notes' in record:
    model_configs_schema_metadata_type.release_notes = record['release_notes']
  return model_configs_schema_metadata_type


def encode_model_registry(
    model_registry: types.ModelRegistry
) -> dict[str, Any]:
  """Serialize ModelRegistry to a dictionary."""
  record = {}
  if model_registry.metadata is not None:
    record['metadata'] = encode_model_configs_schema_metadata_type(
        model_registry.metadata
    )
  if model_registry.default_model_priority_list is not None:
    record['default_model_priority_list'] = (
        encode_default_model_priority_list(
            model_registry.default_model_priority_list
        )
    )
  if model_registry.provider_model_configs is not None:
    record['provider_model_configs'] = (
        encode_provider_model_configs_mapping_type(
            model_registry.provider_model_configs
        )
    )
  return record


def decode_model_registry(
    record: dict[str, Any]
) -> types.ModelRegistry:
  """Deserialize ModelRegistry from a dictionary."""
  metadata = None
  default_model_priority_list = None
  provider_model_configs = None
  if 'metadata' in record:
    metadata = decode_model_configs_schema_metadata_type(record['metadata'])
  if 'default_model_priority_list' in record:
    default_model_priority_list = decode_default_model_priority_list(
        record['default_model_priority_list']
    )
  if 'provider_model_configs' in record:
    provider_model_configs = decode_provider_model_configs_mapping_type(
        record['provider_model_configs']
    )
  return types.ModelRegistry(
      metadata=metadata,
      default_model_priority_list=default_model_priority_list,
      provider_model_configs=provider_model_configs,
  )


def encode_parameter_type(
    parameter_type: types.ParameterType
) -> dict[str, Any]:
  """Serialize ParameterType to a dictionary."""
  record = {}
  if parameter_type.temperature is not None:
    record['temperature'] = str(parameter_type.temperature)
  if parameter_type.max_tokens is not None:
    record['max_tokens'] = str(parameter_type.max_tokens)
  if parameter_type.stop is not None:
    record['stop'] = parameter_type.stop
  if parameter_type.n is not None:
    record['n'] = str(parameter_type.n)
  if parameter_type.thinking is not None:
    record['thinking'] = parameter_type.thinking.value
  return record


def decode_parameter_type(
    record: dict[str, Any]
) -> types.ParameterType:
  """Deserialize ParameterType from a dictionary."""
  parameter_type = types.ParameterType()
  if 'temperature' in record:
    parameter_type.temperature = float(record['temperature'])
  if 'max_tokens' in record:
    parameter_type.max_tokens = int(record['max_tokens'])
  parameter_type.stop = record.get('stop')
  if 'n' in record:
    parameter_type.n = int(record['n'])
  if 'thinking' in record:
    parameter_type.thinking = types.ThinkingType(record['thinking'])
  return parameter_type


def encode_connection_options(
    connection_options: types.ConnectionOptions
) -> dict[str, Any]:
  """Serialize ConnectionOptions to a dictionary."""
  record = {}
  if connection_options.fallback_models is not None:
    record['fallback_models'] = []
    for provider_model in connection_options.fallback_models:
      record['fallback_models'].append(
          encode_provider_model_type(provider_model)
      )
  if connection_options.suppress_provider_errors is not None:
    record['suppress_provider_errors'] = (
        connection_options.suppress_provider_errors
    )
  if connection_options.endpoint is not None:
    record['endpoint'] = connection_options.endpoint
  if connection_options.skip_cache is not None:
    record['skip_cache'] = connection_options.skip_cache
  if connection_options.override_cache_value is not None:
    record['override_cache_value'] = connection_options.override_cache_value
  return record


def decode_connection_options(
    record: dict[str, Any]
) -> types.ConnectionOptions:
  """Deserialize ConnectionOptions from a dictionary."""
  connection_options = types.ConnectionOptions()
  if 'fallback_models' in record:
    connection_options.fallback_models = []
    for provider_model_record in record['fallback_models']:
      connection_options.fallback_models.append(
          decode_provider_model_type(provider_model_record)
      )
  if 'suppress_provider_errors' in record:
    connection_options.suppress_provider_errors = (
        record['suppress_provider_errors']
    )
  connection_options.endpoint = record.get('endpoint')
  if 'skip_cache' in record:
    connection_options.skip_cache = record['skip_cache']
  if 'override_cache_value' in record:
    connection_options.override_cache_value = record['override_cache_value']
  return connection_options


def encode_output_format(
    output_format: types.OutputFormat
) -> dict[str, Any]:
  """Serialize OutputFormat to a dictionary."""
  record = {}
  if output_format.type is not None:
    record['type'] = output_format.type.value
  # Extract from live class if available, else use stored metadata
  pydantic_class_name = output_format.pydantic_class_name
  pydantic_class_json_schema = output_format.pydantic_class_json_schema
  if output_format.pydantic_class is not None:
    pydantic_class_name = output_format.pydantic_class.__name__
    pydantic_class_json_schema = (
        output_format.pydantic_class.model_json_schema()
    )
  if pydantic_class_name is not None:
    record['pydantic_class_name'] = pydantic_class_name
  if pydantic_class_json_schema is not None:
    record['pydantic_class_json_schema'] = json.dumps(
        pydantic_class_json_schema, sort_keys=True
    )
  return record


def decode_output_format(record: dict[str, Any]) -> types.OutputFormat:
  """Deserialize OutputFormat from a dictionary."""
  output_format = types.OutputFormat()
  if 'type' in record:
    output_format.type = types.OutputFormatType(record['type'])
  if 'pydantic_class_name' in record:
    output_format.pydantic_class_name = record['pydantic_class_name']
  if 'pydantic_class_json_schema' in record:
    output_format.pydantic_class_json_schema = json.loads(
        record['pydantic_class_json_schema']
    )
  return output_format


def encode_result_media_content_type(
    result_media_content_type: types.ResultMediaContentType
) -> dict[str, Any]:
  """Serialize ResultMediaContentType to a dictionary."""
  record = {}
  record['data'] = base64.b64encode(
      result_media_content_type.data
  ).decode('utf-8')
  record['media_type'] = result_media_content_type.media_type
  return record


def decode_result_media_content_type(
    record: dict[str, Any]
) -> types.ResultMediaContentType:
  """Deserialize ResultMediaContentType from a dictionary."""
  if 'data' not in record:
    raise ValueError(f'Data not found in record: {record=}')
  if 'media_type' not in record:
    raise ValueError(f'Media type not found in record: {record=}')
  return types.ResultMediaContentType(
      data=base64.b64decode(record['data']),
      media_type=record['media_type'],
  )


def encode_content(
    content: str | list[message_content.MessageContent | str] | None
) -> str | list | None:
  """Serialize content field (str or list of MessageContent/str)."""
  if content is None:
    return None
  if isinstance(content, str):
    return content
  result = []
  for item in content:
    if isinstance(item, str):
      result.append(item)
    else:
      result.append(item.to_dict())
  return result


def decode_content(
    value: str | list | None
) -> str | list[message_content.MessageContent | str] | None:
  """Deserialize content field (str or list of MessageContent/str)."""
  if value is None:
    return None
  if isinstance(value, str):
    return value
  result = []
  for item in value:
    if isinstance(item, dict):
      result.append(message_content.MessageContent.from_dict(item))
    else:
      result.append(item)
  return result


def encode_choice_type(
    choice_type: types.ChoiceType
) -> dict[str, Any]:
  """Serialize ChoiceType to a dictionary."""
  record = {}
  if choice_type.output_text is not None:
    record['output_text'] = choice_type.output_text
  if choice_type.output_image is not None:
    record['output_image'] = choice_type.output_image.to_dict()
  if choice_type.output_audio is not None:
    record['output_audio'] = choice_type.output_audio.to_dict()
  if choice_type.output_video is not None:
    record['output_video'] = choice_type.output_video.to_dict()
  if choice_type.output_json is not None:
    record['output_json'] = choice_type.output_json
  if choice_type.output_pydantic is not None:
    record['output_pydantic'] = {
        'class_name': choice_type.output_pydantic.__class__.__name__,
        'instance_json_value': (
            choice_type.output_pydantic.model_dump(mode='json')
        )
    }
  if choice_type.content is not None:
    record['content'] = encode_content(choice_type.content)
  return record


def decode_choice_type(
    record: dict[str, Any]
) -> types.ChoiceType:
  """Deserialize ChoiceType from a dictionary."""
  choice_type = types.ChoiceType()
  choice_type.output_text = record.get('output_text')
  if 'output_image' in record:
    choice_type.output_image = message_content.MessageContent.from_dict(
        record['output_image']
    )
  if 'output_audio' in record:
    choice_type.output_audio = message_content.MessageContent.from_dict(
        record['output_audio']
    )
  if 'output_video' in record:
    choice_type.output_video = message_content.MessageContent.from_dict(
        record['output_video']
    )
  if 'output_json' in record:
    choice_type.output_json = record['output_json']
  # output_pydantic is intentionally not reconstructed: the encoded metadata
  # (class_name, instance_json_value) is preserved on the wire but the live
  # pydantic.BaseModel class cannot be rebuilt at decode time.
  if 'content' in record:
    choice_type.content = decode_content(record['content'])
  return choice_type


def encode_usage_type(
    usage_type: types.UsageType
) -> dict[str, Any]:
  """Serialize UsageType to a dictionary."""
  record = {}
  if usage_type.input_tokens is not None:
    record['input_tokens'] = str(usage_type.input_tokens)
  if usage_type.output_tokens is not None:
    record['output_tokens'] = str(usage_type.output_tokens)
  if usage_type.total_tokens is not None:
    record['total_tokens'] = str(usage_type.total_tokens)
  if usage_type.estimated_cost is not None:
    record['estimated_cost'] = str(usage_type.estimated_cost)
  return record


def decode_usage_type(
    record: dict[str, Any]
) -> types.UsageType:
  """Deserialize UsageType from a dictionary."""
  usage_type = types.UsageType()
  if 'input_tokens' in record:
    usage_type.input_tokens = int(record['input_tokens'])
  if 'output_tokens' in record:
    usage_type.output_tokens = int(record['output_tokens'])
  if 'total_tokens' in record:
    usage_type.total_tokens = int(record['total_tokens'])
  if 'estimated_cost' in record:
    usage_type.estimated_cost = int(record['estimated_cost'])
  return usage_type


def encode_tool_usage_type(
    tool_usage_type: types.ToolUsageType
) -> dict[str, Any]:
  """Serialize ToolUsageType to a dictionary."""
  record = {}
  if tool_usage_type.web_search_count is not None:
    record['web_search_count'] = str(tool_usage_type.web_search_count)
  if tool_usage_type.web_search_citations is not None:
    record['web_search_citations'] = tool_usage_type.web_search_citations
  return record


def decode_tool_usage_type(
    record: dict[str, Any]
) -> types.ToolUsageType:
  """Deserialize ToolUsageType from a dictionary."""
  tool_usage_type = types.ToolUsageType()
  if 'web_search_count' in record:
    tool_usage_type.web_search_count = int(record['web_search_count'])
  tool_usage_type.web_search_citations = record.get('web_search_citations')
  return tool_usage_type


def encode_timestamp_type(
    timestamp_type: types.TimeStampType
) -> dict[str, Any]:
  """Serialize TimeStampType to a dictionary."""
  record = {}
  if timestamp_type.start_utc_date is not None:
    record['start_utc_date'] = timestamp_type.start_utc_date.isoformat()
  if timestamp_type.end_utc_date is not None:
    record['end_utc_date'] = timestamp_type.end_utc_date.isoformat()
  if timestamp_type.local_time_offset_minute is not None:
    record['local_time_offset_minute'] = (
        timestamp_type.local_time_offset_minute
    )
  if timestamp_type.response_time is not None:
    record['response_time'] = timestamp_type.response_time.total_seconds()
  if timestamp_type.cache_response_time is not None:
    record['cache_response_time'] = (
        timestamp_type.cache_response_time.total_seconds()
    )
  return record


def decode_timestamp_type(
    record: dict[str, Any]
) -> types.TimeStampType:
  """Deserialize TimeStampType from a dictionary."""
  timestamp_type = types.TimeStampType()
  if 'start_utc_date' in record:
    timestamp_type.start_utc_date = datetime.datetime.fromisoformat(
        record['start_utc_date']
    )
  if 'end_utc_date' in record:
    timestamp_type.end_utc_date = datetime.datetime.fromisoformat(
        record['end_utc_date']
    )
  if 'local_time_offset_minute' in record:
    timestamp_type.local_time_offset_minute = (
        record['local_time_offset_minute']
    )
  if 'response_time' in record:
    timestamp_type.response_time = datetime.timedelta(
        seconds=record['response_time']
    )
  if 'cache_response_time' in record:
    timestamp_type.cache_response_time = datetime.timedelta(
        seconds=record['cache_response_time']
    )
  return timestamp_type


def encode_result_record(
    result_record: types.ResultRecord
) -> dict[str, Any]:
  """Serialize ResultRecord to a dictionary."""
  record = {}
  if result_record.status is not None:
    record['status'] = result_record.status.value
  if result_record.role is not None:
    record['role'] = result_record.role.value
  if result_record.output_text is not None:
    record['output_text'] = result_record.output_text
  if result_record.output_image is not None:
    record['output_image'] = result_record.output_image.to_dict()
  if result_record.output_audio is not None:
    record['output_audio'] = result_record.output_audio.to_dict()
  if result_record.output_video is not None:
    record['output_video'] = result_record.output_video.to_dict()
  if result_record.output_json is not None:
    record['output_json'] = result_record.output_json
  if result_record.output_pydantic is not None:
    record['output_pydantic'] = {
        'class_name': result_record.output_pydantic.__class__.__name__,
        'instance_json_value': (
            result_record.output_pydantic.model_dump(mode='json')
        )
    }
  if result_record.content is not None:
    record['content'] = encode_content(result_record.content)
  if result_record.choices is not None:
    record['choices'] = []
    for choice in result_record.choices:
      record['choices'].append(encode_choice_type(choice))
  if result_record.error is not None:
    record['error'] = result_record.error
  if result_record.error_traceback is not None:
    record['error_traceback'] = result_record.error_traceback
  if result_record.usage is not None:
    record['usage'] = encode_usage_type(result_record.usage)
  if result_record.tool_usage is not None:
    record['tool_usage'] = encode_tool_usage_type(result_record.tool_usage)
  if result_record.timestamp is not None:
    record['timestamp'] = encode_timestamp_type(result_record.timestamp)
  return record


def decode_result_record(
    record: dict[str, Any]
) -> types.ResultRecord:
  """Deserialize ResultRecord from a dictionary."""
  result_record = types.ResultRecord()
  if 'status' in record:
    result_record.status = types.ResultStatusType(record['status'])
  if 'role' in record:
    result_record.role = types.MessageRoleType(record['role'])
  result_record.output_text = record.get('output_text')
  if 'output_image' in record:
    result_record.output_image = message_content.MessageContent.from_dict(
        record['output_image'])
  if 'output_audio' in record:
    result_record.output_audio = message_content.MessageContent.from_dict(
        record['output_audio'])
  if 'output_video' in record:
    result_record.output_video = message_content.MessageContent.from_dict(
        record['output_video'])
  if 'output_json' in record:
    result_record.output_json = record['output_json']
  if 'content' in record:
    result_record.content = decode_content(record['content'])
  if 'choices' in record:
    result_record.choices = []
    for choice_record in record['choices']:
      result_record.choices.append(decode_choice_type(choice_record))
  result_record.error = record.get('error')
  result_record.error_traceback = record.get('error_traceback')
  if 'usage' in record:
    result_record.usage = decode_usage_type(record['usage'])
  if 'tool_usage' in record:
    result_record.tool_usage = decode_tool_usage_type(record['tool_usage'])
  if 'timestamp' in record:
    result_record.timestamp = decode_timestamp_type(record['timestamp'])
  return result_record


def encode_query_record(query_record: types.QueryRecord) -> dict[str, Any]:
  """Serialize QueryRecord to a dictionary."""
  record = {}
  if query_record.prompt is not None:
    record['prompt'] = query_record.prompt
  if query_record.chat is not None:
    record['chat'] = query_record.chat.to_dict()
  if query_record.system_prompt is not None:
    record['system_prompt'] = query_record.system_prompt
  if query_record.provider_model is not None:
    record['provider_model'] = encode_provider_model_type(
        query_record.provider_model
    )
  if query_record.parameters is not None:
    record['parameters'] = encode_parameter_type(query_record.parameters)
  if query_record.tools is not None:
    record['tools'] = [tool.value for tool in query_record.tools]
  if query_record.output_format is not None:
    record['output_format'] = encode_output_format(
        query_record.output_format
    )
  if query_record.connection_options is not None:
    record['connection_options'] = encode_connection_options(
        query_record.connection_options
    )
  if query_record.hash_value is not None:
    record['hash_value'] = query_record.hash_value
  return record


def decode_query_record(record: dict[str, Any]) -> types.QueryRecord:
  """Deserialize QueryRecord from a dictionary."""
  query_record = types.QueryRecord()
  query_record.prompt = record.get('prompt')
  if 'chat' in record:
    query_record.chat = chat_session.Chat.from_dict(record['chat'])
  query_record.system_prompt = record.get('system_prompt')
  if 'provider_model' in record:
    query_record.provider_model = decode_provider_model_type(
        record['provider_model']
    )
  if 'parameters' in record:
    query_record.parameters = decode_parameter_type(record['parameters'])
  if 'tools' in record:
    query_record.tools = [
        types.Tools(tool) for tool in record['tools']
    ]
  if 'output_format' in record:
    query_record.output_format = decode_output_format(
        record['output_format']
    )
  elif 'response_format' in record:
    query_record.output_format = decode_output_format(
        record['response_format']
    )
  if 'connection_options' in record:
    query_record.connection_options = decode_connection_options(
        record['connection_options']
    )
  query_record.hash_value = record.get('hash_value')
  return query_record


def encode_connection_metadata(
    connection_metadata: types.ConnectionMetadata
) -> dict[str, Any]:
  """Serialize ConnectionMetadata to a dictionary."""
  record = {}
  if connection_metadata.result_source is not None:
    record['result_source'] = connection_metadata.result_source.value
  if connection_metadata.cache_look_fail_reason is not None:
    record['cache_look_fail_reason'] = (
        connection_metadata.cache_look_fail_reason.value
    )
  if connection_metadata.endpoint_used is not None:
    record['endpoint_used'] = connection_metadata.endpoint_used
  if connection_metadata.failed_fallback_models is not None:
    record['failed_fallback_models'] = []
    for provider_model in connection_metadata.failed_fallback_models:
      record['failed_fallback_models'].append(
          encode_provider_model_type(provider_model)
      )
  if connection_metadata.feature_mapping_strategy is not None:
    record['feature_mapping_strategy'] = (
        connection_metadata.feature_mapping_strategy.value
    )
  return record


def decode_connection_metadata(
    record: dict[str, Any]
) -> types.ConnectionMetadata:
  """Deserialize ConnectionMetadata from a dictionary."""
  connection_metadata = types.ConnectionMetadata()
  if 'result_source' in record:
    connection_metadata.result_source = (
        types.ResultSource(record['result_source'])
    )
  if 'cache_look_fail_reason' in record:
    connection_metadata.cache_look_fail_reason = (
        types.CacheLookFailReason(record['cache_look_fail_reason'])
    )
  connection_metadata.endpoint_used = record.get('endpoint_used')
  if 'failed_fallback_models' in record:
    connection_metadata.failed_fallback_models = []
    for provider_model_record in record['failed_fallback_models']:
      connection_metadata.failed_fallback_models.append(
          decode_provider_model_type(provider_model_record)
      )
  if 'feature_mapping_strategy' in record:
    connection_metadata.feature_mapping_strategy = (
        types.FeatureMappingStrategy(record['feature_mapping_strategy'])
    )
  return connection_metadata


def encode_call_record(
    call_record: types.CallRecord
) -> dict[str, Any]:
  """Serialize CallRecord to a dictionary."""
  record = {}
  if call_record.query is not None:
    record['query'] = encode_query_record(call_record.query)
  if call_record.result is not None:
    record['result'] = encode_result_record(call_record.result)
  if call_record.connection is not None:
    record['connection'] = encode_connection_metadata(call_record.connection)
  # call_record.debug is intentionally NOT serialized: the
  # raw_provider_response field on DebugInfo holds a live provider SDK
  # object that is not portable across the cache or ProxDash boundary.
  # The keep_raw_provider_response client flag is mutually exclusive
  # with cache_options at construction time, so this branch is normally
  # unreachable for cached records anyway.
  return record


def decode_call_record(
    record: dict[str, Any]
) -> types.CallRecord:
  """Deserialize CallRecord from a dictionary."""
  call_record = types.CallRecord()
  if 'query' in record:
    call_record.query = decode_query_record(record['query'])
  if 'result' in record:
    call_record.result = decode_result_record(record['result'])
  if 'connection' in record:
    call_record.connection = decode_connection_metadata(record['connection'])
  return call_record


def encode_cache_record(cache_record: types.CacheRecord) -> dict[str, Any]:
  """Serialize CacheRecord to a dictionary."""
  record = {}
  if cache_record.query is not None:
    record['query'] = encode_query_record(cache_record.query)
  if cache_record.results is not None:
    record['results'] = []
    for result_record in cache_record.results:
      record['results'].append(encode_result_record(result_record))
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


def decode_cache_record(record: dict[str, Any]) -> types.CacheRecord:
  """Deserialize CacheRecord from a dictionary."""
  cache_record = types.CacheRecord()
  if 'query' in record:
    cache_record.query = decode_query_record(record['query'])
  if 'results' in record:
    cache_record.results = []
    for result_record in record['results']:
      cache_record.results.append(decode_result_record(result_record))
  if 'shard_id' in record:
    try:
      cache_record.shard_id = int(record['shard_id'])
    except ValueError:
      cache_record.shard_id = record['shard_id']
  if 'last_access_time' in record:
    cache_record.last_access_time = datetime.datetime.fromisoformat(
        record['last_access_time']
    )
  if 'call_count' in record:
    cache_record.call_count = int(record['call_count'])
  return cache_record


def encode_light_cache_record(
    light_cache_record: types.LightCacheRecord
) -> dict[str, Any]:
  """Serialize LightCacheRecord to a dictionary."""
  record = {}
  if light_cache_record.query_hash is not None:
    record['query_hash'] = light_cache_record.query_hash
  if light_cache_record.results_count is not None:
    record['results_count'] = light_cache_record.results_count
  if light_cache_record.shard_id is not None:
    try:
      record['shard_id'] = int(light_cache_record.shard_id)
    except ValueError:
      record['shard_id'] = light_cache_record.shard_id
  if light_cache_record.last_access_time is not None:
    record['last_access_time'] = (
        light_cache_record.last_access_time.isoformat()
    )
  if light_cache_record.call_count is not None:
    record['call_count'] = light_cache_record.call_count
  return record


def decode_light_cache_record(record: dict[str, Any]) -> types.LightCacheRecord:
  """Deserialize LightCacheRecord from a dictionary."""
  light_cache_record = types.LightCacheRecord()
  light_cache_record.query_hash = record.get('query_hash')
  if 'results_count' in record:
    light_cache_record.results_count = int(record['results_count'])
  if 'shard_id' in record:
    try:
      light_cache_record.shard_id = int(record['shard_id'])
    except ValueError:
      light_cache_record.shard_id = record['shard_id']
  if 'last_access_time' in record:
    light_cache_record.last_access_time = datetime.datetime.fromisoformat(
        record['last_access_time']
    )
  if 'call_count' in record:
    light_cache_record.call_count = int(record['call_count'])
  return light_cache_record


def encode_cache_look_result(
    cache_look_result: types.CacheLookResult
) -> dict[str, Any]:
  """Serialize CacheLookResult to a dictionary."""
  record = {}
  if cache_look_result.result is not None:
    record['result'] = encode_result_record(cache_look_result.result)
  if cache_look_result.cache_look_fail_reason is not None:
    record['cache_look_fail_reason'] = (
        cache_look_result.cache_look_fail_reason.value
    )
  return record


def decode_cache_look_result(
    record: dict[str, Any]
) -> types.CacheLookResult:
  """Deserialize CacheLookResult from a dictionary."""
  cache_look_result = types.CacheLookResult()
  if 'result' in record:
    cache_look_result.result = decode_result_record(record['result'])
  if 'cache_look_fail_reason' in record:
    cache_look_result.cache_look_fail_reason = types.CacheLookFailReason(
        record['cache_look_fail_reason']
    )
  return cache_look_result


def encode_model_status(model_status: types.ModelStatus) -> dict[str, Any]:
  """Serialize ModelStatus to a dictionary."""
  record = {}
  if model_status.unprocessed_models is not None:
    record['unprocessed_models'] = []
    for provider_model in model_status.unprocessed_models:
      record['unprocessed_models'].append(
          encode_provider_model_type(provider_model)
      )
  if model_status.working_models is not None:
    record['working_models'] = []
    for provider_model in model_status.working_models:
      record['working_models'].append(
          encode_provider_model_type(provider_model)
      )
  if model_status.failed_models is not None:
    record['failed_models'] = []
    for provider_model in model_status.failed_models:
      record['failed_models'].append(encode_provider_model_type(provider_model))
  if model_status.filtered_models is not None:
    record['filtered_models'] = []
    for provider_model in model_status.filtered_models:
      record['filtered_models'].append(
          encode_provider_model_type(provider_model)
      )
  if model_status.provider_queries is not None:
    record['provider_queries'] = {}
    for provider_model, provider_query in model_status.provider_queries.items():
      provider_model = json.dumps(encode_provider_model_type(provider_model))
      record['provider_queries'][provider_model] = (
          encode_call_record(provider_query)
      )
  return record


def decode_model_status(record: dict[str, Any]) -> types.ModelStatus:
  """Deserialize ModelStatus from a dictionary."""
  model_status = types.ModelStatus()
  if 'unprocessed_models' in record:
    for provider_model_record in record['unprocessed_models']:
      model_status.unprocessed_models.add(
          decode_provider_model_type(provider_model_record)
      )
  if 'working_models' in record:
    for provider_model_record in record['working_models']:
      model_status.working_models.add(
          decode_provider_model_type(provider_model_record)
      )
  if 'failed_models' in record:
    for provider_model_record in record['failed_models']:
      model_status.failed_models.add(
          decode_provider_model_type(provider_model_record)
      )
  if 'filtered_models' in record:
    for provider_model_record in record['filtered_models']:
      model_status.filtered_models.add(
          decode_provider_model_type(provider_model_record)
      )
  if 'provider_queries' in record:
    for provider_model, provider_query_record in record['provider_queries'
                                                       ].items():
      provider_model = json.loads(provider_model)
      provider_model = decode_provider_model_type(provider_model)
      model_status.provider_queries[provider_model] = (
          decode_call_record(provider_query_record)
      )
  return model_status


def encode_logging_options(
    logging_options: types.LoggingOptions
) -> dict[str, Any]:
  """Serialize LoggingOptions to a dictionary."""
  record = {}
  if logging_options.logging_path is not None:
    record['logging_path'] = logging_options.logging_path
  if logging_options.stdout is not None:
    record['stdout'] = logging_options.stdout
  if logging_options.hide_sensitive_content is not None:
    record['hide_sensitive_content'] = logging_options.hide_sensitive_content
  return record


def decode_logging_options(record: dict[str, Any]) -> types.LoggingOptions:
  """Deserialize LoggingOptions from a dictionary."""
  logging_options = types.LoggingOptions()
  if 'logging_path' in record:
    logging_options.logging_path = record['logging_path']
  if 'stdout' in record:
    logging_options.stdout = record['stdout']
  if 'hide_sensitive_content' in record:
    logging_options.hide_sensitive_content = record['hide_sensitive_content']
  return logging_options


def encode_cache_options(cache_options: types.CacheOptions) -> dict[str, Any]:
  """Serialize CacheOptions to a dictionary."""
  record = {}
  if cache_options.cache_path is not None:
    record['cache_path'] = cache_options.cache_path
  if cache_options.unique_response_limit is not None:
    record['unique_response_limit'] = cache_options.unique_response_limit
  if cache_options.retry_if_error_cached is not None:
    record['retry_if_error_cached'] = cache_options.retry_if_error_cached
  if cache_options.clear_query_cache_on_connect is not None:
    record['clear_query_cache_on_connect'] = (
        cache_options.clear_query_cache_on_connect
    )
  if cache_options.clear_model_cache_on_connect is not None:
    record['clear_model_cache_on_connect'] = (
        cache_options.clear_model_cache_on_connect
    )
  if cache_options.disable_model_cache is not None:
    record['disable_model_cache'] = cache_options.disable_model_cache
  if cache_options.model_cache_duration is not None:
    record['model_cache_duration'] = cache_options.model_cache_duration
  return record


def decode_cache_options(record: dict[str, Any]) -> types.CacheOptions:
  """Deserialize CacheOptions from a dictionary."""
  cache_options = types.CacheOptions()
  if 'cache_path' in record:
    cache_options.cache_path = record['cache_path']
  if 'unique_response_limit' in record:
    cache_options.unique_response_limit = record['unique_response_limit']
  if 'retry_if_error_cached' in record:
    cache_options.retry_if_error_cached = record['retry_if_error_cached']
  if 'clear_query_cache_on_connect' in record:
    cache_options.clear_query_cache_on_connect = (
        record['clear_query_cache_on_connect']
    )
  if 'clear_model_cache_on_connect' in record:
    cache_options.clear_model_cache_on_connect = (
        record['clear_model_cache_on_connect']
    )
  if 'disable_model_cache' in record:
    cache_options.disable_model_cache = record['disable_model_cache']
  if 'model_cache_duration' in record:
    cache_options.model_cache_duration = record['model_cache_duration']
  return cache_options


def encode_proxdash_options(
    proxdash_options: types.ProxDashOptions
) -> dict[str, Any]:
  """Serialize ProxDashOptions to a dictionary."""
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


def decode_proxdash_options(record: dict[str, Any]) -> types.ProxDashOptions:
  """Deserialize ProxDashOptions from a dictionary."""
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


def encode_summary_options(
    summary_options: types.SummaryOptions
) -> dict[str, Any]:
  """Serialize SummaryOptions to a dictionary."""
  record = {}
  record['json'] = summary_options.json
  return record


def decode_summary_options(
    record: dict[str, Any]
) -> types.SummaryOptions:
  """Deserialize SummaryOptions from a dictionary."""
  summary_options = types.SummaryOptions()
  if 'json' in record:
    summary_options.json = record['json']
  return summary_options


def encode_run_options(run_options: types.RunOptions) -> dict[str, Any]:
  """Serialize RunOptions to a dictionary."""
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
        run_options.logging_options
    )
  if run_options.cache_options is not None:
    record['cache_options'] = encode_cache_options(run_options.cache_options)
  if run_options.proxdash_options is not None:
    record['proxdash_options'] = encode_proxdash_options(
        run_options.proxdash_options
    )
  if run_options.allow_multiprocessing is not None:
    record['allow_multiprocessing'] = run_options.allow_multiprocessing
  if run_options.model_test_timeout is not None:
    record['model_test_timeout'] = run_options.model_test_timeout
  if run_options.feature_mapping_strategy is not None:
    record['feature_mapping_strategy'] = (
        run_options.feature_mapping_strategy.value
    )
  if run_options.suppress_provider_errors is not None:
    record['suppress_provider_errors'] = run_options.suppress_provider_errors
  if run_options.keep_raw_provider_response is not None:
    record['keep_raw_provider_response'] = (
        run_options.keep_raw_provider_response
    )
  return record


def decode_run_options(record: dict[str, Any]) -> types.RunOptions:
  """Deserialize RunOptions from a dictionary."""
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
        record['logging_options']
    )
  if 'cache_options' in record:
    run_options.cache_options = decode_cache_options(record['cache_options'])
  if 'proxdash_options' in record:
    run_options.proxdash_options = decode_proxdash_options(
        record['proxdash_options']
    )
  if 'allow_multiprocessing' in record:
    run_options.allow_multiprocessing = record['allow_multiprocessing']
  if 'model_test_timeout' in record:
    run_options.model_test_timeout = record['model_test_timeout']
  if 'feature_mapping_strategy' in record:
    run_options.feature_mapping_strategy = types.FeatureMappingStrategy(
        record['feature_mapping_strategy']
    )
  if 'suppress_provider_errors' in record:
    run_options.suppress_provider_errors = record['suppress_provider_errors']
  if 'keep_raw_provider_response' in record:
    run_options.keep_raw_provider_response = (
        record['keep_raw_provider_response']
    )
  return run_options
