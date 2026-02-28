"""Shared adapter utilities for feature and result adapters."""

import dataclasses

import proxai.types as types

SUPPORT_RANK = {
    types.FeatureSupportType.NOT_SUPPORTED: 0,
    types.FeatureSupportType.BEST_EFFORT: 1,
    types.FeatureSupportType.SUPPORTED: 2,
}

RESPONSE_FORMAT_FIELD_MAP = {
    types.ResponseFormatType.TEXT: "text",
    types.ResponseFormatType.IMAGE: "image",
    types.ResponseFormatType.AUDIO: "audio",
    types.ResponseFormatType.VIDEO: "video",
    types.ResponseFormatType.JSON: "json",
    types.ResponseFormatType.PYDANTIC: "pydantic",
    types.ResponseFormatType.MULTI_MODAL: "multi_modal",
}


def resolve_support(
    support: types.FeatureSupportType | None,
) -> types.FeatureSupportType:
  if support is None:
    return types.FeatureSupportType.NOT_SUPPORTED
  return support


def min_support(
    a: types.FeatureSupportType | None,
    b: types.FeatureSupportType | None,
) -> types.FeatureSupportType:
  """Return the lower-ranked support level, treating None as NOT_SUPPORTED."""
  a = a if a is not None else types.FeatureSupportType.NOT_SUPPORTED
  b = b if b is not None else types.FeatureSupportType.NOT_SUPPORTED
  if SUPPORT_RANK[a] <= SUPPORT_RANK[b]:
    return a
  return b


def merge_support_fields(a, b, dataclass_type):
  """Merge two nested config dataclasses field-by-field using min_support.

  Returns None if both inputs are None.
  """
  if a is None and b is None:
    return None
  a = a or dataclass_type()
  b = b or dataclass_type()
  merged = {}
  for field in dataclasses.fields(dataclass_type):
    merged[field.name] = min_support(
        getattr(a, field.name), getattr(b, field.name))
  return dataclass_type(**merged)


def merge_feature_configs(
    endpoint_config: types.FeatureConfigType,
    model_config: types.FeatureConfigType,
) -> types.FeatureConfigType:
  """Merge endpoint and model feature configs, taking the minimum support."""
  return types.FeatureConfigType(
      prompt=min_support(endpoint_config.prompt, model_config.prompt),
      messages=min_support(endpoint_config.messages, model_config.messages),
      system_prompt=min_support(
          endpoint_config.system_prompt, model_config.system_prompt),
      add_system_to_messages=(
          True if (endpoint_config.add_system_to_messages
                   or model_config.add_system_to_messages)
          else None),
      parameters=merge_support_fields(
          endpoint_config.parameters, model_config.parameters,
          types.ParameterConfigType),
      tools=merge_support_fields(
          endpoint_config.tools, model_config.tools,
          types.ToolConfigType),
      response_format=merge_support_fields(
          endpoint_config.response_format, model_config.response_format,
          types.ResponseFormatConfigType),
  )
