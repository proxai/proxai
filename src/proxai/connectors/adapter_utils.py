"""Shared adapter utilities for feature and result adapters."""

import dataclasses

import proxai.types as types

SUPPORT_RANK = {
    types.FeatureSupportType.NOT_SUPPORTED: 0,
    types.FeatureSupportType.BEST_EFFORT: 1,
    types.FeatureSupportType.SUPPORTED: 2,
}

OUTPUT_FORMAT_FIELD_MAP = {
    types.OutputFormatType.TEXT: "text",
    types.OutputFormatType.IMAGE: "image",
    types.OutputFormatType.AUDIO: "audio",
    types.OutputFormatType.VIDEO: "video",
    types.OutputFormatType.JSON: "json",
    types.OutputFormatType.PYDANTIC: "pydantic",
    types.OutputFormatType.MULTI_MODAL: "multi_modal",
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


def merge_support_fields(
    a: types.FeatureSupportType | None,
    b: types.FeatureSupportType | None,
    dataclass_type: type,
) -> types.FeatureSupportType | None:
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


_INPUT_FORMAT_TYPE_TO_FIELD = {
    types.InputFormatType.TEXT: lambda c: (
        c.input_format.text if c.input_format else None),
    types.InputFormatType.IMAGE: lambda c: (
        c.input_format.image if c.input_format else None),
    types.InputFormatType.DOCUMENT: lambda c: (
        c.input_format.document if c.input_format else None),
    types.InputFormatType.AUDIO: lambda c: (
        c.input_format.audio if c.input_format else None),
    types.InputFormatType.VIDEO: lambda c: (
        c.input_format.video if c.input_format else None),
    types.InputFormatType.JSON: lambda c: (
        c.input_format.json if c.input_format else None),
    types.InputFormatType.PYDANTIC: lambda c: (
        c.input_format.pydantic if c.input_format else None),
}

_OUTPUT_FORMAT_TYPE_TO_FIELD = {
    types.OutputFormatType.TEXT: lambda c: (
        c.output_format.text if c.output_format else None),
    types.OutputFormatType.IMAGE: lambda c: (
        c.output_format.image if c.output_format else None),
    types.OutputFormatType.AUDIO: lambda c: (
        c.output_format.audio if c.output_format else None),
    types.OutputFormatType.VIDEO: lambda c: (
        c.output_format.video if c.output_format else None),
    types.OutputFormatType.JSON: lambda c: (
        c.output_format.json if c.output_format else None),
    types.OutputFormatType.PYDANTIC: lambda c: (
        c.output_format.pydantic if c.output_format else None),
    types.OutputFormatType.MULTI_MODAL: lambda c: (
        c.output_format.multi_modal if c.output_format else None),
}

_TOOL_TAG_TO_FIELD = {
    types.ToolTag.WEB_SEARCH: lambda c: (
        c.tools.web_search if c.tools else None),
}

_FEATURE_TAG_TO_FIELD = {
    types.FeatureTag.PROMPT: lambda c: c.prompt,
    types.FeatureTag.MESSAGES: lambda c: c.messages,
    types.FeatureTag.SYSTEM_PROMPT: lambda c: c.system_prompt,
    types.FeatureTag.TEMPERATURE: lambda c: (
        c.parameters.temperature if c.parameters else None),
    types.FeatureTag.MAX_TOKENS: lambda c: (
        c.parameters.max_tokens if c.parameters else None),
    types.FeatureTag.STOP: lambda c: (
        c.parameters.stop if c.parameters else None),
    types.FeatureTag.N: lambda c: (
        c.parameters.n if c.parameters else None),
    types.FeatureTag.THINKING: lambda c: (
        c.parameters.thinking if c.parameters else None),
}

def resolve_input_format_type_support(
    feature_config: types.FeatureConfigType,
    tag: types.InputFormatType,
) -> types.FeatureSupportType:
  """Return the support level for an input format type."""
  accessor = _INPUT_FORMAT_TYPE_TO_FIELD.get(tag)
  if accessor is None:
    raise ValueError(f"Unknown input format type: {tag}")
  return resolve_support(accessor(feature_config))


def resolve_output_format_type_support(
    feature_config: types.FeatureConfigType,
    tag: types.OutputFormatType,
) -> types.FeatureSupportType:
  """Return the support level for an output format type."""
  accessor = _OUTPUT_FORMAT_TYPE_TO_FIELD.get(tag)
  if accessor is None:
    raise ValueError(f"Unknown output format type: {tag}")
  return resolve_support(accessor(feature_config))


def resolve_tool_tag_support(
    feature_config: types.FeatureConfigType,
    tag: types.ToolTag,
) -> types.FeatureSupportType:
  """Return the support level for a tool tag."""
  accessor = _TOOL_TAG_TO_FIELD.get(tag)
  if accessor is None:
    raise ValueError(f"Unknown tool tag: {tag}")
  return resolve_support(accessor(feature_config))


def resolve_feature_tag_support(
    feature_config: types.FeatureConfigType,
    tag: types.FeatureTag,
) -> types.FeatureSupportType:
  """Return the support level for a feature tag."""
  accessor = _FEATURE_TAG_TO_FIELD.get(tag)
  if accessor is None:
    raise ValueError(f"Unknown feature tag: {tag}")
  return resolve_support(accessor(feature_config))


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
      output_format=merge_support_fields(
          endpoint_config.output_format, model_config.output_format,
          types.OutputFormatConfigType),
      input_format=merge_support_fields(
          endpoint_config.input_format, model_config.input_format,
          types.InputFormatConfigType),
  )
