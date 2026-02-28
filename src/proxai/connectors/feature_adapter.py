"""Feature adapter for mapping query features to provider capabilities."""

import copy
import json

import proxai.types as types
from proxai.connectors.adapter_utils import (
    RESPONSE_FORMAT_FIELD_MAP,
    SUPPORT_RANK,
    merge_feature_configs,
    resolve_support,
    resolve_tag_support,
)


class FeatureAdapter:
  """Adapts query records to match a provider endpoint's feature support."""

  _NO_BEST_EFFORT_RESPONSE_FORMATS = (
      "text", "image", "audio", "video", "multi_modal",
  )

  def __init__(
      self,
      endpoint: str,
      endpoint_feature_config: types.FeatureConfigType | None = None,
      model_feature_config: types.FeatureConfigType | None = None,
  ):
    if endpoint_feature_config is None and model_feature_config is None:
      raise ValueError(
          "At least one of 'endpoint_feature_config' or "
          "'model_feature_config' must be set."
      )
    self.endpoint = endpoint
    self.endpoint_feature_config = endpoint_feature_config
    self.model_feature_config = model_feature_config
    if (endpoint_feature_config is not None
        and model_feature_config is not None):
      self.feature_config = merge_feature_configs(
          endpoint_feature_config, model_feature_config)
    elif model_feature_config is not None:
      self.feature_config = model_feature_config
    else:
      self.feature_config = endpoint_feature_config

  def get_feature_tags_support_level(
      self, feature_tags: list[types.FeatureTagType],
  ) -> types.FeatureSupportType:
    """Return the minimum support level across the given feature tags.

    Returns SUPPORTED if the list is empty.
    """
    if not feature_tags:
      return types.FeatureSupportType.SUPPORTED
    levels = [
        resolve_tag_support(self.feature_config, tag) for tag in feature_tags
    ]
    return min(levels, key=lambda l: SUPPORT_RANK[l])

  def get_query_record_support_level(
      self, query_record: types.QueryRecord,
  ) -> types.FeatureSupportType:
    """Return the minimum support level across all features used in the query.

    Checks each feature set in the query against the endpoint's feature config.
    Returns NOT_SUPPORTED < BEST_EFFORT < SUPPORTED minimum.
    If no features are set, returns SUPPORTED.
    """
    levels = []

    if query_record.prompt is not None:
      levels.append(resolve_support(self.feature_config.prompt))
    if query_record.chat is not None:
      levels.append(resolve_support(self.feature_config.messages))
      if query_record.chat.system_prompt is not None:
        levels.append(resolve_support(self.feature_config.system_prompt))
    if query_record.system_prompt is not None:
      levels.append(resolve_support(self.feature_config.system_prompt))

    if query_record.parameters is not None:
      self._collect_parameter_levels(query_record.parameters, levels)

    if query_record.tools:
      self._collect_tool_levels(query_record.tools, levels)

    if (query_record.response_format is None
        or query_record.response_format.type is None):
      raise ValueError("'response_format.type' must be set.")
    self._collect_response_format_level(
        query_record.response_format, levels)

    if not levels:
      return types.FeatureSupportType.SUPPORTED

    return min(levels, key=lambda l: SUPPORT_RANK[l])

  def _collect_parameter_levels(
      self,
      parameters: types.ParameterType,
      levels: list[types.FeatureSupportType],
  ):
    param_config = self.feature_config.parameters
    if parameters.temperature is not None:
      levels.append(resolve_support(
          param_config.temperature if param_config else None))
    if parameters.max_tokens is not None:
      levels.append(resolve_support(
          param_config.max_tokens if param_config else None))
    if parameters.stop is not None:
      levels.append(resolve_support(
          param_config.stop if param_config else None))
    if parameters.n is not None:
      levels.append(resolve_support(
          param_config.n if param_config else None))
    if parameters.thinking is not None:
      levels.append(resolve_support(
          param_config.thinking if param_config else None))

  def _collect_tool_levels(
      self,
      tools: list[types.Tools],
      levels: list[types.FeatureSupportType],
  ):
    tool_config = self.feature_config.tools
    for tool in tools:
      if tool == types.Tools.WEB_SEARCH:
        levels.append(resolve_support(
            tool_config.web_search if tool_config else None))

  def _collect_response_format_level(
      self,
      response_format: types.ResponseFormat,
      levels: list[types.FeatureSupportType],
  ):
    rf_config = self.feature_config.response_format
    field_name = RESPONSE_FORMAT_FIELD_MAP.get(response_format.type)
    if field_name and rf_config:
      levels.append(resolve_support(getattr(rf_config, field_name, None)))
    else:
      levels.append(types.FeatureSupportType.NOT_SUPPORTED)

  def _adapt_prompt(self, query_record: types.QueryRecord,
                    json_guidance: bool = False,
                    pydantic_schema: dict | None = None):
    """Adapt system_prompt and response format guidance for prompt queries."""
    if query_record.system_prompt is not None:
      level = resolve_support(self.feature_config.system_prompt)
      if level == types.FeatureSupportType.BEST_EFFORT:
        query_record.prompt = (
            f"{query_record.system_prompt}\n\n{query_record.prompt}")
        query_record.system_prompt = None
      elif level == types.FeatureSupportType.NOT_SUPPORTED:
        raise ValueError(
            f"Feature 'system_prompt' is not supported "
            f"by endpoint '{self.endpoint}'."
        )
    if json_guidance:
      query_record.prompt = (
          f"{query_record.prompt}\n\nYou must respond with valid JSON.")
    if pydantic_schema is not None:
      schema_str = json.dumps(pydantic_schema, indent=2)
      query_record.prompt = (
          f"{query_record.prompt}\n\n"
          f"You must respond with valid JSON that follows this schema:\n"
          f"{schema_str}")

  def _adapt_chat(self, query_record: types.QueryRecord,
                  json_guidance: bool = False,
                  pydantic_schema: dict | None = None):
    """Adapt chat, system_prompt, and response format guidance for chat."""
    # Resolve system_prompt support for chat.
    system_best_effort = False
    add_system_to_messages = False
    if query_record.chat.system_prompt is not None:
      level = resolve_support(self.feature_config.system_prompt)
      if level == types.FeatureSupportType.SUPPORTED:
        if self.feature_config.add_system_to_messages:
          add_system_to_messages = True
      elif level == types.FeatureSupportType.BEST_EFFORT:
        system_best_effort = True
      elif level == types.FeatureSupportType.NOT_SUPPORTED:
        raise ValueError(
            f"Feature 'system_prompt' is not supported "
            f"by endpoint '{self.endpoint}'."
        )
      

    # Resolve messages support and export.
    messages_best_effort = False
    messages_level = resolve_support(self.feature_config.messages)
    if messages_level == types.FeatureSupportType.BEST_EFFORT:
      messages_best_effort = True
    elif messages_level == types.FeatureSupportType.NOT_SUPPORTED:
      raise ValueError(
          f"Feature 'messages' is not supported "
          f"by endpoint '{self.endpoint}'."
      )

    exported_chat = query_record.chat.export(
        add_system_to_first_user_message=system_best_effort,
        add_system_to_messages=add_system_to_messages,
        add_json_guidance_to_system=json_guidance,
        add_json_guidance_to_user_prompt=json_guidance,
        add_json_schema_guidance_to_system=pydantic_schema,
        add_json_schema_guidance_to_user_prompt=pydantic_schema,
        export_single_prompt=messages_best_effort)
    if messages_best_effort:
      query_record.prompt = exported_chat
      query_record.chat = None
    else:
      query_record.chat = exported_chat

  def _adapt_tools(self, query_record: types.QueryRecord):
    """Adapt tools for the query."""
    for tool in query_record.tools:
      if tool != types.Tools.WEB_SEARCH:
        raise ValueError(
            f"Unknown tool: {tool}. Only 'WEB_SEARCH' is supported."
        )
    tool_config = self.feature_config.tools
    level = resolve_support(
        tool_config.web_search if tool_config else None)
    if level == types.FeatureSupportType.SUPPORTED:
      return
    if level == types.FeatureSupportType.NOT_SUPPORTED:
      raise ValueError(
          f"Feature 'web_search' is not supported "
          f"by endpoint '{self.endpoint}'."
      )
    raise Exception(
        'web_search feature config cannot be best effort. '
        'Code should never reach here.\n'
        'Open bug report at https://github.com/proxai/proxai/issues'
    )

  def _adapt_response_format(
      self, query_record: types.QueryRecord,
  ) -> tuple[bool, dict | None]:
    """Validate response format and return guidance flags.

    Returns (json_guidance, pydantic_schema):
      json_guidance: True if JSON guidance should be added.
      pydantic_schema: JSON schema dict if pydantic schema guidance needed.
    """
    if (query_record.response_format is None
        or query_record.response_format.type is None):
      return False, None

    rf_config = self.feature_config.response_format
    rf_type = query_record.response_format.type
    field_name = RESPONSE_FORMAT_FIELD_MAP.get(rf_type)
    if not field_name or not rf_config:
      level = types.FeatureSupportType.NOT_SUPPORTED
    else:
      level = resolve_support(getattr(rf_config, field_name, None))

    if field_name in self._NO_BEST_EFFORT_RESPONSE_FORMATS:
      if level == types.FeatureSupportType.BEST_EFFORT:
        raise Exception(
            f"'{field_name}' response format config cannot be best effort. "
            f"Code should never reach here.\n"
            f"Open bug report at https://github.com/proxai/proxai/issues"
        )

    if level == types.FeatureSupportType.NOT_SUPPORTED:
      raise ValueError(
          f"Feature 'response_format.{rf_type.value}' is not supported "
          f"by endpoint '{self.endpoint}'."
      )

    if rf_type == types.ResponseFormatType.JSON:
      return True, None

    if rf_type == types.ResponseFormatType.PYDANTIC:
      if level == types.FeatureSupportType.BEST_EFFORT:
        schema = query_record.response_format.pydantic_class.model_json_schema()
        return False, schema

    return False, None

  def _adapt_parameters(self, query_record: types.QueryRecord):
    params = query_record.parameters
    param_config = self.feature_config.parameters
    if params.temperature is not None:
      if self._should_remove(
          param_config.temperature if param_config else None, "temperature"):
        params.temperature = None
    if params.max_tokens is not None:
      if self._should_remove(
          param_config.max_tokens if param_config else None, "max_tokens"):
        params.max_tokens = None
    if params.stop is not None:
      if self._should_remove(
          param_config.stop if param_config else None, "stop"):
        params.stop = None
    if params.n is not None:
      if self._should_remove(
          param_config.n if param_config else None, "n"):
        params.n = None
    if params.thinking is not None:
      if self._should_remove(
          param_config.thinking if param_config else None, "thinking"):
        params.thinking = None
    if all(v is None for v in [
        params.temperature, params.max_tokens,
        params.stop, params.n, params.thinking]):
      query_record.parameters = None

  def _should_remove(self, support: types.FeatureSupportType | None,
                     feature_name: str) -> bool:
    """Return True if the feature should be removed (best-effort).

    Raises ValueError if the feature is not supported.
    """
    level = resolve_support(support)
    if level == types.FeatureSupportType.SUPPORTED:
      return False
    if level == types.FeatureSupportType.BEST_EFFORT:
      return True
    raise ValueError(
        f"Feature '{feature_name}' is not supported "
        f"by endpoint '{self.endpoint}'."
    )

  def adapt_query_record(
      self, query_record: types.QueryRecord,
  ) -> types.QueryRecord:
    """Remove best-effort features from the query, raise on not-supported.

    For each feature set in the query:
      SUPPORTED     → keep as-is.
      BEST_EFFORT   → set to None (remove from query).
      NOT_SUPPORTED → raise ValueError.

    Returns a deep copy of the query record with best-effort features removed.
    """
    query_record = copy.deepcopy(query_record)

    if query_record.prompt is not None and query_record.chat is not None:
      raise ValueError("'prompt' and 'chat' cannot both be set.")

    json_guidance, pydantic_schema = self._adapt_response_format(query_record)

    if query_record.chat is not None:
      self._adapt_chat(query_record, json_guidance, pydantic_schema)
    elif query_record.prompt is not None:
      self._adapt_prompt(query_record, json_guidance, pydantic_schema)

    if query_record.tools:
      self._adapt_tools(query_record)

    if query_record.parameters is not None:
      self._adapt_parameters(query_record)

    return query_record

