"""Feature adapter for mapping query features to provider capabilities."""

import copy
import json

import proxai.connectors.adapter_utils as adapter_utils
import proxai.types as types


class FeatureAdapter:
  """Adapts query records to match a provider endpoint's feature support."""

  _NO_BEST_EFFORT_RESPONSE_FORMATS = (
      "text", "image", "audio", "video",
  )

  @staticmethod
  def get_query_signature(query_record: types.QueryRecord) -> dict:
    """Build a structured signature of features used in the query."""
    signature = {}
    if query_record.prompt is not None:
      signature['prompt'] = True
    if query_record.chat is not None:
      signature['messages'] = True
      if query_record.chat.system_prompt is not None:
        signature['system_prompt'] = True
      content_types = set()
      for message in query_record.chat.messages:
        if isinstance(message.content, list):
          for block in message.content:
            content_types.add(block.type.value)
      if content_types:
        signature['input_format'] = sorted(content_types)
    if query_record.system_prompt is not None:
      signature['system_prompt'] = True
    if query_record.parameters is not None:
      params = {}
      if query_record.parameters.temperature is not None:
        params['temperature'] = query_record.parameters.temperature
      if query_record.parameters.max_tokens is not None:
        params['max_tokens'] = query_record.parameters.max_tokens
      if query_record.parameters.stop is not None:
        params['stop'] = query_record.parameters.stop
      if query_record.parameters.n is not None:
        params['n'] = query_record.parameters.n
      if query_record.parameters.thinking is not None:
        params['thinking'] = query_record.parameters.thinking.value.lower()
      if params:
        signature['parameters'] = params
    if query_record.tools:
      signature['tools'] = [t.value.lower() for t in query_record.tools]
    if query_record.output_format and query_record.output_format.type:
      signature['output_format'] = (
          query_record.output_format.type.value.lower())
    return signature

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
      self.feature_config = adapter_utils.merge_feature_configs(
          endpoint_feature_config, model_feature_config)
    elif model_feature_config is not None:
      self.feature_config = model_feature_config
    else:
      self.feature_config = endpoint_feature_config

  def get_feature_tags_support_level(
      self, feature_tags: list[types.FeatureTag],
  ) -> types.FeatureSupportType:
    """Return the minimum support level across the given feature tags.

    Returns SUPPORTED if the list is empty.
    """
    if not feature_tags:
      return types.FeatureSupportType.SUPPORTED
    levels = [
        adapter_utils.resolve_feature_tag_support(self.feature_config, tag)
        for tag in feature_tags
    ]
    return min(levels, key=lambda l: adapter_utils.SUPPORT_RANK[l])

  def get_query_support_details(
      self, query_record: types.QueryRecord,
  ) -> dict[str, str]:
    """Return per-feature support levels for features used in the query."""
    details = {}
    S = types.FeatureSupportType

    if query_record.prompt is not None:
      details['prompt'] = adapter_utils.resolve_support(
          self.feature_config.prompt).value.lower()
    if query_record.chat is not None:
      details['messages'] = adapter_utils.resolve_support(
          self.feature_config.messages).value.lower()
      if query_record.chat.system_prompt is not None:
        details['system_prompt'] = adapter_utils.resolve_support(
            self.feature_config.system_prompt).value.lower()
      seen = set()
      for message in query_record.chat.messages:
        if isinstance(message.content, list):
          for block in message.content:
            ct = block.type.value
            if ct in seen:
              continue
            seen.add(ct)
            fmt_type = self._CONTENT_TYPE_TO_INPUT_FORMAT_TYPE.get(ct)
            if fmt_type is not None:
              level = adapter_utils.resolve_input_format_type_support(
                  self.feature_config, fmt_type)
              details[f'input:{ct}'] = level.value.lower()
    if query_record.system_prompt is not None:
      details['system_prompt'] = adapter_utils.resolve_support(
          self.feature_config.system_prompt).value.lower()
    if query_record.parameters is not None:
      p = query_record.parameters
      pc = self.feature_config.parameters
      if p.temperature is not None:
        details['temperature'] = adapter_utils.resolve_support(
            pc.temperature if pc else None).value.lower()
      if p.max_tokens is not None:
        details['max_tokens'] = adapter_utils.resolve_support(
            pc.max_tokens if pc else None).value.lower()
      if p.stop is not None:
        details['stop'] = adapter_utils.resolve_support(
            pc.stop if pc else None).value.lower()
      if p.n is not None:
        details['n'] = adapter_utils.resolve_support(
            pc.n if pc else None).value.lower()
      if p.thinking is not None:
        details['thinking'] = adapter_utils.resolve_support(
            pc.thinking if pc else None).value.lower()
    if query_record.tools:
      tc = self.feature_config.tools
      for tool in query_record.tools:
        if tool == types.Tools.WEB_SEARCH:
          details['web_search'] = adapter_utils.resolve_support(
              tc.web_search if tc else None).value.lower()
    if query_record.output_format and query_record.output_format.type:
      rf_config = self.feature_config.output_format
      field_name = adapter_utils.OUTPUT_FORMAT_FIELD_MAP.get(
          query_record.output_format.type)
      if field_name and rf_config:
        details['output_format'] = adapter_utils.resolve_support(
            getattr(rf_config, field_name, None)).value.lower()
      else:
        details['output_format'] = 'not_supported'
    return details

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
      levels.append(adapter_utils.resolve_support(self.feature_config.prompt))
    if query_record.chat is not None:
      levels.append(adapter_utils.resolve_support(self.feature_config.messages))
      if query_record.chat.system_prompt is not None:
        levels.append(adapter_utils.resolve_support(self.feature_config.system_prompt))
      self._collect_input_format_levels(query_record.chat, levels)
    if query_record.system_prompt is not None:
      levels.append(adapter_utils.resolve_support(self.feature_config.system_prompt))

    if query_record.parameters is not None:
      self._collect_parameter_levels(query_record.parameters, levels)

    if query_record.tools:
      self._collect_tool_levels(query_record.tools, levels)

    if (query_record.output_format is None
        or query_record.output_format.type is None):
      raise ValueError("'output_format.type' must be set.")
    self._collect_output_format_level(
        query_record.output_format, levels)

    if not levels:
      return types.FeatureSupportType.SUPPORTED

    return min(levels, key=lambda l: adapter_utils.SUPPORT_RANK[l])

  def _collect_parameter_levels(
      self,
      parameters: types.ParameterType,
      levels: list[types.FeatureSupportType],
  ):
    param_config = self.feature_config.parameters
    if parameters.temperature is not None:
      levels.append(adapter_utils.resolve_support(
          param_config.temperature if param_config else None))
    if parameters.max_tokens is not None:
      levels.append(adapter_utils.resolve_support(
          param_config.max_tokens if param_config else None))
    if parameters.stop is not None:
      levels.append(adapter_utils.resolve_support(
          param_config.stop if param_config else None))
    if parameters.n is not None:
      levels.append(adapter_utils.resolve_support(
          param_config.n if param_config else None))
    if parameters.thinking is not None:
      levels.append(adapter_utils.resolve_support(
          param_config.thinking if param_config else None))

  _CONTENT_TYPE_TO_INPUT_FORMAT_TYPE = {
      'text': types.InputFormatType.TEXT,
      'image': types.InputFormatType.IMAGE,
      'document': types.InputFormatType.DOCUMENT,
      'audio': types.InputFormatType.AUDIO,
      'video': types.InputFormatType.VIDEO,
      'json': types.InputFormatType.JSON,
      'pydantic_instance': types.InputFormatType.PYDANTIC,
  }

  def _collect_input_format_levels(
      self,
      chat: types.Chat,
      levels: list[types.FeatureSupportType],
  ):
    """Collect support levels for content types found in chat messages."""
    seen = set()
    for message in chat.messages:
      if isinstance(message.content, str):
        continue
      for block in message.content:
        content_type_value = block.type.value
        if content_type_value in seen:
          continue
        seen.add(content_type_value)
        input_format_type = self._CONTENT_TYPE_TO_INPUT_FORMAT_TYPE.get(
            content_type_value)
        if input_format_type is None:
          continue
        levels.append(adapter_utils.resolve_input_format_type_support(
            self.feature_config, input_format_type))

  def _collect_tool_levels(
      self,
      tools: list[types.Tools],
      levels: list[types.FeatureSupportType],
  ):
    tool_config = self.feature_config.tools
    for tool in tools:
      if tool == types.Tools.WEB_SEARCH:
        levels.append(adapter_utils.resolve_support(
            tool_config.web_search if tool_config else None))

  def _collect_output_format_level(
      self,
      output_format: types.OutputFormat,
      levels: list[types.FeatureSupportType],
  ):
    rf_config = self.feature_config.output_format
    field_name = adapter_utils.OUTPUT_FORMAT_FIELD_MAP.get(output_format.type)
    if field_name and rf_config:
      levels.append(adapter_utils.resolve_support(getattr(rf_config, field_name, None)))
    else:
      levels.append(types.FeatureSupportType.NOT_SUPPORTED)

  def _adapt_prompt(self, query_record: types.QueryRecord,
                    json_guidance: bool = False,
                    pydantic_schema: dict | None = None):
    """Adapt system_prompt and response format guidance for prompt queries."""
    if query_record.system_prompt is not None:
      level = adapter_utils.resolve_support(self.feature_config.system_prompt)
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

    # Pattern 2: endpoint has no native system kwarg and expects the system
    # prompt as the first entry in `messages`. Fold prompt + system_prompt
    # into a chat-shaped dict matching `chat.export()` output, so executors
    # see the same shape on the chat and prompt paths and never need to
    # read `query_record.system_prompt`.
    if (query_record.system_prompt is not None
        and self.feature_config.add_system_to_messages):
      query_record.chat = {
          'messages': [
              {'role': 'system', 'content': query_record.system_prompt},
              {'role': 'user', 'content': query_record.prompt},
          ],
      }
      query_record.prompt = None
      query_record.system_prompt = None

  def _adapt_chat(self, query_record: types.QueryRecord,
                  json_guidance: bool = False,
                  pydantic_schema: dict | None = None):
    """Adapt chat, system_prompt, and response format guidance for chat."""
    # Resolve system_prompt support for chat.
    system_best_effort = False
    add_system_to_messages = False
    if query_record.chat.system_prompt is not None:
      level = adapter_utils.resolve_support(self.feature_config.system_prompt)
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
    messages_level = adapter_utils.resolve_support(self.feature_config.messages)
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
    level = adapter_utils.resolve_support(
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

  def _adapt_output_format(
      self, query_record: types.QueryRecord,
  ) -> tuple[bool, dict | None]:
    """Validate output format and return guidance flags.

    Returns (json_guidance, pydantic_schema):
      json_guidance: True if JSON guidance should be added.
      pydantic_schema: JSON schema dict if pydantic schema guidance needed.
    """
    if (query_record.output_format is None
        or query_record.output_format.type is None):
      return False, None

    rf_config = self.feature_config.output_format
    rf_type = query_record.output_format.type
    field_name = adapter_utils.OUTPUT_FORMAT_FIELD_MAP.get(rf_type)
    if not field_name or not rf_config:
      level = types.FeatureSupportType.NOT_SUPPORTED
    else:
      level = adapter_utils.resolve_support(getattr(rf_config, field_name, None))

    if field_name in self._NO_BEST_EFFORT_RESPONSE_FORMATS:
      if level == types.FeatureSupportType.BEST_EFFORT:
        raise Exception(
            f"'{field_name}' output format config cannot be best effort. "
            f"Code should never reach here.\n"
            f"Open bug report at https://github.com/proxai/proxai/issues"
        )

    if level == types.FeatureSupportType.NOT_SUPPORTED:
      raise ValueError(
          f"Feature 'output_format.{rf_type.value}' is not supported "
          f"by endpoint '{self.endpoint}'."
      )

    if rf_type == types.OutputFormatType.JSON:
      return True, None

    if rf_type == types.OutputFormatType.PYDANTIC:
      if level == types.FeatureSupportType.BEST_EFFORT:
        schema = query_record.output_format.pydantic_class.model_json_schema()
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
    level = adapter_utils.resolve_support(support)
    if level == types.FeatureSupportType.SUPPORTED:
      return False
    if level == types.FeatureSupportType.BEST_EFFORT:
      return True
    raise ValueError(
        f"Feature '{feature_name}' is not supported "
        f"by endpoint '{self.endpoint}'."
    )

  @staticmethod
  def _json_block_to_text(block: dict) -> dict:
    return {
        'type': 'text',
        'text': json.dumps(block['json'], indent=2),
    }

  @staticmethod
  def _pydantic_block_to_text(block: dict) -> dict:
    pydantic = block.get('pydantic_content', {})
    name = pydantic.get('class_name', 'Unknown')
    value = pydantic.get('instance_json_value', {})
    return {
        'type': 'text',
        'text': (
            f'class name: {name}\n'
            f'class value:\n{json.dumps(value, indent=2)}'
        ),
    }

  _CONTENT_TYPE_TO_INPUT_FORMAT = {
      'text': (types.InputFormatType.TEXT, None),
      'image': (types.InputFormatType.IMAGE, None),
      'document': (types.InputFormatType.DOCUMENT, None),
      'audio': (types.InputFormatType.AUDIO, None),
      'video': (types.InputFormatType.VIDEO, None),
      'json': (types.InputFormatType.JSON, '_json_block_to_text'),
      'pydantic_instance': (
          types.InputFormatType.PYDANTIC, '_pydantic_block_to_text'),
  }

  # Types that should be passed through on BEST_EFFORT instead of
  # dropped, so the connector can apply its own conversion (e.g.,
  # PDF text extraction via content_utils).
  _BEST_EFFORT_PASSTHROUGH_TYPES = frozenset({'document'})

  def _adapt_content_block(self, block: dict) -> dict:
    """Adapt a single content block based on input format support.

    SUPPORTED     → pass through.
    BEST_EFFORT   → convert to text (if converter exists),
                    pass through (if in _BEST_EFFORT_PASSTHROUGH_TYPES),
                    else drop.
    NOT_SUPPORTED → raise.
    """
    entry = self._CONTENT_TYPE_TO_INPUT_FORMAT.get(block.get('type'))
    if entry is None:
      return block
    input_format_type, converter_name = entry
    level = adapter_utils.resolve_input_format_type_support(
        self.feature_config, input_format_type)
    if level == types.FeatureSupportType.NOT_SUPPORTED:
      raise ValueError(
          f"Input format '{input_format_type.value.lower()}' is not "
          f"supported by endpoint '{self.endpoint}'."
      )
    if level == types.FeatureSupportType.BEST_EFFORT:
      if converter_name is not None:
        return getattr(self, converter_name)(block)
      if block.get('type') in self._BEST_EFFORT_PASSTHROUGH_TYPES:
        return block
      return None
    return block

  def _adapt_input_format(self, query_record: types.QueryRecord):
    """Adapt content blocks in exported chat messages."""
    chat = query_record.chat
    if not isinstance(chat, dict) or 'messages' not in chat:
      return
    for message in chat['messages']:
      content = message.get('content')
      if not isinstance(content, list):
        continue
      adapted_content = []
      for block in content:
        adapted = self._adapt_content_block(block)
        if adapted is not None:
          adapted_content.append(adapted)
      message['content'] = adapted_content

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

    json_guidance, pydantic_schema = self._adapt_output_format(query_record)

    if query_record.chat is not None:
      self._adapt_chat(query_record, json_guidance, pydantic_schema)
      self._adapt_input_format(query_record)
    elif query_record.prompt is not None:
      self._adapt_prompt(query_record, json_guidance, pydantic_schema)

    if query_record.tools:
      self._adapt_tools(query_record)

    if query_record.parameters is not None:
      self._adapt_parameters(query_record)

    return query_record

