"""Result adapter for mapping query response formats to provider capabilities."""

import json

import proxai.chat.message_content as message_content
import proxai.connectors.adapter_utils as adapter_utils
import proxai.types as types

ContentType = message_content.ContentType
MessageContent = message_content.MessageContent
PydanticContent = message_content.PydanticContent


class ResultAdapter:
  """Adapts query records to match a provider endpoint's response format support."""

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

  def get_query_record_support_level(
      self, query_record: types.QueryRecord,
  ) -> types.FeatureSupportType:
    """Return the support level for the response format in the query.

    Checks the response format in the query against the endpoint's feature
    config. Returns NOT_SUPPORTED if the response format is not supported,
    BEST_EFFORT if partially supported, or SUPPORTED if fully supported.
    """
    if (query_record.output_format is None
        or query_record.output_format.type is None):
      raise ValueError("'output_format.type' must be set.")

    rf_config = self.feature_config.output_format
    field_name = adapter_utils.OUTPUT_FORMAT_FIELD_MAP.get(
        query_record.output_format.type)
    if field_name and rf_config:
      return adapter_utils.resolve_support(getattr(rf_config, field_name, None))
    return types.FeatureSupportType.NOT_SUPPORTED

  def adapt_result_record(
      self,
      query_record: types.QueryRecord,
      result_record: types.ResultRecord,
  ):
    if result_record.content:
      result_record.content = self._adapt_content(
          query_record=query_record,
          content=result_record.content)
      self._adapt_output_values(result_obj=result_record)
    if result_record.choices:
      for choice in result_record.choices:
        choice.content = self._adapt_content(
            query_record=query_record,
            content=choice.content)
        self._adapt_output_values(result_obj=choice)

  def _adapt_content(
      self,
      query_record: types.QueryRecord,
      content: list[MessageContent],
  ) -> list[MessageContent]:
    result = []
    for content_item in content:
      result.append(self._adapt_message_content(query_record, content_item))
    return result

  def _adapt_message_content(
      self,
      query_record: types.QueryRecord,
      content_item: MessageContent,
  ) -> MessageContent:
    """Adapt content value to the expected response format."""
    output_format = query_record.output_format
    if content_item.type in [
        ContentType.THINKING,
        ContentType.IMAGE,
        ContentType.DOCUMENT,
        ContentType.AUDIO,
        ContentType.VIDEO,
        ContentType.TOOL,
    ]:
      return content_item

    if content_item.type == ContentType.TEXT:
      if output_format.type == types.OutputFormatType.TEXT:
        return content_item
      if output_format.type == types.OutputFormatType.JSON:
        return MessageContent(
            type=ContentType.JSON,
            json=json.loads(content_item.text))
      if output_format.type == types.OutputFormatType.PYDANTIC:
        json_value = json.loads(content_item.text)
        pydantic = PydanticContent(
            class_name=output_format.pydantic_class.__name__,
            class_value=output_format.pydantic_class,
            instance_value=output_format.pydantic_class.model_validate(
                json_value),
            instance_json_value=json_value,
        )
        return MessageContent(
            type=ContentType.PYDANTIC_INSTANCE,
            pydantic_content=pydantic)

    if content_item.type == ContentType.JSON:
      if output_format.type == types.OutputFormatType.JSON:
        return content_item
      if output_format.type == types.OutputFormatType.PYDANTIC:
        json_value = content_item.json
        pydantic = PydanticContent(
            class_name=output_format.pydantic_class.__name__,
            class_value=output_format.pydantic_class,
            instance_value=output_format.pydantic_class.model_validate(
                json_value),
            instance_json_value=json_value,
        )
        return MessageContent(
            type=ContentType.PYDANTIC_INSTANCE,
            pydantic_content=pydantic)

    return content_item

  _MEDIA_PLACEHOLDER_TYPES = {
      ContentType.IMAGE: 'image',
      ContentType.AUDIO: 'audio',
      ContentType.VIDEO: 'video',
      ContentType.DOCUMENT: 'document',
  }

  @staticmethod
  def _media_ref(content_item: MessageContent) -> str:
    """Short human-readable reference for inline media placeholders.

    Prefers explicit identifiers (source URL, local path) over raw bytes.
    """
    if content_item.source is not None:
      return content_item.source
    if content_item.path is not None:
      return content_item.path
    return '<data>'

  def _adapt_output_values(
      self,
      result_obj: types.ResultRecord,
  ):
    """Derive output_* shortcuts from result_obj.content.

    Forward iteration, no break:
      - TEXT           : concatenated into output_text (no separator).
      - IMAGE/AUDIO/VIDEO/DOCUMENT: inline placeholder text like
                         "[image: <ref>]" appended to output_text; the
                         matching typed output_image / output_audio /
                         output_video field is set to the last block
                         of that type (DOCUMENT has no typed output).
      - JSON           : output_json = last JSON block's json.
      - PYDANTIC_INSTANCE: output_pydantic = last block's live instance.
      - THINKING / TOOL: skipped (not surfaced in output_*).

    output_text stays None if no TEXT/media block is present; it starts
    as "" the first time one is seen, so an empty-text response still
    reports output_text="" rather than None.
    """
    text_seen = False
    for content_item in result_obj.content:
      if content_item.type == ContentType.TEXT:
        if not text_seen:
          result_obj.output_text = ''
          text_seen = True
        if content_item.text is not None:
          result_obj.output_text += content_item.text
      elif content_item.type in self._MEDIA_PLACEHOLDER_TYPES:
        label = self._MEDIA_PLACEHOLDER_TYPES[content_item.type]
        if not text_seen:
          result_obj.output_text = ''
          text_seen = True
        result_obj.output_text += (
            f'[{label}: {self._media_ref(content_item)}]'
        )
        if content_item.type == ContentType.IMAGE:
          result_obj.output_image = content_item
        elif content_item.type == ContentType.AUDIO:
          result_obj.output_audio = content_item
        elif content_item.type == ContentType.VIDEO:
          result_obj.output_video = content_item
      elif content_item.type == ContentType.JSON:
        result_obj.output_json = content_item.json
      elif content_item.type == ContentType.PYDANTIC_INSTANCE:
        result_obj.output_pydantic = (
            content_item.pydantic_content.instance_value)
      # THINKING and TOOL contribute nothing to output_*.

