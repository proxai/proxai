"""Result adapter for mapping query response formats to provider capabilities."""

import base64
import json
from typing import List

import pydantic

from proxai.chat.message_content import ContentType
from proxai.chat.message_content import MessageContent
from proxai.chat.message_content import PydanticContent
import proxai.types as types
from proxai.connectors.adapter_utils import (
    RESPONSE_FORMAT_FIELD_MAP,
    merge_feature_configs,
    resolve_support,
)

_RESPONSE_FORMAT_TO_CONTENT_TYPE = {
    types.ResponseFormatType.IMAGE: ContentType.IMAGE,
    types.ResponseFormatType.AUDIO: ContentType.AUDIO,
    types.ResponseFormatType.VIDEO: ContentType.VIDEO,
}


class ResultAdapter:
  """Adapts query records to match a provider endpoint's response format support."""

  def __init__(
      self,
      endpoint: str,
      endpoint_feature_config: types.FeatureConfigType,
      model_feature_config: types.FeatureConfigType | None = None,
  ):
    self.endpoint = endpoint
    self.endpoint_feature_config = endpoint_feature_config
    self.model_feature_config = model_feature_config
    if model_feature_config is not None:
      self.feature_config = merge_feature_configs(
          endpoint_feature_config, model_feature_config)
    else:
      self.feature_config = endpoint_feature_config

  def get_support_level(
      self, query_record: types.QueryRecord,
  ) -> types.FeatureSupportType:
    """Return the support level for the response format in the query.

    Checks the response format in the query against the endpoint's feature
    config. Returns NOT_SUPPORTED if the response format is not supported,
    BEST_EFFORT if partially supported, or SUPPORTED if fully supported.
    """
    if (query_record.response_format is None
        or query_record.response_format.type is None):
      raise ValueError("'response_format.type' must be set.")

    rf_config = self.feature_config.response_format
    field_name = RESPONSE_FORMAT_FIELD_MAP.get(
        query_record.response_format.type)
    if field_name and rf_config:
      return resolve_support(getattr(rf_config, field_name, None))
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
    for message_content in content:
      result.append(self._adapt_message_content(query_record, message_content))
    return result

  def _adapt_message_content(
      self,
      query_record: types.QueryRecord,
      message_content: MessageContent,
  ) -> MessageContent:
    """Adapt content value to the expected response format."""
    response_format = query_record.response_format
    if message_content.type in [
        ContentType.THINKING,
        ContentType.IMAGE,
        ContentType.DOCUMENT,
        ContentType.AUDIO,
        ContentType.VIDEO,
        ContentType.TOOL,
    ]:
      return message_content

    if message_content.type == ContentType.TEXT:
      if response_format.type == types.ResponseFormatType.TEXT:
        return message_content
      if response_format.type == types.ResponseFormatType.JSON:
        return MessageContent(
            type=ContentType.JSON,
            json=json.loads(message_content.text))
      if response_format.type == types.ResponseFormatType.PYDANTIC:
        json_value = json.loads(message_content.text)
        pydantic_content = PydanticContent(
            class_name=response_format.pydantic_class.__name__,
            class_value=response_format.pydantic_class,
            instance_value=response_format.pydantic_class.model_validate(
                json_value),
            instance_json_value=json_value,
        )
        return MessageContent(
            type=ContentType.PYDANTIC_INSTANCE,
            pydantic_content=pydantic_content)

    if message_content.type == ContentType.JSON:
      if response_format.type == types.ResponseFormatType.JSON:
        return message_content
      if response_format.type == types.ResponseFormatType.PYDANTIC:
        json_value = message_content.json
        pydantic_content = PydanticContent(
            class_name=response_format.pydantic_class.__name__,
            class_value=response_format.pydantic_class,
            instance_value=response_format.pydantic_class.model_validate(
                json_value),
            instance_json_value=json_value,
        )
        return MessageContent(
            type=ContentType.PYDANTIC_INSTANCE,
            pydantic_content=pydantic_content)
        
    return message_content

  def _adapt_output_values(
      self,
      result_obj: types.ResultRecord,
  ):
    for message_content in reversed(result_obj.content):
      if message_content.type == ContentType.TEXT:
        result_obj.output_text = message_content.text
      elif message_content.type == ContentType.JSON:
        result_obj.output_json = message_content.json
      elif message_content.type == ContentType.PYDANTIC_INSTANCE:
        result_obj.output_pydantic = (
            message_content.pydantic_content.instance_value)
      elif message_content.type == ContentType.IMAGE:
        result_obj.output_image = message_content
      elif message_content.type == ContentType.AUDIO:
        result_obj.output_audio = message_content
      elif message_content.type == ContentType.VIDEO:
        result_obj.output_video = message_content

