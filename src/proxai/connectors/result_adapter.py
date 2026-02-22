"""Result adapter for mapping query response formats to provider capabilities."""

import base64
import json
from typing import List

import pydantic

from proxai.chat.message_content import ContentType
from proxai.chat.message_content import MessageContent
from proxai.chat.message_content import PydanticContent
import proxai.types as types

_SUPPORT_RANK = {
    types.FeatureSupportType.NOT_SUPPORTED: 0,
    types.FeatureSupportType.BEST_EFFORT: 1,
    types.FeatureSupportType.SUPPORTED: 2,
}

_RESPONSE_FORMAT_TO_CONTENT_TYPE = {
    types.ResponseFormatType.IMAGE: ContentType.IMAGE,
    types.ResponseFormatType.AUDIO: ContentType.AUDIO,
    types.ResponseFormatType.VIDEO: ContentType.VIDEO,
}

_RESPONSE_FORMAT_FIELD_MAP = {
    types.ResponseFormatType.TEXT: "text",
    types.ResponseFormatType.IMAGE: "image",
    types.ResponseFormatType.AUDIO: "audio",
    types.ResponseFormatType.VIDEO: "video",
    types.ResponseFormatType.JSON: "json",
    types.ResponseFormatType.PYDANTIC: "pydantic",
    types.ResponseFormatType.MULTI_MODAL: "multi_modal",
}


class ResultAdapter:
  """Adapts query records to match a provider endpoint's response format support."""

  def __init__(
      self,
      endpoint: str,
      feature_config: types.FeatureConfigType,
  ):
    self.endpoint = endpoint
    self.feature_config = feature_config

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
    field_name = _RESPONSE_FORMAT_FIELD_MAP.get(
        query_record.response_format.type)
    if field_name and rf_config:
      return self._resolve(getattr(rf_config, field_name, None))
    return types.FeatureSupportType.NOT_SUPPORTED

  def adapt_result_content(
      self,
      query_record: types.QueryRecord,
      content: (
          str | bytes | dict | list
          | pydantic.BaseModel | types.ResultMediaContentType),
  ) -> List[MessageContent]:
    """Adapt raw provider content to the expected response format.

    Dispatches to format-specific handlers that validate the content type
    and convert it to a list of MessageContent.

    Args:
      query_record: The query record with response_format info.
      content: Raw content from the provider (str, bytes, dict, list,
          or pydantic.BaseModel).

    Returns:
      List[MessageContent] for all response format types.
    """
    rf_type = query_record.response_format.type

    if rf_type == types.ResponseFormatType.TEXT:
      return self._adapt_text(content)
    if rf_type in _RESPONSE_FORMAT_TO_CONTENT_TYPE:
      return self._adapt_media(rf_type, content)
    if rf_type == types.ResponseFormatType.JSON:
      return self._adapt_json(content)
    if rf_type == types.ResponseFormatType.PYDANTIC:
      return self._adapt_pydantic(query_record, content)
    if rf_type == types.ResponseFormatType.MULTI_MODAL:
      return self._adapt_multi_modal(content)
    return content

  def _adapt_text(self, content) -> List[MessageContent]:
    """Adapt content for TEXT response format."""
    if not isinstance(content, str):
      raise ValueError(
          f"Expected str content for TEXT response format, "
          f"got {type(content).__name__}.")
    return [MessageContent(type=ContentType.TEXT, text=content)]

  def _adapt_media(
      self,
      rf_type: types.ResponseFormatType,
      content,
  ) -> List[MessageContent]:
    """Adapt content for IMAGE, AUDIO, or VIDEO response format."""
    if not isinstance(content, bytes):
      raise ValueError(
          f"Expected bytes content for {rf_type.value} response format, "
          f"got {type(content).__name__}.")
    content_type = _RESPONSE_FORMAT_TO_CONTENT_TYPE[rf_type]
    data = base64.b64encode(content).decode("utf-8")
    return [MessageContent(type=content_type, data=data)]

  def _adapt_json(
      self,
      content,
  ) -> List[MessageContent]:
    """Adapt content for JSON response format."""
    if isinstance(content, str):
      content = json.loads(content)
    elif not isinstance(content, dict):
      raise ValueError(
          f"Expected str or dict content for JSON response format, "
          f"got {type(content).__name__}.")
    return [MessageContent(type=ContentType.JSON, json=content)]

  def _adapt_pydantic(
      self,
      query_record: types.QueryRecord,
      content,
  ) -> List[MessageContent]:
    """Adapt content for PYDANTIC response format."""
    pydantic_class = query_record.response_format.pydantic_class
    if type(content) == str:
      content = json.loads(content)
    if isinstance(content, dict):
      content = pydantic_class.model_validate(content)
    if not isinstance(content, pydantic.BaseModel):
      raise ValueError(
          f"Expected pydantic.BaseModel content for PYDANTIC response "
          f"format, got {type(content).__name__}.")
    return [MessageContent(
        type=ContentType.PYDANTIC_INSTANCE,
        pydantic_content=PydanticContent(
            class_name=pydantic_class.__name__,
            class_value=pydantic_class,
            instance_value=content,
            instance_json_value=content.model_dump(),
        )
    )]

  def _adapt_multi_modal(self, content) -> List[MessageContent]:
    """Adapt content for MULTI_MODAL response format."""
    if not isinstance(content, list):
      raise ValueError(
          f"Expected list content for MULTI_MODAL response format, "
          f"got {type(content).__name__}.")
    result = []
    for item in content:
      if isinstance(item, str):
        result.append(MessageContent(type=ContentType.TEXT, text=item))
      elif isinstance(item, types.ResultMediaContentType):
        result.append(self._adapt_multi_modal_media(item))
      elif isinstance(item, dict):
        result.append(MessageContent(type=ContentType.JSON, json=item))
      elif isinstance(item, pydantic.BaseModel):
        result.append(MessageContent(
            type=ContentType.PYDANTIC_INSTANCE,
            pydantic_content=PydanticContent(
                class_name=type(item).__name__,
                class_value=type(item),
                instance_value=item,
                instance_json_value=item.model_dump(),
            )
        ))
      else:
        raise ValueError(
            f"Expected str, ResultMediaContentType, dict, or "
            f"pydantic.BaseModel elements for MULTI_MODAL response "
            f"format, got {type(item).__name__}.")
    return result

  _MIME_PREFIX_TO_CONTENT_TYPE = {
      "image": ContentType.IMAGE,
      "audio": ContentType.AUDIO,
      "video": ContentType.VIDEO,
  }

  def _adapt_multi_modal_media(
      self, item: types.ResultMediaContentType,
  ) -> MessageContent:
    """Adapt a ResultMediaContentType element in a MULTI_MODAL list."""
    prefix = item.media_type.split("/")[0]
    content_type = self._MIME_PREFIX_TO_CONTENT_TYPE.get(prefix)
    if content_type is None:
      raise ValueError(
          f"Unsupported media type '{item.media_type}' in MULTI_MODAL "
          f"element. Expected MIME type starting with: "
          f"{list(self._MIME_PREFIX_TO_CONTENT_TYPE.keys())}.")
    data = base64.b64encode(item.data).decode("utf-8")
    return MessageContent(
        type=content_type,
        data=data,
        media_type=item.media_type,
    )

  @staticmethod
  def _resolve(
      support: types.FeatureSupportType | None,
  ) -> types.FeatureSupportType:
    if support is None:
      return types.FeatureSupportType.NOT_SUPPORTED
    return support
