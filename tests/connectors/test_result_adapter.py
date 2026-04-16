"""Tests for ResultAdapter."""

import pytest

from proxai.chat.message_content import ContentType
from proxai.chat.message_content import MessageContent
import proxai.types as types
from proxai.connectors.result_adapter import ResultAdapter

S = types.FeatureSupportType.SUPPORTED
BE = types.FeatureSupportType.BEST_EFFORT
NS = types.FeatureSupportType.NOT_SUPPORTED


# ===================================================================
# ResultAdapter init config resolution
# ===================================================================

class TestResultAdapterInitConfig:
  """Tests for ResultAdapter config resolution in __init__."""

  def test_both_none_raises(self):
    with pytest.raises(ValueError, match="At least one"):
      ResultAdapter(endpoint="test")

  def test_only_endpoint_config(self):
    ep_config = types.FeatureConfigType(prompt=S)
    adapter = ResultAdapter(
        endpoint="test", endpoint_feature_config=ep_config)
    assert adapter.feature_config is ep_config
    assert adapter.model_feature_config is None

  def test_only_model_config(self):
    model_config = types.FeatureConfigType(prompt=S)
    adapter = ResultAdapter(
        endpoint="test", model_feature_config=model_config)
    assert adapter.feature_config is model_config
    assert adapter.endpoint_feature_config is None

  def test_both_configs_merges(self):
    ep_config = types.FeatureConfigType(
        response_format=types.ResponseFormatConfigType(text=S, json=S))
    model_config = types.FeatureConfigType(
        response_format=types.ResponseFormatConfigType(text=S, json=BE))
    adapter = ResultAdapter(
        endpoint="test",
        endpoint_feature_config=ep_config,
        model_feature_config=model_config,
    )
    assert adapter.feature_config.response_format.json == BE
    assert adapter.feature_config.response_format.text == S

  def test_originals_stored(self):
    ep_config = types.FeatureConfigType(prompt=S)
    model_config = types.FeatureConfigType(prompt=BE)
    adapter = ResultAdapter(
        endpoint="test",
        endpoint_feature_config=ep_config,
        model_feature_config=model_config,
    )
    assert adapter.endpoint_feature_config is ep_config
    assert adapter.model_feature_config is model_config


# ===================================================================
# get_feature_tags_support_level
# ===================================================================

def _adapter(
    prompt=None, messages=None, system_prompt=None,
    temperature=None, max_tokens=None, stop=None, n=None, thinking=None,
    web_search=None,
    text=None, image=None, audio=None, video=None,
    json_fmt=None, pydantic_fmt=None, multi_modal=None,
) -> ResultAdapter:
  """Build a ResultAdapter with the given support levels."""
  return ResultAdapter(
      endpoint="test_endpoint",
      endpoint_feature_config=types.FeatureConfigType(
          prompt=prompt,
          messages=messages,
          system_prompt=system_prompt,
          parameters=types.ParameterConfigType(
              temperature=temperature,
              max_tokens=max_tokens,
              stop=stop,
              n=n,
              thinking=thinking,
          ),
          tools=types.ToolConfigType(web_search=web_search),
          response_format=types.ResponseFormatConfigType(
              text=text, image=image, audio=audio, video=video,
              json=json_fmt, pydantic=pydantic_fmt, multi_modal=multi_modal,
          ),
      ),
  )


class TestGetFeatureTagsSupportLevel:
  """Tests for get_feature_tags_support_level on ResultAdapter."""

  def test_empty_tags_returns_supported(self):
    adapter = _adapter(prompt=NS, text=NS)
    assert adapter.get_feature_tags_support_level([]) == S

  def test_single_supported_tag(self):
    adapter = _adapter(text=S)
    assert adapter.get_feature_tags_support_level(
        [types.FeatureTagType.RESPONSE_TEXT]) == S

  def test_single_not_supported_tag(self):
    adapter = _adapter(text=NS)
    assert adapter.get_feature_tags_support_level(
        [types.FeatureTagType.RESPONSE_TEXT]) == NS

  def test_minimum_across_tags(self):
    adapter = _adapter(text=S, image=BE)
    result = adapter.get_feature_tags_support_level([
        types.FeatureTagType.RESPONSE_TEXT,
        types.FeatureTagType.RESPONSE_IMAGE,
    ])
    assert result == BE

  def test_not_supported_dominates(self):
    adapter = _adapter(text=S, json_fmt=NS)
    result = adapter.get_feature_tags_support_level([
        types.FeatureTagType.RESPONSE_TEXT,
        types.FeatureTagType.RESPONSE_JSON,
    ])
    assert result == NS

  def test_parameter_tags(self):
    adapter = _adapter(temperature=S, max_tokens=BE)
    result = adapter.get_feature_tags_support_level([
        types.FeatureTagType.TEMPERATURE,
        types.FeatureTagType.MAX_TOKENS,
    ])
    assert result == BE

  def test_all_tag_types(self):
    adapter = _adapter(
        prompt=S, messages=S, system_prompt=S,
        temperature=S, max_tokens=S, stop=S, n=S, thinking=S,
        web_search=S,
        text=S, image=S, audio=S, video=S,
        json_fmt=S, pydantic_fmt=S, multi_modal=S,
    )
    all_tags = list(types.FeatureTagType)
    assert adapter.get_feature_tags_support_level(all_tags) == S


# ===================================================================
# _adapt_output_values — media output projection
# ===================================================================

class TestAdaptOutputValues:
  """Tests for _adapt_output_values on ResultAdapter."""

  def _make_adapter(self) -> ResultAdapter:
    return _adapter(
        prompt=S, text=S, image=S, audio=S, video=S)

  def test_image_content_populates_output_image(self):
    adapter = self._make_adapter()
    img = MessageContent(
        type=ContentType.IMAGE,
        source='https://example.com/img.png',
    )
    result = types.ResultRecord(content=[img])
    adapter._adapt_output_values(result)
    assert result.output_image is img

  def test_audio_content_populates_output_audio(self):
    adapter = self._make_adapter()
    audio = MessageContent(
        type=ContentType.AUDIO,
        data=b'audio_bytes',
        media_type='audio/mpeg',
    )
    result = types.ResultRecord(content=[audio])
    adapter._adapt_output_values(result)
    assert result.output_audio is audio

  def test_video_content_populates_output_video(self):
    adapter = self._make_adapter()
    video = MessageContent(
        type=ContentType.VIDEO,
        source='https://example.com/video.mp4',
    )
    result = types.ResultRecord(content=[video])
    adapter._adapt_output_values(result)
    assert result.output_video is video

  def test_mixed_text_and_image(self):
    adapter = self._make_adapter()
    text = MessageContent(type=ContentType.TEXT, text='hello')
    img = MessageContent(
        type=ContentType.IMAGE,
        source='https://example.com/img.png',
    )
    result = types.ResultRecord(content=[text, img])
    adapter._adapt_output_values(result)
    assert result.output_text == 'hello'
    assert result.output_image is img

  def test_multiple_images_first_wins(self):
    """Reversed scan means the first content block overwrites later ones."""
    adapter = self._make_adapter()
    img1 = MessageContent(
        type=ContentType.IMAGE,
        source='https://example.com/first.png',
    )
    img2 = MessageContent(
        type=ContentType.IMAGE,
        source='https://example.com/second.png',
    )
    result = types.ResultRecord(content=[img1, img2])
    adapter._adapt_output_values(result)
    assert result.output_image is img1

  def test_no_media_leaves_output_none(self):
    adapter = self._make_adapter()
    text = MessageContent(type=ContentType.TEXT, text='hello')
    result = types.ResultRecord(content=[text])
    adapter._adapt_output_values(result)
    assert result.output_image is None
    assert result.output_audio is None
    assert result.output_video is None
