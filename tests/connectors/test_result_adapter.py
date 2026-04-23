"""Tests for ResultAdapter."""

import pydantic
import pytest

import proxai.chat.message_content as message_content
from proxai.chat.message_content import ContentType
from proxai.chat.message_content import MessageContent
from proxai.chat.message_content import PydanticContent
import proxai.types as types
from proxai.connectors.result_adapter import ResultAdapter

S = types.FeatureSupportType.SUPPORTED
BE = types.FeatureSupportType.BEST_EFFORT
NS = types.FeatureSupportType.NOT_SUPPORTED


class _TestModel(pydantic.BaseModel):
  name: str
  age: int


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
        output_format=types.OutputFormatConfigType(text=S, json=S))
    model_config = types.FeatureConfigType(
        output_format=types.OutputFormatConfigType(text=S, json=BE))
    adapter = ResultAdapter(
        endpoint="test",
        endpoint_feature_config=ep_config,
        model_feature_config=model_config,
    )
    assert adapter.feature_config.output_format.json == BE
    assert adapter.feature_config.output_format.text == S

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
          output_format=types.OutputFormatConfigType(
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
    adapter = _adapter(prompt=S, text=S)
    assert adapter.get_feature_tags_support_level(
        [types.FeatureTag.PROMPT]) == S

  def test_single_not_supported_tag(self):
    adapter = _adapter(prompt=NS, text=S)
    assert adapter.get_feature_tags_support_level(
        [types.FeatureTag.PROMPT]) == NS

  def test_minimum_across_tags(self):
    adapter = _adapter(prompt=S, messages=BE, text=S)
    result = adapter.get_feature_tags_support_level([
        types.FeatureTag.PROMPT,
        types.FeatureTag.MESSAGES,
    ])
    assert result == BE

  def test_not_supported_dominates(self):
    adapter = _adapter(prompt=S, messages=NS, text=S)
    result = adapter.get_feature_tags_support_level([
        types.FeatureTag.PROMPT,
        types.FeatureTag.MESSAGES,
    ])
    assert result == NS

  def test_parameter_tags(self):
    adapter = _adapter(temperature=S, max_tokens=BE)
    result = adapter.get_feature_tags_support_level([
        types.FeatureTag.TEMPERATURE,
        types.FeatureTag.MAX_TOKENS,
    ])
    assert result == BE

  def test_all_feature_tags(self):
    adapter = _adapter(
        prompt=S, messages=S, system_prompt=S,
        temperature=S, max_tokens=S, stop=S, n=S, thinking=S,
        text=S,
    )
    all_tags = list(types.FeatureTag)
    assert adapter.get_feature_tags_support_level(all_tags) == S


# ===================================================================
# get_query_record_support_level
# ===================================================================

class TestResultAdapterGetQueryRecordSupportLevel:
  """Tests for get_query_record_support_level."""

  def test_no_output_format_raises(self):
    adapter = _adapter(text=S)
    with pytest.raises(ValueError, match="output_format.type.*must be set"):
      adapter.get_query_record_support_level(types.QueryRecord())

  def test_returns_configured_level(self):
    adapter = _adapter(json_fmt=BE)
    query = types.QueryRecord(output_format=types.OutputFormat(
        type=types.OutputFormatType.JSON))
    assert adapter.get_query_record_support_level(query) == BE


# ===================================================================
# _adapt_message_content — content transformation
# ===================================================================

class TestAdaptMessageContent:
  """Tests for _adapt_message_content core transformation logic."""

  def test_passthrough_types(self):
    """Non-text, non-json content types are returned unchanged."""
    adapter = _adapter(text=S)
    query = types.QueryRecord(output_format=types.OutputFormat(
        type=types.OutputFormatType.TEXT))
    passthrough = [
        MessageContent(type=ContentType.TEXT, text="hi"),
        MessageContent(type=ContentType.THINKING, text="reasoning"),
        MessageContent(
            type=ContentType.IMAGE, source="https://ex.com/a.png"),
        MessageContent(
            type=ContentType.DOCUMENT, source="https://ex.com/a.pdf",
            media_type="application/pdf"),
        MessageContent(
            type=ContentType.AUDIO, source="https://ex.com/a.mp3"),
        MessageContent(
            type=ContentType.VIDEO, source="https://ex.com/a.mp4"),
    ]
    for content in passthrough:
      assert adapter._adapt_message_content(query, content) is content

  def test_text_to_json_parses(self):
    adapter = _adapter(json_fmt=S)
    query = types.QueryRecord(output_format=types.OutputFormat(
        type=types.OutputFormatType.JSON))
    content = MessageContent(type=ContentType.TEXT,
                             text='{"key": "value", "n": 42}')
    result = adapter._adapt_message_content(query, content)
    assert result.type == ContentType.JSON
    assert result.json == {"key": "value", "n": 42}

  def test_text_to_pydantic_parses_and_validates(self):
    adapter = _adapter(pydantic_fmt=S)
    query = types.QueryRecord(output_format=types.OutputFormat(
        type=types.OutputFormatType.PYDANTIC, pydantic_class=_TestModel))
    content = MessageContent(type=ContentType.TEXT,
                             text='{"name": "Alice", "age": 30}')
    result = adapter._adapt_message_content(query, content)
    assert result.type == ContentType.PYDANTIC_INSTANCE
    assert result.pydantic_content.class_name == "_TestModel"
    assert result.pydantic_content.class_value is _TestModel
    assert result.pydantic_content.instance_value == _TestModel(
        name="Alice", age=30)
    assert result.pydantic_content.instance_json_value == {
        "name": "Alice", "age": 30}

  def test_json_to_json_passthrough(self):
    adapter = _adapter(json_fmt=S)
    query = types.QueryRecord(output_format=types.OutputFormat(
        type=types.OutputFormatType.JSON))
    content = MessageContent(type=ContentType.JSON, json={"key": "value"})
    assert adapter._adapt_message_content(query, content) is content

  def test_json_to_pydantic_validates(self):
    adapter = _adapter(pydantic_fmt=S)
    query = types.QueryRecord(output_format=types.OutputFormat(
        type=types.OutputFormatType.PYDANTIC, pydantic_class=_TestModel))
    content = MessageContent(type=ContentType.JSON,
                             json={"name": "Bob", "age": 25})
    result = adapter._adapt_message_content(query, content)
    assert result.type == ContentType.PYDANTIC_INSTANCE
    assert result.pydantic_content.instance_value == _TestModel(
        name="Bob", age=25)


# ===================================================================
# _adapt_output_values — media output projection
# ===================================================================

class TestAdaptOutputValues:
  """Tests for _adapt_output_values on ResultAdapter."""

  def _make_adapter(self) -> ResultAdapter:
    return _adapter(
        prompt=S, text=S, image=S, audio=S, video=S)

  def test_image_content_populates_output_image_and_placeholder_text(self):
    adapter = self._make_adapter()
    img = MessageContent(
        type=ContentType.IMAGE,
        source='https://example.com/img.png',
    )
    result = types.ResultRecord(content=[img])
    adapter._adapt_output_values(result)
    assert result.output_image is img
    # Media blocks emit an inline placeholder into output_text.
    assert result.output_text == '[image: https://example.com/img.png]'

  def test_audio_content_populates_output_audio_and_placeholder_text(self):
    adapter = self._make_adapter()
    audio = MessageContent(
        type=ContentType.AUDIO,
        data=b'audio_bytes',
        media_type='audio/mpeg',
    )
    result = types.ResultRecord(content=[audio])
    adapter._adapt_output_values(result)
    assert result.output_audio is audio
    # Raw bytes have no URL or path — ref falls back to "<data>".
    assert result.output_text == '[audio: <data>]'

  def test_video_content_populates_output_video_and_placeholder_text(self):
    adapter = self._make_adapter()
    video = MessageContent(
        type=ContentType.VIDEO,
        source='https://example.com/video.mp4',
    )
    result = types.ResultRecord(content=[video])
    adapter._adapt_output_values(result)
    assert result.output_video is video
    assert result.output_text == '[video: https://example.com/video.mp4]'

  def test_document_emits_placeholder_but_has_no_typed_output(self):
    adapter = self._make_adapter()
    doc = MessageContent(
        type=ContentType.DOCUMENT,
        path='/tmp/report.pdf',
    )
    result = types.ResultRecord(content=[doc])
    adapter._adapt_output_values(result)
    assert result.output_text == '[document: /tmp/report.pdf]'
    # DOCUMENT has no typed output_* on ResultRecord.

  def test_mixed_text_and_image_inlines_placeholder_in_text(self):
    adapter = self._make_adapter()
    pre = MessageContent(type=ContentType.TEXT, text='Here: ')
    img = MessageContent(
        type=ContentType.IMAGE,
        source='https://example.com/img.png',
    )
    post = MessageContent(type=ContentType.TEXT, text=' — good?')
    result = types.ResultRecord(content=[pre, img, post])
    adapter._adapt_output_values(result)
    # TEXT blocks concatenate with no separator; media placeholder sits
    # inline at its position.
    assert result.output_text == 'Here: [image: https://example.com/img.png] — good?'
    assert result.output_image is img

  def test_multiple_text_blocks_concatenate_without_separator(self):
    adapter = self._make_adapter()
    result = types.ResultRecord(content=[
        MessageContent(type=ContentType.TEXT, text='foo'),
        MessageContent(type=ContentType.TEXT, text='bar'),
        MessageContent(type=ContentType.TEXT, text='baz'),
    ])
    adapter._adapt_output_values(result)
    assert result.output_text == 'foobarbaz'

  def test_multiple_images_last_wins_in_typed_output(self):
    """output_image is the last IMAGE block; placeholders appear for all."""
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
    assert result.output_image is img2
    assert result.output_text == (
        '[image: https://example.com/first.png]'
        '[image: https://example.com/second.png]'
    )

  def test_thinking_blocks_are_excluded_from_output_text(self):
    adapter = self._make_adapter()
    result = types.ResultRecord(content=[
        MessageContent(type=ContentType.THINKING, text='reasoning...'),
        MessageContent(type=ContentType.TEXT, text='the answer'),
    ])
    adapter._adapt_output_values(result)
    assert result.output_text == 'the answer'

  def test_no_media_no_text_leaves_output_text_none(self):
    adapter = self._make_adapter()
    # A JSON-only response should not fabricate an output_text value.
    result = types.ResultRecord(content=[
        MessageContent(type=ContentType.JSON, json={'k': 'v'}),
    ])
    adapter._adapt_output_values(result)
    assert result.output_text is None
    assert result.output_json == {'k': 'v'}

  def test_no_media_leaves_output_media_none(self):
    adapter = self._make_adapter()
    text = MessageContent(type=ContentType.TEXT, text='hello')
    result = types.ResultRecord(content=[text])
    adapter._adapt_output_values(result)
    assert result.output_image is None
    assert result.output_audio is None
    assert result.output_video is None
    # Plain text still surfaces as output_text.
    assert result.output_text == 'hello'

  def test_json_content_populates_output_json(self):
    adapter = self._make_adapter()
    content = MessageContent(type=ContentType.JSON, json={'k': 'v'})
    result = types.ResultRecord(content=[content])
    adapter._adapt_output_values(result)
    assert result.output_json == {'k': 'v'}

  def test_multiple_json_blocks_last_wins(self):
    adapter = self._make_adapter()
    result = types.ResultRecord(content=[
        MessageContent(type=ContentType.JSON, json={'k': 1}),
        MessageContent(type=ContentType.JSON, json={'k': 2}),
    ])
    adapter._adapt_output_values(result)
    assert result.output_json == {'k': 2}

  def test_pydantic_content_populates_output_pydantic(self):
    adapter = self._make_adapter()
    instance = _TestModel(name='Alice', age=30)
    content = MessageContent(
        type=ContentType.PYDANTIC_INSTANCE,
        pydantic_content=PydanticContent(
            class_value=_TestModel, instance_value=instance))
    result = types.ResultRecord(content=[content])
    adapter._adapt_output_values(result)
    assert result.output_pydantic is instance

  def test_tool_blocks_do_not_contribute_to_output_text(self):
    adapter = self._make_adapter()
    result = types.ResultRecord(content=[
        MessageContent(
            type=ContentType.TOOL,
            tool_content=message_content.ToolContent(
                name='web_search',
                kind=message_content.ToolKind.RESULT,
            ),
        ),
        MessageContent(type=ContentType.TEXT, text='final answer'),
    ])
    adapter._adapt_output_values(result)
    assert result.output_text == 'final answer'


# ===================================================================
# adapt_result_record — end-to-end
# ===================================================================

class TestAdaptResultRecord:
  """End-to-end tests for adapt_result_record."""

  def test_adapts_content_and_populates_output(self):
    adapter = _adapter(json_fmt=S)
    query = types.QueryRecord(output_format=types.OutputFormat(
        type=types.OutputFormatType.JSON))
    result = types.ResultRecord(content=[
        MessageContent(type=ContentType.TEXT, text='{"result": "ok"}'),
    ])
    adapter.adapt_result_record(query, result)
    assert result.content[0].type == ContentType.JSON
    assert result.content[0].json == {'result': 'ok'}
    assert result.output_json == {'result': 'ok'}

  def test_adapts_each_choice(self):
    adapter = _adapter(json_fmt=S)
    query = types.QueryRecord(output_format=types.OutputFormat(
        type=types.OutputFormatType.JSON))
    result = types.ResultRecord(choices=[
        types.ChoiceType(content=[
            MessageContent(type=ContentType.TEXT, text='{"a": 1}')]),
        types.ChoiceType(content=[
            MessageContent(type=ContentType.TEXT, text='{"b": 2}')]),
    ])
    adapter.adapt_result_record(query, result)
    assert result.choices[0].content[0].json == {'a': 1}
    assert result.choices[0].output_json == {'a': 1}
    assert result.choices[1].content[0].json == {'b': 2}
    assert result.choices[1].output_json == {'b': 2}
