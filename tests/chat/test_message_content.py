"""Tests for MessageContent dataclass."""

import pytest

import proxai.types as types
from proxai.chat.message_content import MessageContent, SUPPORTED_MEDIA_TYPES


class TestMessageContentCreation:
  """Test creating MessageContent with different types and fields."""

  def test_text_with_string_type(self):
    mc = MessageContent(type="text", text="Hello")
    assert mc.type == types.ContentType.TEXT
    assert mc.text == "Hello"

  def test_text_with_enum_type(self):
    mc = MessageContent(type=types.ContentType.TEXT, text="Hello")
    assert mc.type == types.ContentType.TEXT
    assert mc.text == "Hello"

  def test_thinking_content(self):
    mc = MessageContent(type="thinking", text="Let me reason...")
    assert mc.type == types.ContentType.THINKING
    assert mc.text == "Let me reason..."

  def test_image_from_source(self):
    mc = MessageContent(type="image", source="https://example.com/img.png")
    assert mc.type == types.ContentType.IMAGE
    assert mc.source == "https://example.com/img.png"

  def test_image_from_data_with_media_type(self):
    mc = MessageContent(type="image", data="base64data", media_type="image/png")
    assert mc.data == "base64data"
    assert mc.media_type == "image/png"

  def test_image_from_path(self):
    mc = MessageContent(type="image", path="./photo.png")
    assert mc.path == "./photo.png"

  def test_document_from_source(self):
    mc = MessageContent(type="document", source="https://example.com/doc.pdf")
    assert mc.type == types.ContentType.DOCUMENT

  def test_audio_from_path(self):
    mc = MessageContent(type="audio", path="./recording.mp3")
    assert mc.type == types.ContentType.AUDIO

  def test_video_from_source(self):
    mc = MessageContent(type="video", source="https://example.com/clip.mp4")
    assert mc.type == types.ContentType.VIDEO


class TestMessageContentValidation:
  """Test validation rules for MessageContent."""

  def test_invalid_type_raises_error(self):
    with pytest.raises(ValueError, match="Invalid content type"):
      MessageContent(type="invalid", text="Hello")

  def test_text_without_text_field_raises_error(self):
    with pytest.raises(ValueError, match="'text' field is required"):
      MessageContent(type="text")

  def test_thinking_without_text_field_raises_error(self):
    with pytest.raises(ValueError, match="'text' field is required"):
      MessageContent(type="thinking")

  def test_text_with_source_raises_error(self):
    with pytest.raises(ValueError, match="source cannot be set"):
      MessageContent(type="text", text="Hello", source="https://example.com")

  def test_text_with_media_type_raises_error(self):
    with pytest.raises(ValueError, match="media_type cannot be set"):
      MessageContent(type="text", text="Hello", media_type="image/png")

  def test_thinking_with_data_raises_error(self):
    with pytest.raises(ValueError, match="data cannot be set"):
      MessageContent(type="thinking", text="Hello", data="base64")

  def test_media_without_source_data_or_path_raises_error(self):
    with pytest.raises(ValueError, match="At least one of"):
      MessageContent(type="image")

  def test_unsupported_media_type_raises_error(self):
    with pytest.raises(ValueError, match="Unsupported media_type"):
      MessageContent(type="image", source="https://x.com/a", media_type="image/bmp")

  def test_all_supported_media_types_are_accepted(self):
    for media_type in SUPPORTED_MEDIA_TYPES:
      mc = MessageContent(
          type="image", source="https://example.com/file", media_type=media_type
      )
      assert mc.media_type == media_type


class TestMessageContentSerialization:
  """Test to_dict and from_dict round-trip."""

  def test_text_to_dict(self):
    mc = MessageContent(type="text", text="Hello")
    d = mc.to_dict()
    assert d == {"type": "text", "text": "Hello"}

  def test_media_to_dict_omits_none_fields(self):
    mc = MessageContent(type="image", source="https://example.com/img.png")
    d = mc.to_dict()
    assert d == {"type": "image", "source": "https://example.com/img.png"}
    assert "text" not in d
    assert "data" not in d
    assert "path" not in d
    assert "media_type" not in d

  def test_media_to_dict_with_all_fields(self):
    mc = MessageContent(
        type="image",
        source="https://example.com/img.png",
        data="base64data",
        path="./photo.png",
        media_type="image/png",
    )
    d = mc.to_dict()
    assert d["source"] == "https://example.com/img.png"
    assert d["data"] == "base64data"
    assert d["path"] == "./photo.png"
    assert d["media_type"] == "image/png"

  def test_round_trip_text(self):
    original = MessageContent(type="text", text="Hello world")
    restored = MessageContent.from_dict(original.to_dict())
    assert restored == original

  def test_round_trip_media(self):
    original = MessageContent(
        type="document", source="https://example.com/doc.pdf",
        media_type="application/pdf",
    )
    restored = MessageContent.from_dict(original.to_dict())
    assert restored == original


class TestMessageContentCopyAndRepr:
  """Test copy and repr behavior."""

  def test_copy_is_independent(self):
    original = MessageContent(type="text", text="Hello")
    copied = original.copy()
    copied.text = "Changed"
    assert original.text == "Hello"

  def test_repr_text(self):
    mc = MessageContent(type="text", text="Hello")
    assert "type='text'" in repr(mc)
    assert "text='Hello'" in repr(mc)

  def test_repr_media(self):
    mc = MessageContent(
        type="image", source="https://example.com/img.png",
        media_type="image/png",
    )
    assert "type='image'" in repr(mc)
    assert "source='https://example.com/img.png'" in repr(mc)
    assert "media_type='image/png'" in repr(mc)
