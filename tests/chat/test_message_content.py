"""Tests for MessageContent dataclass."""

import datetime
import json
import uuid
from decimal import Decimal

import pytest

import pydantic
import proxai.types as types
from proxai.chat.message_content import (
    Citation,
    MessageContent,
    PydanticContent,
    SUPPORTED_MEDIA_TYPES,
    ToolContent,
    ToolKind,
)


class EventModel(pydantic.BaseModel):
  """Pydantic model spanning common non-JSON field types."""

  name: str
  when: datetime.datetime
  id: uuid.UUID
  amount: Decimal


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
    mc = MessageContent(type="image", data=b"rawbytes", media_type="image/png")
    assert mc.data == b"rawbytes"
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

  def test_json_content(self):
    mc = MessageContent(type="json", json={"key": "value", "count": 42})
    assert mc.type == types.ContentType.JSON
    assert mc.json == {"key": "value", "count": 42}

  def test_pydantic_instance_content(self):
    class UserModel(pydantic.BaseModel):
      name: str
      age: int

    instance = UserModel(name="Alice", age=30)
    mc = MessageContent(
        type="pydantic_instance",
        pydantic_content=PydanticContent(
            class_name="UserModel",
            class_value=UserModel,
            instance_value=instance,
        ),
    )
    assert mc.type == types.ContentType.PYDANTIC_INSTANCE
    assert mc.pydantic_content.class_name == "UserModel"
    assert mc.pydantic_content.instance_value == instance

  def test_tool_call_content(self):
    mc = MessageContent(
        type="tool",
        tool_content=ToolContent(
            name="search",
            kind=ToolKind.CALL,
        ),
    )
    assert mc.type == types.ContentType.TOOL
    assert mc.tool_content.name == "search"
    assert mc.tool_content.kind == ToolKind.CALL

  def test_tool_result_content_with_citations(self):
    mc = MessageContent(
        type="tool",
        tool_content=ToolContent(
            name="search",
            kind=ToolKind.RESULT,
            citations=[
                Citation(title="Page 1", url="https://example.com/1"),
                Citation(title="Page 2"),
                Citation(url="https://example.com/3"),
            ],
        ),
    )
    assert mc.tool_content.kind == ToolKind.RESULT
    assert len(mc.tool_content.citations) == 3
    assert mc.tool_content.citations[0].title == "Page 1"
    assert mc.tool_content.citations[0].url == "https://example.com/1"
    assert mc.tool_content.citations[1].url is None
    assert mc.tool_content.citations[2].title is None


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

  def test_json_without_json_field_raises_error(self):
    with pytest.raises(ValueError, match="'json' field is required"):
      MessageContent(type="json")

  def test_pydantic_instance_without_pydantic_content_raises_error(self):
    with pytest.raises(ValueError, match="'pydantic_content' field is required"):
      MessageContent(type="pydantic_instance")

  def test_tool_without_tool_content_raises_error(self):
    with pytest.raises(ValueError, match="'tool_content' field is required"):
      MessageContent(type="tool")

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
        data=b"\x89PNG\r\n",
        path="./photo.png",
        media_type="image/png",
    )
    d = mc.to_dict()
    assert d["source"] == "https://example.com/img.png"
    assert d["data"] == "iVBORw0K"
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

  def test_round_trip_media_with_bytes_data(self):
    original = MessageContent(
        type="image", data=b"\x89PNG\r\n\x1a\n",
        media_type="image/png",
    )
    restored = MessageContent.from_dict(original.to_dict())
    assert restored == original
    assert isinstance(restored.data, bytes)

  def test_json_to_dict(self):
    mc = MessageContent(type="json", json={"key": "value", "nested": {"a": 1}})
    d = mc.to_dict()
    assert d == {
        "type": "json",
        "json": {"key": "value", "nested": {"a": 1}},
    }

  def test_round_trip_json(self):
    original = MessageContent(
        type="json", json={"items": [1, 2, 3], "flag": True}
    )
    restored = MessageContent.from_dict(original.to_dict())
    assert restored == original

  def test_pydantic_instance_to_dict_with_instance_value(self):
    class UserModel(pydantic.BaseModel):
      name: str
      age: int

    instance = UserModel(name="Alice", age=30)
    mc = MessageContent(
        type="pydantic_instance",
        pydantic_content=PydanticContent(
            class_name="UserModel",
            instance_value=instance,
        ),
    )
    d = mc.to_dict()
    assert d["type"] == "pydantic_instance"
    assert d["pydantic_content"]["class_name"] == "UserModel"
    assert d["pydantic_content"]["instance_json_value"] == {
        "name": "Alice", "age": 30,
    }

  def test_pydantic_instance_to_dict_with_instance_json_value(self):
    mc = MessageContent(
        type="pydantic_instance",
        pydantic_content=PydanticContent(
            class_name="UserModel",
            instance_json_value={"name": "Bob", "age": 25},
        ),
    )
    d = mc.to_dict()
    assert d["pydantic_content"]["instance_json_value"] == {
        "name": "Bob", "age": 25,
    }

  def test_pydantic_instance_round_trip_is_lossy(self):
    class UserModel(pydantic.BaseModel):
      name: str
      age: int

    instance = UserModel(name="Alice", age=30)
    mc = MessageContent(
        type="pydantic_instance",
        pydantic_content=PydanticContent(
            class_name="UserModel",
            instance_value=instance,
        ),
    )
    d = mc.to_dict()
    assert d["pydantic_content"]["class_name"] == "UserModel"
    assert d["pydantic_content"]["instance_json_value"] == {
        "name": "Alice", "age": 30,
    }
    restored = MessageContent.from_dict(d)
    assert restored.type == types.ContentType.PYDANTIC_INSTANCE
    assert restored.pydantic_content.class_name == "UserModel"
    assert restored.pydantic_content.instance_json_value == {
        "name": "Alice", "age": 30,
    }
    assert restored.pydantic_content.class_value is None
    assert restored.pydantic_content.instance_value is None


class TestPydanticContentPostInit:
  """PydanticContent.__post_init__ normalizes class_name and instance_json."""

  def test_derives_class_name_from_class_value(self):
    pc = PydanticContent(class_value=EventModel)
    assert pc.class_name == "EventModel"

  def test_keeps_explicit_class_name(self):
    pc = PydanticContent(class_name="OverriddenName", class_value=EventModel)
    assert pc.class_name == "OverriddenName"

  def test_derives_instance_json_value_from_instance(self):
    event = EventModel(
        name="launch",
        when=datetime.datetime(2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc),
        id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        amount=Decimal("1.5"),
    )
    pc = PydanticContent(class_value=EventModel, instance_value=event)
    # Must be JSON-safe: no datetime/UUID/Decimal Python objects remain.
    assert pc.instance_json_value == {
        "name": "launch",
        "when": "2024-01-01T12:00:00Z",
        "id": "12345678-1234-5678-1234-567812345678",
        "amount": "1.5",
    }

  def test_keeps_explicit_instance_json_value(self):
    explicit = {"name": "override", "when": "1970-01-01T00:00:00Z",
                "id": "00000000-0000-0000-0000-000000000000", "amount": "0"}
    event = EventModel(
        name="launch",
        when=datetime.datetime(2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc),
        id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        amount=Decimal("1.5"),
    )
    pc = PydanticContent(
        class_value=EventModel,
        instance_value=event,
        instance_json_value=explicit,
    )
    assert pc.instance_json_value is explicit


class TestPydanticContentDatetimeRoundTrip:
  """End-to-end round-trip for pydantic instances with non-JSON field types."""

  def _build_event_instance(self):
    return EventModel(
        name="launch",
        when=datetime.datetime(2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc),
        id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        amount=Decimal("1.5"),
    )

  def test_to_dict_is_json_dumps_safe(self):
    event = self._build_event_instance()
    mc = MessageContent(
        type="pydantic_instance",
        pydantic_content=PydanticContent(
            class_value=EventModel, instance_value=event
        ),
    )
    # Must not raise: every field value is JSON-native after mode='json'.
    serialized = json.dumps(mc.to_dict(), sort_keys=True)
    assert "2024-01-01T12:00:00Z" in serialized
    assert "1.5" in serialized

  def test_model_validate_recovers_native_instance(self):
    event = self._build_event_instance()
    pc = PydanticContent(class_value=EventModel, instance_value=event)
    restored = EventModel.model_validate(pc.instance_json_value)
    assert restored == event
    assert isinstance(restored.when, datetime.datetime)
    assert isinstance(restored.id, uuid.UUID)
    assert isinstance(restored.amount, Decimal)

  def test_full_serialization_round_trip_preserves_data(self):
    event = self._build_event_instance()
    mc = MessageContent(
        type="pydantic_instance",
        pydantic_content=PydanticContent(
            class_value=EventModel, instance_value=event
        ),
    )
    # Simulate cache-style persistence: to_dict -> json.dumps -> json.loads
    # -> from_dict -> model_validate.
    wire = json.loads(json.dumps(mc.to_dict(), sort_keys=True))
    restored_mc = MessageContent.from_dict(wire)
    restored_event = EventModel.model_validate(
        restored_mc.pydantic_content.instance_json_value
    )
    assert restored_event == event

  def test_tool_call_to_dict(self):
    mc = MessageContent(
        type="tool",
        tool_content=ToolContent(name="search", kind=ToolKind.CALL),
    )
    d = mc.to_dict()
    assert d == {
        "type": "tool",
        "tool_content": {"name": "search", "kind": "CALL"},
    }

  def test_tool_result_with_citations_to_dict(self):
    mc = MessageContent(
        type="tool",
        tool_content=ToolContent(
            name="search",
            kind=ToolKind.RESULT,
            citations=[
                Citation(title="Page 1", url="https://example.com/1"),
                Citation(title="Page 2"),
            ],
        ),
    )
    d = mc.to_dict()
    assert d["tool_content"]["name"] == "search"
    assert d["tool_content"]["kind"] == "RESULT"
    assert d["tool_content"]["citations"] == [
        {"title": "Page 1", "url": "https://example.com/1"},
        {"title": "Page 2"},
    ]

  def test_round_trip_tool_call(self):
    original = MessageContent(
        type="tool",
        tool_content=ToolContent(name="search", kind=ToolKind.CALL),
    )
    restored = MessageContent.from_dict(original.to_dict())
    assert restored == original

  def test_round_trip_tool_result_with_citations(self):
    original = MessageContent(
        type="tool",
        tool_content=ToolContent(
            name="search",
            kind=ToolKind.RESULT,
            citations=[
                Citation(title="Page 1", url="https://example.com/1"),
                Citation(url="https://example.com/2"),
            ],
        ),
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
