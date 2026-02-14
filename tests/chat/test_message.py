"""Tests for Message dataclass."""

import pytest

import proxai.types as types
from proxai.chat.message import Message
from proxai.chat.message_content import MessageContent


class TestMessageCreation:
  """Test creating Message with different roles and content types."""

  def test_string_content_with_string_role(self):
    msg = Message(role="user", content="Hello")
    assert msg.role == types.MessageRoleType.USER
    assert msg.content == "Hello"

  def test_string_content_with_enum_role(self):
    msg = Message(role=types.MessageRoleType.ASSISTANT, content="Hi there")
    assert msg.role == types.MessageRoleType.ASSISTANT
    assert msg.content == "Hi there"

  def test_list_content_with_message_content_objects(self):
    msg = Message(role="user", content=[
        MessageContent(type="image", source="https://example.com/img.png"),
        MessageContent(type="text", text="What is this?"),
    ])
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert msg.content[0].type == types.ContentType.IMAGE
    assert msg.content[1].type == types.ContentType.TEXT

  def test_list_content_normalizes_bare_strings(self):
    msg = Message(role="user", content=["Hello", "World"])
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert all(isinstance(item, MessageContent) for item in msg.content)
    assert msg.content[0].type == types.ContentType.TEXT
    assert msg.content[0].text == "Hello"
    assert msg.content[1].text == "World"

  def test_list_content_mixed_strings_and_message_content(self):
    msg = Message(role="user", content=[
        "Describe this image:",
        MessageContent(type="image", source="https://example.com/img.png"),
    ])
    assert len(msg.content) == 2
    assert msg.content[0].text == "Describe this image:"
    assert msg.content[1].type == types.ContentType.IMAGE


class TestMessageValidation:
  """Test validation rules for Message."""

  def test_invalid_role_raises_error(self):
    with pytest.raises(ValueError, match="Invalid role"):
      Message(role="system", content="Hello")

  def test_invalid_content_list_item_raises_error(self):
    with pytest.raises(TypeError, match="Content list items must be"):
      Message(role="user", content=[123])


class TestMessageSerialization:
  """Test to_dict and from_dict round-trip."""

  def test_string_content_to_dict(self):
    msg = Message(role="user", content="Hello")
    d = msg.to_dict()
    assert d == {"role": "user", "content": "Hello"}

  def test_list_content_to_dict(self):
    msg = Message(role="user", content=[
        MessageContent(type="text", text="Hello"),
        MessageContent(type="image", source="https://example.com/img.png"),
    ])
    d = msg.to_dict()
    assert d["role"] == "user"
    assert len(d["content"]) == 2
    assert d["content"][0] == {"type": "text", "text": "Hello"}
    assert d["content"][1] == {"type": "image", "source": "https://example.com/img.png"}

  def test_round_trip_string_content(self):
    original = Message(role="assistant", content="Hello world")
    restored = Message.from_dict(original.to_dict())
    assert restored == original

  def test_round_trip_list_content(self):
    original = Message(role="user", content=[
        MessageContent(type="text", text="Check this"),
        MessageContent(type="image", source="https://example.com/img.png"),
    ])
    restored = Message.from_dict(original.to_dict())
    assert restored == original


class TestMessageCopyAndRepr:
  """Test copy and repr behavior."""

  def test_copy_is_independent(self):
    original = Message(role="user", content="Hello")
    copied = original.copy()
    copied.content = "Changed"
    assert original.content == "Hello"

  def test_repr_string_content(self):
    msg = Message(role="user", content="Hello")
    assert repr(msg) == "Message(role='user', content='Hello')"

  def test_repr_list_content(self):
    msg = Message(role="user", content=[
        MessageContent(type="text", text="Hello"),
        MessageContent(type="image", source="https://example.com/img.png"),
    ])
    assert repr(msg) == "Message(role='user', content=[2 items])"
