"""Tests for Chat class."""

import pytest

from proxai.chat.chat_session import Chat
from proxai.chat.message import Message
from proxai.chat.message_content import MessageContent


def _user_msg(text):
  return Message(role="user", content=text)


def _assistant_msg(text):
  return Message(role="assistant", content=text)


class TestChatCreation:
  """Test creating Chat with different initialization options."""

  def test_empty_chat(self):
    chat = Chat()
    assert len(chat) == 0
    assert chat.system_prompt is None

  def test_chat_with_system_prompt(self):
    chat = Chat(system_prompt="You are helpful.")
    assert chat.system_prompt == "You are helpful."
    assert len(chat) == 0

  def test_chat_with_messages(self):
    chat = Chat(messages=[_user_msg("Hello"), _assistant_msg("Hi")])
    assert len(chat) == 2
    assert chat[0].content == "Hello"
    assert chat[1].content == "Hi"

  def test_chat_from_dicts(self):
    chat = Chat(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ])
    assert len(chat) == 2
    assert chat[0].content == "Hello"


class TestChatSystemPrompt:
  """Test system prompt property."""

  def test_set_and_get_system_prompt(self):
    chat = Chat()
    chat.system_prompt = "Be concise."
    assert chat.system_prompt == "Be concise."

  def test_clear_system_prompt(self):
    chat = Chat(system_prompt="Hello")
    chat.system_prompt = None
    assert chat.system_prompt is None

  def test_invalid_system_prompt_raises_error(self):
    chat = Chat()
    with pytest.raises(TypeError, match="system_prompt must be a string"):
      chat.system_prompt = 123


class TestChatMutation:
  """Test append, extend, insert, pop, clear."""

  def test_append(self):
    chat = Chat()
    chat.append(_user_msg("Hello"))
    assert len(chat) == 1
    assert chat[0].content == "Hello"

  def test_append_dict(self):
    chat = Chat()
    chat.append({"role": "user", "content": "Hello"})
    assert len(chat) == 1
    assert chat[0].content == "Hello"

  def test_append_invalid_raises_error(self):
    chat = Chat()
    with pytest.raises(TypeError, match="Expected Message or dict"):
      chat.append("not a message")

  def test_extend(self):
    chat = Chat()
    chat.extend([_user_msg("Hello"), _assistant_msg("Hi")])
    assert len(chat) == 2

  def test_insert(self):
    chat = Chat(messages=[_user_msg("First"), _user_msg("Third")])
    chat.insert(1, _user_msg("Second"))
    assert len(chat) == 3
    assert chat[1].content == "Second"

  def test_pop(self):
    chat = Chat(messages=[_user_msg("Hello"), _assistant_msg("Hi")])
    popped = chat.pop()
    assert popped.content == "Hi"
    assert len(chat) == 1

  def test_clear(self):
    chat = Chat(messages=[_user_msg("Hello"), _assistant_msg("Hi")])
    chat.clear()
    assert len(chat) == 0


class TestChatListInterface:
  """Test list-like operations: indexing, slicing, iteration, deletion."""

  def test_getitem(self):
    chat = Chat(messages=[_user_msg("A"), _user_msg("B"), _user_msg("C")])
    assert chat[0].content == "A"
    assert chat[-1].content == "C"

  def test_slice_returns_chat(self):
    chat = Chat(
        messages=[_user_msg("A"), _user_msg("B"), _user_msg("C")],
        system_prompt="Test",
    )
    sliced = chat[1:]
    assert isinstance(sliced, Chat)
    assert len(sliced) == 2
    assert sliced[0].content == "B"
    assert sliced.system_prompt == "Test"

  def test_setitem(self):
    chat = Chat(messages=[_user_msg("Hello")])
    chat[0] = _assistant_msg("Replaced")
    assert chat[0].content == "Replaced"

  def test_delitem(self):
    chat = Chat(messages=[_user_msg("A"), _user_msg("B")])
    del chat[0]
    assert len(chat) == 1
    assert chat[0].content == "B"

  def test_iter(self):
    messages = [_user_msg("A"), _assistant_msg("B")]
    chat = Chat(messages=messages)
    contents = [msg.content for msg in chat]
    assert contents == ["A", "B"]

  def test_len(self):
    chat = Chat(messages=[_user_msg("A"), _user_msg("B")])
    assert len(chat) == 2


class TestChatOperators:
  """Test __add__, __iadd__, __eq__."""

  def test_add_creates_new_chat(self):
    chat1 = Chat(messages=[_user_msg("A")], system_prompt="Sys")
    chat2 = Chat(messages=[_assistant_msg("B")])
    combined = chat1 + chat2
    assert len(combined) == 2
    assert combined.system_prompt == "Sys"
    assert len(chat1) == 1

  def test_iadd(self):
    chat1 = Chat(messages=[_user_msg("A")])
    chat2 = Chat(messages=[_assistant_msg("B")])
    chat1 += chat2
    assert len(chat1) == 2

  def test_equality(self):
    chat1 = Chat(
        messages=[_user_msg("Hello")], system_prompt="Sys"
    )
    chat2 = Chat(
        messages=[_user_msg("Hello")], system_prompt="Sys"
    )
    assert chat1 == chat2

  def test_inequality_different_messages(self):
    chat1 = Chat(messages=[_user_msg("A")])
    chat2 = Chat(messages=[_user_msg("B")])
    assert chat1 != chat2

  def test_inequality_different_system_prompt(self):
    chat1 = Chat(messages=[_user_msg("A")], system_prompt="X")
    chat2 = Chat(messages=[_user_msg("A")], system_prompt="Y")
    assert chat1 != chat2


class TestChatCopyAndRepr:
  """Test copy and repr behavior."""

  def test_copy_is_independent(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Sys")
    copied = chat.copy()
    copied.append(_assistant_msg("Hi"))
    copied.system_prompt = "Changed"
    assert len(chat) == 1
    assert chat.system_prompt == "Sys"

  def test_repr(self):
    assert repr(Chat()) == "Chat(0 messages)"
    assert repr(Chat(messages=[_user_msg("A")])) == "Chat(1 message)"
    assert repr(Chat(messages=[_user_msg("A"), _user_msg("B")])) == "Chat(2 messages)"


class TestChatSerialization:
  """Test to_dict and from_dict round-trip."""

  def test_to_dict_without_system_prompt(self):
    chat = Chat(messages=[_user_msg("Hello")])
    d = chat.to_dict()
    assert "system_prompt" not in d
    assert len(d["messages"]) == 1

  def test_to_dict_with_system_prompt(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    d = chat.to_dict()
    assert d["system_prompt"] == "Be helpful."

  def test_round_trip(self):
    original = Chat(system_prompt="Test prompt")
    original.append(_user_msg("Hello"))
    original.append(Message(role="assistant", content=[
        MessageContent(type="text", text="Here's an image:"),
        MessageContent(type="image", source="https://example.com/img.png"),
    ]))
    restored = Chat.from_dict(original.to_dict())
    assert restored == original
