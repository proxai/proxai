"""MessageContent dataclass for multi-modal message content blocks."""

import copy
import dataclasses

from proxai import types

SUPPORTED_MEDIA_TYPES = frozenset({
    # Image
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/heic",
    "image/heif",
    # Document
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv",
    "text/plain",
    "text/markdown",
    # Audio
    "audio/mpeg",
    "audio/wav",
    "audio/flac",
    "audio/aac",
    "audio/ogg",
    "audio/aiff",
    # Video
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",
    "video/mpeg",
    "video/x-matroska",
})


@dataclasses.dataclass
class MessageContent:
  """A single content block within a message.

  Represents one piece of content in a message, such as text, an image,
  a document, audio, or video. Each content block has a type and the
  appropriate fields for that type.

  Attributes:
    type: The content type ("text", "image", "document", "audio", "video").
    text: The text content. Required when type is TEXT.
    source: URL for media content.
    data: Base64-encoded inline data for media content.
    path: Local file path for media content.
    media_type: MIME type string (e.g., "image/png", "application/pdf").

  Example:
    >>> import proxai as px
    >>> # Text content
    >>> px.MessageContent(type="text", text="Hello!")
    >>> # Image from URL
    >>> px.MessageContent(type="image", source="https://example.com/img.png")
    >>> # Image from file
    >>> px.MessageContent(type=px.MessageContent.Type.IMAGE, path="./photo.png")
  """

  type: types.ContentType | str
  text: str | None = None
  source: str | None = None
  data: str | None = None
  path: str | None = None
  media_type: str | None = None

  Type = types.ContentType

  def __post_init__(self):
    if isinstance(self.type, str):
      try:
        self.type = types.ContentType(self.type)
      except ValueError:
        raise ValueError(
            f"Invalid content type: {self.type!r}. "
            f"Must be one of: {[t.value for t in types.ContentType]}"
        )
    if self.type in (types.ContentType.TEXT, types.ContentType.THINKING):
      if self.text is None:
        raise ValueError(
            f"'text' field is required when type is '{self.type.value}'."
        )
      invalid_fields = []
      if self.source is not None:
        invalid_fields.append("source")
      if self.data is not None:
        invalid_fields.append("data")
      if self.path is not None:
        invalid_fields.append("path")
      if self.media_type is not None:
        invalid_fields.append("media_type")
      if invalid_fields:
        raise ValueError(
            f"{', '.join(invalid_fields)} cannot be set "
            f"when type is '{self.type.value}'."
        )
    else:
      if self.source is None and self.data is None and self.path is None:
        raise ValueError(
            f"At least one of 'source', 'data', or 'path' is required "
            f"when type is '{self.type.value}'."
        )
    if self.media_type is not None:
      self._validate_media_type(self.media_type)

  @staticmethod
  def _validate_media_type(media_type: str):
    """Validate that media_type is a supported MIME type."""
    if media_type not in SUPPORTED_MEDIA_TYPES:
      raise ValueError(
          f"Unsupported media_type: {media_type!r}. "
          f"Supported types: {sorted(SUPPORTED_MEDIA_TYPES)}."
      )

  def to_dict(self) -> dict:
    """Serialize to a dictionary, omitting None fields."""
    result = {"type": self.type.value}
    if self.text is not None:
      result["text"] = self.text
    if self.source is not None:
      result["source"] = self.source
    if self.data is not None:
      result["data"] = self.data
    if self.path is not None:
      result["path"] = self.path
    if self.media_type is not None:
      result["media_type"] = self.media_type
    return result

  @classmethod
  def from_dict(cls, data: dict) -> "MessageContent":
    """Create a MessageContent from a dictionary."""
    return cls(
        type=data["type"],
        text=data.get("text"),
        source=data.get("source"),
        data=data.get("data"),
        path=data.get("path"),
        media_type=data.get("media_type"),
    )

  def copy(self) -> "MessageContent":
    """Return a deep copy of this MessageContent."""
    return copy.deepcopy(self)

  def __repr__(self) -> str:
    parts = [f"type='{self.type.value}'"]
    if self.text is not None:
      display_text = self.text if len(self.text) <= 50 else self.text[:47] + "..."
      parts.append(f"text='{display_text}'")
    if self.source is not None:
      parts.append(f"source='{self.source}'")
    if self.data is not None:
      preview = self.data[:20] + "..." if len(self.data) > 20 else self.data
      parts.append(f"data='{preview}'")
    if self.path is not None:
      parts.append(f"path='{self.path}'")
    if self.media_type is not None:
      parts.append(f"media_type='{self.media_type}'")
    return f"MessageContent({', '.join(parts)})"
