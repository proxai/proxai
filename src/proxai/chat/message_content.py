"""MessageContent dataclass for multi-modal message content blocks."""
from __future__ import annotations

import base64
import copy
import dataclasses
import enum
from typing import Any

import pydantic

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


class MessageRoleType(str, enum.Enum):
  """Role of the message sender in a conversation.

  Attributes:
    USER: Message from the user.
    ASSISTANT: Message from the AI assistant.
  """

  USER = "user"
  ASSISTANT = "assistant"


class ContentType(str, enum.Enum):
  """Type of content in a message block.

  Attributes:
    TEXT: Plain text content.
    THINKING: Model thinking/reasoning text content.
    IMAGE: Image content (URL, base64, or file path).
    DOCUMENT: Document content (PDF, DOCX, etc.).
    AUDIO: Audio content (MP3, WAV, etc.).
    VIDEO: Video content (MP4, WebM, etc.).
  """

  TEXT = "text"
  THINKING = "thinking"
  IMAGE = "image"
  DOCUMENT = "document"
  AUDIO = "audio"
  VIDEO = "video"
  JSON = "json"
  PYDANTIC_INSTANCE = "pydantic_instance"
  TOOL = "tool"


@dataclasses.dataclass
class PydanticContent:
  """Pydantic model information for structured response parsing.

  The serializable fields (`class_name`, `instance_json_value`) are the
  authoritative source for equality, serialization, and cache hashing.
  `class_value` and `instance_value` are convenience handles to the live
  Python objects and may be dropped after a serialization round-trip.

  __post_init__ enforces the invariant that whenever a rich handle is
  provided, its serializable counterpart is materialized: `class_name` is
  derived from `class_value.__name__`, and `instance_json_value` is
  derived from `instance_value.model_dump(mode='json')`. `mode='json'`
  converts datetime/UUID/Decimal/Enum/Path/set/nested-model fields into
  JSON-native forms; `class_value.model_validate(instance_json_value)`
  round-trips back to the original native instance. Callers that supply
  inconsistent name/class or instance/json pairs keep whatever they
  passed in — the hook only fills missing fields, it never overwrites.
  """

  class_name: str | None = None
  class_value: type[pydantic.BaseModel] | None = None
  instance_value: pydantic.BaseModel | None = None
  instance_json_value: dict[str, Any] | None = None

  def __post_init__(self):
    if self.class_name is None and self.class_value is not None:
      self.class_name = self.class_value.__name__
    if self.instance_json_value is None and self.instance_value is not None:
      self.instance_json_value = self.instance_value.model_dump(mode='json')


class ToolKind(str, enum.Enum):
  CALL = "CALL"
  RESULT = "RESULT"


@dataclasses.dataclass
class Citation:
  title: str | None = None
  url: str | None = None


@dataclasses.dataclass
class ToolContent:
  """Tool content for structured response parsing."""

  name: str | None = None
  kind: ToolKind | None = None
  citations: list[Citation] | None = None


class FileUploadState(str, enum.Enum):
  """Processing state of a file upload."""

  PENDING = "pending"
  ACTIVE = "active"
  FAILED = "failed"


@dataclasses.dataclass
class FileUploadMetadata:
  """Metadata from a provider File API upload."""

  file_id: str
  provider: str | None = None
  filename: str | None = None
  size_bytes: int | None = None
  mime_type: str | None = None
  created_at: str | None = None
  expires_at: str | None = None
  uri: str | None = None
  state: FileUploadState | None = None
  sha256_hash: str | None = None

  def to_dict(self) -> dict:
    result = {'file_id': self.file_id}
    if self.provider is not None:
      result['provider'] = self.provider
    if self.filename is not None:
      result['filename'] = self.filename
    if self.size_bytes is not None:
      result['size_bytes'] = self.size_bytes
    if self.mime_type is not None:
      result['mime_type'] = self.mime_type
    if self.created_at is not None:
      result['created_at'] = self.created_at
    if self.expires_at is not None:
      result['expires_at'] = self.expires_at
    if self.uri is not None:
      result['uri'] = self.uri
    if self.state is not None:
      result['state'] = self.state.value
    if self.sha256_hash is not None:
      result['sha256_hash'] = self.sha256_hash
    return result

  @classmethod
  def from_dict(cls, data: dict) -> "FileUploadMetadata":
    return cls(
        file_id=data["file_id"],
        provider=data.get("provider"),
        filename=data.get("filename"),
        size_bytes=data.get("size_bytes"),
        mime_type=data.get("mime_type"),
        created_at=data.get("created_at"),
        expires_at=data.get("expires_at"),
        uri=data.get("uri"),
        state=(
            FileUploadState(data["state"])
            if data.get("state") else None
        ),
        sha256_hash=data.get("sha256_hash"),
    )


@dataclasses.dataclass
class ProxDashFileStatus:
  """Metadata from a ProxDash file upload.

  ProxDash is the central file control center, separate from provider
  file APIs. Provider file IDs are ephemeral transport metadata that
  expire and get cleaned up. The ProxDash file ID is the permanent
  reference to the canonical copy stored in ProxDash's S3 bucket.
  """

  file_id: str
  s3_key: str | None = None
  upload_confirmed: bool = False
  source: str | None = None
  created_at: str | None = None
  updated_at: str | None = None

  def to_dict(self) -> dict:
    result = {'file_id': self.file_id}
    if self.s3_key is not None:
      result['s3_key'] = self.s3_key
    result['upload_confirmed'] = self.upload_confirmed
    if self.source is not None:
      result['source'] = self.source
    if self.created_at is not None:
      result['created_at'] = self.created_at
    if self.updated_at is not None:
      result['updated_at'] = self.updated_at
    return result

  @classmethod
  def from_dict(cls, data: dict) -> "ProxDashFileStatus":
    return cls(
        file_id=data["file_id"],
        s3_key=data.get("s3_key"),
        upload_confirmed=data.get("upload_confirmed", False),
        source=data.get("source"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


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
    path: Local file path for media content. The query cache key includes
      the file's mtime_ns and size (in addition to the path string), so an
      in-place edit invalidates the cache. Replacing the file with one of
      identical size and mtime (e.g., `touch -r`) will NOT be detected —
      pass `data` bytes directly if byte-exact cache semantics matter.
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

  type: ContentType | str | None = None

  text: str | None = None

  json: dict[str, Any] | None = None
  pydantic_content: PydanticContent | None = None
  tool_content: ToolContent | None = None

  source: str | None = None
  data: bytes | None = None
  path: str | None = None
  media_type: str | None = None
  filename: str | None = None

  provider_file_api_status: dict[str, FileUploadMetadata] | None = None
  provider_file_api_ids: dict[str, str] | None = None

  proxdash_file_id: str | None = None
  proxdash_file_status: ProxDashFileStatus | None = None

  def __post_init__(self):
    if self.type is None:
      if self.media_type is not None:
        inferred = self._infer_content_type(self.media_type)
        if inferred is None:
          raise ValueError(
              f"Cannot infer content type from media_type "
              f"{self.media_type!r}. Provide 'type' explicitly."
          )
        self.type = inferred
      else:
        raise ValueError(
            "'type' is required when 'media_type' is not provided."
        )
    if isinstance(self.type, str):
      try:
        self.type = ContentType(self.type)
      except ValueError:
        raise ValueError(
            f"Invalid content type: {self.type!r}. "
            f"Must be one of: {[t.value for t in ContentType]}"
        )
    if self.media_type is not None and self.type in (
        ContentType.IMAGE,
        ContentType.DOCUMENT,
        ContentType.AUDIO,
        ContentType.VIDEO,
    ):
      expected = self._infer_content_type(self.media_type)
      if expected is not None and expected != self.type:
        raise ValueError(
            f"Content type '{self.type.value}' does not match "
            f"media_type '{self.media_type}' "
            f"(expected type '{expected.value}')."
        )
    if self.type in (ContentType.TEXT, ContentType.THINKING):
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
    elif self.type == ContentType.JSON:
      if self.json is None:
        raise ValueError("'json' field is required when type is 'json'.")
    elif self.type == ContentType.PYDANTIC_INSTANCE:
      if self.pydantic_content is None:
        raise ValueError(
            "'pydantic_content' field is required when type is "
            "'pydantic_instance'."
        )
    elif self.type == ContentType.TOOL:
      if self.tool_content is None:
        raise ValueError(
            "'tool_content' field is required when type is 'tool'."
        )
    elif self.type in (
        ContentType.IMAGE,
        ContentType.DOCUMENT,
        ContentType.AUDIO,
        ContentType.VIDEO,
    ):
      has_local_content = (
          self.source is not None or self.data is not None or
          self.path is not None
      )
      has_remote_reference = (
          self.provider_file_api_ids is not None and
          len(self.provider_file_api_ids) > 0
      )
      if not has_local_content and not has_remote_reference:
        raise ValueError(
            f"At least one of 'source', 'data', 'path', or "
            f"'provider_file_api_ids' is required "
            f"when type is '{self.type.value}'."
        )
    if self.media_type is not None:
      self._validate_media_type(self.media_type)

  _MIME_PREFIX_TO_CONTENT_TYPE = {
      'image/': ContentType.IMAGE,
      'audio/': ContentType.AUDIO,
      'video/': ContentType.VIDEO,
  }
  _MIME_TO_CONTENT_TYPE = {
      'application/pdf': ContentType.DOCUMENT,
      'application/vnd.openxmlformats-officedocument'
      '.wordprocessingml.document': ContentType.DOCUMENT,
      'application/vnd.openxmlformats-officedocument'
      '.spreadsheetml.sheet': ContentType.DOCUMENT,
      'text/csv': ContentType.DOCUMENT,
      'text/plain': ContentType.DOCUMENT,
      'text/markdown': ContentType.DOCUMENT,
  }

  @staticmethod
  def _infer_content_type(media_type: str) -> ContentType | None:
    """Infer ContentType from a MIME type string."""
    if media_type in MessageContent._MIME_TO_CONTENT_TYPE:
      return MessageContent._MIME_TO_CONTENT_TYPE[media_type]
    for prefix, content_type in (
        MessageContent._MIME_PREFIX_TO_CONTENT_TYPE.items()
    ):
      if media_type.startswith(prefix):
        return content_type
    return None

  @staticmethod
  def _validate_media_type(media_type: str):
    """Validate that media_type is a supported MIME type."""
    if media_type not in SUPPORTED_MEDIA_TYPES:
      raise ValueError(
          f"Unsupported media_type: {media_type!r}. "
          f"Supported  {sorted(SUPPORTED_MEDIA_TYPES)}."
      )

  def to_dict(self) -> dict:
    """Serialize to a dictionary, omitting None fields."""
    result = {"type": self.type.value}
    if self.text is not None:
      result["text"] = self.text
    if self.json is not None:
      result["json"] = self.json
    if self.pydantic_content is not None:
      pydantic_dict = {}
      if self.pydantic_content.class_name is not None:
        pydantic_dict["class_name"] = self.pydantic_content.class_name
      if self.pydantic_content.instance_json_value is not None:
        pydantic_dict["instance_json_value"] = (
            self.pydantic_content.instance_json_value
        )
      if pydantic_dict:
        result["pydantic_content"] = pydantic_dict
    if self.tool_content is not None:
      tool_dict = {}
      if self.tool_content.name is not None:
        tool_dict["name"] = self.tool_content.name
      if self.tool_content.kind is not None:
        tool_dict["kind"] = self.tool_content.kind.value
      if self.tool_content.citations is not None:
        tool_dict["citations"] = []
        for citation in self.tool_content.citations:
          citation_dict = {}
          if citation.title is not None:
            citation_dict["title"] = citation.title
          if citation.url is not None:
            citation_dict["url"] = citation.url
          tool_dict["citations"].append(citation_dict)
      if tool_dict:
        result["tool_content"] = tool_dict
    if self.source is not None:
      result["source"] = self.source
    if self.data is not None:
      result["data"] = base64.b64encode(self.data).decode('utf-8')
    if self.path is not None:
      result["path"] = self.path
    if self.media_type is not None:
      result["media_type"] = self.media_type
    if self.filename is not None:
      result["filename"] = self.filename
    if self.provider_file_api_ids is not None:
      result["provider_file_api_ids"] = self.provider_file_api_ids
    if self.provider_file_api_status is not None:
      status_dict = {}
      for provider, meta in self.provider_file_api_status.items():
        status_dict[provider] = meta.to_dict()
      result["provider_file_api_status"] = status_dict
    if self.proxdash_file_id is not None:
      result["proxdash_file_id"] = self.proxdash_file_id
    if self.proxdash_file_status is not None:
      result["proxdash_file_status"] = self.proxdash_file_status.to_dict()
    return result

  @classmethod
  def from_dict(cls, data: dict) -> "MessageContent":
    """Create a MessageContent from a dictionary."""
    pydantic_content = None
    if data.get("pydantic_content") is not None:
      pydantic_content = PydanticContent(**data["pydantic_content"])
    tool_content = None
    if data.get("tool_content") is not None:
      tc_data = data["tool_content"]
      citations = None
      if tc_data.get("citations") is not None:
        citations = [Citation(**c) for c in tc_data["citations"]]
      tool_content = ToolContent(
          name=tc_data.get("name"),
          kind=ToolKind(tc_data["kind"]) if tc_data.get("kind") else None,
          citations=citations,
      )
    provider_file_api_ids = data.get("provider_file_api_ids")
    provider_file_api_status = None
    if data.get("provider_file_api_status") is not None:
      provider_file_api_status = {
          provider: FileUploadMetadata.from_dict(meta_dict)
          for provider, meta_dict in data["provider_file_api_status"].items()
      }
    proxdash_file_id = data.get("proxdash_file_id")
    proxdash_file_status = None
    if data.get("proxdash_file_status") is not None:
      proxdash_file_status = ProxDashFileStatus.from_dict(
          data["proxdash_file_status"]
      )
    return cls(
        type=data["type"],
        text=data.get("text"),
        json=data.get("json"),
        pydantic_content=pydantic_content,
        tool_content=tool_content,
        source=data.get("source"),
        data=(base64.b64decode(data["data"]) if "data" in data else None),
        path=data.get("path"),
        media_type=data.get("media_type"),
        filename=data.get("filename"),
        provider_file_api_status=provider_file_api_status,
        provider_file_api_ids=provider_file_api_ids,
        proxdash_file_id=proxdash_file_id,
        proxdash_file_status=proxdash_file_status,
    )

  def copy(self) -> "MessageContent":
    """Return a deep copy of this MessageContent."""
    return copy.deepcopy(self)

  def __repr__(self) -> str:
    parts = [f"type='{self.type.value}'"]
    if self.text is not None:
      display_text = self.text if len(self.text
                                     ) <= 50 else self.text[:47] + "..."
      parts.append(f"text='{display_text}'")
    if self.source is not None:
      parts.append(f"source='{self.source}'")
    if self.data is not None:
      preview = (
          base64.b64encode(self.data[:20]).decode('utf-8') + "..." if
          len(self.data) > 20 else base64.b64encode(self.data).decode('utf-8')
      )
      parts.append(f"data='{preview}'")
    if self.path is not None:
      parts.append(f"path='{self.path}'")
    if self.media_type is not None:
      parts.append(f"media_type='{self.media_type}'")
    return f"MessageContent({', '.join(parts)})"
