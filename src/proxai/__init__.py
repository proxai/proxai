# read version from installed package
from importlib.metadata import version

from proxai.chat import Chat
from proxai.chat import Message
from proxai.chat import MessageContent
from proxai.client import ModelConnector
from proxai.proxai import (
    CacheOptions,
    Chat,
    Client,
    DefaultModelsConnector,
    FeatureMappingStrategy,
    LoggingOptions,
    ParameterType,
    ProviderModelType,
    ProxDashOptions,
    ResponseFormatType,
    Tools,
    check_health,
    connect,
    generate,
    generate_text,
    get_current_options,
    get_default_proxai_client,
    reset_state,
    set_model,
)
from proxai.types import MessageRoleType

__all__ = [
    "CacheOptions",
    "Chat",
    "Client",
    "DefaultModelsConnector",
    "FeatureMappingStrategy",
    "LoggingOptions",
    "ParameterType",
    "Tools",
    "Message",
    "MessageContent",
    "MessageRoleType",
    "ModelConnector",
    "ProxDashOptions",
    "ProviderModelType",
    "ResponseFormatType",
    "check_health",
    "connect",
    "generate",
    "generate_text",
    "get_current_options",
    "get_default_proxai_client",
    "models",
    "reset_state",
    "set_model",
]

__version__ = version("proxai")
models = DefaultModelsConnector()
