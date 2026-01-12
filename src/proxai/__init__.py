# read version from installed package
from importlib.metadata import version

from proxai.proxai import (
    CacheOptions,
    Client,
    DefaultModelsConnector,
    LoggingOptions,
    ProxDashOptions,
    ResponseFormat,
    ResponseFormatType,
    check_health,
    connect,
    generate_text,
    get_current_options,
    get_default_proxai_client,
    reset_state,
    set_model,
)

__all__ = [
    "CacheOptions",
    "Client",
    "DefaultModelsConnector",
    "LoggingOptions",
    "ProxDashOptions",
    "ResponseFormat",
    "ResponseFormatType",
    "check_health",
    "connect",
    "generate_text",
    "get_current_options",
    "get_default_proxai_client",
    "models",
    "reset_state",
    "set_model",
]

__version__ = version("proxai")
models = DefaultModelsConnector()
