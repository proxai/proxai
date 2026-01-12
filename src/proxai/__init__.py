# read version from installed package
from importlib.metadata import version
from proxai.proxai import (
    CacheOptions,
    LoggingOptions,
    ProxDashOptions,
    Client,
    ResponseFormat,
    ResponseFormatType,
    connect,
    check_health,
    generate_text,
    get_current_options,
    get_default_proxai_client,
    set_model,
    reset_state,
    DefaultModelsConnector,
)


__version__ = version("proxai")
models = DefaultModelsConnector()
