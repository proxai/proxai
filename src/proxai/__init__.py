# read version from installed package
from importlib.metadata import version
from proxai.proxai import (
    generate_text,
    get_summary,
    get_available_models,
    set_model,
    CacheOptions,
    LoggingOptions,
    connect,
)


__version__ = version("proxai")
models = get_available_models()
