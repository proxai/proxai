# read version from installed package
from importlib.metadata import version
from proxai.proxai import (
    generate_text,
    set_model,
    LoggingOptions,
    connect,
    AvailableModels
)


__version__ = version("proxai")
models = AvailableModels()
