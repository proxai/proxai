# read version from installed package
from importlib.metadata import version
from proxai.proxai import (
     connect,
     generate_text,
     get_available_models,
)


__version__ = version("proxai")
models = get_available_models()
