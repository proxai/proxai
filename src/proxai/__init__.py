# read version from installed package
from importlib.metadata import version
from proxai.proxai import (
     connect,
     check_health,
     generate_text,
     get_available_models,
     get_current_options,
     set_model
)


__version__ = version("proxai")
models = get_available_models()
