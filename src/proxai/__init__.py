# read version from installed package
from importlib.metadata import version
from proxai.proxai import (
     connect,
     check_health,
     generate_text,
     get_current_options,
     get_default_proxai_client,
     set_model,
)


__version__ = version("proxai")
models = get_default_proxai_client().available_models_instance
