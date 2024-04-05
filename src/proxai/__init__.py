# read version from installed package
from importlib.metadata import version
from proxai.proxai import generate_text, register_model, local_logging_path


__version__ = version("proxai")
