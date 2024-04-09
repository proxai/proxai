# read version from installed package
from importlib.metadata import version
from proxai.proxai import generate_text, register_model, logging_options


__version__ = version("proxai")
