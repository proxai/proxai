# read version from installed package
from importlib.metadata import version
from proxai.proxai import generate_text, logging_options, set_model


__version__ = version("proxai")
