# read version from installed package
from importlib.metadata import version
from proxai.proxai import (
    CacheOptions,
    LoggingOptions,
    ProxDashOptions,
    check_health,
    connect,
    set_model,
    generate_text,
    get_summary,
    get_available_models,
    get_current_options,
    _init_hidden_run_key,
)


__version__ = version("proxai")
_init_hidden_run_key()
models = get_available_models()
