# read version from installed package
from importlib.metadata import version
from proxai.proxai import (
    CacheOptions,
    LoggingOptions,
    ProxDashOptions,
    set_run_type,
    check_health,
    connect,
    set_model,
    generate_text,
    get_summary,
    get_available_models,
    get_current_options,
    reset_state,
    reset_platform_cache,
    _init_globals,
    _get_model_configs,
)


__version__ = version("proxai")
_init_globals()
models = get_available_models()
