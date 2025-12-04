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
    export_client_state,
    import_client_state,
    get_client,
)
from proxai.client import ProxAIClient


__version__ = version("proxai")
models = get_available_models()
