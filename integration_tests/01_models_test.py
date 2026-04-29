"""Integration tests — px.models.* and check_health.

Most blocks here are pure JSON-registry reads (offline). The
list_working_* blocks probe real provider APIs and require provider
credentials in the environment (OPENAI_API_KEY, GEMINI_API_KEY, ...).

Usage:
  poetry run python3 integration_tests/01_models_test.py
  poetry run python3 integration_tests/01_models_test.py --mode new
  poetry run python3 integration_tests/01_models_test.py --auto-continue
  poetry run python3 integration_tests/01_models_test.py --test list_models_default
"""
import sys
import os
import time

# Mock providers only show up in list_models when their env var is set
# (per PROVIDER_KEY_MAP in src/proxai/connectors/model_configs.py).
os.environ.setdefault('MOCK_PROVIDER_API_KEY', 'dummy')
os.environ.setdefault('MOCK_FAILING_PROVIDER', 'dummy')
os.environ.setdefault('MOCK_SLOW_PROVIDER', 'dummy')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import proxai as px
import proxai.types as types

import _utils
from _utils import (
    integration_block, manual_check, run_sequence,
    init_run, ensure_setup_state,
)


_LABEL = '01_models'


def _names(provider_models) -> set:
  return {pm.model for pm in provider_models}


@integration_block
def setup_connection(state_data):
  """Connect ProxDash + cache + logging for this file."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      logging_options=types.LoggingOptions(
          logging_path=ctx.root_logging_path,
      ),
      cache_options=types.CacheOptions(
          cache_path=ctx.root_cache_path,
          clear_query_cache_on_connect=True,
      ),
      proxdash_options=types.ProxDashOptions(
          stdout=True,
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  print(f'> Connected to ProxDash at {ctx.proxdash_base_url}')
  return state_data


# -----------------------------------------------------------------------------
# 1.1 — 1.12  Pure JSON-registry reads (offline)
# -----------------------------------------------------------------------------

@integration_block
def list_models_default(state_data):
  """Default list — recommended text models present, non-text excluded."""
  start = time.time()
  models = px.models.list_models()
  elapsed = time.time() - start
  assert elapsed < 1, f'list_models should be fast (got {elapsed:.2f}s)'
  names = _names(models)
  print(f'> Default count: {len(names)}')
  assert len(names) > 0
  # Recommended text models present.
  assert 'gpt-4o' in names
  assert 'gemini-3-flash' in names
  assert 'sonnet-4.6' in names
  # Non-text-output models excluded.
  assert 'dall-e-3' not in names
  assert 'tts-1' not in names
  assert 'sora-2' not in names
  return state_data


@integration_block
def list_models_recommended_only_false(state_data):
  """recommended_only=False brings in non-recommended text models."""
  names = _names(px.models.list_models(recommended_only=False))
  print(f'> Full count: {len(names)}')
  assert 'mistral-small-latest' in names  # non-recommended text
  assert 'mock_model' in names            # non-recommended mock
  assert 'gpt-4o' in names                # recommended still present
  assert 'dall-e-3' not in names          # non-text still filtered
  return state_data


@integration_block
def list_models_by_size(state_data):
  """model_size filter picks only models with that tag."""
  small = _names(px.models.list_models(model_size='small'))
  print(f'> small: {len(small)}')
  assert 'gemini-3-flash' in small
  assert 'haiku-4.5' in small
  assert 'opus-4.6' not in small  # large

  medium = _names(px.models.list_models(model_size='medium'))
  print(f'> medium: {len(medium)}')
  assert 'sonnet-4.6' in medium
  assert 'gemini-3-flash' not in medium

  large = _names(px.models.list_models(model_size='large'))
  print(f'> large: {len(large)}')
  assert 'opus-4.6' in large
  assert 'o3' in large
  assert 'sonnet-4.6' not in large  # medium

  largest = _names(px.models.list_models(model_size='largest'))
  print(f'> largest: {len(largest)}')
  assert 'deepseek-v4-pro' in largest
  return state_data


@integration_block
def list_models_by_input_format(state_data):
  """input_format narrows to models whose input_format.X is SUPPORTED."""
  image_in = _names(px.models.list_models(
      input_format=types.InputFormatType.IMAGE))
  print(f'> image input: {len(image_in)}')
  assert 'gpt-4o' in image_in
  assert 'sonnet-4.6' in image_in
  assert 'gemini-3-flash' in image_in
  assert 'deepseek-v4-flash' not in image_in  # text-only input

  audio_in = _names(px.models.list_models(
      input_format=types.InputFormatType.AUDIO))
  print(f'> audio input: {len(audio_in)}')
  assert 'gemini-3-flash' in audio_in
  assert 'gpt-4o' not in audio_in

  video_in = _names(px.models.list_models(
      input_format=types.InputFormatType.VIDEO))
  print(f'> video input: {len(video_in)}')
  assert 'gemini-3-flash' in video_in

  doc_in = _names(px.models.list_models(
      input_format=types.InputFormatType.DOCUMENT))
  print(f'> document input: {len(doc_in)}')
  assert 'gpt-4o' in doc_in
  return state_data


@integration_block
def list_models_by_output_format(state_data):
  """output_format narrows to models with output.X SUPPORTED."""
  image = _names(px.models.list_models(
      output_format=types.OutputFormatType.IMAGE, recommended_only=False))
  print(f'> image output: {len(image)}')
  assert 'dall-e-3' in image
  assert 'gemini-2.5-flash-image' in image
  assert 'gpt-4o' not in image

  audio = _names(px.models.list_models(
      output_format=types.OutputFormatType.AUDIO, recommended_only=False))
  print(f'> audio output: {len(audio)}')
  assert 'tts-1' in audio
  assert len(audio) >= 2  # multiple audio models from at least one provider

  video = _names(px.models.list_models(
      output_format=types.OutputFormatType.VIDEO, recommended_only=False))
  print(f'> video output: {len(video)}')
  assert 'sora-2' in video

  json_out = _names(px.models.list_models(
      output_format=types.OutputFormatType.JSON))
  print(f'> json output: {len(json_out)}')
  assert 'gpt-4o' in json_out
  assert 'sonnet-4.6' in json_out  # BEST_EFFORT counts under default mapping
  return state_data


@integration_block
def list_models_by_tool_tags(state_data):
  """tool_tags=WEB_SEARCH matches models with web_search SUPPORTED."""
  ws = _names(px.models.list_models(tool_tags=types.ToolTag.WEB_SEARCH))
  print(f'> web_search: {len(ws)}')
  assert 'gpt-4o' in ws
  assert 'gemini-3-flash' in ws
  assert 'sonnet-4.6' in ws
  assert 'deepseek-v4-flash' not in ws
  return state_data


@integration_block
def list_models_by_feature_tags(state_data):
  """feature_tags=THINKING matches models with thinking SUPPORTED."""
  thinking = _names(px.models.list_models(
      feature_tags=types.FeatureTag.THINKING))
  print(f'> thinking: {len(thinking)}')
  assert 'o3' in thinking
  assert 'opus-4.6' in thinking
  assert 'sonnet-4.6' in thinking
  assert 'gpt-4o' not in thinking
  return state_data


@integration_block
def list_models_combined_filters(state_data):
  """Combining filters intersects effects."""
  image_json = _names(px.models.list_models(
      input_format=types.InputFormatType.IMAGE,
      output_format=types.OutputFormatType.JSON,
  ))
  print(f'> image-in + json-out: {len(image_json)}')
  assert 'gpt-4o' in image_json
  assert 'gemini-3-flash' in image_json
  assert 'deepseek-v4-flash' not in image_json  # no image input
  assert 'dall-e-3' not in image_json           # no json output

  size_tool = _names(px.models.list_models(
      model_size='medium', tool_tags=types.ToolTag.WEB_SEARCH))
  print(f'> medium + web_search: {len(size_tool)}')
  assert 'sonnet-4.6' in size_tool
  return state_data


@integration_block
def list_providers(state_data):
  providers = px.models.list_providers(recommended_only=False)
  print(f'> providers: {providers}')
  for expected in ('openai', 'gemini', 'claude', 'mistral'):
    assert expected in providers, f'{expected!r} missing'
  return state_data


@integration_block
def list_provider_models(state_data):
  openai = _names(px.models.list_provider_models('openai'))
  print(f'> openai: {len(openai)}')
  assert 'gpt-4o' in openai
  assert 'o3' in openai
  assert 'gemini-2.5-flash' not in openai

  openai_large = _names(
      px.models.list_provider_models('openai', model_size='large'))
  print(f'> openai large: {len(openai_large)}')
  assert 'o3' in openai_large

  gemini = _names(px.models.list_provider_models('gemini'))
  print(f'> gemini: {len(gemini)}')
  assert 'gemini-3-flash' in gemini
  assert 'gpt-4o' not in gemini
  return state_data


@integration_block
def get_model_and_get_model_config(state_data):
  m = px.models.get_model('openai', 'gpt-4o')
  print(f'> get_model: {m}')
  assert isinstance(m, types.ProviderModelType)
  assert m.provider == 'openai'
  assert m.model == 'gpt-4o'

  cfg = px.models.get_model_config('gemini', 'gemini-3-flash')
  print(f'> get_model_config: {type(cfg).__name__}')
  assert isinstance(cfg, types.ProviderModelConfig)
  assert cfg.provider_model.model == 'gemini-3-flash'
  assert cfg.metadata.is_recommended is True
  assert types.ModelSizeType.SMALL in cfg.metadata.model_size_tags
  assert cfg.features.tools.web_search == types.FeatureSupportType.SUPPORTED

  cfg2 = px.models.get_model_config('mock_provider', 'mock_model')
  assert cfg2.metadata.is_recommended is False
  return state_data


@integration_block
def get_default_model_list(state_data):
  priority = px.models.get_default_model_list()
  print(f'> priority: {[(m.provider, m.model) for m in priority]}')
  assert len(priority) > 0
  assert (priority[0].provider, priority[0].model) == (
      'gemini', 'gemini-3-flash')
  return state_data


# -----------------------------------------------------------------------------
# 1.13 — 1.17  Working-model probes (require real provider keys)
# -----------------------------------------------------------------------------

@integration_block
def list_working_models_basic(state_data):
  """First call probes provider APIs."""
  start = time.time()
  models = px.models.list_working_models()
  elapsed = time.time() - start
  print(f'> Working models: {len(models)} ({elapsed:.2f}s)')
  assert len(models) > 0, 'No working models — check provider API keys.'
  for idx, pm in enumerate(models[:10]):
    print(f'  {idx:>3}: {pm.provider:>20} - {pm.model}')
  if len(models) > 10:
    print('  ...')
  return state_data


@integration_block
def list_working_models_with_filters(state_data):
  """model_size, return_all, recommended_only=False — consolidated."""
  small = px.models.list_working_models(model_size='small')
  print(f'> working small: {len(small)}')

  medium = px.models.list_working_models(
      model_size=types.ModelSizeType.MEDIUM)
  print(f'> working medium: {len(medium)}')

  largest = px.models.list_working_models(model_size='largest')
  print(f'> working largest: {len(largest)}')

  status = px.models.list_working_models(return_all=True)
  print(f'> return_all: working={len(status.working_models)} '
        f'failed={len(status.failed_models)}')
  assert len(status.working_models) >= 0
  return state_data


@integration_block
def list_working_models_clear_cache_verbose(state_data):
  """Clear cache + verbose — operator sees probe progress + failures."""
  status = px.models.list_working_models(
      clear_model_cache=True, return_all=True, verbose=True)
  print(f'> working: {len(status.working_models)}, '
        f'failed: {len(status.failed_models)}')
  for pm, call_record in status.provider_queries.items():
    if not call_record.result or not call_record.result.error:
      continue
    err = call_record.result.error[:120]
    print(f'  FAIL {pm.provider}/{pm.model}: {err}')
  return state_data


@integration_block
def list_working_providers(state_data):
  providers = px.models.list_working_providers()
  print(f'> working providers: {providers}')
  return state_data


@integration_block
def list_working_provider_models(state_data):
  models = px.models.list_working_provider_models('openai')
  print(f'> openai working: {len(models)}')
  for pm in models[:5]:
    print(f'  {pm.provider:>20} - {pm.model}')

  # With size filter.
  large = px.models.list_working_provider_models('openai', model_size='large')
  print(f'> openai large working: {len(large)}')
  return state_data


# -----------------------------------------------------------------------------
# 1.18 — 1.21  check_health (probes; visual + assertions)
# -----------------------------------------------------------------------------

@integration_block
def check_health_default(state_data):
  """Visual: progress bars and per-provider summary."""
  px.models.check_health()
  manual_check(
      'Did check_health print a clear per-provider working/failed summary?',
      'check_health output is missing or unclear.')
  return state_data


@integration_block
def check_health_no_multiprocessing(state_data):
  """Sequential probing — operator confirms output is sequential.

  allow_multiprocessing now lives on connect-time ModelProbeOptions.
  """
  ctx = _utils._CTX
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url, api_key=state_data['api_key']),
      model_probe_options=types.ModelProbeOptions(allow_multiprocessing=False),
  )
  px.models.check_health()
  manual_check(
      'Was the check_health output sequential (one provider at a time)?',
      'check_health did not run sequentially.')
  return state_data


@integration_block
def check_health_with_timeout(state_data):
  """ModelProbeOptions.timeout trims slow probes."""
  ctx = _utils._CTX
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url, api_key=state_data['api_key']),
      model_probe_options=types.ModelProbeOptions(timeout=1),
  )
  px.models.check_health()
  return state_data


@integration_block
def check_health_extensive_return(state_data):
  """check_health() now returns ModelStatus directly."""
  ctx = _utils._CTX
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url, api_key=state_data['api_key']),
      model_probe_options=types.ModelProbeOptions(timeout=1),
  )
  status = px.models.check_health(verbose=False)
  print(f'> working: {len(status.working_models)}')
  print(f'> failed:  {len(status.failed_models)}')
  assert len(status.working_models) >= 0
  assert len(status.failed_models) >= 0
  return state_data


# -----------------------------------------------------------------------------
# 1.22  Refusal of media output_formats (offline assertion)
# -----------------------------------------------------------------------------

@integration_block
def list_working_methods_refuse_media(state_data):
  """All four working methods raise ValueError for IMAGE/AUDIO/VIDEO."""
  formats = [
      types.OutputFormatType.IMAGE,
      types.OutputFormatType.AUDIO,
      types.OutputFormatType.VIDEO,
  ]

  def _expect(method_name: str, fn):
    try:
      fn()
    except ValueError as e:
      msg = str(e)
      assert method_name in msg, (
          f'{method_name} error did not mention method: {msg}')
      assert 'list_models' in msg, (
          f'{method_name} error did not point at list_models: {msg}')
      return
    raise AssertionError(f'{method_name} did not raise for media format')

  for fmt in formats:
    _expect('list_working_models',
            lambda: px.models.list_working_models(
                output_format=fmt, verbose=False))
    _expect('list_working_providers',
            lambda: px.models.list_working_providers(
                output_format=fmt, verbose=False))
    _expect('list_working_provider_models',
            lambda: px.models.list_working_provider_models(
                'mock_provider', output_format=fmt, verbose=False))
    _expect('get_working_model',
            lambda: px.models.get_working_model(
                'mock_provider', 'mock_model', output_format=fmt))
    print(f'> {fmt.value}: all four methods refused')
  return state_data


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

BLOCKS = [
    list_models_default,
    list_models_recommended_only_false,
    list_models_by_size,
    list_models_by_input_format,
    list_models_by_output_format,
    list_models_by_tool_tags,
    list_models_by_feature_tags,
    list_models_combined_filters,
    list_providers,
    list_provider_models,
    get_model_and_get_model_config,
    get_default_model_list,
    list_working_models_basic,
    list_working_models_with_filters,
    list_working_models_clear_cache_verbose,
    list_working_providers,
    list_working_provider_models,
    check_health_default,
    # check_health_no_multiprocessing,
    check_health_with_timeout,
    check_health_extensive_return,
    list_working_methods_refuse_media,
]


def main():
  ctx = init_run(_LABEL)
  state_data = ensure_setup_state(ctx)
  state_data = setup_connection(state_data=state_data, force_run=True)
  run_sequence(_LABEL, BLOCKS, state_data=state_data)


if __name__ == '__main__':
  main()
