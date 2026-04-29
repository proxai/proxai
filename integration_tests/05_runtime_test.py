"""Integration tests — runtime configuration: connect, options, query
cache, logging, and error-handling.

These tests exercise px.connect / px.get_current_options / px.reset_state
and the four cross-cutting concerns: cache, logs, errors, multiprocessing.

Usage:
  poetry run python3 integration_tests/05_runtime_test.py
  poetry run python3 integration_tests/05_runtime_test.py --auto-continue
  poetry run python3 integration_tests/05_runtime_test.py --test query_cache_basic
"""
import json
import os
import random
import sys
from dataclasses import asdict
from pprint import pprint

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
    DEFAULT_TEXT_MODEL, FAILING_MODEL,
)


_LABEL = '05_runtime'


# -----------------------------------------------------------------------------
# 5.1 — 5.3  Connection lifecycle
# -----------------------------------------------------------------------------

@integration_block
def connect_empty(state_data):
  """px.connect() with no args sets default options."""
  px.reset_state()
  px.connect()
  options = px.get_current_options()
  print('> Current options after empty connect:')
  pprint(asdict(options))
  assert options.experiment_path is None
  assert options.logging_options is None or options.logging_options.logging_path is None
  assert options.cache_options is None
  assert options.provider_call_options.suppress_provider_errors is False
  assert (options.provider_call_options.feature_mapping_strategy ==
          types.FeatureMappingStrategy.BEST_EFFORT)
  return state_data


@integration_block
def connect_full_options(state_data):
  """px.connect() with all options round-trips through get_current_options."""
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
      provider_call_options=types.ProviderCallOptions(
          feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
          suppress_provider_errors=True,
      ),
      model_probe_options=types.ModelProbeOptions(allow_multiprocessing=False),
  )
  options = px.get_current_options()
  print('> Current options after full connect:')
  pprint(asdict(options))
  assert options.experiment_path == ctx.experiment_path
  assert options.cache_options.cache_path == ctx.root_cache_path
  assert options.provider_call_options.suppress_provider_errors is True
  assert (options.provider_call_options.feature_mapping_strategy ==
          types.FeatureMappingStrategy.STRICT)
  assert options.model_probe_options.allow_multiprocessing is False
  return state_data


@integration_block
def reset_state_clears_session(state_data):
  """px.reset_state() drops the in-process client config."""
  ctx = _utils._CTX
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  before = px.get_current_options()
  assert before.experiment_path == ctx.experiment_path

  px.reset_state()
  after = px.get_current_options()
  assert after.experiment_path is None, (
      f'reset_state did not clear experiment_path (got {after.experiment_path})')
  print('> reset_state correctly cleared experiment_path')
  return state_data


# -----------------------------------------------------------------------------
# 5.4 — 5.6  Local logging files
# -----------------------------------------------------------------------------

@integration_block
def logging_to_merged_log(state_data):
  """Connect writes connection messages to merged.log."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      logging_options=types.LoggingOptions(logging_path=ctx.root_logging_path),
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  log_dir = os.path.join(ctx.root_logging_path, ctx.experiment_path)
  print(f'> log_dir contents: {os.listdir(log_dir)}')
  log_path = os.path.join(log_dir, 'merged.log')

  entries = []
  with open(log_path) as f:
    for line in f:
      entries.append(json.loads(line))
  assert len(entries) > 0, 'merged.log is empty'
  assert any(
      'Connected to ProxDash' in e.get('message', '') for e in entries), (
      f'No "Connected to ProxDash" entry in merged.log')
  assert any(
      'experiment' in e.get('message', '').lower() for e in entries), (
      'No experiment-path entry in merged.log')
  print(f'> merged.log has {len(entries)} entries')
  return state_data


@integration_block
def logging_with_stdout(state_data):
  """logging_options.stdout=True echoes connection messages."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      logging_options=types.LoggingOptions(
          logging_path=ctx.root_logging_path, stdout=True),
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  print('> Connection messages should have been printed to stdout above.')
  manual_check(
      'Did connection log records print to stdout?',
      'logging_options.stdout did not echo connect messages.')
  return state_data


# -----------------------------------------------------------------------------
# 5.7 — 5.11  Query cache
# -----------------------------------------------------------------------------

@integration_block
def query_cache_basic(state_data):
  """First call PROVIDER, subsequent calls CACHE."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      cache_options=types.CacheOptions(
          cache_path=ctx.root_cache_path,
          clear_query_cache_on_connect=True,
      ),
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  prompt = 'Hello model, what is 23 times 23?'

  result = px.generate(prompt, provider_model=DEFAULT_TEXT_MODEL)
  print(f'> result_source: {result.connection.result_source}')
  assert result.connection.result_source == types.ResultSource.PROVIDER

  result = px.generate(prompt, provider_model=DEFAULT_TEXT_MODEL)
  print(f'> result_source: {result.connection.result_source}')
  assert result.connection.result_source == types.ResultSource.CACHE

  result = px.generate(prompt, provider_model=DEFAULT_TEXT_MODEL)
  assert result.connection.result_source == types.ResultSource.CACHE
  return state_data


@integration_block
def query_cache_unique_response_limit(state_data):
  """unique_response_limit=3: first 3 PROVIDER, then CACHE for the rest."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      cache_options=types.CacheOptions(
          cache_path=ctx.root_cache_path,
          unique_response_limit=3,
          clear_query_cache_on_connect=True,
      ),
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  prompt = (
      'Pick 100 random positive integers under 3000 and explain why. '
      'Open with a short poem.'
  )
  for idx in range(6):
    result = px.generate(
        prompt, provider_model=DEFAULT_TEXT_MODEL,
        parameters=types.ParameterType(temperature=0.3),
    )
    src = result.connection.result_source
    print(f'> {idx}: {src}')
    if idx < 3:
      assert src == types.ResultSource.PROVIDER, (
          f'iteration {idx} expected PROVIDER got {src}')
    else:
      assert src == types.ResultSource.CACHE, (
          f'iteration {idx} expected CACHE got {src}')
  return state_data


@integration_block
def query_cache_skip_cache_per_call(state_data):
  """connection_options.skip_cache=True bypasses cache for that call."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      cache_options=types.CacheOptions(
          cache_path=ctx.root_cache_path,
          clear_query_cache_on_connect=True,
      ),
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  prompt = 'Hello model, what is 23 times 23?'

  r1 = px.generate(prompt, provider_model=DEFAULT_TEXT_MODEL)
  assert r1.connection.result_source == types.ResultSource.PROVIDER

  r2 = px.generate(prompt, provider_model=DEFAULT_TEXT_MODEL)
  assert r2.connection.result_source == types.ResultSource.CACHE

  r3 = px.generate(
      prompt, provider_model=DEFAULT_TEXT_MODEL,
      connection_options=types.ConnectionOptions(skip_cache=True))
  print(f'> skip_cache result_source: {r3.connection.result_source}')
  assert r3.connection.result_source == types.ResultSource.PROVIDER
  return state_data


@integration_block
def query_cache_override_cache_value(state_data):
  """connection_options.override_cache_value=True forces fresh provider call."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      cache_options=types.CacheOptions(
          cache_path=ctx.root_cache_path,
          clear_query_cache_on_connect=True,
      ),
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  prompt = 'Write a short poem about 23 + 7 in exactly 3 lines.'

  r1 = px.generate(
      prompt, provider_model=DEFAULT_TEXT_MODEL,
      parameters=types.ParameterType(temperature=0.7))
  assert r1.connection.result_source == types.ResultSource.PROVIDER
  text1 = r1.result.output_text

  r2 = px.generate(
      prompt, provider_model=DEFAULT_TEXT_MODEL,
      parameters=types.ParameterType(temperature=0.7),
      connection_options=types.ConnectionOptions(override_cache_value=True))
  print(f'> override_cache_value result_source: {r2.connection.result_source}')
  assert r2.connection.result_source == types.ResultSource.PROVIDER
  text2 = r2.result.output_text
  assert text1 != text2, 'override_cache_value should produce a fresh response'

  r3 = px.generate(
      prompt, provider_model=DEFAULT_TEXT_MODEL,
      parameters=types.ParameterType(temperature=0.7))
  assert r3.connection.result_source == types.ResultSource.CACHE
  return state_data


@integration_block
def query_cache_pydantic_response(state_data):
  """Cached pydantic responses round-trip equality."""
  from pydantic import BaseModel

  class ColorInfo(BaseModel):
    color_name: str
    hex_code: str

  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      cache_options=types.CacheOptions(
          cache_path=ctx.root_cache_path,
          clear_query_cache_on_connect=True,
      ),
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  prompt = 'Give me a random color name and its hex code.'

  r1 = px.generate(
      prompt, provider_model=DEFAULT_TEXT_MODEL, output_format=ColorInfo)
  print(f'> r1 source: {r1.connection.result_source}')
  assert r1.connection.result_source == types.ResultSource.PROVIDER
  v1 = r1.result.output_pydantic
  assert isinstance(v1, ColorInfo)

  r2 = px.generate(
      prompt, provider_model=DEFAULT_TEXT_MODEL, output_format=ColorInfo)
  print(f'> r2 source: {r2.connection.result_source}')
  assert r2.connection.result_source == types.ResultSource.CACHE
  v2 = r2.result.output_pydantic
  assert v2.color_name == v1.color_name
  assert v2.hex_code == v1.hex_code
  return state_data


# -----------------------------------------------------------------------------
# 5.12 — 5.13  Error handling at connect time
# -----------------------------------------------------------------------------

@integration_block
def connect_suppress_provider_errors(state_data):
  """provider_call_options.suppress_provider_errors=True returns the
  error in CallRecord rather than raising."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
      provider_call_options=types.ProviderCallOptions(
          suppress_provider_errors=True),
  )
  prompt = (
      f'If {random.randint(1, 1000)} + {random.randint(1, 1000)} would be a '
      'poem, what life be look like?'
  )
  result = px.generate(prompt, provider_model=FAILING_MODEL)
  print(f'> result.status: {result.result.status}')
  print(f'> result.error: {(result.result.error or "")[:120]}')
  assert result.result.status == types.ResultStatusType.FAILED
  assert result.result.error
  assert result.result.error_traceback
  return state_data


@integration_block
def connect_feature_mapping_strategy_strict(state_data):
  """STRICT mode raises when a feature is BEST_EFFORT or NOT_SUPPORTED."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
      provider_call_options=types.ProviderCallOptions(
          feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
      ),
  )
  raised = None
  try:
    px.generate_text(
        'Hello model',
        provider_model=('openai', 'o4-mini'),
        parameters=types.ParameterType(temperature=0.5))
  except Exception as e:
    raised = str(e)
    print(f'> raised: {raised[:200]}')
  assert raised is not None, 'STRICT mode should raise'
  return state_data


# -----------------------------------------------------------------------------
# 5.14  allow_multiprocessing
# -----------------------------------------------------------------------------

@integration_block
def connect_allow_multiprocessing_false(state_data):
  """Sequential probing — operator sees one provider at a time + warning."""
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
      model_probe_options=types.ModelProbeOptions(allow_multiprocessing=False),
  )
  px.models.list_working_models(verbose=True)
  manual_check(
      'Did a warning print about sequential testing, and did probes run one '
      'at a time?',
      'allow_multiprocessing=False did not produce expected warning / output.')
  return state_data


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

BLOCKS = [
    connect_empty,
    connect_full_options,
    reset_state_clears_session,
    logging_to_merged_log,
    logging_with_stdout,
    query_cache_basic,
    query_cache_unique_response_limit,
    query_cache_skip_cache_per_call,
    query_cache_override_cache_value,
    query_cache_pydantic_response,
    connect_suppress_provider_errors,
    connect_feature_mapping_strategy_strict,
    connect_allow_multiprocessing_false,
]


def main():
  ctx = init_run(_LABEL)
  state_data = ensure_setup_state(ctx)
  run_sequence(_LABEL, BLOCKS, state_data=state_data)


if __name__ == '__main__':
  main()
