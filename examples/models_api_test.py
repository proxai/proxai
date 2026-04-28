"""Systematic end-to-end test for the px.models.* list / get APIs.

Runs plain `assert`s against each px.models.* endpoint using the actual
v1.3.0 model registry; no try/except — first failure stops the run.

Usage:
  poetry run python3 examples/models_api_test.py
  poetry run python3 examples/models_api_test.py --test list_models_by_size
  poetry run python3 examples/models_api_test.py --test all
"""

import argparse
import os

import proxai as px
import proxai.types as types

# Provide dummy values for mock providers so they show up in list_models
# output (api_key_manager filters by os.environ).
os.environ.setdefault('MOCK_PROVIDER_API_KEY', 'dummy')
os.environ.setdefault('MOCK_FAILING_PROVIDER', 'dummy')
os.environ.setdefault('MOCK_SLOW_PROVIDER', 'dummy')


# -----------------------------------------------------------------------------
# Assertion helpers
# -----------------------------------------------------------------------------

def _names(models) -> set[str]:
  return {m.model for m in models}


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_list_models_default():
  """list_models() with defaults = text output, recommended_only=True."""
  print('\n=== test_list_models_default ===')
  names = _names(px.models.list_models())
  print('  default count:', len(names))

  # Recommended text models are present.
  assert 'gpt-4o' in names
  assert 'o3' in names
  assert 'gemini-3-flash' in names
  assert 'gemini-2.5-flash' in names
  assert 'sonnet-4.6' in names
  assert 'opus-4.6' in names
  assert 'haiku-4.5' in names

  # Non-text-output models are excluded.
  assert 'dall-e-3' not in names
  assert 'tts-1' not in names
  assert 'sora-2' not in names

  # Non-recommended models are excluded.
  assert 'mistral-small-latest' not in names
  assert 'mock_model' not in names


def test_list_models_recommended_only_false():
  """recommended_only=False brings in is_recommended=False models."""
  print('\n=== test_list_models_recommended_only_false ===')
  names = _names(px.models.list_models(recommended_only=False))
  print('  full count:', len(names))

  # Non-recommended text models become visible.
  assert 'mistral-small-latest' in names
  assert 'mock_model' in names

  # Recommended ones still present.
  assert 'gpt-4o' in names
  assert 'opus-4.6' in names

  # Non-text-output still excluded by output filter.
  assert 'dall-e-3' not in names


def test_list_models_by_size():
  """model_size filter picks only models with that tag in metadata."""
  print('\n=== test_list_models_by_size ===')

  small = _names(px.models.list_models(model_size='small'))
  print('  small count:', len(small))
  assert 'gemini-3-flash' in small
  assert 'gemini-2.5-flash' in small
  assert 'haiku-4.5' in small
  assert 'deepseek-v4-flash' in small
  assert 'o3' not in small         # large
  assert 'opus-4.6' not in small   # large
  assert 'sonnet-4.6' not in small # medium

  medium = _names(px.models.list_models(model_size='medium'))
  print('  medium count:', len(medium))
  assert 'sonnet-4.6' in medium
  assert 'gemini-3-flash' not in medium  # small
  assert 'opus-4.6' not in medium        # large

  large = _names(px.models.list_models(model_size='large'))
  print('  large count:', len(large))
  assert 'o3' in large
  assert 'opus-4.6' in large
  assert 'deepseek-v4-pro' in large  # ['large', 'largest']
  assert 'grok-3' in large
  assert 'sonnet-4.6' not in large   # medium

  # 'largest' is a subset of 'large'.
  largest = _names(px.models.list_models(model_size='largest'))
  print('  largest count:', len(largest))
  assert 'deepseek-v4-pro' in largest
  assert 'opus-4.6' not in largest  # only large, not largest


def test_list_models_by_output_format():
  """output_format narrows to models whose output_format.X is SUPPORTED."""
  print('\n=== test_list_models_by_output_format ===')

  # IMAGE / AUDIO / VIDEO models in the catalog are mostly non-recommended,
  # so use recommended_only=False to surface dall-e-3 / tts-1 / sora-2.
  image = _names(px.models.list_models(
      output_format=types.OutputFormatType.IMAGE, recommended_only=False))
  print('  image count:', len(image))
  assert 'dall-e-3' in image
  assert 'gemini-2.5-flash-image' in image
  assert 'gpt-4o' not in image

  audio = _names(px.models.list_models(
      output_format=types.OutputFormatType.AUDIO, recommended_only=False))
  print('  audio count:', len(audio))
  assert 'tts-1' in audio
  assert 'gemini-2.5-flash-tts' in audio
  assert 'gpt-4o' not in audio

  video = _names(px.models.list_models(
      output_format=types.OutputFormatType.VIDEO, recommended_only=False))
  print('  video count:', len(video))
  assert 'sora-2' in video
  assert 'veo-3.1-generate' in video
  assert 'gpt-4o' not in video

  json_out = _names(px.models.list_models(
      output_format=types.OutputFormatType.JSON))
  print('  json count:', len(json_out))
  assert 'gpt-4o' in json_out
  assert 'sonnet-4.6' in json_out  # BEST_EFFORT counts under default mapping
  assert 'dall-e-3' not in json_out


def test_list_models_by_input_format():
  """input_format narrows to models whose input_format.X is SUPPORTED."""
  print('\n=== test_list_models_by_input_format ===')

  image_in = _names(
      px.models.list_models(input_format=types.InputFormatType.IMAGE))
  print('  image count:', len(image_in))
  assert 'gpt-4o' in image_in
  assert 'sonnet-4.6' in image_in
  assert 'gemini-3-flash' in image_in
  assert 'deepseek-v4-flash' not in image_in  # text-only input

  doc_in = _names(
      px.models.list_models(input_format=types.InputFormatType.DOCUMENT))
  print('  document count:', len(doc_in))
  assert 'gpt-4o' in doc_in
  assert 'opus-4.6' in doc_in

  audio_in = _names(
      px.models.list_models(input_format=types.InputFormatType.AUDIO))
  print('  audio count:', len(audio_in))
  assert 'gemini-3-flash' in audio_in
  assert 'gemini-2.5-flash' in audio_in
  assert 'gpt-4o' not in audio_in
  assert 'opus-4.6' not in audio_in

  video_in = _names(
      px.models.list_models(input_format=types.InputFormatType.VIDEO))
  print('  video count:', len(video_in))
  assert 'gemini-3-flash' in video_in
  assert 'gemini-2.5-flash' in video_in
  assert 'gpt-4o' not in video_in


def test_list_models_by_tool_tags():
  """tool_tags=WEB_SEARCH matches models with tools.web_search=SUPPORTED."""
  print('\n=== test_list_models_by_tool_tags ===')
  ws = _names(px.models.list_models(tool_tags=types.ToolTag.WEB_SEARCH))
  print('  web_search count:', len(ws))
  assert 'gpt-4o' in ws
  assert 'gemini-3-flash' in ws
  assert 'sonnet-4.6' in ws
  assert 'o3' in ws
  assert 'deepseek-v4-flash' not in ws
  assert 'grok-3' not in ws


def test_list_models_by_feature_tags():
  """feature_tags=THINKING matches models with parameters.thinking=SUPPORTED."""
  print('\n=== test_list_models_by_feature_tags ===')
  thinking = _names(
      px.models.list_models(feature_tags=types.FeatureTag.THINKING))
  print('  thinking count:', len(thinking))
  assert 'o3' in thinking
  assert 'opus-4.6' in thinking
  assert 'sonnet-4.6' in thinking
  assert 'haiku-4.5' in thinking
  assert 'gpt-4o' not in thinking


def test_list_models_combined():
  """Combining filters intersects their effects."""
  print('\n=== test_list_models_combined ===')
  # Image input AND json output.
  combined = _names(px.models.list_models(
      input_format=types.InputFormatType.IMAGE,
      output_format=types.OutputFormatType.JSON,
  ))
  print('  image-in + json-out:', len(combined))
  assert 'gpt-4o' in combined
  assert 'gemini-3-flash' in combined
  assert 'sonnet-4.6' in combined
  assert 'deepseek-v4-flash' not in combined  # no image input
  assert 'dall-e-3' not in combined           # no json output

  # medium size AND web_search.
  size_and_tool = _names(px.models.list_models(
      model_size='medium',
      tool_tags=types.ToolTag.WEB_SEARCH,
  ))
  print('  medium + web_search:', len(size_and_tool))
  assert 'sonnet-4.6' in size_and_tool       # medium + web_search
  assert 'gemini-3-flash' not in size_and_tool  # has web_search but small


def test_list_providers():
  """list_providers() returns unique providers for the active filter set."""
  print('\n=== test_list_providers ===')
  providers = px.models.list_providers(recommended_only=False)
  print('  providers:', providers)
  for expected in ('openai', 'gemini', 'claude', 'deepseek', 'grok',
                   'mistral', 'cohere'):
    assert expected in providers, f'{expected!r} missing from {providers!r}'


def test_list_provider_models():
  """list_provider_models() restricts to one provider."""
  print('\n=== test_list_provider_models ===')
  openai = _names(px.models.list_provider_models('openai'))
  print('  openai count:', len(openai))
  assert 'gpt-4o' in openai
  assert 'o3' in openai
  assert 'gemini-2.5-flash' not in openai
  assert 'sonnet-4.6' not in openai

  # With size filter — o3 is large in the JSON.
  openai_large = _names(
      px.models.list_provider_models('openai', model_size='large'))
  print('  openai large count:', len(openai_large))
  assert 'o3' in openai_large

  gemini = _names(px.models.list_provider_models('gemini'))
  print('  gemini count:', len(gemini))
  assert 'gemini-3-flash' in gemini
  assert 'gemini-2.5-flash' in gemini
  assert 'gpt-4o' not in gemini


def test_get_model():
  """get_model() returns a ProviderModelType for a registered model."""
  print('\n=== test_get_model ===')
  m = px.models.get_model('openai', 'gpt-4o')
  print('  get_model(openai, gpt-4o) =', m)
  assert isinstance(m, types.ProviderModelType)
  assert m.provider == 'openai'
  assert m.model == 'gpt-4o'


def test_get_model_config():
  """get_model_config() returns the full config including features."""
  print('\n=== test_get_model_config ===')
  cfg = px.models.get_model_config('gemini', 'gemini-3-flash')
  print('  config type:', type(cfg).__name__)
  assert isinstance(cfg, types.ProviderModelConfig)
  assert cfg.provider_model.model == 'gemini-3-flash'
  assert cfg.metadata.is_recommended is True
  assert types.ModelSizeType.SMALL in cfg.metadata.model_size_tags
  # Features.
  assert cfg.features.input_format.image == types.FeatureSupportType.SUPPORTED
  assert cfg.features.input_format.audio == types.FeatureSupportType.SUPPORTED
  assert cfg.features.tools.web_search == types.FeatureSupportType.SUPPORTED

  # Non-recommended has is_recommended=False (mock_provider in JSON).
  cfg2 = px.models.get_model_config('mock_provider', 'mock_model')
  assert cfg2.metadata.is_recommended is False


def test_get_default_model_list():
  """get_default_model_list() returns the priority list from JSON."""
  print('\n=== test_get_default_model_list ===')
  priority = px.models.get_default_model_list()
  print('  priority:', [(m.provider, m.model) for m in priority])
  assert len(priority) > 0
  # JSON v1.3.0 priority leads with gemini/gemini-3-flash.
  assert (priority[0].provider, priority[0].model) == (
      'gemini', 'gemini-3-flash')


# -----------------------------------------------------------------------------
# Working-method tests (exercise the health-check harness)
# -----------------------------------------------------------------------------
# These probe each model via connector.generate(). mock_provider /
# mock_failing_provider answer without network I/O. Real providers
# (openai / claude / ...) only probe if their real API keys are in env —
# keeping the tests resilient to CI vs. dev setup, we only assert on
# the mocks.


def test_list_working_models_default():
  """Default probe: mock_model succeeds, mock_failing_model fails."""
  print('\n=== test_list_working_models_default ===')
  status = px.models.list_working_models(
      return_all=True, verbose=False, recommended_only=False,
  )
  working = {m.model for m in status.working_models}
  failed = {m.model for m in status.failed_models}
  print(f'  working (subset): mock_model={"mock_model" in working}')
  print(f'  failed  (subset): mock_failing_model='
        f'{"mock_failing_model" in failed}')
  assert 'mock_model' in working
  assert 'mock_failing_model' in failed


def test_list_working_methods_refuse_media_output_formats():
  """All four working methods raise for IMAGE / AUDIO / VIDEO."""
  print('\n=== test_list_working_methods_refuse_media_output_formats ===')
  media_formats = [
      types.OutputFormatType.IMAGE,
      types.OutputFormatType.AUDIO,
      types.OutputFormatType.VIDEO,
  ]

  def _expect_raises(method_name, fn):
    try:
      fn()
    except ValueError as e:
      msg = str(e)
      assert method_name in msg, (
          f'error message for {method_name} did not mention the method: {msg}')
      assert 'list_models' in msg, (
          f'error message for {method_name} did not point at list_models: {msg}'
      )
      return
    raise AssertionError(f'{method_name} did not raise for media format')

  for fmt in media_formats:
    _expect_raises(
        'list_working_models',
        lambda: px.models.list_working_models(output_format=fmt, verbose=False))
    _expect_raises(
        'list_working_providers',
        lambda: px.models.list_working_providers(
            output_format=fmt, verbose=False))
    _expect_raises(
        'list_working_provider_models',
        lambda: px.models.list_working_provider_models(
            'mock_provider', output_format=fmt, verbose=False))
    _expect_raises(
        'get_working_model',
        lambda: px.models.get_working_model(
            'mock_provider', 'mock_model', output_format=fmt))
    print(f'  {fmt.value}: all four methods refused')


def test_list_working_models_new_filters():
  """input_format / feature_tags / tool_tags pre-filter before probing.

  mock_provider's model-level feature config (per JSON v1.3.0):
    input_format  = {text}   (no image/document/audio/video/json/pydantic)
    feature_tags  = {prompt, messages, system_prompt}  (no thinking)
    tool_tags     = {}                                 (no web_search)

  Asking for any of those missing capabilities drops the model before the
  probe fires — it never reaches working_models.
  """
  print('\n=== test_list_working_models_new_filters ===')

  # feature_tags=THINKING — mock_provider has no thinking.
  thinking_status = px.models.list_working_models(
      feature_tags=types.FeatureTag.THINKING, return_all=True,
      verbose=False, recommended_only=False,
  )
  thinking_working = {m.model for m in thinking_status.working_models}
  print(f'  thinking working: mock_model in? '
        f'{"mock_model" in thinking_working}')
  assert 'mock_model' not in thinking_working

  # input_format=AUDIO — mock_provider has no audio input.
  audio_status = px.models.list_working_models(
      input_format=types.InputFormatType.AUDIO, return_all=True,
      verbose=False, recommended_only=False,
  )
  audio_working = {m.model for m in audio_status.working_models}
  print(f'  audio-input working: mock_model in? '
        f'{"mock_model" in audio_working}')
  assert 'mock_model' not in audio_working

  # tool_tags=WEB_SEARCH — mock_provider has no web search.
  ws_status = px.models.list_working_models(
      tool_tags=types.ToolTag.WEB_SEARCH, return_all=True,
      verbose=False, recommended_only=False,
  )
  ws_working = {m.model for m in ws_status.working_models}
  print(f'  web_search working: mock_model in? '
        f'{"mock_model" in ws_working}')
  assert 'mock_model' not in ws_working

  # Positive control: feature_tags=PROMPT is supported; mock_model stays.
  prompt_status = px.models.list_working_models(
      feature_tags=types.FeatureTag.PROMPT, return_all=True,
      verbose=False, recommended_only=False,
  )
  prompt_working = {m.model for m in prompt_status.working_models}
  assert 'mock_model' in prompt_working


def test_list_working_models_safe_output_formats():
  """JSON / PYDANTIC output_formats route through the probe.

  mock_provider declares JSON and PYDANTIC supported, so both probes
  land in working_models with the right output_format on the record.
  """
  print('\n=== test_list_working_models_safe_output_formats ===')
  for fmt in [types.OutputFormatType.JSON, types.OutputFormatType.PYDANTIC]:
    status = px.models.list_working_models(
        output_format=fmt, return_all=True,
        verbose=False, recommended_only=False,
    )
    working = {m.model for m in status.working_models}
    mock_recs = [
        rec for m, rec in status.provider_queries.items()
        if m.model == 'mock_model'
    ]
    assert 'mock_model' in working, f'{fmt}: mock_model not in working set'
    assert mock_recs, f'{fmt}: no provider_queries record for mock_model'
    rec = mock_recs[0]
    assert rec.query.output_format.type == fmt, (
        f'{fmt}: probe query recorded output_format='
        f'{rec.query.output_format.type}, expected {fmt}')
    print(f'  {fmt.value}: mock_model probed with correct output_format')


TEST_SEQUENCE = [
    ('list_models_default', test_list_models_default),
    ('list_models_recommended_only_false',
     test_list_models_recommended_only_false),
    ('list_models_by_size', test_list_models_by_size),
    ('list_models_by_output_format', test_list_models_by_output_format),
    ('list_models_by_input_format', test_list_models_by_input_format),
    ('list_models_by_tool_tags', test_list_models_by_tool_tags),
    ('list_models_by_feature_tags', test_list_models_by_feature_tags),
    ('list_models_combined', test_list_models_combined),
    ('list_providers', test_list_providers),
    ('list_provider_models', test_list_provider_models),
    ('get_model', test_get_model),
    ('get_model_config', test_get_model_config),
    ('get_default_model_list', test_get_default_model_list),
    ('list_working_models_default', test_list_working_models_default),
    ('list_working_methods_refuse_media_output_formats',
     test_list_working_methods_refuse_media_output_formats),
    ('list_working_models_new_filters',
     test_list_working_models_new_filters),
    ('list_working_models_safe_output_formats',
     test_list_working_models_safe_output_formats),
]
TEST_MAP = dict(TEST_SEQUENCE)


def main():
  parser = argparse.ArgumentParser(description='px.models.* API smoke test')
  test_names = [name for name, _ in TEST_SEQUENCE]
  parser.add_argument(
      '--test', default='all',
      help=f'Test to run: {", ".join(test_names)}, or "all"')
  args = parser.parse_args()

  if args.test == 'all':
    for name, fn in TEST_SEQUENCE:
      fn()
      print(f'  PASS [{name}]')
  else:
    if args.test not in TEST_MAP:
      print(f'Unknown test: {args.test}')
      print(f'Available: {", ".join(test_names)}')
      return
    TEST_MAP[args.test]()
    print(f'  PASS [{args.test}]')


if __name__ == '__main__':
  main()
