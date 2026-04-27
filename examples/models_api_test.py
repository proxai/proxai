"""Systematic end-to-end test for the px.models.* list / get APIs.

Builds a registry that mirrors proxdash_test.py / alias_test.py but tags every
model with an explicit model_size + is_recommended so the size and recommended
filters actually have something to match. Runs plain `assert`s against each
px.models.* endpoint; no try/except — first failure stops the run.

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
# Expected registry. The test assertions derive from this.
# -----------------------------------------------------------------------------

# Every entry here is registered via register_models(). Tests assert on the
# subsets that follow ("small" size means it appears in model_size_tags=small,
# etc.).
#
# Fields:
#   sizes           : list of 'small'/'medium'/'large'/'largest', or None for
#                     models with no size tags (image/audio/video-only). In the
#                     real world the valid combinations are ['small'],
#                     ['medium'], ['large'], or ['large', 'largest'] — the
#                     biggest models carry both 'large' and 'largest' so they
#                     show up under both filters.
#   is_recommended  : bool
#   web_search      : whether tools.web_search is SUPPORTED
#   thinking        : whether parameters.thinking is SUPPORTED
#   input_format    : list of 'text'/'image'/'document'/'audio'/'video'/'json'/
#                     'pydantic' — each maps to SUPPORTED on input_format.X
#   output_format   : list of 'text'/'image'/'audio'/'video'/'json'/'pydantic'
#                     — each maps to SUPPORTED on output_format.X

_MODEL_CATALOG = [
    # --- Text-generating mock models (small) ---
    dict(provider='mock_provider', model='mock_model', sizes=['small'],
         is_recommended=True,
         input_format=['text', 'image', 'document', 'json', 'pydantic'],
         output_format=['text', 'json', 'pydantic']),
    dict(provider='mock_failing_provider', model='mock_failing_model',
         sizes=['small'], is_recommended=True,
         input_format=['text'],
         output_format=['text', 'json', 'pydantic']),

    # --- OpenAI ---
    dict(provider='openai', model='gpt-4o', sizes=['small'],
         is_recommended=True, web_search=True,
         input_format=['text', 'image', 'document'],
         output_format=['text', 'json', 'pydantic']),
    dict(provider='openai', model='o3', sizes=['medium'],
         is_recommended=True, thinking=True,
         input_format=['text', 'image', 'document'],
         output_format=['text', 'json', 'pydantic']),
    dict(provider='openai', model='dall-e-3', sizes=None, is_recommended=True,
         output_format=['image']),
    dict(provider='openai', model='tts-1', sizes=None, is_recommended=True,
         output_format=['audio']),
    dict(provider='openai', model='sora-2', sizes=None, is_recommended=True,
         output_format=['video']),

    # --- Gemini ---
    dict(provider='gemini', model='gemini-3-flash-preview', sizes=['medium'],
         is_recommended=True, web_search=True,
         input_format=['text', 'image', 'document', 'audio', 'video'],
         output_format=['text', 'json', 'pydantic']),
    dict(provider='gemini', model='gemini-2.5-flash', sizes=['small'],
         is_recommended=True,
         input_format=['text', 'image', 'document', 'audio', 'video'],
         output_format=['text', 'json', 'pydantic']),
    dict(provider='gemini', model='gemini-2.5-flash-image', sizes=None,
         is_recommended=True, output_format=['image']),
    dict(provider='gemini', model='gemini-2.5-flash-preview-tts', sizes=None,
         is_recommended=False, output_format=['audio']),
    dict(provider='gemini', model='veo-3.1-generate-preview', sizes=None,
         is_recommended=True, output_format=['video']),

    # --- Claude ---
    dict(provider='claude', model='claude-sonnet-4-6', sizes=['large'],
         is_recommended=True, web_search=True,
         input_format=['text', 'image', 'document'],
         output_format=['text', 'json', 'pydantic']),
    dict(provider='claude', model='claude-opus-4-6',
         sizes=['large', 'largest'],
         is_recommended=True, thinking=True,
         input_format=['text', 'image', 'document'],
         output_format=['text', 'json', 'pydantic']),

    # --- DeepSeek (non-recommended) ---
    dict(provider='deepseek', model='deepseek-chat', sizes=['medium'],
         is_recommended=False,
         input_format=['text'],
         output_format=['text', 'json', 'pydantic']),
    dict(provider='deepseek', model='deepseek-reasoner', sizes=['large'],
         is_recommended=False, thinking=True,
         input_format=['text'],
         output_format=['text', 'json', 'pydantic']),

    # --- Grok (non-recommended) ---
    dict(provider='grok', model='grok-3', sizes=['medium'],
         is_recommended=False,
         input_format=['text', 'document'],
         output_format=['text', 'json', 'pydantic']),
]


_DEFAULT_PRIORITY_LIST = [
    ('gemini', 'gemini-3-flash-preview'),
    ('openai', 'gpt-4o'),
    ('claude', 'claude-sonnet-4-6'),
]


# -----------------------------------------------------------------------------
# Registry construction
# -----------------------------------------------------------------------------

def _get_model_config(
    provider: str,
    model: str,
    sizes: list[str] | None,
    is_recommended: bool,
    web_search: bool = False,
    thinking: bool = False,
    input_format: list[str] | None = None,
    output_format: list[str] | None = None,
) -> types.ProviderModelConfig:
  """Build a ProviderModelConfig that populates every filterable field."""
  S = types.FeatureSupportType.SUPPORTED
  NS = types.FeatureSupportType.NOT_SUPPORTED

  input_format = input_format or ['text']
  output_format = output_format or ['text']

  model_size_tags = None
  if sizes is not None:
    model_size_tags = [types.ModelSizeType(s) for s in sizes]

  return types.ProviderModelConfig(
      provider_model=types.ProviderModelType(
          provider=provider, model=model, provider_model_identifier=model
      ),
      pricing=types.ProviderModelPricingType(
          input_token_cost=1,
          output_token_cost=2,
      ),
      metadata=types.ProviderModelMetadataType(
          is_recommended=is_recommended,
          model_size_tags=model_size_tags,
      ),
      features=types.FeatureConfigType(
          prompt=S, messages=S, system_prompt=S,
          parameters=types.ParameterConfigType(
              temperature=S, max_tokens=S, stop=S, n=NS,
              thinking=S if thinking else NS,
          ),
          tools=types.ToolConfigType(
              web_search=S if web_search else NS,
          ),
          input_format=types.InputFormatConfigType(
              text=S if 'text' in input_format else NS,
              image=S if 'image' in input_format else NS,
              document=S if 'document' in input_format else NS,
              audio=S if 'audio' in input_format else NS,
              video=S if 'video' in input_format else NS,
              json=S if 'json' in input_format else NS,
              pydantic=S if 'pydantic' in input_format else NS,
          ),
          output_format=types.OutputFormatConfigType(
              text=S if 'text' in output_format else NS,
              json=S if 'json' in output_format else NS,
              pydantic=S if 'pydantic' in output_format else NS,
              image=S if 'image' in output_format else NS,
              audio=S if 'audio' in output_format else NS,
              video=S if 'video' in output_format else NS,
          ),
      ),
  )


def register_models(client: px.Client):
  client.models.model_config.unregister_all_models()
  for entry in _MODEL_CATALOG:
    client.models.model_config.register_provider_model_config(
        _get_model_config(**entry)
    )
  client.models.model_config.override_default_model_priority_list([
      px.models.get_model(provider, model)
      for provider, model in _DEFAULT_PRIORITY_LIST
  ])


# -----------------------------------------------------------------------------
# Assertion helpers
# -----------------------------------------------------------------------------

def _names(models) -> set[str]:
  return {m.model for m in models}


def _sorted_names(models) -> list[str]:
  return sorted(m.model for m in models)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_list_models_default():
  """list_models() with defaults = text output, recommended_only=True."""
  print('\n=== test_list_models_default ===')
  names = _names(px.models.list_models())
  print('  default list:', sorted(names))

  # Recommended text models are present.
  assert 'gpt-4o' in names
  assert 'o3' in names
  assert 'gemini-3-flash-preview' in names
  assert 'gemini-2.5-flash' in names
  assert 'claude-sonnet-4-6' in names
  assert 'claude-opus-4-6' in names

  # Non-recommended text models are excluded by default.
  assert 'deepseek-chat' not in names
  assert 'deepseek-reasoner' not in names
  assert 'grok-3' not in names

  # Non-text models are excluded by default output filter.
  assert 'dall-e-3' not in names
  assert 'tts-1' not in names
  assert 'sora-2' not in names


def test_list_models_recommended_only_false():
  """recommended_only=False brings in the is_recommended=False models."""
  print('\n=== test_list_models_recommended_only_false ===')
  names = _names(px.models.list_models(recommended_only=False))
  print('  full list:', sorted(names))

  assert 'deepseek-chat' in names
  assert 'deepseek-reasoner' in names
  assert 'grok-3' in names
  # Still recommended ones.
  assert 'gpt-4o' in names
  assert 'claude-opus-4-6' in names
  # Still non-text excluded by output filter.
  assert 'dall-e-3' not in names


def test_list_models_by_size():
  """model_size filter picks only models with that tag in metadata."""
  print('\n=== test_list_models_by_size ===')

  small = _names(
      px.models.list_models(model_size='small', recommended_only=False))
  print('  small:', sorted(small))
  assert 'gpt-4o' in small
  assert 'gemini-2.5-flash' in small
  assert 'mock_model' in small
  assert 'o3' not in small
  assert 'claude-opus-4-6' not in small

  medium = _names(
      px.models.list_models(model_size='medium', recommended_only=False))
  print('  medium:', sorted(medium))
  assert 'o3' in medium
  assert 'gemini-3-flash-preview' in medium
  assert 'deepseek-chat' in medium
  assert 'grok-3' in medium
  assert 'gpt-4o' not in medium
  assert 'claude-opus-4-6' not in medium

  # 'large' includes both 'large'-only models and those also tagged 'largest'.
  large = _names(
      px.models.list_models(model_size='large', recommended_only=False))
  print('  large:', sorted(large))
  assert 'claude-sonnet-4-6' in large
  assert 'deepseek-reasoner' in large
  assert 'claude-opus-4-6' in large      # ['large', 'largest']
  assert 'gpt-4o' not in large
  assert 'o3' not in large

  # 'largest' is a subset of 'large' — only models explicitly tagged 'largest'.
  largest = _names(
      px.models.list_models(model_size='largest', recommended_only=False))
  print('  largest:', sorted(largest))
  assert largest == {'claude-opus-4-6'}


def test_list_models_by_output_format():
  """output_format narrows to models whose output_format.X is SUPPORTED."""
  print('\n=== test_list_models_by_output_format ===')

  image = _names(
      px.models.list_models(output_format=types.OutputFormatType.IMAGE))
  print('  image:', sorted(image))
  assert 'dall-e-3' in image
  assert 'gemini-2.5-flash-image' in image
  assert 'gpt-4o' not in image

  audio = _names(
      px.models.list_models(output_format=types.OutputFormatType.AUDIO,
                            recommended_only=False))
  print('  audio:', sorted(audio))
  assert 'tts-1' in audio
  assert 'gemini-2.5-flash-preview-tts' in audio  # non-recommended
  assert 'gpt-4o' not in audio

  video = _names(
      px.models.list_models(output_format=types.OutputFormatType.VIDEO))
  print('  video:', sorted(video))
  assert 'sora-2' in video
  assert 'veo-3.1-generate-preview' in video
  assert 'gpt-4o' not in video

  json_out = _names(
      px.models.list_models(output_format=types.OutputFormatType.JSON))
  print('  json:', sorted(json_out))
  assert 'gpt-4o' in json_out
  assert 'claude-sonnet-4-6' in json_out
  assert 'dall-e-3' not in json_out


def test_list_models_by_input_format():
  """input_format narrows to models whose input_format.X is SUPPORTED."""
  print('\n=== test_list_models_by_input_format ===')

  image_in = _names(
      px.models.list_models(input_format=types.InputFormatType.IMAGE))
  print('  image:', sorted(image_in))
  assert 'gpt-4o' in image_in
  assert 'claude-sonnet-4-6' in image_in
  assert 'gemini-3-flash-preview' in image_in
  assert 'deepseek-chat' not in image_in
  assert 'mock_failing_model' not in image_in

  doc_in = _names(
      px.models.list_models(input_format=types.InputFormatType.DOCUMENT))
  print('  document:', sorted(doc_in))
  assert 'gpt-4o' in doc_in
  assert 'claude-opus-4-6' in doc_in
  assert 'deepseek-chat' not in doc_in

  audio_in = _names(
      px.models.list_models(input_format=types.InputFormatType.AUDIO))
  print('  audio:', sorted(audio_in))
  assert 'gemini-3-flash-preview' in audio_in
  assert 'gemini-2.5-flash' in audio_in
  assert 'gpt-4o' not in audio_in
  assert 'claude-opus-4-6' not in audio_in

  video_in = _names(
      px.models.list_models(input_format=types.InputFormatType.VIDEO))
  print('  video:', sorted(video_in))
  assert 'gemini-3-flash-preview' in video_in
  assert 'gemini-2.5-flash' in video_in
  assert 'gpt-4o' not in video_in


def test_list_models_by_tool_tags():
  """tool_tags=WEB_SEARCH matches models with tools.web_search=SUPPORTED."""
  print('\n=== test_list_models_by_tool_tags ===')
  ws = _names(px.models.list_models(tool_tags=types.ToolTag.WEB_SEARCH))
  print('  web_search:', sorted(ws))
  assert 'gpt-4o' in ws
  assert 'gemini-3-flash-preview' in ws
  assert 'claude-sonnet-4-6' in ws
  assert 'o3' not in ws
  assert 'claude-opus-4-6' not in ws


def test_list_models_by_feature_tags():
  """feature_tags=THINKING matches models with parameters.thinking=SUPPORTED."""
  print('\n=== test_list_models_by_feature_tags ===')
  thinking = _names(
      px.models.list_models(feature_tags=types.FeatureTag.THINKING,
                            recommended_only=False))
  print('  thinking:', sorted(thinking))
  assert 'o3' in thinking
  assert 'claude-opus-4-6' in thinking
  assert 'deepseek-reasoner' in thinking  # non-recommended
  assert 'gpt-4o' not in thinking
  assert 'claude-sonnet-4-6' not in thinking


def test_list_models_combined():
  """Combining filters intersects their effects."""
  print('\n=== test_list_models_combined ===')
  # Image input AND json output.
  combined = _names(px.models.list_models(
      input_format=types.InputFormatType.IMAGE,
      output_format=types.OutputFormatType.JSON,
  ))
  print('  image-in + json-out:', sorted(combined))
  assert 'gpt-4o' in combined
  assert 'gemini-3-flash-preview' in combined
  assert 'claude-sonnet-4-6' in combined
  assert 'deepseek-chat' not in combined   # no image input
  assert 'dall-e-3' not in combined        # no json output

  # medium size AND web_search.
  size_and_tool = _names(px.models.list_models(
      model_size='medium',
      tool_tags=types.ToolTag.WEB_SEARCH,
  ))
  print('  medium + web_search:', sorted(size_and_tool))
  assert 'gemini-3-flash-preview' in size_and_tool
  assert 'o3' not in size_and_tool       # medium but no web_search
  assert 'gpt-4o' not in size_and_tool   # has web_search but small


def test_list_providers():
  """list_providers() returns unique providers for the active filter set."""
  print('\n=== test_list_providers ===')
  providers = px.models.list_providers(recommended_only=False)
  print('  providers:', providers)
  for expected in ('openai', 'gemini', 'claude', 'deepseek', 'grok'):
    assert expected in providers, f'{expected!r} missing from {providers!r}'


def test_list_provider_models():
  """list_provider_models() restricts to one provider."""
  print('\n=== test_list_provider_models ===')
  openai_models = _names(
      px.models.list_provider_models('openai', recommended_only=False))
  print('  openai:', sorted(openai_models))
  assert 'gpt-4o' in openai_models
  assert 'o3' in openai_models
  assert 'gemini-2.5-flash' not in openai_models
  assert 'claude-sonnet-4-6' not in openai_models

  # With size filter.
  openai_small = _names(
      px.models.list_provider_models('openai', model_size='small',
                                     recommended_only=False))
  print('  openai small:', sorted(openai_small))
  assert 'gpt-4o' in openai_small
  assert 'o3' not in openai_small  # o3 is medium

  gemini_models = _names(
      px.models.list_provider_models('gemini', recommended_only=False))
  print('  gemini:', sorted(gemini_models))
  assert 'gemini-3-flash-preview' in gemini_models
  assert 'gemini-2.5-flash' in gemini_models
  assert 'gpt-4o' not in gemini_models


def test_get_model():
  """get_model() returns a ProviderModelType for a registered model."""
  print('\n=== test_get_model ===')
  m = px.models.get_model('openai', 'gpt-4o')
  print('  get_model(openai, gpt-4o) =', m)
  assert isinstance(m, types.ProviderModelType)
  assert m.provider == 'openai'
  assert m.model == 'gpt-4o'
  assert m.provider_model_identifier == 'gpt-4o'


def test_get_model_config():
  """get_model_config() returns the full config including features."""
  print('\n=== test_get_model_config ===')
  cfg = px.models.get_model_config('gemini', 'gemini-3-flash-preview')
  print('  config type:', type(cfg).__name__)
  assert isinstance(cfg, types.ProviderModelConfig)
  assert cfg.provider_model.model == 'gemini-3-flash-preview'
  assert cfg.metadata.is_recommended is True
  assert types.ModelSizeType.MEDIUM in cfg.metadata.model_size_tags
  # Features.
  assert cfg.features.input_format.image == types.FeatureSupportType.SUPPORTED
  assert cfg.features.input_format.audio == types.FeatureSupportType.SUPPORTED
  assert cfg.features.tools.web_search == types.FeatureSupportType.SUPPORTED

  # Non-recommended has is_recommended=False.
  cfg2 = px.models.get_model_config('deepseek', 'deepseek-chat')
  assert cfg2.metadata.is_recommended is False


def test_get_default_model_list():
  """get_default_model_list() returns the priority list we set."""
  print('\n=== test_get_default_model_list ===')
  priority = px.models.get_default_model_list()
  print('  priority:', [(m.provider, m.model) for m in priority])
  assert len(priority) == len(_DEFAULT_PRIORITY_LIST)
  for expected, actual in zip(_DEFAULT_PRIORITY_LIST, priority):
    assert (actual.provider, actual.model) == expected


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

  mock_provider's model-level feature config:
    input_format  = {text, image, document, json, pydantic}   (no audio/video)
    feature_tags  = {prompt, messages, system_prompt}          (no thinking)
    tool_tags     = {}                                         (no web_search)

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
  """JSON / PYDANTIC output_formats now probe successfully.

  Before: only TEXT was routed through the probe; anything else crashed
  with 'Output format type not supported'. The dispatch table now picks
  the matching _test_generate_<format> per the requested output_format,
  and the probe's CallRecord carries that output_format through.

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

  register_models(px.get_default_proxai_client())

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
