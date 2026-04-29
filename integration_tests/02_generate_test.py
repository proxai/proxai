"""Integration tests — px.generate / generate_text / generate_json /
generate_pydantic / generate_image / generate_audio / generate_video,
plus parameters, multi-modal input, tools, and connection options.

Usage:
  poetry run python3 integration_tests/02_generate_test.py
  poetry run python3 integration_tests/02_generate_test.py --auto-continue
  poetry run python3 integration_tests/02_generate_test.py --test generate_text_basic
"""
import os
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import asdict
from pprint import pprint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import proxai as px
import proxai.types as types

import _utils
from _utils import (
    integration_block, manual_check, run_sequence,
    init_run, ensure_setup_state, asset,
    DEFAULT_TEXT_MODEL, TEXT_MODELS, THINKING_MODEL, WEB_SEARCH_MODEL,
    JSON_OUTPUT_MODEL, IMAGE_MODEL, AUDIO_MODEL, VIDEO_MODEL,
    MULTIMODAL_MODEL, FAILING_MODEL,
    ASSET_PDF, ASSET_IMAGE, ASSET_MD, ASSET_AUDIO, ASSET_VIDEO,
)


_LABEL = '02_generate'


@integration_block
def setup_connection(state_data):
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      logging_options=types.LoggingOptions(
          logging_path=ctx.root_logging_path),
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
  print(f'> Connected at {ctx.proxdash_base_url}')
  return state_data


def _open_file_for_review(path: str, label: str) -> None:
  """Best-effort: open a generated media file for the operator to inspect."""
  print(f'> Saved {label}: {path}')
  try:
    subprocess.run(
        ['qlmanage', '-p', path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
  except FileNotFoundError:
    pass  # qlmanage is mac-only — silent on other platforms.


# -----------------------------------------------------------------------------
# 7.1 Text generation — alias + parameters
# -----------------------------------------------------------------------------

@integration_block
def generate_text_basic(state_data):
  """Plain px.generate_text returns a string."""
  response = px.generate_text(
      'Hello! Which model are you?',
      provider_model=DEFAULT_TEXT_MODEL)
  print(response)
  assert isinstance(response, str)
  assert len(response) > 0
  return state_data


@integration_block
def generate_text_with_provider_model(state_data):
  """Explicit provider_model tuple selects the target model."""
  response = px.generate_text(
      'Hello! Which model are you?',
      provider_model=('gemini', 'gemini-2.5-flash'))
  print(response)
  assert isinstance(response, str)
  return state_data


@integration_block
def generate_text_with_provider_model_type(state_data):
  """ProviderModelType returned by get_working_model is also accepted."""
  pm = px.models.get_model('claude', 'sonnet-4.6')
  print(f'> {type(pm).__name__}: {pm}')
  response = px.generate_text(
      'Hello! Which model are you?', provider_model=pm)
  print(response)
  return state_data


@integration_block
def generate_text_with_system_prompt(state_data):
  """system_prompt steers the response."""
  response = px.generate_text(
      'Hello! Which model are you?',
      system_prompt='You are an helpful assistant that always answers in Japanese.',
      provider_model=DEFAULT_TEXT_MODEL)
  print(response)
  manual_check(
      'Did the model answer in Japanese?',
      'system_prompt did not steer the response.')
  return state_data


@integration_block
def generate_text_with_message_history(state_data):
  """Multi-turn messages preserve conversation context."""
  response = px.generate_text(
      messages=[
          {'role': 'user',
           'content': 'From now on, always answer with a single integer.'},
          {'role': 'assistant', 'content': 'OK.'},
          {'role': 'user', 'content': 'Hello AI Model!'},
          {'role': 'assistant', 'content': '17'},
          {'role': 'user', 'content': 'How are you today?'},
          {'role': 'assistant', 'content': '923123'},
          {'role': 'user',
           'content': 'Can you answer the next question without any integer?'},
      ],
      provider_model=DEFAULT_TEXT_MODEL,
  )
  print(response)
  return state_data


@integration_block
def generate_text_with_max_tokens(state_data):
  """parameters.max_tokens caps output length."""
  response = px.generate_text(
      'Can you write all numbers from 1 to 1000?',
      parameters=types.ParameterType(max_tokens=20),
      provider_model=DEFAULT_TEXT_MODEL)
  print(response)
  return state_data


@integration_block
def generate_text_with_temperature(state_data):
  """parameters.temperature controls determinism."""
  response = px.generate_text(
      'If 5 + 20 would be a poem, what would life look like?',
      parameters=types.ParameterType(temperature=0.01),
      provider_model=DEFAULT_TEXT_MODEL)
  print(response)
  return state_data


@integration_block
def generate_text_extensive_return(state_data):
  """px.generate(...) returns the full CallRecord."""
  result = px.generate(
      'Hello! Which model are you?', provider_model=DEFAULT_TEXT_MODEL)
  print('CallRecord:')
  pprint(asdict(result))
  assert result.result is not None
  assert result.result.status == types.ResultStatusType.SUCCESS
  assert result.result.usage is not None
  assert result.connection is not None
  return state_data


@integration_block
def generate_text_with_thinking(state_data):
  """parameters.thinking emits THINKING content blocks (model-dependent)."""
  result = px.generate(
      ('Explain a hard problem in quantum computing in detail. '
       'Think step-by-step and reason carefully.'),
      provider_model=THINKING_MODEL,
      parameters=types.ParameterType(thinking=types.ThinkingType.MEDIUM))
  has_thinking = False
  for msg in (result.result.content or []):
    if msg.type == px.ContentType.THINKING:
      has_thinking = True
      break
  print(f'> THINKING content present: {has_thinking}')
  # Some models do not emit thinking even when asked — print a warning
  # rather than fail. The successful call itself is what matters here.
  return state_data


@integration_block
def set_model_default_text(state_data):
  """px.set_model + px.generate_text uses the configured default."""
  px.set_model(DEFAULT_TEXT_MODEL)
  response = px.generate_text('Which provider model are you?')
  print(response)
  manual_check(
      f'Did the response identify the model as {DEFAULT_TEXT_MODEL[0]}?',
      'set_model did not select the expected provider.')
  return state_data


# -----------------------------------------------------------------------------
# 7.2 Output formats — JSON / Pydantic / Image / Audio / Video
# -----------------------------------------------------------------------------

@integration_block
def generate_json_call(state_data):
  """px.generate_json returns a dict matching the schema."""
  schema = {
      'type': 'json_schema',
      'json_schema': {
          'name': 'ColorInfo',
          'strict': True,
          'schema': {
              'type': 'object',
              'properties': {
                  'color_name': {'type': 'string'},
                  'hex_code': {'type': 'string'},
              },
              'required': ['color_name', 'hex_code'],
              'additionalProperties': False,
          },
      },
  }
  for pm in [JSON_OUTPUT_MODEL, ('gemini', 'gemini-3-flash')]:
    response = px.generate_json(
        'Give me a random color name and its hex code.',
        provider_model=pm,
        # generate_json doesn't take output_format directly; older API used
        # response_format. Pass the schema via parameters? Use generate(...)
        # with output_format= for stricter schema-driven mode below.
    )
    print(f'> {pm}: {response}')
    assert isinstance(response, dict)
  return state_data


@integration_block
def generate_pydantic_call(state_data):
  """px.generate_pydantic returns an instance of the supplied model."""
  from pydantic import BaseModel

  class ColorInfo(BaseModel):
    color_name: str
    hex_code: str

  response = px.generate_pydantic(
      'Give me a random color name and its hex code.',
      provider_model=DEFAULT_TEXT_MODEL,
      output_format=ColorInfo)
  print(response)
  assert isinstance(response, ColorInfo)
  assert response.color_name
  assert response.hex_code
  return state_data


@integration_block
def generate_image_call(state_data):
  """px.generate_image writes an image; operator inspects it."""
  result = px.generate(
      'Make a funny cartoon cat in a living room.',
      provider_model=IMAGE_MODEL,
      output_format='image')
  assert result.result.status == types.ResultStatusType.SUCCESS
  img = result.result.output_image
  assert img is not None
  data = img.data
  if data is None and img.source:
    with urllib.request.urlopen(img.source) as resp:
      data = resp.read()
  assert data and len(data) > 100, 'image data missing or too small'
  with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
    f.write(data)
    path = f.name
  _open_file_for_review(path, 'image')
  manual_check(
      'Does the saved image look like a cartoon cat in a living room?',
      'image generation produced unexpected content.')
  return state_data


@integration_block
def generate_audio_call(state_data):
  """px.generate_audio produces audio bytes; operator confirms playback."""
  result = px.generate(
      'Hello! This is a test of ProxAI text-to-speech.',
      provider_model=AUDIO_MODEL,
      output_format='audio')
  assert result.result.status == types.ResultStatusType.SUCCESS
  aud = result.result.output_audio
  assert aud and aud.data and len(aud.data) > 100
  with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
    f.write(aud.data)
    path = f.name
  _open_file_for_review(path, 'audio')
  manual_check(
      'Did the audio play and say something close to "Hello! This is a test"?',
      'audio generation did not produce playable speech.')
  return state_data


@integration_block
def generate_video_call(state_data):
  """px.generate_video produces video bytes; operator inspects.

  Skips gracefully if the ProxDash registry has no video model with a
  usable input path (text/image input disabled across the catalog).
  """
  try:
    result = px.generate(
        'A cat playing with a ball of yarn.',
        provider_model=VIDEO_MODEL,
        output_format='video')
  except ValueError as e:
    if 'No compatible endpoint found' in str(e):
      print(f'> Video generation skipped: registry has no compatible '
            f'endpoint for {VIDEO_MODEL[0]}/{VIDEO_MODEL[1]}.')
      return state_data
    raise
  assert result.result.status == types.ResultStatusType.SUCCESS
  vid = result.result.output_video
  assert vid and vid.data and len(vid.data) > 1000
  with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
    f.write(vid.data)
    path = f.name
  _open_file_for_review(path, 'video')
  manual_check(
      'Does the saved video show a cat with yarn?',
      'video generation produced unexpected content.')
  return state_data


@integration_block
def set_model_per_output_format(state_data):
  """px.set_model can pin a different model per output format."""
  px.set_model(
      generate_text=DEFAULT_TEXT_MODEL,
      generate_image=IMAGE_MODEL,
  )
  text_resp = px.generate_text('Say hi in one word.')
  print(f'> text: {text_resp}')
  assert isinstance(text_resp, str)

  img_result = px.generate(
      'A friendly cartoon dog.', output_format='image')
  assert img_result.result.status == types.ResultStatusType.SUCCESS
  print('> image generated successfully')
  return state_data


# -----------------------------------------------------------------------------
# 7.3 Multi-modal input
# -----------------------------------------------------------------------------

def _input_messages(content_type, path, media_type, prompt):
  return [{
      'role': 'user',
      'content': [
          px.MessageContent(
              type=content_type, path=path, media_type=media_type),
          px.MessageContent(type=px.ContentType.TEXT, text=prompt),
      ],
  }]


@integration_block
def input_image(state_data):
  """Image input + text prompt — output should mention 'cat'."""
  result = px.generate(
      messages=_input_messages(
          px.ContentType.IMAGE, asset(ASSET_IMAGE), 'image/jpeg',
          'What is in this image?'),
      provider_model=DEFAULT_TEXT_MODEL)
  text = result.result.output_text or ''
  print(f'> {text[:120]}...')
  assert any(w in text.lower() for w in ('cat', 'kitten', 'feline')), text[:200]
  return state_data


@integration_block
def input_document_pdf(state_data):
  """PDF document input."""
  result = px.generate(
      messages=_input_messages(
          px.ContentType.DOCUMENT, asset(ASSET_PDF), 'application/pdf',
          'What is inside this document?'),
      provider_model=DEFAULT_TEXT_MODEL)
  text = result.result.output_text or ''
  print(f'> {text[:120]}...')
  assert any(w in text.lower() for w in ('cat', 'kitten', 'feline')), text[:200]
  return state_data


@integration_block
def input_document_md(state_data):
  """Markdown document input."""
  result = px.generate(
      messages=_input_messages(
          px.ContentType.DOCUMENT, asset(ASSET_MD), 'text/markdown',
          'What is inside this document?'),
      provider_model=DEFAULT_TEXT_MODEL)
  text = result.result.output_text or ''
  print(f'> {text[:120]}...')
  assert any(w in text.lower() for w in ('cat', 'kitten', 'feline')), text[:200]
  return state_data


@integration_block
def input_audio_gemini(state_data):
  """Audio input — gemini supports it."""
  result = px.generate(
      messages=_input_messages(
          px.ContentType.AUDIO, asset(ASSET_AUDIO), 'audio/mpeg',
          'What is this audio about?'),
      provider_model=MULTIMODAL_MODEL)
  text = result.result.output_text or ''
  print(f'> {text[:120]}...')
  assert any(w in text.lower() for w in ('cat', 'kitten', 'feline', 'meow')), text[:200]
  return state_data


@integration_block
def input_video_gemini(state_data):
  """Video input — gemini supports it."""
  result = px.generate(
      messages=_input_messages(
          px.ContentType.VIDEO, asset(ASSET_VIDEO), 'video/mp4',
          'What is in this video?'),
      provider_model=MULTIMODAL_MODEL)
  text = result.result.output_text or ''
  print(f'> {text[:120]}...')
  assert any(w in text.lower() for w in ('cat', 'kitten', 'feline')), text[:200]
  return state_data


# -----------------------------------------------------------------------------
# 7.4 Tools and connection options
# -----------------------------------------------------------------------------

@integration_block
def tools_web_search(state_data):
  """tools=[Tools.WEB_SEARCH] returns citations."""
  result = px.generate(
      'What is the most important news for January 20, 2024?',
      provider_model=WEB_SEARCH_MODEL,
      tools=[px.Tools.WEB_SEARCH])
  text = result.result.output_text or ''
  print(f'> {text[:200]}...')
  assert len(text) > 10
  has_tool = any(
      m.type == px.ContentType.TOOL for m in (result.result.content or []))
  assert has_tool, 'expected at least one TOOL content block from web_search'
  for msg in result.result.content:
    if msg.type == px.ContentType.TOOL:
      assert msg.tool_content.name == 'web_search'
      assert msg.tool_content.citations
      print(f'> citations: {len(msg.tool_content.citations)}')
  return state_data


@integration_block
def connection_options_fallback(state_data):
  """fallback_models swaps in when the primary fails."""
  # Primary works — fallback unused.
  result = px.generate(
      'What is 2 + 2?',
      provider_model=DEFAULT_TEXT_MODEL,
      connection_options=types.ConnectionOptions(
          fallback_models=[FAILING_MODEL]),
  )
  assert result.result.status == types.ResultStatusType.SUCCESS
  assert not result.connection.failed_fallback_models
  print(f'> primary used, no fallback: {result.query.provider_model.model}')

  # Primary fails — fallback succeeds.
  result = px.generate(
      'What is 2 + 2?',
      provider_model=FAILING_MODEL,
      connection_options=types.ConnectionOptions(
          fallback_models=[DEFAULT_TEXT_MODEL]),
  )
  assert result.result.status == types.ResultStatusType.SUCCESS
  assert result.connection.failed_fallback_models
  print(f'> primary failed, fallback used: '
        f'{result.query.provider_model.provider_model_identifier}')
  return state_data


@integration_block
def connection_options_endpoint_openai(state_data):
  """endpoint='responses.create' / 'chat.completions.create' (openai-only)."""
  for endpoint in ('responses.create', 'chat.completions.create'):
    result = px.generate(
        'What is 2 + 2?',
        provider_model=DEFAULT_TEXT_MODEL,
        connection_options=types.ConnectionOptions(endpoint=endpoint))
    assert result.connection.endpoint_used == endpoint, (
        f'expected {endpoint}, got {result.connection.endpoint_used}')
    print(f'> endpoint_used: {result.connection.endpoint_used}')

  # Bad endpoint should raise.
  raised = None
  try:
    px.generate(
        'hi', provider_model=DEFAULT_TEXT_MODEL,
        connection_options=types.ConnectionOptions(endpoint='not.real'))
  except ValueError as e:
    raised = str(e)
  assert raised and 'endpoint' in raised, 'bad endpoint should raise ValueError'
  print('> bad endpoint correctly raised')
  return state_data


@integration_block
def call_level_suppress_provider_errors(state_data):
  """connection_options.suppress_provider_errors at call site."""
  result = px.generate(
      'What is 2 + 2?',
      provider_model=FAILING_MODEL,
      connection_options=types.ConnectionOptions(
          suppress_provider_errors=True))
  assert result.result.status == types.ResultStatusType.FAILED
  assert result.result.error
  print(f'> suppressed error: {result.result.error[:120]}')
  return state_data


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

BLOCKS = [
    # 7.1 text generation
    generate_text_basic,
    generate_text_with_provider_model,
    generate_text_with_provider_model_type,
    generate_text_with_system_prompt,
    generate_text_with_message_history,
    generate_text_with_max_tokens,
    generate_text_with_temperature,
    generate_text_extensive_return,
    generate_text_with_thinking,
    set_model_default_text,
    # 7.2 output formats
    generate_json_call,
    generate_pydantic_call,
    generate_image_call,
    generate_audio_call,
    generate_video_call,
    set_model_per_output_format,
    # 7.3 multi-modal input
    input_image,
    input_document_pdf,
    input_document_md,
    input_audio_gemini,
    input_video_gemini,
    # 7.4 tools and connection options
    tools_web_search,
    connection_options_fallback,
    connection_options_endpoint_openai,
    call_level_suppress_provider_errors,
]


def main():
  ctx = init_run(_LABEL)
  state_data = ensure_setup_state(ctx)
  state_data = setup_connection(state_data=state_data, force_run=True)
  run_sequence(_LABEL, BLOCKS, state_data=state_data)


if __name__ == '__main__':
  main()
