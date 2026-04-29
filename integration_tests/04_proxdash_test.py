"""Integration tests — ProxDash UI human verification.

This file is mostly manual_check. The operator should have the ProxDash
UI open at _WEBVIEW_BASE_URL and is expected to switch focus to the
browser repeatedly.

Run AFTER 02_generate_test.py and 03_files_test.py so ProxDash already
contains records to inspect.

Usage:
  poetry run python3 integration_tests/04_proxdash_test.py
  poetry run python3 integration_tests/04_proxdash_test.py --auto-continue
"""
import json
import os
import sys
from pprint import pprint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pydantic

import proxai as px
import proxai.types as types
import proxai.serializers.type_serializer as type_serializer

import _utils
from _utils import (
    integration_block, manual_check, manual_check_with_url, run_sequence,
    init_run, ensure_setup_state, asset,
    DEFAULT_TEXT_MODEL, MULTIMODAL_MODEL, IMAGE_MODEL, AUDIO_MODEL, VIDEO_MODEL,
    WORKING_MOCK, FAILING_MODEL,
    ASSET_PDF, ASSET_IMAGE, ASSET_MD, ASSET_AUDIO, ASSET_VIDEO,
)


_LABEL = '04_proxdash'


def _connect_basic(state_data, **proxdash_extra):
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      logging_options=types.LoggingOptions(
          logging_path=ctx.root_logging_path),
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
          **proxdash_extra,
      ),
  )


def _truncate(obj, max_len=80):
  if isinstance(obj, dict):
    return {k: _truncate(v, max_len) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_truncate(v, max_len) for v in obj]
  if isinstance(obj, str) and len(obj) > max_len:
    return obj[:max_len] + f'... ({len(obj)} chars)'
  return obj


def _print_call_record(result) -> None:
  data = type_serializer.encode_call_record(result)
  pprint(_truncate(data))


# -----------------------------------------------------------------------------
# 9.1  Connection / stdout / disable / local log
# -----------------------------------------------------------------------------

@integration_block
def proxdash_stdout_connection_message(state_data):
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          stdout=True,
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  print('> Expected stdout messages:')
  print(f'    * Connected to ProxDash at {ctx.proxdash_base_url}')
  print(f'    * Connected to ProxDash experiment: {ctx.experiment_path}')
  manual_check(
      'Were both connection messages printed by ProxDash stdout?',
      'ProxDash stdout did not print the expected connection messages.')
  return state_data


@integration_block
def proxdash_disable(state_data):
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          stdout=True,
          disable_proxdash=True,
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  print('> Expected stdout: "ProxDash connection disabled."')
  manual_check(
      'Did stdout print that ProxDash is disabled?',
      'ProxDash disable did not produce expected stdout.')
  return state_data


@integration_block
def proxdash_local_log_file(state_data):
  ctx = _utils._CTX
  log_path = os.path.join(
      ctx.root_logging_path, ctx.experiment_path, 'proxdash.log')
  if os.path.exists(log_path):
    os.remove(log_path)

  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      logging_options=types.LoggingOptions(
          logging_path=ctx.root_logging_path),
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )

  logs = []
  with open(log_path) as f:
    for line in f:
      logs.append(json.loads(line))
  assert any(
      l['message'].startswith(f'Connected to ProxDash at {ctx.proxdash_base_url}')
      for l in logs), 'connection-base log entry missing'
  assert any(
      l['message'].startswith(
          f'Connected to ProxDash experiment: {ctx.experiment_path}')
      for l in logs), 'connection-experiment log entry missing'
  print(f'> proxdash.log entries: {len(logs)}')
  for l in logs[:3]:
    print(f'  - {l["message"][:100]}')
  return state_data


# -----------------------------------------------------------------------------
# 9.2  Records visibility
# -----------------------------------------------------------------------------

@integration_block
def proxdash_logging_record_basic(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data)
  prompt = 'Logging record test. Hello model, integration test.'
  px.generate_text(prompt, provider_model=DEFAULT_TEXT_MODEL)

  url = f'{ctx.webview_base_url}/dashboard/logging'
  print(f'1 - Open: {url}')
  print(f'2 - Latest record prompt should start with:')
  print(f'    "{prompt}"')
  manual_check(
      'Is the latest record visible and prompt correct?',
      'logging record missing or incorrect.')
  print('3 - Click "Open" on the latest record.')
  manual_check(
      'Are details (prompt, response, provider, model, tokens) visible?',
      'single-record detail page broken.')
  return state_data


@integration_block
def proxdash_logging_record_full_options(state_data):
  """Verify ProxDash UI renders all parameter fields."""
  ctx = _utils._CTX
  _connect_basic(state_data)
  px.generate_text(
      messages=[
          {'role': 'user',     'content': 'Always answer with a single integer.'},
          {'role': 'assistant', 'content': 'OK.'},
          {'role': 'user',     'content': 'Hello AI Model!'},
          {'role': 'assistant', 'content': '17'},
          {'role': 'user',     'content': 'How are you today?'},
          {'role': 'assistant', 'content': '923123'},
          {'role': 'user',     'content': 'Answer without any integer.'},
      ],
      parameters=types.ParameterType(
          temperature=0.3, max_tokens=2000, stop=['STOP']),
      provider_model=DEFAULT_TEXT_MODEL,
  )
  url = f'{ctx.webview_base_url}/dashboard/logging'
  print(f'1 - Open: {url} and open the latest record.')
  print('2 - Verify these fields are correct:')
  print('    * Messages: 7 entries (user/assistant interleaved)')
  print('    * Temperature: 0.3')
  print('    * Max Tokens: 2000')
  print('    * Stop: ["STOP"]')
  print(f'    * Provider Model: {DEFAULT_TEXT_MODEL[0]}/{DEFAULT_TEXT_MODEL[1]}')
  manual_check(
      'Are all fields rendered correctly?',
      'some fields are missing or wrong on the detail page.')
  return state_data


# -----------------------------------------------------------------------------
# 9.3  Sensitive content
# -----------------------------------------------------------------------------

@integration_block
def proxdash_hide_sensitive_prompt(state_data):
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          hide_sensitive_content=True,
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  prompt = ('This record should appear on ProxDash but the prompt content '
            'and the response content should not appear.')
  px.generate_text(prompt, provider_model=DEFAULT_TEXT_MODEL)

  url = f'{ctx.webview_base_url}/dashboard/logging'
  print(f'1 - Open: {url}')
  print('2 - Latest record should show:')
  print('    * Prompt:   <sensitive content hidden>')
  print('    * Response: <sensitive content hidden>')
  manual_check(
      'Is the prompt/response content hidden in the list view?',
      'sensitive content not hidden in list.')
  print('3 - Open the record details.')
  manual_check(
      'Is the prompt/response content hidden in the detail view?',
      'sensitive content not hidden in detail.')
  return state_data


@integration_block
def proxdash_hide_sensitive_messages(state_data):
  ctx = _utils._CTX
  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          hide_sensitive_content=True,
          base_url=ctx.proxdash_base_url,
          api_key=state_data['api_key'],
      ),
  )
  px.generate_text(
      messages=[
          {'role': 'user', 'content': 'These contents should not appear.'},
          {'role': 'assistant', 'content': 'Ok, I will not show them.'},
          {'role': 'user', 'content': 'Be sure to hide them.'},
      ],
      provider_model=DEFAULT_TEXT_MODEL,
  )

  url = f'{ctx.webview_base_url}/dashboard/logging'
  print(f'1 - Open: {url}, open the latest record.')
  print('2 - Verify:')
  print('    * System:   <sensitive content hidden>')
  print('    * Messages: each content is <sensitive content hidden>')
  print('    * Response: <sensitive content hidden>')
  manual_check(
      'Are all message contents and system hidden?',
      'sensitive message contents leaked into the UI.')
  return state_data


@integration_block
def proxdash_limited_api_key(state_data):
  ctx = _utils._CTX
  # Allow automation by setting PROXAI_LIMITED_API_KEY in the env. When
  # unset, prompt the operator to create a limited key in the UI.
  limited = os.environ.get('PROXAI_LIMITED_API_KEY', '').strip()
  if not limited:
    print(f'1 - Open API Keys page: {ctx.webview_base_url}/dashboard/api-keys')
    print('2 - Click "+ Generate Key".')
    print('3 - Choose "Without prompt, messages, system instruction".')
    print('4 - Copy the key and paste below.')
    limited = input('> Limited API key: ').strip()
  state_data['limited_api_key'] = limited

  px.reset_state()
  px.connect(
      experiment_path=ctx.experiment_path,
      proxdash_options=types.ProxDashOptions(
          base_url=ctx.proxdash_base_url,
          api_key=state_data['limited_api_key'],
      ),
  )
  prompt = 'This record should appear without prompt or response content.'
  px.generate_text(prompt, provider_model=DEFAULT_TEXT_MODEL)

  url = f'{ctx.webview_base_url}/dashboard/logging'
  print(f'5 - Open: {url}')
  print('6 - Latest record should show:')
  print('    * Prompt:   <sensitive content hidden>')
  print('    * Response: <sensitive content hidden>')
  manual_check(
      'Are prompt/response hidden by the limited key?',
      'limited key did not strip sensitive content.')
  return state_data


# -----------------------------------------------------------------------------
# 9.4  Experiment path
# -----------------------------------------------------------------------------

@integration_block
def proxdash_experiment_path(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data)
  px.generate_text(
      'Experiment path test. Hello model.',
      provider_model=DEFAULT_TEXT_MODEL)

  url = f'{ctx.webview_base_url}/dashboard/experiments'
  print(f'1 - Open: {url}')
  print('2 - Click "Refresh".')
  print(f'3 - Verify experiment folder exists: {ctx.experiment_path}')
  manual_check(
      'Does the experiment folder show up?',
      'experiment folder missing in /dashboard/experiments.')
  print('4 - Open the experiment, click "Logging Records" tab.')
  print(f'5 - Latest record prompt: "Experiment path test. Hello model."')
  manual_check(
      'Latest logging record visible and correct?',
      'experiment-scoped logging record missing.')
  return state_data


# -----------------------------------------------------------------------------
# 9.5  Multi-modal records — content rendering
# -----------------------------------------------------------------------------

@integration_block
def proxdash_renders_text(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  result = px.generate(
      'Say hello in one word.', provider_model=DEFAULT_TEXT_MODEL)
  _print_call_record(result)
  manual_check_with_url(
      'Does the latest record render plain text input/output correctly?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'a single-text prompt and a short text response')
  return state_data


@integration_block
def proxdash_renders_json_input(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  result = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.JSON,
                  json={'title': 'love letter to cats',
                        'description': 'feelings as JSON.'}),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is this about?'),
          ],
      }],
      provider_model=DEFAULT_TEXT_MODEL)
  _print_call_record(result)
  manual_check_with_url(
      'Does the record show the JSON input pretty-printed?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'JSON content block before the text prompt')
  return state_data


@integration_block
def proxdash_renders_pydantic_input(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)

  class CatDescription(pydantic.BaseModel):
    name: str = 'Whiskers'
    color: str = 'orange'
    favorite_food: str = 'tuna'

  result = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.PYDANTIC_INSTANCE,
                  pydantic_content=types.PydanticContent(
                      instance_value=CatDescription())),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is this about?'),
          ],
      }],
      provider_model=DEFAULT_TEXT_MODEL)
  _print_call_record(result)
  manual_check_with_url(
      'Does the record show the Pydantic instance?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'a CatDescription pydantic block before the text prompt')
  return state_data


@integration_block
def proxdash_renders_image_input(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  result = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.IMAGE,
                  path=asset(ASSET_IMAGE), media_type='image/jpeg'),
              px.MessageContent(
                  type=px.ContentType.TEXT, text='What is in this image?'),
          ],
      }],
      provider_model=DEFAULT_TEXT_MODEL)
  _print_call_record(result)
  manual_check_with_url(
      'Does the record show the cat.jpeg image preview?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'a thumbnail of the uploaded cat image')
  return state_data


@integration_block
def proxdash_renders_audio_input(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  result = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.AUDIO,
                  path=asset(ASSET_AUDIO), media_type='audio/mpeg'),
              px.MessageContent(
                  type=px.ContentType.TEXT, text='What is this audio about?'),
          ],
      }],
      provider_model=MULTIMODAL_MODEL)
  _print_call_record(result)
  manual_check_with_url(
      'Does the record show audio playback or download for cat.mp3?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'an audio player or filename for the uploaded mp3')
  return state_data


@integration_block
def proxdash_renders_video_input(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  result = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.VIDEO,
                  path=asset(ASSET_VIDEO), media_type='video/mp4'),
              px.MessageContent(
                  type=px.ContentType.TEXT, text='What is in this video?'),
          ],
      }],
      provider_model=MULTIMODAL_MODEL)
  _print_call_record(result)
  manual_check_with_url(
      'Does the record show video preview or download for cat.mp4?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'a video preview or filename for the uploaded mp4')
  return state_data


@integration_block
def proxdash_renders_document_md(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  result = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.DOCUMENT,
                  path=asset(ASSET_MD), media_type='text/markdown'),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is inside this document?'),
          ],
      }],
      provider_model=DEFAULT_TEXT_MODEL)
  _print_call_record(result)
  manual_check_with_url(
      'Does the record show the markdown document content?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'rendered markdown body for cat.md')
  return state_data


@integration_block
def proxdash_renders_image_output(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  result = px.generate(
      'Make a funny cartoon cat in a living room.',
      provider_model=IMAGE_MODEL,
      output_format='image')
  _print_call_record(result)
  manual_check_with_url(
      'Does the record show the generated cartoon cat image?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'an image preview of the generated cat')
  return state_data


@integration_block
def proxdash_renders_audio_output(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  result = px.generate(
      'Hello! This is a ProxAI test.',
      provider_model=AUDIO_MODEL,
      output_format='audio')
  _print_call_record(result)
  manual_check_with_url(
      'Does the record show audio playback?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'a playable audio output element')
  return state_data


@integration_block
def proxdash_renders_video_output(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  try:
    result = px.generate(
        'A cat playing with a ball of yarn.',
        provider_model=VIDEO_MODEL,
        output_format='video')
  except ValueError as e:
    if 'No compatible endpoint found' in str(e):
      print(f'> Video render skipped: registry has no compatible '
            f'endpoint for {VIDEO_MODEL[0]}/{VIDEO_MODEL[1]}.')
      return state_data
    raise
  _print_call_record(result)
  manual_check_with_url(
      'Does the record show video playback?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'a playable video output element')
  return state_data


# -----------------------------------------------------------------------------
# 9.6  Files in ProxDash UI
# -----------------------------------------------------------------------------

@integration_block
def proxdash_files_appear_in_ui(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  media = px.MessageContent(
      path=asset(ASSET_PDF), media_type='application/pdf')
  px.files.upload(media=media, providers=[])
  print(f'> Uploaded to ProxDash: {media.proxdash_file_id}')

  url = f'{ctx.webview_base_url}/dashboard/files'
  manual_check_with_url(
      'Does the file appear in the ProxDash file list?',
      url,
      f'cat.pdf with proxdash_file_id={media.proxdash_file_id}')
  px.files.remove(media=media)
  return state_data


@integration_block
def proxdash_file_attached_to_call_record(state_data):
  ctx = _utils._CTX
  _connect_basic(state_data, stdout=True)
  result = px.generate(
      messages=[{
          'role': 'user',
          'content': [
              px.MessageContent(
                  type=px.ContentType.DOCUMENT,
                  path=asset(ASSET_PDF), media_type='application/pdf'),
              px.MessageContent(
                  type=px.ContentType.TEXT,
                  text='What is this document about?'),
          ],
      }],
      provider_model=('gemini', 'gemini-2.5-flash'))
  _print_call_record(result)
  manual_check_with_url(
      'Does the call record show the attached PDF (link to file or preview)?',
      f'{ctx.webview_base_url}/dashboard/logging',
      'a record with a clickable PDF reference')
  return state_data


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

BLOCKS = [
    proxdash_stdout_connection_message,
    proxdash_disable,
    proxdash_local_log_file,
    proxdash_logging_record_basic,
    proxdash_logging_record_full_options,
    proxdash_hide_sensitive_prompt,
    proxdash_hide_sensitive_messages,
    proxdash_limited_api_key,
    proxdash_experiment_path,
    proxdash_renders_text,
    proxdash_renders_json_input,
    proxdash_renders_pydantic_input,
    proxdash_renders_image_input,
    proxdash_renders_audio_input,
    proxdash_renders_video_input,
    proxdash_renders_document_md,
    proxdash_renders_image_output,
    proxdash_renders_audio_output,
    proxdash_renders_video_output,
    proxdash_files_appear_in_ui,
    proxdash_file_attached_to_call_record,
]


def main():
  ctx = init_run(_LABEL)
  state_data = ensure_setup_state(ctx)
  run_sequence(_LABEL, BLOCKS, state_data=state_data)


if __name__ == '__main__':
  main()
