"""ProxAI API Integration Test

Flags:
  --mode: latest, new, or a specific test ID. Default is latest.
  --print-code: print code blocks

Examples:
  python3 integrations_tests/proxai_api_test.py
  python3 integrations_tests/proxai_api_test.py --mode new
  python3 integrations_tests/proxai_api_test.py --print-code
"""
import os
import inspect
from pathlib import Path
import proxai as px
import random
import time
from pprint import pprint
import json
import argparse
from dataclasses import asdict

_ROOT_INTEGRATION_TEST_PATH = f'{Path.home()}/proxai_integration_test/'
_WEBVIEW_BASE_URL = 'http://localhost:3000'
_PROXDASH_BASE_URL = 'http://localhost:3001'
_TEST_ID = None
_TEST_NAME = None
_TEST_PATH = None
_ROOT_LOGGING_PATH = None
_ROOT_CACHE_PATH = None
_EXPERIMENT_PATH = None
_PRINT_CODE = False


def init_test_path():
  global _TEST_ID
  global _TEST_NAME
  global _TEST_PATH
  global _ROOT_LOGGING_PATH
  global _ROOT_CACHE_PATH
  global _EXPERIMENT_PATH
  global _PRINT_CODE
  os.makedirs(_ROOT_INTEGRATION_TEST_PATH, exist_ok=True)
  parser = argparse.ArgumentParser(description='ProxAI Integration Test')
  parser.add_argument('--mode', type=str, default='latest',
                      help='Execution mode for the integration test')
  parser.add_argument('--print-code', action='store_true',
                      help='Print code blocks')
  args = parser.parse_args()
  _PRINT_CODE = args.print_code
  dir_list = os.listdir(_ROOT_INTEGRATION_TEST_PATH)
  test_ids = [
      int(dir_name.split('test_')[1])
      for dir_name in dir_list if dir_name.startswith('test_')]
  if args.mode == 'latest':
    _TEST_ID = max(test_ids) if test_ids else '1'
  elif args.mode == 'new':
    _TEST_ID = max(test_ids) + 1 if test_ids else '1'
  else:
    try:
      _TEST_ID = int(args.mode)
    except ValueError:
      raise ValueError(f'Invalid test ID: {args.mode}')
  _TEST_NAME = f'test_{_TEST_ID}'
  _TEST_PATH = os.path.join(_ROOT_INTEGRATION_TEST_PATH, _TEST_NAME)
  _ROOT_LOGGING_PATH = f'{Path.home()}/proxai_log'
  _ROOT_CACHE_PATH = os.path.join(_TEST_PATH, 'proxai_cache')
  _EXPERIMENT_PATH = f'integration_tests/proxai_api_test/{_TEST_NAME}'
  os.makedirs(_TEST_PATH, exist_ok=True)
  os.makedirs(_ROOT_LOGGING_PATH, exist_ok=True)
  os.makedirs(_ROOT_CACHE_PATH, exist_ok=True)
  return


def integration_block(func):
  def wrapper(
      force_run=False,
      skip=None,
      **kwargs):
    state_path = os.path.join(_TEST_PATH, f'{func.__name__}.state')
    separator = "-" * (60 - len(func.__name__))
    if skip:
      print(f'\033[33m------------- [SKIPPED] {func.__name__} {separator}\033[0m')
      return kwargs.get('state_data', {})
    elif os.path.exists(state_path) and not force_run:
      print(f'\033[33m------------- [SKIPPED] {func.__name__} {separator}\033[0m')
      return json.load(open(state_path))
    else:
      print(f'\033[32m------------- [RUNNING] {func.__name__} {separator}\033[0m')
      if _PRINT_CODE:
        print(f'\033[32m<Code Block> \033[0m')
        print(inspect.getsource(func).strip())
        print(f'\033[32m</Code Block> \033[0m')
      state_data = func(**kwargs)
      json.dump(state_data, open(state_path, 'w'))
      input('> Press Enter to continue...')
      return state_data
  return wrapper


def _manual_user_check(test_message, fail_message):
  while True:
    answer = input(f'{test_message} [y/n]: ')
    if answer == 'y':
      break
    elif answer == 'n':
      raise Exception(fail_message)
    else:
      print('Please enter "y" or "n".')


@integration_block
def create_user(state_data):
  print(f'1 - Go to {_WEBVIEW_BASE_URL}/signup')
  print('2 - Create an account')
  print('    * Username: manueltest')
  print('    * Email   : manueltest@proxai.co')
  print('    * Password: test123!')
  print(f'3 - Create API key: {_WEBVIEW_BASE_URL}/dashboard/api-keys')
  print('> Enter the API key: ', end='')
  state_data['api_key'] = input()
  return state_data


@integration_block
def local_proxdash_connection(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      logging_options=px.types.LoggingOptions(
          logging_path=_ROOT_LOGGING_PATH,
      ),
      cache_options=px.types.CacheOptions(
          clear_model_cache_on_connect=False,
          clear_query_cache_on_connect=True,
          cache_path=_ROOT_CACHE_PATH,
      ),
      proxdash_options=px.types.ProxDashOptions(
          stdout=True,
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  print(f'> Connection working for {_PROXDASH_BASE_URL}')
  return state_data


@integration_block
def list_models(state_data):
  provider_models = px.models.list_models()
  for provider_model in provider_models:
    print(f'{provider_model.provider:>25} - {provider_model.model}')
  return state_data


@integration_block
def list_models_with_only_large_models(state_data):
  provider_models = px.models.list_models(only_largest_models=True)
  for provider_model in provider_models:
    print(f'{provider_model.provider:>25} - {provider_model.model}')
  return state_data


@integration_block
def list_models_with_return_all(state_data):
  model_status = px.models.list_models(return_all=True)
  print(str(model_status)[:150] + '...')
  print(f'Available models: {len(model_status.working_models)}')
  print(f'Failed models: {len(model_status.failed_models)}')
  return state_data


@integration_block
def list_models_with_clear_model_cache_and_verbose_output(state_data):
  model_status = px.models.list_models(
      clear_model_cache=True,
      return_all=True,
      verbose=True
  )
  print(f'Available models: {len(model_status.working_models)}')
  print(f'Failed models: {len(model_status.failed_models)}')
  for provider_model, logging_queries in model_status.provider_queries.items():
    if not logging_queries.response_record.error:
      continue
    print(
        f'{provider_model.provider} - {provider_model.model}: ',
        f'{logging_queries.response_record.error[:120]}')
  return state_data


@integration_block
def generate_text(state_data):
  response = px.generate_text('Hello! Which model are you?')
  print(response)
  return state_data


@integration_block
def generate_text_with_provider_model(state_data):
  response = px.generate_text(
      'Hello! Which model are you?',
      provider_model=('cohere', 'command'))
  print(response)
  return state_data


@integration_block
def generate_text_with_provider_model_type(state_data):
  provider_model = px.models.get_model('claude', 'opus')
  print(type(provider_model))
  print(f'{provider_model.provider=}')
  print(f'{provider_model.model=}')
  print(f'{provider_model.provider_model_identifier=}')

  response = px.generate_text(
      'Hello! Which model are you?',
      provider_model=provider_model)
  print(response)
  return state_data


@integration_block
def generate_text_with_system_prompt(state_data):
  response = px.generate_text(
      'Hello! Which model are you?',
      system="You are an helpful assistant that always answers in Japan.",
      provider_model=('openai', 'gpt-4.1'))
  print(response)
  return state_data


@integration_block
def generate_text_with_message_history(state_data):
  response = px.generate_text(
      system="No matter what, always answer with single integer.",
      messages=[
          {"role": "user", "content": "Hello AI Model!"},
          {"role": "assistant", "content": "17"},
          {"role": "user", "content": "How are you today?"},
          {"role": "assistant", "content": "923123"},
          {"role": "user",
          "content": "Can you answer question without any integer?"}
      ],
      provider_model=('openai', 'gpt-4.1')
  )
  print(response)
  return state_data


@integration_block
def generate_text_with_max_tokens(state_data):
  response = px.generate_text(
    'Can you write all numbers from 1 to 1000?',
    max_tokens=20,
    provider_model=('openai', 'gpt-4.1'))
  print(response)
  return state_data


@integration_block
def generate_text_with_temperature(state_data):
  response = px.generate_text(
      'If 5 + 20 would be a poem, what life be look like?',
      temperature=0.01,
      provider_model=('openai', 'gpt-4.1'))
  print(response)
  return state_data


@integration_block
def generate_text_with_extensive_return(state_data):
  response = px.generate_text(
      'Hello! Which model are you?',
      extensive_return=True)
  pprint(asdict(response))
  return state_data


@integration_block
def generate_text_with_suppress_provider_errors(state_data):
  model_status = px.models.list_models(return_all=True)
  assert model_status.failed_models, (
      'There is no failed models to try \'suppress_provider_errors\' option.')

  provider_model = list(model_status.failed_models)[0]
  response = px.generate_text(
      f'If {random.randint(1, 1000)} + {random.randint(1, 1000)} would be a '
      'poem, what life be look like?',
      provider_model=provider_model,
      suppress_provider_errors=True,
      extensive_return=True)
  assert response.response_record.error, (
      f'Could not reproduce error for {provider_model}.'
      f'This test failed.')

  print(f'Provider Model: {provider_model}')
  print('Response Source:', response.response_source)
  print('Error:', response.response_record.error.strip())
  error_traceback = response.response_record.error_traceback.strip()
  error_traceback = '\n'.join(error_traceback.split('\n')[:5]) + '\n...'
  print('Error Traceback:', error_traceback)

  return state_data


@integration_block
def set_model(state_data):
  def get_response_of_simple_request():
    return px.generate_text(
        'Hey AI model! This is simple request. Give an answer.',
        max_tokens=20,
        suppress_provider_errors=True
    ).strip().replace('\n', ' ')[:80]

  for provider_model in px.models.list_models()[:10]:
    px.set_model(provider_model)

    response = get_response_of_simple_request()
    print(
        f'{provider_model.provider:>25} - {provider_model.model:45} - {response}')
  return state_data


@integration_block
def check_health(state_data):
  px.check_health()
  return state_data


@integration_block
def check_health_without_multiprocessing(state_data):
  px.check_health(allow_multiprocessing=False)
  return state_data


@integration_block
def check_health_with_timeout(state_data):
  px.check_health(model_test_timeout=1)
  return state_data


@integration_block
def check_health_return(state_data):
  model_status = px.check_health(
      verbose=False,
      extensive_return=True,
      model_test_timeout=1)
  print(f'--- model_status.working_models: {len(model_status.working_models)}')
  print(f'--- model_status.failed_models: {len(model_status.failed_models)}')
  return state_data


@integration_block
def logging(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      logging_options=px.types.LoggingOptions(
          logging_path=_ROOT_LOGGING_PATH,
      ),
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  logging_path = os.path.join(_ROOT_LOGGING_PATH, _EXPERIMENT_PATH)
  px.generate_text('Hello model! This test for log manager.')

  pprint(os.listdir(logging_path))
  last_log_data = None
  with open(os.path.join(logging_path, 'provider_queries.log'), 'r') as f:
    for line in f:
      last_log_data = json.loads(line)

  assert (
      last_log_data['query_record']['prompt'] ==
      'Hello model! This test for log manager.'
  ), 'Last log data is not correct.'
  pprint(last_log_data)
  return state_data


@integration_block
def logging_with_hide_sensitive_content(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      logging_options=px.types.LoggingOptions(
          logging_path=_ROOT_LOGGING_PATH,
          hide_sensitive_content=True,
      ),
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  logging_path = os.path.join(_ROOT_LOGGING_PATH, _EXPERIMENT_PATH)

  print('1 - Check prompt and response are hidden.')
  px.generate_text('Hello model! This test for log manager.')
  last_log_data = None
  with open(os.path.join(logging_path, 'provider_queries.log'), 'r') as f:
    for line in f:
      last_log_data = json.loads(line)
  assert (
      last_log_data['query_record']['prompt'] ==
      '<sensitive content hidden>'
  ), 'Sensitive content is not hidden.'
  assert (
      last_log_data['response_record']['response'] ==
      '<sensitive content hidden>'
  ), 'Sensitive content is not hidden.'
  pprint(last_log_data)
  print()

  print('2 - Check system and messages are hidden.')
  px.generate_text(
      system='You are a helpful assistant that always answers in Japan.',
      messages=[
        {'role': 'user', 'content': 'Hello! This is a test.'},
        {'role': 'assistant', 'content': 'Hello! How can I help you with your test?'},
        {'role': 'user', 'content': 'What is the capital of Japan?'},
      ],
  )
  last_log_data = None
  with open(os.path.join(logging_path, 'provider_queries.log'), 'r') as f:
    for line in f:
      last_log_data = json.loads(line)
  assert (
      last_log_data['query_record']['system'] ==
      '<sensitive content hidden>'
  ), 'Sensitive content is not hidden.'
  assert (
      last_log_data['query_record']['messages'][0] ==
      {
        'role': 'assistant',
        'content': '<sensitive content hidden>'
      }
  ), 'Sensitive content is not hidden.'
  pprint(last_log_data)

  return state_data


@integration_block
def logging_with_stdout(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      logging_options=px.types.LoggingOptions(
          logging_path=_ROOT_LOGGING_PATH,
          stdout=True,
      ),
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  print('1 - Check following generate_text logging_record is visible:')
  px.generate_text(
      'Hello model! This test for log manager. I need to check if this '
      'message is visible in stdout.')
  _manual_user_check(
      test_message='Logging record is visible in stdout?',
      fail_message='Logging record is not visible in stdout.')
  return state_data


@integration_block
def query_cache(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      cache_options=px.types.CacheOptions(
          cache_path=_ROOT_CACHE_PATH
      ),
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  response = px.generate_text(
      f'Hello model, what is 23 times 23?',
      extensive_return=True)
  print(f'> Response Source: {response.response_source}')
  assert response.response_source == 'PROVIDER'

  response = px.generate_text(
      f'Hello model, what is 23 times 23?',
      extensive_return=True)
  print(f'> Response Source: {response.response_source}')
  assert response.response_source == 'CACHE'

  response = px.generate_text(
      f'Hello model, what is 23 times 23?',
      extensive_return=True)
  print(f'> Response Source: {response.response_source}')
  assert response.response_source == 'CACHE'
  return state_data


@integration_block
def query_cache_with_unique_response_limit(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      cache_options=px.types.CacheOptions(
          cache_path=_ROOT_CACHE_PATH,
          unique_response_limit=3
      ),
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  for idx in range(6):
    print(f'> {idx} | ', end='')
    response = px.generate_text(
        'Can you pick 100 different random positive integers which are less '
        f'than 3000? Can you also explain why you picked these numbers? '
        'Please think deeply about your decision and answer accordingly. '
        'Start your sentence with random simple poem.',
        provider_model=('openai', 'gpt-4.1'),
        temperature=0.3,
        extensive_return=True)
    if idx < 3:
      assert response.response_source == 'PROVIDER'
    else:
      assert response.response_source == 'CACHE'
    response_text = response.response_record.response[:100].replace('\n', '')
    response_time = response.response_record.response_time.total_seconds()
    print(f'{response.response_source:9} | '
          f'{response_time:9.2f}s | '
          f'{response_text}...')

  return state_data


@integration_block
def query_cache_with_use_cache_false(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      cache_options=px.types.CacheOptions(
          cache_path=_ROOT_CACHE_PATH
      ),
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  response = px.generate_text(
      f'Hello model, what is 23 times 23?',
      extensive_return=True)
  print(f'> Response Source: {response.response_source}')
  assert response.response_source == 'PROVIDER'

  response = px.generate_text(
      f'Hello model, what is 23 times 23?',
      extensive_return=True)
  print(f'> Response Source: {response.response_source}')
  assert response.response_source == 'CACHE'

  response = px.generate_text(
      f'Hello model, what is 23 times 23?',
      use_cache=False,
      extensive_return=True)
  print(f'> Response Source: {response.response_source}')
  assert response.response_source == 'PROVIDER'
  return state_data


@integration_block
def query_cache_with_clear_cache_and_override_unique_responses(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      cache_options=px.types.CacheOptions(
          cache_path=_ROOT_CACHE_PATH,
          unique_response_limit=3,
          clear_query_cache_on_connect=True,
      ),
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )

  response = px.generate_text(
      f'Hello model, what is 23 times 23?',
      unique_response_limit=1,
      extensive_return=True)
  print(f'> Response Source: {response.response_source}')
  assert response.response_source == 'PROVIDER'

  response = px.generate_text(
      f'Hello model, what is 23 times 23?',
      unique_response_limit=1,
      extensive_return=True)
  print(f'> Response Source: {response.response_source}')
  assert response.response_source == 'CACHE'

  response = px.generate_text(
      f'Hello model, what is 23 times 23?',
      unique_response_limit=1,
      extensive_return=True)
  print(f'> Response Source: {response.response_source}')
  assert response.response_source == 'CACHE'
  return state_data


@integration_block
def proxdash_logging_record(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  px.generate_text(
      'Logging record test. Hello model, I am testing my api logging record.')
  print(f'1 - Go to ProxDash page: {_WEBVIEW_BASE_URL}/dashboard/logging')
  print('2 - Check if the latest logging record starts with:')
  print('    * "Logging record test. Hello model, I am testing my api '
        'logging record."')
  _manual_user_check(
      test_message='Logging record is visible and correct?',
      fail_message='Logging record is not visible or correct.')
  print('3 - Click "Open" button to see the details of the logging record.')
  print('4 - Check if the details are correct.')
  _manual_user_check(
      test_message='Details are correct on single logging record view page?',
      fail_message='Single logging record view page is broken.')
  return state_data


@integration_block
def proxdash_logging_record_with_all_options(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  px.generate_text(
      system="No matter what, always answer with single integer.",
      messages=[
          {"role": "user", "content": "Hello AI Model!"},
          {"role": "assistant", "content": "17"},
          {"role": "user", "content": "How are you today?"},
          {"role": "assistant", "content": "923123"},
          {"role": "user",
          "content": "Can you answer question without any integer?"}
      ],
      provider_model=('openai', 'gpt-4.1'),
      temperature=0.3,
      max_tokens=2000,
      stop=['STOP'])
  print(f'1 - Go to ProxDash page: {_WEBVIEW_BASE_URL}/dashboard/logging')
  print('2 - Open the latest logging record.')
  print('3 - Check if the details are correct:')
  print('    * System: No matter what, always answer with single integer.')
  print('    * Messages:')
  print('    * Temperature: 0.3')
  print('    * Max Tokens: 2000')
  print('    * Stop: ["\\n\\n"]')
  print('    * Provider Model: openai/gpt-4.1')
  _manual_user_check(
      test_message='All details are correct?',
      fail_message='Some details are not correct.')
  return state_data


@integration_block
def proxdash_logging_record_with_hide_sensitive_content_prompt(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      proxdash_options=px.types.ProxDashOptions(
          hide_sensitive_content=True,
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  prompt = (
      'This record should appear on ProxDash but the prompt '
      'content and the response content from AI provider shouldn\'t appear '
      'on ProxDash.'
  )
  px.generate_text(prompt)
  print(f'1 - Go to ProxDash page: {_WEBVIEW_BASE_URL}/dashboard/logging')
  print('2 - Check if the latest logging record is:')
  print('    * Prompt: <sensitive content hidden> ')
  print('    * Response: <sensitive content hidden> ')
  _manual_user_check(
      test_message='Latest logging record prompt and response are hidden?',
      fail_message='Latest logging record prompt or response is not hidden.')
  print('3 - Click "Open" button to see the details of the logging record.')
  print('4 - Check if all of the followings are hidden:')
  print('    * Prompt: <sensitive content hidden> ')
  print('    * Response: <sensitive content hidden> ')
  _manual_user_check(
      test_message='All contents are hidden in single logging record view page?',
      fail_message='Some contents are not hidden in single logging record view page.')
  return state_data


@integration_block
def proxdash_logging_record_with_hide_sensitive_content_message(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      proxdash_options=px.types.ProxDashOptions(
          hide_sensitive_content=True,
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  messages = [
    {'role': 'user', 'content': 'These contents should not appear on ProxDash.'},
    {'role': 'assistant', 'content': 'Ok, I will not show them.'},
    {'role': 'user', 'content': 'Be sure to hide them.'},
  ]
  px.generate_text(
      system='You are a helpful assistant that always answers in Japan.',
      messages=messages)
  print(f'1 - Go to ProxDash page: {_WEBVIEW_BASE_URL}/dashboard/logging')
  print('2 - Check if the latest logging record is:')
  print('    * Prompt: <sensitive content hidden> ')
  print('    * Response: <sensitive content hidden> ')
  _manual_user_check(
      test_message='Latest logging record prompt and response are hidden?',
      fail_message='Latest logging record prompt or response is not hidden.')
  print('3 - Click "Open" button to see the details of the logging record.')
  print('4 - Check all of the followings are hidden:')
  print('    * Prompt: <sensitive content hidden> ')
  print('    * Response: <sensitive content hidden> ')
  print('    * System: <sensitive content hidden> ')
  print('    * Messages: <sensitive content hidden> ')
  _manual_user_check(
      test_message='All contents are hidden in single logging record view page?',
      fail_message='Some contents are not hidden in single logging record view page.')
  return state_data


@integration_block
def proxdash_experiment_path(state_data):
  px.connect(
      experiment_path=_EXPERIMENT_PATH,
      proxdash_options=px.types.ProxDashOptions(
          base_url=_PROXDASH_BASE_URL,
          api_key=state_data['api_key'],
      ),
  )
  px.generate_text(
      'Experiment path test. Hello model, I am testing my api experiment path.')
  print(f'1 - Go to ProxDash page: {_WEBVIEW_BASE_URL}/dashboard/experiments')
  print('2 - Click "Refresh" button')
  print(f'3 - Check experiment path exist in experiment folders: {_EXPERIMENT_PATH}')
  _manual_user_check(
      test_message='Experiment path exists in experiment folders?',
      fail_message='Experiment path is not exist in experiment folders.')
  print('4 - Click "Open" button to open single experiment view page.')
  print('5 - Click "Logging Records" tab to list all logging records.')
  print('6 - Check if the latest logging record starts with:')
  print('    * "Experiment path test. Hello model, I am testing my api '
        'experiment path."')
  _manual_user_check(
      test_message='Latest logging record is visible and correct?',
      fail_message='Latest logging record is not visible or correct.')
  print('7 - Click "Open" button to see the details of the logging record.')
  print('8 - Check if the details are correct.')
  _manual_user_check(
      test_message='Details are correct on single logging record view page?',
      fail_message='Single logging record view page is broken.')
  return state_data


def main():
  init_test_path()
  state_data = {}
  state_data = create_user(state_data=state_data)
  state_data = local_proxdash_connection(state_data=state_data, force_run=True)
  state_data = list_models(state_data=state_data)
  state_data = list_models_with_only_large_models(state_data=state_data)
  state_data = list_models_with_return_all(state_data=state_data)
  state_data = list_models_with_clear_model_cache_and_verbose_output(state_data=state_data)
  state_data = generate_text(state_data=state_data)
  state_data = generate_text_with_provider_model(state_data=state_data)
  state_data = generate_text_with_provider_model_type(state_data=state_data)
  state_data = generate_text_with_system_prompt(state_data=state_data)
  state_data = generate_text_with_message_history(state_data=state_data)
  state_data = generate_text_with_max_tokens(state_data=state_data)
  state_data = generate_text_with_temperature(state_data=state_data)
  state_data = generate_text_with_extensive_return(state_data=state_data)
  state_data = generate_text_with_suppress_provider_errors(state_data=state_data)
  state_data = set_model(state_data=state_data)
  state_data = check_health(state_data=state_data)
  state_data = check_health_without_multiprocessing(state_data=state_data, skip=True)
  state_data = check_health_with_timeout(state_data=state_data)
  state_data = check_health_return(state_data=state_data)
  state_data = logging(state_data=state_data)
  state_data = logging_with_hide_sensitive_content(state_data=state_data)
  state_data = logging_with_stdout(state_data=state_data)
  state_data = query_cache(state_data=state_data)
  state_data = query_cache_with_unique_response_limit(state_data=state_data)
  state_data = query_cache_with_use_cache_false(state_data=state_data)
  state_data = query_cache_with_clear_cache_and_override_unique_responses(state_data=state_data)
  state_data = proxdash_logging_record(state_data=state_data)
  state_data = proxdash_logging_record_with_all_options(state_data=state_data)
  state_data = proxdash_logging_record_with_hide_sensitive_content_prompt(state_data=state_data)
  state_data = proxdash_logging_record_with_hide_sensitive_content_message(state_data=state_data)
  state_data = proxdash_experiment_path(state_data=state_data)


if __name__ == '__main__':
  main()
