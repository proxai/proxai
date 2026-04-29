"""Shared utilities for ProxAI integration tests.

This module is intentionally not a test file (leading underscore). Each
test file (01_models_test.py, 02_generate_test.py, ...) imports from
here.

Public API:
  RunContext              dataclass — paths, base URLs, CLI flags
  init_run(label)         parse argv, set up paths, return RunContext
  ensure_setup_state(ctx) load or create the shared _setup.state with api_key
  integration_block       decorator for state-persisting test blocks
  manual_check            y/n prompt; raise on n
  manual_check_with_url   print URL + prompt
  print_separator         colored status line

Provider model constants (TEXT_MODELS, THINKING_MODEL, IMAGE_MODEL, ...)
and asset path helpers (asset(name)) live here too — one source of truth
across all 5 test files.
"""

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path

_ROOT_INTEGRATION_TEST_PATH = f'{Path.home()}/proxai_integration_test/'
_ROOT_LOGGING_PATH = f'{Path.home()}/proxai_log'

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), '_assets')


@dataclass
class RunContext:
  """Per-run state shared across all integration_block calls in one file."""
  test_id: int
  test_path: str                # ~/proxai_integration_test/test_<id>/
  file_path: str                # ~/proxai_integration_test/test_<id>/<label>/
  setup_state_path: str         # ~/proxai_integration_test/test_<id>/_setup.state
  root_logging_path: str
  root_cache_path: str
  experiment_path: str          # integration_tests/<label>/test_<id>
  webview_base_url: str
  proxdash_base_url: str
  print_code: bool
  auto_continue: bool


_CTX: RunContext | None = None


def _color_code(color: str) -> str:
  return {
      'green': '\033[32m',
      'yellow': '\033[33m',
      'red': '\033[31m',
      'magenta': '\033[35m',
      'cyan': '\033[36m',
  }.get(color, '\033[0m')


def print_separator(status: str, message: str, color: str) -> None:
  """Print a colored separator line with status and message."""
  code = _color_code(color)
  separator = '-' * max(0, 59 - len(message))
  print(f'{code}------------- [{status:8}] {message} {separator}\033[0m')


def init_run(label: str) -> RunContext:
  """Parse argv, set up directories, and return a RunContext for this file.

  `label` identifies the test file (e.g. '01_models', '03_files'). It
  segments per-file state from the shared _setup.state.
  """
  global _CTX
  os.makedirs(_ROOT_INTEGRATION_TEST_PATH, exist_ok=True)

  parser = argparse.ArgumentParser(description=f'ProxAI integration: {label}')
  parser.add_argument('--mode', type=str, default='latest',
                      help='latest | new | <test_id>')
  parser.add_argument('--print-code', action='store_true',
                      help='Print the source of each integration block')
  parser.add_argument('--auto-continue', action='store_true',
                      help='Skip the inter-block "Press Enter" pause. Manual '
                           'check y/n prompts still require input.')
  parser.add_argument('--env', choices=['dev', 'prod'], default='dev',
                      help='ProxDash UI / nest environment')
  parser.add_argument('--test', type=str, default=None,
                      help='Run only the named integration block')
  args, _unknown = parser.parse_known_args()

  if args.env == 'prod':
    webview_base_url = 'https://proxai.co'
    proxdash_base_url = 'https://proxainest-production.up.railway.app'
  else:
    webview_base_url = 'http://localhost:3000'
    proxdash_base_url = 'http://localhost:3001'

  dir_list = os.listdir(_ROOT_INTEGRATION_TEST_PATH)
  test_ids = [
      int(name.split('test_')[1])
      for name in dir_list
      if name.startswith('test_') and name.split('test_')[1].isdigit()
  ]
  if args.mode == 'latest':
    test_id = max(test_ids) if test_ids else 1
  elif args.mode == 'new':
    test_id = max(test_ids) + 1 if test_ids else 1
  else:
    try:
      test_id = int(args.mode)
    except ValueError as err:
      raise ValueError(f'Invalid --mode: {args.mode}') from err

  test_path = os.path.join(_ROOT_INTEGRATION_TEST_PATH, f'test_{test_id}')
  file_path = os.path.join(test_path, label)
  setup_state_path = os.path.join(test_path, '_setup.state')
  root_cache_path = os.path.join(test_path, 'proxai_cache')
  experiment_path = f'integration_tests/{label}/test_{test_id}'

  os.makedirs(test_path, exist_ok=True)
  os.makedirs(file_path, exist_ok=True)
  os.makedirs(_ROOT_LOGGING_PATH, exist_ok=True)
  os.makedirs(root_cache_path, exist_ok=True)

  ctx = RunContext(
      test_id=test_id,
      test_path=test_path,
      file_path=file_path,
      setup_state_path=setup_state_path,
      root_logging_path=_ROOT_LOGGING_PATH,
      root_cache_path=root_cache_path,
      experiment_path=experiment_path,
      webview_base_url=webview_base_url,
      proxdash_base_url=proxdash_base_url,
      print_code=args.print_code,
      auto_continue=args.auto_continue,
  )
  _CTX = ctx
  print_separator('STARTING', experiment_path, 'magenta')
  return ctx


def get_run_args() -> argparse.Namespace:
  """Re-parse argv to read --test (used by file main() to dispatch)."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='latest')
  parser.add_argument('--print-code', action='store_true')
  parser.add_argument('--auto-continue', action='store_true')
  parser.add_argument('--env', choices=['dev', 'prod'], default='dev')
  parser.add_argument('--test', default=None)
  args, _unknown = parser.parse_known_args()
  return args


def ensure_setup_state(ctx: RunContext) -> dict:
  """Load shared _setup.state or walk operator through first-time setup.

  Idempotent: any file's main() can call this; only the first call in a
  given test_<id> pays the cost of asking the operator for an api_key.
  """
  if os.path.exists(ctx.setup_state_path):
    with open(ctx.setup_state_path) as f:
      return json.load(f)

  print_separator('SETUP', 'first-time setup for this test_id', 'magenta')
  print(f'1 - Go to {ctx.webview_base_url}/signup')
  print('2 - Create an account')
  print('    * Username: manualtest')
  print('    * Email   : manualtest@proxai.co')
  print('    * Password: test123!')
  print(f'3 - Create API key: {ctx.webview_base_url}/dashboard/api-keys')
  api_key = input('> Enter the API key: ').strip()
  if not api_key:
    raise RuntimeError('Empty API key — aborting.')
  state = {'api_key': api_key}
  with open(ctx.setup_state_path, 'w') as f:
    json.dump(state, f)
  return state


def _integration_block_state_path(ctx: RunContext, name: str) -> str:
  return os.path.join(ctx.file_path, f'{name}.state')


def integration_block(func):
  """Decorator: persist test state, skip on rerun, manage flow.

  Wrapped function signature: `func(state_data, **kwargs) -> state_data`.
  state_data is a dict that flows through every block in a file.

  kwargs:
    force_run=True    re-run even if state exists
    skip=True         mark as skipped, return state_data unchanged
  """
  def wrapper(state_data: dict, force_run: bool = False, skip: bool = False,
              **kwargs) -> dict:
    if _CTX is None:
      raise RuntimeError(
          'integration_block called before init_run() — '
          'every test file must call _utils.init_run(label) first.')

    name = func.__name__
    state_path = _integration_block_state_path(_CTX, name)

    if skip:
      print_separator('SKIPPED', name, 'yellow')
      return state_data
    if os.path.exists(state_path) and not force_run:
      print_separator('SKIPPED', name, 'yellow')
      with open(state_path) as f:
        return json.load(f)

    print_separator('RUNNING', name, 'green')
    if _CTX.print_code:
      print('\033[32m<Code Block>\033[0m')
      print(inspect.getsource(func).strip())
      print('\033[32m</Code Block>\033[0m')

    state_data = func(state_data=state_data, **kwargs)

    if not _CTX.auto_continue:
      input('> Press Enter to continue...')

    with open(state_path, 'w') as f:
      json.dump(state_data, f)
    print_separator('PASSED', name, 'green')
    return state_data
  wrapper.__name__ = func.__name__
  wrapper.__wrapped__ = func
  return wrapper


def manual_check(test_message: str, fail_message: str) -> None:
  """Prompt operator y/n; raise on n. Honors no flags — manual checks
  always require human input."""
  while True:
    answer = input(f'{test_message} [y/n]: ').strip().lower()
    if answer == 'y':
      return
    if answer == 'n':
      raise AssertionError(fail_message)
    print('Please enter "y" or "n".')


def manual_check_with_url(prompt: str, url: str, expectation: str,
                          fail_message: str | None = None) -> None:
  """Print a URL and an expectation, then ask y/n."""
  print(f'  Open: {url}')
  print(f'  Expect: {expectation}')
  manual_check(prompt, fail_message or f'UI mismatch at {url}')


# -----------------------------------------------------------------------------
# Provider model selection — one source of truth for every test file.
#
# Use these constants in tests so that when a model is renamed in the JSON
# (or retired), there is exactly one place to change. Each constant points
# to a known-good, recommended, documented v1.3.0 model.
# -----------------------------------------------------------------------------

# Primary text-generation model for assert-driven tests where any working
# text model is fine.
DEFAULT_TEXT_MODEL = ('openai', 'gpt-4o')

# Three text models exercised by tests that loop over providers.
TEXT_MODELS = [
    ('openai',   'gpt-4o'),
    ('gemini',   'gemini-3-flash'),
    ('claude',   'sonnet-4.6'),
]

# Models with specific feature support.
THINKING_MODEL    = ('openai', 'o3')
WEB_SEARCH_MODEL  = ('gemini', 'gemini-3-flash')
JSON_OUTPUT_MODEL = ('openai', 'gpt-4o')

# Output-format-specific models.
IMAGE_MODEL = ('gemini', 'gemini-2.5-flash-image')
AUDIO_MODEL = ('gemini', 'lyria-3-clip')
VIDEO_MODEL = ('openai', 'sora-2')

# Gemini variants for multi-modal input (gemini supports the most modalities).
MULTIMODAL_MODEL = ('gemini', 'gemini-3-flash')

# Mock providers for error-handling tests.
FAILING_MODEL = ('mock_failing_provider', 'mock_failing_model')
WORKING_MOCK  = ('mock_provider', 'mock_model')

# Per-provider canonical model for files-API multi-provider tests.
FILES_PROVIDER_MODELS = {
    'gemini':  ('gemini',  'gemini-2.5-flash'),
    'claude':  ('claude',  'sonnet-4.6'),
    'openai':  ('openai',  'gpt-4o'),
    'mistral': ('mistral', 'mistral-small-latest'),
}


# -----------------------------------------------------------------------------
# Assets — cat.{pdf,jpeg,webp,md,mp3,mp4} live in _assets/
# -----------------------------------------------------------------------------

def asset(name: str) -> str:
  """Resolve a file under integration_tests/_assets/."""
  path = os.path.join(_ASSETS_DIR, name)
  if not os.path.exists(path):
    raise FileNotFoundError(
        f'Missing asset: {path}. Run "cp examples/refactoring_test_assets/* '
        f'{_ASSETS_DIR}/" if migrating from examples.')
  return path


ASSET_PDF   = 'cat.pdf'
ASSET_IMAGE = 'cat.jpeg'
ASSET_WEBP  = 'cat.webp'
ASSET_MD    = 'cat.md'
ASSET_AUDIO = 'cat.mp3'
ASSET_VIDEO = 'cat.mp4'


# -----------------------------------------------------------------------------
# Test runner helper — used by every file's main()
# -----------------------------------------------------------------------------

def run_sequence(label: str, blocks: list, state_data: dict | None = None,
                 args=None) -> dict:
  """Run a list of decorated integration blocks in order.

  blocks is a list of either:
    - the decorated function itself (no extra kwargs)
    - a tuple (func, kwargs_dict) for blocks that need force_run/skip

  Honors --test <name> to run only one block.
  """
  if args is None:
    args = get_run_args()
  state_data = state_data or {}

  if args.test is not None:
    matched = False
    for entry in blocks:
      func, extra_kwargs = (entry, {}) if not isinstance(entry, tuple) else entry
      if func.__name__ == args.test:
        state_data = func(state_data=state_data, **extra_kwargs)
        matched = True
        break
    if not matched:
      names = ', '.join(
          (e if not isinstance(e, tuple) else e[0]).__name__ for e in blocks)
      raise SystemExit(f'Unknown --test {args.test}. Available: {names}')
    return state_data

  for entry in blocks:
    func, extra_kwargs = (entry, {}) if not isinstance(entry, tuple) else entry
    state_data = func(state_data=state_data, **extra_kwargs)
  return state_data
