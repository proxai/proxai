import json
import os
import tempfile

import pytest

import proxai.logging.utils as logging_utils
import proxai.types as types


def _get_query_record_examples():
  return [
      {'call_type': types.CallType.GENERATE_TEXT},
      {'prompt': 'Test prompt'},
      {'system': 'Test system'},
      {'messages': [{'role': 'user', 'content': 'Test message'}]},
      {'call_type': types.CallType.GENERATE_TEXT,
       'provider_model': pytest.model_configs_instance.get_provider_model(
           ('openai', 'gpt-4')),
       'prompt': 'Test prompt',
       'system': 'Test system',
       'messages': [{'role': 'user', 'content': 'Test message'}]},
  ]


def _get_query_response_record_examples():
  return [
      {'response': types.Response(
          type=types.ResponseType.TEXT,
          value='Test response')},
      {'error': 'Test error'},
      {'response': types.Response(
          type=types.ResponseType.TEXT,
          value='Test response'), 'error': None},
  ]


def _get_logging_record_examples():
  return [
      {'query_record': types.QueryRecord(**_get_query_record_examples()[0])},
      {'response_record': types.QueryResponseRecord(
          **_get_query_response_record_examples()[0])},
      {'query_record': types.QueryRecord(**_get_query_record_examples()[4]),
       'response_record': types.QueryResponseRecord(
           **_get_query_response_record_examples()[0])},
  ]


class TestHideSensitiveContent:
  @pytest.mark.parametrize('query_record_options', _get_query_record_examples())
  def test_hide_sensitive_content_query_record(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    hidden_record = logging_utils._hide_sensitive_content_query_record(
        query_record)

    # Original should be unchanged
    assert query_record == types.QueryRecord(**query_record_options)

    # Check sensitive fields are hidden
    if hidden_record.system:
      assert hidden_record.system == '<sensitive content hidden>'
    if hidden_record.prompt:
      assert hidden_record.prompt == '<sensitive content hidden>'
    if hidden_record.messages:
      assert len(hidden_record.messages) == 1
      assert hidden_record.messages[0]['role'] == 'assistant'
      assert hidden_record.messages[0]['content'] == (
          '<sensitive content hidden>')

  @pytest.mark.parametrize(
      'response_record_options', _get_query_response_record_examples())
  def test_hide_sensitive_content_query_response_record(
      self, response_record_options):
    response_record = types.QueryResponseRecord(**response_record_options)
    hidden_record = logging_utils._hide_sensitive_content_query_response_record(
        response_record)

    # Original should be unchanged
    assert response_record == types.QueryResponseRecord(
        **response_record_options)

    # Check response is hidden if present
    if hidden_record.response:
      assert hidden_record.response.value == '<sensitive content hidden>'
    # Error should remain unchanged
    assert hidden_record.error == response_record.error

  @pytest.mark.parametrize(
      'logging_record_options', _get_logging_record_examples())
  def test_hide_sensitive_content_logging_record(self, logging_record_options):
    logging_record = types.LoggingRecord(**logging_record_options)
    hidden_record = logging_utils._hide_sensitive_content_logging_record(
        logging_record)

    # Original should be unchanged
    assert logging_record == types.LoggingRecord(**logging_record_options)

    # Check both query and response records are hidden appropriately
    if hidden_record.query_record:
      if hidden_record.query_record.system:
        assert hidden_record.query_record.system == '<sensitive content hidden>'
      if hidden_record.query_record.prompt:
        assert hidden_record.query_record.prompt == '<sensitive content hidden>'
      if hidden_record.query_record.messages:
        assert hidden_record.query_record.messages[0]['content'] == (
            '<sensitive content hidden>')

    if hidden_record.response_record and hidden_record.response_record.response:
      assert hidden_record.response_record.response.value == (
          '<sensitive content hidden>')


class TestLogging:
  def test_write_log(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      logging_options = types.LoggingOptions(logging_path=temp_dir)
      test_data = {'test': 'data'}
      test_filename = 'test.log'

      logging_utils._write_log(
          logging_options=logging_options,
          file_name=test_filename,
          data=test_data)

      file_path = os.path.join(temp_dir, test_filename)
      assert os.path.exists(file_path)

      with open(file_path) as f:
        content = json.loads(f.read().strip())
        assert content == test_data

  def test_log_logging_record_no_options(self):
    # Should return silently with no options
    logging_utils.log_logging_record(None, types.LoggingRecord())

  def test_log_logging_record_stdout_only(self):
    logging_options = types.LoggingOptions(stdout=True)
    logging_record = types.LoggingRecord(
        **_get_logging_record_examples()[2])

    # Should not raise any errors for stdout only
    logging_utils.log_logging_record(logging_options, logging_record)

  def test_log_logging_record_with_file(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      logging_options = types.LoggingOptions(
          logging_path=temp_dir,
          hide_sensitive_content=True)
      logging_record = types.LoggingRecord(
          **_get_logging_record_examples()[2])

      logging_utils.log_logging_record(logging_options, logging_record)

      file_path = os.path.join(temp_dir, logging_utils.QUERY_LOGGING_FILE_NAME)
      assert os.path.exists(file_path)

  def test_log_message(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      logging_options = types.LoggingOptions(
          logging_path=temp_dir,
          hide_sensitive_content=True)

      test_message = "Test message"
      query_record = types.QueryRecord(**_get_query_record_examples()[4])

      # Test ERROR type
      logging_utils.log_message(
          logging_options=logging_options,
          message=test_message,
          type=types.LoggingType.ERROR,
          query_record=query_record)

      assert os.path.exists(os.path.join(
          temp_dir, logging_utils.ERROR_LOGGING_FILE_NAME))
      assert os.path.exists(os.path.join(
          temp_dir, logging_utils.MERGED_LOGGING_FILE_NAME))

      # Test WARNING type
      logging_utils.log_message(
          logging_options=logging_options,
          message=test_message,
          type=types.LoggingType.WARNING)

      assert os.path.exists(os.path.join(
          temp_dir, logging_utils.WARNING_LOGGING_FILE_NAME))

      # Test INFO type
      logging_utils.log_message(
          logging_options=logging_options,
          message=test_message,
          type=types.LoggingType.INFO)

      assert os.path.exists(os.path.join(
          temp_dir, logging_utils.INFO_LOGGING_FILE_NAME))

  def test_log_proxdash_message(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      logging_options = types.LoggingOptions(
          logging_path=temp_dir,
          hide_sensitive_content=True)
      proxdash_options = types.ProxDashOptions(stdout=True)

      test_message = "Test proxdash message"
      query_record = types.QueryRecord(**_get_query_record_examples()[4])

      # Test ERROR type
      logging_utils.log_proxdash_message(
          logging_options=logging_options,
          proxdash_options=proxdash_options,
          message=test_message,
          type=types.LoggingType.ERROR,
          query_record=query_record)

      assert os.path.exists(os.path.join(
          temp_dir, logging_utils.ERROR_LOGGING_FILE_NAME))
      assert os.path.exists(os.path.join(
          temp_dir, logging_utils.PROXDASH_LOGGING_FILE_NAME))

      # Test WARNING type
      logging_utils.log_proxdash_message(
          logging_options=logging_options,
          proxdash_options=proxdash_options,
          message=test_message,
          type=types.LoggingType.WARNING)

      assert os.path.exists(os.path.join(
          temp_dir, logging_utils.WARNING_LOGGING_FILE_NAME))
