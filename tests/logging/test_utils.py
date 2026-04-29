import json

import proxai.logging.utils as logging_utils
import proxai.types as types


_HIDDEN = '<sensitive content hidden>'
_MODEL = types.ProviderModelType(
    provider='openai', model='gpt-4', provider_model_identifier='gpt-4'
)


class TestHideSensitiveContentQueryRecord:

  def test_masks_all_sensitive_fields(self):
    chat = types.Chat(
        system_prompt='chat sys',
        messages=[types.Message(role='user', content='hello')],
    )
    qr = types.QueryRecord(
        prompt='the prompt',
        system_prompt='the system',
        chat=chat,
        provider_model=_MODEL,
    )
    hidden = logging_utils._hide_sensitive_content_query_record(qr)

    assert hidden.prompt == _HIDDEN
    assert hidden.system_prompt == _HIDDEN
    assert hidden.chat.system_prompt == _HIDDEN
    assert hidden.chat.messages == [
        types.Message(role=types.MessageRoleType.ASSISTANT, content=_HIDDEN)
    ]
    assert hidden.provider_model == _MODEL

  def test_does_not_mutate_original(self):
    qr = types.QueryRecord(prompt='p', system_prompt='s')
    logging_utils._hide_sensitive_content_query_record(qr)
    assert qr.prompt == 'p'
    assert qr.system_prompt == 's'


class TestWriteLog:

  def test_write_log(self, tmp_path):
    opts = types.LoggingOptions(logging_path=str(tmp_path))
    logging_utils._write_log(
        logging_options=opts, file_name='test.log', data={'k': 'v'}
    )
    content = (tmp_path / 'test.log').read_text().strip()
    assert json.loads(content) == {'k': 'v'}


class TestLogMessage:

  def test_routes_by_type(self, tmp_path):
    opts = types.LoggingOptions(logging_path=str(tmp_path))
    for level in [
        types.LoggingType.ERROR,
        types.LoggingType.WARNING,
        types.LoggingType.INFO,
    ]:
      logging_utils.log_message(
          logging_options=opts, message='msg', type=level
      )
    assert (tmp_path / logging_utils.ERROR_LOGGING_FILE_NAME).exists()
    assert (tmp_path / logging_utils.WARNING_LOGGING_FILE_NAME).exists()
    assert (tmp_path / logging_utils.INFO_LOGGING_FILE_NAME).exists()
    merged_lines = (
        tmp_path / logging_utils.MERGED_LOGGING_FILE_NAME
    ).read_text().splitlines()
    assert len(merged_lines) == 3

  def test_hides_sensitive_query_record(self, tmp_path):
    opts = types.LoggingOptions(
        logging_path=str(tmp_path), hide_sensitive_content=True
    )
    qr = types.QueryRecord(prompt='secret')
    logging_utils.log_message(
        logging_options=opts, message='msg', type=types.LoggingType.INFO,
        query_record=qr,
    )
    content = (tmp_path / logging_utils.MERGED_LOGGING_FILE_NAME).read_text()
    assert 'secret' not in content
    assert _HIDDEN in content


class TestLogProxdashMessage:

  def test_routes_by_type(self, tmp_path):
    opts = types.LoggingOptions(logging_path=str(tmp_path))
    proxdash_opts = types.ProxDashOptions()
    for level in [
        types.LoggingType.ERROR,
        types.LoggingType.WARNING,
        types.LoggingType.INFO,
    ]:
      logging_utils.log_proxdash_message(
          logging_options=opts, proxdash_options=proxdash_opts,
          message='msg', type=level,
      )
    assert (tmp_path / logging_utils.ERROR_LOGGING_FILE_NAME).exists()
    assert (tmp_path / logging_utils.WARNING_LOGGING_FILE_NAME).exists()
    assert not (tmp_path / logging_utils.INFO_LOGGING_FILE_NAME).exists()
    assert len(
        (tmp_path / logging_utils.PROXDASH_LOGGING_FILE_NAME)
        .read_text().splitlines()
    ) == 3
    assert len(
        (tmp_path / logging_utils.MERGED_LOGGING_FILE_NAME)
        .read_text().splitlines()
    ) == 3
