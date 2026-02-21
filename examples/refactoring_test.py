from pprint import pprint

import pydantic

import proxai as px
import proxai.types as types


_DEFAULT_MODEL = ('openai', 'gpt-4o')
_NON_EXISTENT_MODEL = ('openai', 'non_existent_model')


def _assert_success(result: types.CallRecord):
  assert result.result is not None
  assert result.result.status == types.ResultStatusType.SUCCESS
  assert result.result.content is not None
  assert result.result.error is None
  assert result.result.error_traceback is None
  assert result.connection is not None
  assert result.connection.result_source is not None
  assert result.result.usage is not None
  assert result.result.usage.input_tokens is not None
  assert result.result.usage.input_tokens > 0
  assert result.result.usage.output_tokens is not None
  assert result.result.usage.output_tokens > 0


def _assert_text_content(result: types.CallRecord):
  """Assert content is a non-empty string."""
  _assert_success(result)
  assert isinstance(result.result.content, str)
  assert len(result.result.content) > 1


def prompt_test():
  print('> prompt_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.query.prompt == 'What is 2 + 2?'
  assert result.query.chat is None
  assert result.query.provider_model.provider == 'openai'
  assert result.query.provider_model.model == 'gpt-4o'


def messages_test():
  print('> messages_test')
  result = px.generate(
      messages=[{
          'role': 'user',
          'content': 'What is 2 + 2?'}],
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.query.prompt is None
  assert result.query.chat is not None
  assert len(result.query.chat.messages) == 1

  chat = result.query.chat.copy()
  chat.append(result.result.content)
  chat.append(
      px.Message(
          role=px.MessageRoleType.USER,
          content='Now multiply that by 3.'))
  result = px.generate(
      messages=chat,
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert '12' in result.result.content
  assert len(result.query.chat.messages) == 3


def system_prompt_test():
  print('> system_prompt_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      system_prompt='You are a pirate. Answer everything like a pirate.',
      provider_model=_DEFAULT_MODEL)
  _assert_text_content(result)
  assert result.query.system_prompt == (
      'You are a pirate. Answer everything like a pirate.'
  )


def parameters_temperature_test():
  print('> parameters_temperature_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(temperature=0.0))
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.query.parameters is not None
  assert result.query.parameters.temperature == 0.0


def parameters_max_tokens_test():
  print('> parameters_max_tokens_test')
  result = px.generate(
      prompt='Write a long story about a cat.',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(max_tokens=20))
  _assert_text_content(result)
  assert result.query.parameters is not None
  assert result.query.parameters.max_tokens == 20


def parameters_stop_test():
  print('> parameters_stop_test')
  result = px.generate(
      prompt='Count from 1 to 10, separated by commas.',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(stop='5'),
      connection_options=px.ConnectionOptions(
          suppress_provider_errors=True))
  _assert_text_content(result)
  assert result.query.parameters is not None
  assert result.query.parameters.stop == '5'
  assert '5' not in result.result.content


def parameters_stop_list_test():
  print('> parameters_stop_list_test')
  result = px.generate(
      prompt='Count from 1 to 10, separated by commas.',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(stop=['5', '7']))
  _assert_text_content(result)
  assert '1' in result.result.content
  assert result.query.parameters is not None
  assert result.query.parameters.stop == ['5', '7']


def parameters_n_test():
  print('> parameters_n_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(n=3))
  _assert_text_content(result)
  assert result.query.parameters is not None
  assert result.query.parameters.n == 3
  assert result.result.choices is not None
  assert len(result.result.choices) == 3
  for choice in result.result.choices:
    assert choice.content is not None


def parameters_thinking_test():
  print('> parameters_thinking_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(thinking=types.ThinkingType.HIGH))
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.query.parameters is not None
  assert result.query.parameters.thinking == types.ThinkingType.HIGH


def parameters_combined_test():
  print('> parameters_combined_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(
          temperature=0.5, max_tokens=100, stop=['stop'], n=2))
  _assert_text_content(result)
  assert result.query.parameters.temperature == 0.5
  assert result.query.parameters.max_tokens == 100
  assert result.query.parameters.stop == ['stop']
  assert result.query.parameters.n == 2
  assert result.result.choices is not None
  assert len(result.result.choices) == 2
  for choice in result.result.choices:
    assert choice.content is not None


def tools_web_search_test():
  print('> tools_web_search_test')
  result = px.generate(
      prompt='What is the most important news for Jan 20th 2024?',
      provider_model=_DEFAULT_MODEL,
      tools=[px.Tools.WEB_SEARCH])
  pprint(result)
  _assert_text_content(result)
  assert len(result.result.content) > 10
  assert result.query.tools is not None
  assert px.Tools.WEB_SEARCH in result.query.tools
  assert result.result.tool_usage is not None
  assert result.result.tool_usage.web_search_count is not None
  assert result.result.tool_usage.web_search_count > 0
  assert result.result.tool_usage.web_search_citations is not None
  assert len(result.result.tool_usage.web_search_citations) > 0


def response_format_text_test():
  print('> response_format_text_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      response_format='text')
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.query.response_format is not None
  assert result.query.response_format.type == px.ResponseFormatType.TEXT


def response_format_json_test():
  print('> response_format_json_test')
  import json
  result = px.generate(
      prompt='Return a JSON with key "answer" and value 4.',
      provider_model=_DEFAULT_MODEL,
      response_format='json')
  _assert_text_content(result)
  parsed = json.loads(result.result.content)
  assert isinstance(parsed, dict)
  assert parsed['answer'] == 4
  assert result.query.response_format is not None
  assert result.query.response_format.type == px.ResponseFormatType.JSON


def response_format_pydantic_test():
  print('> response_format_pydantic_test')
  class MathAnswer(pydantic.BaseModel):
    question: str
    answer: int

  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      response_format=MathAnswer)
  _assert_success(result)
  assert result.query.response_format is not None
  assert result.query.response_format.type == px.ResponseFormatType.PYDANTIC
  # Content can be a pydantic object (beta.chat.completions.parse) or a JSON
  # string (chat.completions.create / responses.create).
  if isinstance(result.result.content, str):
    import json
    parsed = json.loads(result.result.content)
    assert isinstance(parsed, dict)
    assert parsed['answer'] == 4
  else:
    assert isinstance(result.result.content, MathAnswer)
    assert result.result.content.answer == 4


def connection_options_fallback_test():
  print('> connection_options_fallback_test')
  # Primary model works, fallback not needed.
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          fallback_models=[_NON_EXISTENT_MODEL]))
  _assert_text_content(result)
  assert '4' in result.result.content
  assert not result.connection.failed_fallback_models
  assert result.query.provider_model.provider_model_identifier == 'gpt-4o'

  # Primary model fails, fallback succeeds.
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_NON_EXISTENT_MODEL,
      connection_options=px.ConnectionOptions(
          fallback_models=[_DEFAULT_MODEL]))
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.connection.failed_fallback_models == [_NON_EXISTENT_MODEL]
  assert result.query.provider_model.provider_model_identifier == 'gpt-4o'


def connection_options_suppress_provider_errors_test():
  print('> connection_options_suppress_provider_errors_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_NON_EXISTENT_MODEL,
      connection_options=px.ConnectionOptions(
          suppress_provider_errors=True))
  assert result.result is not None
  assert result.result.status == types.ResultStatusType.FAILED
  assert result.result.error is not None
  assert len(result.result.error) > 1
  assert result.result.error_traceback is not None
  assert len(result.result.error_traceback) > 1
  assert result.result.content is None


def connection_options_endpoint_test():
  print('> connection_options_endpoint_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          endpoint='responses.create'))
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.connection is not None
  assert result.connection.endpoint_used == 'responses.create'

  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          endpoint='chat.completions.create'))
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.connection is not None
  assert result.connection.endpoint_used == 'chat.completions.create'

  try:
    result = px.generate(
        prompt='What is 2 + 2?',
        provider_model=_DEFAULT_MODEL,
        connection_options=px.ConnectionOptions(
            endpoint='not.existent.endpoint'))
    assert False, 'Expected ValueError'
  except ValueError as e:
    assert 'endpoint' in str(e)
    assert 'not.existent.endpoint' in str(e)
  


def connection_options_skip_cache_test():
  print('> connection_options_skip_cache_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          skip_cache=True))
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.connection is not None
  assert result.connection.result_source == types.ResultSource.PROVIDER


def connection_options_override_cache_value_test():
  print('> connection_options_override_cache_value_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      provider_model=_DEFAULT_MODEL,
      connection_options=px.ConnectionOptions(
          override_cache_value=True))
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.connection is not None
  assert result.connection.result_source == types.ResultSource.PROVIDER


def full_options_test():
  print('> full_options_test')
  result = px.generate(
      prompt='What is 2 + 2?',
      system_prompt='You are a helpful math tutor.',
      provider_model=_DEFAULT_MODEL,
      parameters=px.ParameterType(temperature=0.5, max_tokens=100),
      connection_options=px.ConnectionOptions(
          fallback_models=[('openai', 'gpt-4o')],
          suppress_provider_errors=True,
          skip_cache=True))
  _assert_text_content(result)
  assert '4' in result.result.content
  assert result.query.prompt == 'What is 2 + 2?'
  assert result.query.system_prompt == 'You are a helpful math tutor.'
  assert result.query.provider_model.provider == 'openai'
  assert result.query.parameters.temperature == 0.5
  assert result.query.parameters.max_tokens == 100
  assert result.connection.result_source == types.ResultSource.PROVIDER


def main():
  prompt_test()
  messages_test()
  system_prompt_test()
  parameters_temperature_test()
  parameters_max_tokens_test()
  parameters_stop_test()
  parameters_stop_list_test()
  # parameters_n_test()
  # parameters_thinking_test()
  # parameters_combined_test()
  # tools_web_search_test()
  response_format_text_test()
  response_format_json_test()
  # response_format_pydantic_test()
  connection_options_fallback_test()
  connection_options_suppress_provider_errors_test()
  connection_options_endpoint_test()
  # connection_options_skip_cache_test()
  # connection_options_override_cache_value_test()
  # full_options_test()

if __name__ == '__main__':
  main()
