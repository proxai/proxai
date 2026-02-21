from dataclasses import asdict
from pprint import pprint

import proxai as px


def prompt_test():
  result = px.generate(
      prompt='When is the first galatasaray and fenerbahce?',
      provider_model=('openai', 'gpt-5.2'))
  pprint(result)
  print(result.result.error)
  print(result.result.error_traceback)


def messages_test():
  result = px.generate(
      messages=[{
          'role': 'user',
          'content': 'What is the date of the first galatasaray and fenerbahce match in 2026 season?'}],
      provider_model=('openai', 'gpt-5.2'))
  
  chat = result.query.chat.copy()
  chat.append(result.result.content)

  chat.append(
      px.Message(
          role=px.MessageRoleType.USER,
          content='Where this match will be played?'))
  result = px.generate(
      messages=chat,
      provider_model=('openai', 'gpt-5.2'))

  chat.append(result.result.content)
  pprint(chat.to_dict())


def fallback_test():
  result = px.generate(
      prompt='When is the first galatasaray and fenerbahce?',
      provider_model=('openai', 'gpt-5.2'),
      connection_options=px.ConnectionOptions(
          fallback_models=[('openai', 'non_existent_model')]))
  assert not result.connection.failed_fallback_models
  assert result.query.provider_model.provider_model_identifier == 'gpt-5.2'

  result = px.generate(
      prompt='When is the first galatasaray and fenerbahce?',
      provider_model=('openai', 'non_existent_model'),
      connection_options=px.ConnectionOptions(
          fallback_models=[('openai', 'gpt-5.2')]))
  assert result.connection.failed_fallback_models == [('openai', 'non_existent_model')]
  assert result.query.provider_model.provider_model_identifier == 'gpt-5.2'


def main():
  # simple_model_test()
  # messages_test()
  fallback_test()

if __name__ == '__main__':
  main()
