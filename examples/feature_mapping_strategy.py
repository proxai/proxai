import proxai as px


def simple_test():
  response = px.generate_text(
      'When is the Galatasary and Besiktas game for this season?',
      provider_model=('openai', 'gpt-5.1'),
      web_search=True)
  print(response)


if __name__ == '__main__':
  simple_test()
