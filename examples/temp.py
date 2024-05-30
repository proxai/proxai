from openai import OpenAI

openai = OpenAI()
response = openai.chat.completions.create(
    model='gpt-4',
    messages=[
        {'role': 'user',
         'content': 'Which company created you and what is your model name?'}
    ],
    max_tokens=100,)
print(response.choices[0].message.content)
