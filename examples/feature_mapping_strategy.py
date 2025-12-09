import proxai as px
import proxai.types as px_types
from pydantic import BaseModel

PROMPT = f'A=5, B=10. Give me {{"A": A, "B": B, "C": A + B}}.'

MESSAGES = [
  {"role": "user", "content": PROMPT}
]

JSON_SCHEMA = {
  "json_schema": {
    "name": "SumOfNumbers",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "A": {
          "type": "integer",
          "description": "The value of A."
        },
        "B": {
          "type": "integer",
          "description": "The value of B."
        },
        "C": {
          "type": "number",
          "description": "Calculated: A + B"
        }
      },
      "required": ["A", "B", "C"],
      "additionalProperties": False
    }
  }
}

class SumOfNumbers(BaseModel):
  A: int
  B: int
  C: int

TEST_FEATURES = {
    'messages': MESSAGES,
    'system': 'You are a helpful assistant.',
    'max_tokens': 1000,
    'temperature': 0.5,
    # 'stop': '\n\n', # Fails for Claude
    'stop': 'STOP',
    'response_format::text': px.ResponseFormat(type=px.ResponseFormatType.TEXT),
    'response_format::json': px.ResponseFormat(type=px.ResponseFormatType.JSON),
    'response_format::json_schema': px.ResponseFormat(type=px.ResponseFormatType.JSON_SCHEMA, schema=JSON_SCHEMA),
    'response_format::pydantic': px.ResponseFormat(type=px.ResponseFormatType.PYDANTIC, schema=SumOfNumbers),
    'web_search': True,
  }


def simple_test():
  px.connect(
    feature_mapping_strategy=px_types.FeatureMappingStrategy.PASSTHROUGH)
  response = px.generate_text(
      PROMPT,
      provider_model=('claude', 'haiku-3'),
      response_format=TEST_FEATURES['response_format::json_schema'])
  print(response)


if __name__ == '__main__':
  simple_test()
