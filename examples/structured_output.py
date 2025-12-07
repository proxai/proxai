import copy
import json
from pathlib import Path
import proxai as px
from pprint import pprint
from pydantic import BaseModel, Field

PROVIDER_MODEL = ('deepseek', 'deepseek-v3')

TEST_PROMPT = """
I need you to calculate the payroll costs for a small team.
Here is the scenario:
- 5 people are on the team.
- Each person works 40 hours per week.
- They are paid $50 per hour.

Please output the raw input data and the calculated total costs (weekly,
monthly, and annual) strictly adhering to the requested JSON structure.
Assume a month is 4 weeks and a year is 52 weeks.
"""

JSON_SCHEMA = {
  "type": "json_schema",
  "json_schema": {
    "name": "payroll_calculation",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "number_of_people": {
          "type": "integer",
          "description": "The count of employees."
        },
        "hours_per_week": {
          "type": "integer",
          "description": "Hours worked per person per week."
        },
        "hourly_rate": {
          "type": "integer",
          "description": "The hourly wage in dollars."
        },
        "total_weekly_cost": {
          "type": "integer",
          "description": "Calculated: people * hours * rate."
        },
        "total_monthly_cost": {
          "type": "number",
          "description": "Calculated: total_weekly_cost * 4.33"
        },
        "total_annual_cost": {
          "type": "integer",
          "description": "Calculated: total_weekly_cost * 52"
        }
      },
      "required": [
        "number_of_people",
        "hours_per_week",
        "hourly_rate",
        "total_weekly_cost",
        "total_monthly_cost",
        "total_annual_cost"
      ],
      "additionalProperties": False
    }
  }
}



class PayrollCalculation(BaseModel):
    number_of_people: int = Field(
        ...,
        description="The count of employees."
    )
    hours_per_week: int = Field(
        ...,
        description="Hours worked per person per week."
    )
    hourly_rate: int = Field(
        ...,
        description="The hourly wage in dollars."
    )
    total_weekly_cost: int = Field(
        ...,
        description="Calculated cost: people * hours * rate."
    )
    total_monthly_cost: float = Field(
        ...,
        description="Calculated cost: weekly_cost * 4.33."
    )
    total_annual_cost: int = Field(
        ...,
        description="Calculated cost: weekly_cost * 52."
    )



def test_response_format_options():
  px.set_model(generate_text=PROVIDER_MODEL)

  print('\n============== NO RESPONSE FORMAT ==============')
  result = px.generate_text(TEST_PROMPT)
  print(f'TYPE: {type(result)}')
  pprint(result)
  assert type(result) == str

  print('\n============== JSON RESPONSE FORMAT ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format='json')
  print(f'TYPE: {type(result)}')
  pprint(result)
  assert type(result) == dict

  print('\n============== JSON SCHEMA RESPONSE FORMAT ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=JSON_SCHEMA)
  print(f'TYPE: {type(result)}')
  pprint(result)
  assert type(result) == dict

  print('\n============== PYDANTIC RESPONSE FORMAT ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=PayrollCalculation)
  print(f'TYPE: {type(result)}')
  pprint(result)
  assert type(result) == PayrollCalculation

  print('\n============== px.ResponseFormat FORMAT TEXT ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=px.ResponseFormat(
        type=px.ResponseFormatType.TEXT))
  print(f'TYPE: {type(result)}')
  pprint(result)
  assert type(result) == str

  print('\n============== px.ResponseFormat FORMAT JSON ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=px.ResponseFormat(
        type=px.ResponseFormatType.JSON))
  print(f'TYPE: {type(result)}')
  pprint(result)
  assert type(result) == dict

  print('\n============== px.ResponseFormat FORMAT JSON SCHEMA ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=px.ResponseFormat(
        type=px.ResponseFormatType.JSON_SCHEMA,
        value=JSON_SCHEMA))
  print(f'TYPE: {type(result)}')
  pprint(result)
  assert type(result) == dict

  print('\n============== px.ResponseFormat FORMAT PYDANTIC ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=px.ResponseFormat(
        type=px.ResponseFormatType.PYDANTIC,
        value=px.ResponseFormatPydanticValue(
            class_name=PayrollCalculation.__name__,
            class_value=PayrollCalculation)))
  print(f'TYPE: {type(result)}')
  pprint(result)
  assert type(result) == PayrollCalculation


def test_cached_response():
  px.connect(
      cache_path=f'{Path.home()}/proxai_cache/',
      cache_options=px.CacheOptions(
          clear_query_cache_on_connect=True))
  px.set_model(generate_text=PROVIDER_MODEL)

  print('\n============== CACHED RESPONSE TEST FOR TEXT ==============')
  for i in range(3):
    result = px.generate_text(
        TEST_PROMPT,
        extensive_return=True)
    print(f'TYPE: {type(result)}')
    print(f'RESPONSE SOURCE: {result.response_source}')
    if i == 0:
      assert result.response_source == 'PROVIDER'
    else:
      assert result.response_source == 'CACHE'

  print('\n============== CACHED RESPONSE TEST FOR JSON ==============')
  for i in range(3):
    result = px.generate_text(
        TEST_PROMPT,
        response_format='json',
        extensive_return=True)
    print(f'TYPE: {type(result)}')
    print(f'RESPONSE SOURCE: {result.response_source}')
    if i == 0:
      assert result.response_source == 'PROVIDER'
    else:
      assert result.response_source == 'CACHE'

  print('\n============== CACHED RESPONSE TEST FOR JSON SCHEMA ==============')
  for i in range(3):
    result = px.generate_text(
        TEST_PROMPT,
        response_format=JSON_SCHEMA,
        extensive_return=True)
    print(f'TYPE: {type(result)}')
    print(f'RESPONSE SOURCE: {result.response_source}')
    if i == 0:
      assert result.response_source == 'PROVIDER'
    else:
      assert result.response_source == 'CACHE'

  print('\n============== CACHED RESPONSE TEST FOR PYDANTIC ==============')
  for i in range(3):
    result = px.generate_text(
        TEST_PROMPT,
        response_format=PayrollCalculation,
        extensive_return=True)
    print(f'TYPE: {type(result)}')
    print(f'RESPONSE SOURCE: {result.response_source}')
    if i == 0:
      assert result.response_source == 'PROVIDER'
    else:
      assert result.response_source == 'CACHE'

  print('\n============== CACHED RESPONSE TEST RESPONSE TYPE INVARIANT ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=px.ResponseFormat(
        type=px.ResponseFormatType.TEXT),
      extensive_return=True)
  print(f'TYPE: {type(result)}')
  print(f'RESPONSE SOURCE: {result.response_source}')
  assert result.response_source == 'CACHE'

  result = px.generate_text(
      TEST_PROMPT,
      response_format=px.ResponseFormat(
        type=px.ResponseFormatType.JSON),
      extensive_return=True)
  print(f'TYPE: {type(result)}')
  print(f'RESPONSE SOURCE: {result.response_source}')
  assert result.response_source == 'CACHE'

  result = px.generate_text(
      TEST_PROMPT,
      response_format=px.ResponseFormat(
        type=px.ResponseFormatType.JSON_SCHEMA,
        value=JSON_SCHEMA),
      extensive_return=True)
  print(f'TYPE: {type(result)}')
  print(f'RESPONSE SOURCE: {result.response_source}')
  assert result.response_source == 'CACHE'

  result = px.generate_text(
      TEST_PROMPT,
      response_format=px.ResponseFormat(
        type=px.ResponseFormatType.PYDANTIC,
        value=px.ResponseFormatPydanticValue(
            class_name=PayrollCalculation.__name__,
            class_value=PayrollCalculation)),
      extensive_return=True)
  print(f'TYPE: {type(result)}')
  print(f'RESPONSE SOURCE: {result.response_source}')
  assert result.response_source == 'CACHE'


def test_cached_response_with_modified_schemas():
  px.connect(
      cache_path=f'{Path.home()}/proxai_cache/',
      cache_options=px.CacheOptions(
          clear_query_cache_on_connect=True))
  px.set_model(generate_text=PROVIDER_MODEL)

  print('\n============== CHECK JSON SCHEMA MODIFICATION ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=JSON_SCHEMA,
      extensive_return=True)
  print(f'TYPE: {type(result)}')
  print(f'RESPONSE SOURCE: {result.response_source}')
  assert result.response_source == 'PROVIDER'

  json_schema_modified = copy.deepcopy(JSON_SCHEMA)
  json_schema_modified['json_schema']['schema']['properties'][
      'total_annual_cost']['description'] = 'Calculated cost: weekly_cost * 40.'
  result = px.generate_text(
      TEST_PROMPT,
      response_format=json_schema_modified,
      extensive_return=True)
  print(f'TYPE: {type(result)}')
  print(f'RESPONSE SOURCE: {result.response_source}')
  assert result.response_source == 'PROVIDER'

  print('\n============== CHECK PYDANTIC MODIFICATION ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=PayrollCalculation,
      extensive_return=True)
  print(f'TYPE: {type(result)}')
  print(f'RESPONSE SOURCE: {result.response_source}')
  assert result.response_source == 'PROVIDER'


  # Different class name but same schema
  class ModifiedPayrollCalculation(PayrollCalculation):
    pass

  result = px.generate_text(
      TEST_PROMPT,
      response_format=ModifiedPayrollCalculation,
      extensive_return=True)
  print(f'TYPE: {type(result)}')
  print(f'RESPONSE SOURCE: {result.response_source}')
  assert result.response_source == 'PROVIDER'

  # Same class name but different schema
  class PayrollCalculation2(PayrollCalculation):
    total_annual_cost: int = Field(
        ...,
        description="Calculated cost: weekly_cost * 40.")

  def _test_wrapper():
    class PayrollCalculation(PayrollCalculation2):
      pass

    result = px.generate_text(
        TEST_PROMPT,
        response_format=PayrollCalculation,
        extensive_return=True)
    print(f'TYPE: {type(result)}')
    print(f'RESPONSE SOURCE: {result.response_source}')
    assert result.response_source == 'PROVIDER'

  _test_wrapper()


def main():
  test_response_format_options()
  # test_cached_response()
  # test_cached_response_with_modified_schemas()


if __name__ == '__main__':
  main()
