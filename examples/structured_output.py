import json
import proxai as px
from pprint import pprint
from pydantic import BaseModel, Field

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



def all_user_options_test():
  px.set_model(generate_text=('openai', 'gpt-5.1'))

  # print('============== NO RESPONSE FORMAT ==============')
  # result = px.generate_text(TEST_PROMPT)
  # pprint(result)

  # print('============== JSON RESPONSE FORMAT ==============')
  # result = px.generate_text(
  #     TEST_PROMPT,
  #     response_format='json')
  # pprint(json.loads(result))

  # print('============== JSON SCHEMA RESPONSE FORMAT ==============')
  # result = px.generate_text(
  #     TEST_PROMPT,
  #     response_format=JSON_SCHEMA)
  # pprint(json.loads(result))

  print('============== PYDANTIC RESPONSE FORMAT ==============')
  result = px.generate_text(
      TEST_PROMPT,
      response_format=PayrollCalculation)
  pprint(result)

  # print('============== px.ResponseFormat FORMAT TEXT ==============')
  # result = px.generate_text(
  #     TEST_PROMPT,
  #     response_format=px.ResponseFormat(
  #       type=px.ResponseFormatType.TEXT))
  # pprint(result)

  # print('============== px.ResponseFormat FORMAT JSON ==============')
  # result = px.generate_text(
  #     TEST_PROMPT,
  #     response_format=px.ResponseFormat(
  #       type=px.ResponseFormatType.JSON))
  # pprint(result)

  # print('============== px.ResponseFormat FORMAT JSON SCHEMA ==============')
  # result = px.generate_text(
  #     TEST_PROMPT,
  #     response_format=px.ResponseFormat(
  #       type=px.ResponseFormatType.JSON_SCHEMA,
  #       value=JSON_SCHEMA))
  # pprint(result)

  # print('============== px.ResponseFormat FORMAT PYDANTIC ==============')
  # result = px.generate_text(
  #     TEST_PROMPT,
  #     response_format=px.ResponseFormat(
  #       type=px.ResponseFormatType.PYDANTIC,
  #       value=PayrollCalculation))
  # pprint(result)


def main():
  all_user_options_test()


if __name__ == '__main__':
  main()
