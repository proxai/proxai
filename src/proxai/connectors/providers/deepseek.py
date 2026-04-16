import functools

from openai import OpenAI

import proxai.connectors.provider_connector as provider_connector
import proxai.connectors.providers.deepseek_mock as deepseek_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
ResponseFormatConfigType = types.ResponseFormatConfigType


class DeepSeekConnector(provider_connector.ProviderConnector):
  """Connector for DeepSeek models.

  DeepSeek exposes an OpenAI-compatible HTTP API, so we reuse the official
  `openai` SDK with a base_url override. Only the `chat.completions.create`
  endpoint is implemented — DeepSeek does not currently expose a native
  structured-outputs parse endpoint, web search tool, or `n > 1` sampling.
  """

  def init_model(self):
    return OpenAI(
        api_key=self.provider_token_value_map['DEEPSEEK_API_KEY'],
        base_url='https://api.deepseek.com',
    )

  def init_mock_model(self):
    return deepseek_mock.DeepSeekMock()

  PROVIDER_NAME = 'deepseek'

  PROVIDER_API_KEYS = ['DEEPSEEK_API_KEY']

  ENDPOINT_PRIORITY = [
      'chat.completions.create',
  ]

  ENDPOINT_CONFIG = {
      'chat.completions.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          add_system_to_messages=True,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              # The DeepSeek API rejects `n > 1` with a 400 error, so we
              # surface this as NOT_SUPPORTED rather than silently dropping it.
              n=FeatureSupportType.NOT_SUPPORTED,
              # DeepSeek has no `reasoning_effort` knob. The deepseek-reasoner
              # model always emits reasoning_content; deepseek-chat never does.
              # BEST_EFFORT lets the framework drop the parameter silently.
              thinking=FeatureSupportType.BEST_EFFORT,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.NOT_SUPPORTED,
          ),
          response_format=ResponseFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
      ),
  }

  def _chat_completions_create_executor(
      self,
      query_record: types.QueryRecord) -> types.ResultRecord:
    create = functools.partial(self.api.chat.completions.create)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      create = functools.partial(
          create, messages=[
              {'role': 'user', 'content': query_record.prompt}])

    if query_record.chat is not None:
      create = functools.partial(
          create, messages=query_record.chat['messages'])

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        create = functools.partial(
            create, max_completion_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature)

      if query_record.parameters.stop is not None:
        create = functools.partial(create, stop=query_record.parameters.stop)

    # DeepSeek's chat.completions.create supports a native JSON object mode.
    # Pydantic stays at BEST_EFFORT (no native parse endpoint), but we can
    # still flip on json_object so the framework's downstream
    # json.loads + model_validate is reliable.
    if query_record.response_format.type in (
        types.ResponseFormatType.JSON,
        types.ResponseFormatType.PYDANTIC,
    ):
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    parsed = []
    message = response.choices[0].message

    # deepseek-reasoner exposes its chain of thought via `reasoning_content`.
    # deepseek-chat leaves the attribute unset / None.
    reasoning = getattr(message, 'reasoning_content', None)
    if reasoning:
      parsed.append(
          message_content.MessageContent(
              type=message_content.ContentType.THINKING,
              text=reasoning,
          )
      )

    parsed.append(
        message_content.MessageContent(
            type=message_content.ContentType.TEXT,
            text=message.content,
        )
    )

    result_record.content = parsed
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  ENDPOINT_EXECUTORS = {
      'chat.completions.create': '_chat_completions_create_executor',
  }
