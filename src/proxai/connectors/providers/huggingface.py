import functools

from huggingface_hub import InferenceClient

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.huggingface_mock as huggingface_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
ResponseFormatConfigType = types.ResponseFormatConfigType


class HuggingFaceConnector(model_connector.ProviderModelConnector):
  """Connector for Hugging Face Inference API models.

  Hugging Face exposes an OpenAI-compatible chat router via
  `huggingface_hub.InferenceClient(provider='auto', ...)`. We use a single
  `chat.completions.create` endpoint that covers text, JSON, and Pydantic
  outputs. The router does not expose a generic `reasoning_effort` knob, so
  `thinking` is BEST_EFFORT (silently dropped). Reasoning models that emit
  chain-of-thought via the `message.reasoning` field are still surfaced as
  THINKING blocks. There is no native web search tool.
  """

  def init_model(self):
    return InferenceClient(
        provider='auto',
        token=self.provider_token_value_map['HF_TOKEN'],
    )

  def init_mock_model(self):
    return huggingface_mock.HuggingFaceMock()

  PROVIDER_NAME = 'huggingface'

  PROVIDER_API_KEYS = ['HF_TOKEN']

  ENDPOINT_PRIORITY = [
      'chat.completions.create',
  ]

  ENDPOINT_CONFIG = {
      'chat.completions.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          # The HF router has no separate system field — system messages
          # ride at the head of the `messages` list as a `role=system` entry.
          add_system_to_messages=True,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              # The HF router rejects `n > 1` with a 422, so surface this as
              # NOT_SUPPORTED instead of silently dropping it.
              n=FeatureSupportType.NOT_SUPPORTED,
              # No generic `reasoning_effort` knob across HF providers.
              # Reasoning models still emit chain-of-thought on their own;
              # we just can't ask non-reasoning models to think harder.
              thinking=FeatureSupportType.BEST_EFFORT,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.NOT_SUPPORTED,
          ),
          response_format=ResponseFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              # Native `response_format={'type': 'json_object'}` is honored.
              json=FeatureSupportType.SUPPORTED,
              # Pydantic is upgraded to SUPPORTED via the `json_schema`
              # response_format. The router returns the JSON as a text string
              # (no `parsed` field), so the executor runs
              # `model_validate_json` itself before emitting the
              # PYDANTIC_INSTANCE block.
              pydantic=FeatureSupportType.SUPPORTED,
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
            create, max_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature)

      if query_record.parameters.stop is not None:
        if isinstance(query_record.parameters.stop, str):
          create = functools.partial(
              create, stop=[query_record.parameters.stop])
        else:
          create = functools.partial(
              create, stop=query_record.parameters.stop)

    needs_pydantic = (
        query_record.response_format is not None
        and query_record.response_format.type
        == types.ResponseFormatType.PYDANTIC)

    if query_record.response_format.type == types.ResponseFormatType.JSON:
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    if needs_pydantic:
      pydantic_class = query_record.response_format.pydantic_class
      create = functools.partial(
          create,
          response_format={
              'type': 'json_schema',
              'json_schema': {
                  'name': pydantic_class.__name__,
                  'schema': pydantic_class.model_json_schema(),
                  'strict': False,
              },
          })

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return result_record

    parsed = []
    message = response.choices[0].message

    # Some routed reasoning models (e.g., gpt-oss) expose chain-of-thought
    # via `message.reasoning`. Non-reasoning models leave it as None.
    reasoning = getattr(message, 'reasoning', None)
    if reasoning:
      parsed.append(
          message_content.MessageContent(
              type=message_content.ContentType.THINKING,
              text=reasoning,
          )
      )

    if needs_pydantic:
      instance = pydantic_class.model_validate_json(message.content)
      parsed.append(
          message_content.MessageContent(
              type=message_content.ContentType.PYDANTIC_INSTANCE,
              pydantic_content=message_content.PydanticContent(
                  class_name=pydantic_class.__name__,
                  class_value=pydantic_class,
                  instance_value=instance,
              ),
          )
      )
    else:
      parsed.append(
          message_content.MessageContent(
              type=message_content.ContentType.TEXT,
              text=message.content,
          )
      )

    result_record.content = parsed
    return result_record

  ENDPOINT_EXECUTORS = {
      'chat.completions.create': '_chat_completions_create_executor',
  }
