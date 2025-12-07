from typing import Any, Callable
import functools
import json
from databricks.sdk import WorkspaceClient
import proxai.types as types
import proxai.connectors.providers.databricks_mock as databricks_mock
import proxai.connectors.model_connector as model_connector


class DatabricksConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'databricks'

  def init_model(self):
    w = WorkspaceClient()
    return w.serving_endpoints.get_open_ai_client()

  def init_mock_model(self):
    return databricks_mock.DatabricksMock()

  def _get_api_call_function(
      self,
      query_record: types.QueryRecord) -> Callable:
    # Use beta.chat.completions.parse for Pydantic models
    if (query_record.response_format is not None and
        query_record.response_format.type == types.ResponseFormatType.PYDANTIC):
      return functools.partial(self.api.beta.chat.completions.parse)
    else:
      return functools.partial(self.api.chat.completions.create)

  def _feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    # Note: Databricks uses OpenAI-compatible API with 'system', 'user', and
    # 'assistant' as roles.
    # Some parameters may not work as expected for some models. For example,
    # the system instruction doesn't have any effect on the completion for
    # databricks-dbrx-instruct. But the stop parameter works as expected for
    # this model. However, system instruction works for
    # databricks-llama-2-70b-chat.
    query_messages = []
    if query_record.system != None:
      query_messages.append({'role': 'system', 'content': query_record.system})
    if query_record.prompt != None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages != None:
      query_messages.extend(query_record.messages)
    create = functools.partial(create, messages=query_messages)

    if query_record.max_tokens != None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop != None:
      create = functools.partial(create, stop=query_record.stop)

    # Handle response format configuration
    if query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.TEXT:
        pass
      elif query_record.response_format.type == types.ResponseFormatType.JSON:
        create = functools.partial(
            create,
            response_format={'type': 'json_object'})
      elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
        create = functools.partial(
            create,
            response_format=query_record.response_format.value)
      elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
        create = functools.partial(
            create,
            response_format=query_record.response_format.value.class_value)

    return create

  def _response_mapping(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    # Handle response based on format type
    if query_record.response_format is None:
      return types.Response(
          value=response.choices[0].message.content,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value=response.choices[0].message.content,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value=self._extract_json_from_text(response.choices[0].message.content),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return types.Response(
          value=self._extract_json_from_text(response.choices[0].message.content),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=response.choices[0].message.parsed),
          type=types.ResponseType.PYDANTIC)

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self._response_mapping(response, query_record)
