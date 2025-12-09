from typing import Any, Callable, Optional
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
        query_record.response_format.type ==
        types.ResponseFormatType.PYDANTIC and
        'response_format::pydantic' in
        self.provider_model_config.features.supported):
      return functools.partial(self.api.beta.chat.completions.parse)
    else:
      return functools.partial(self.api.chat.completions.create)

  def system_feature_mapping(
      self,
      query_function: Callable,
      system_message: Optional[str] = None) -> Callable:
    if system_message is None:
      return query_function
    messages = query_function.keywords.get('messages')
    if messages is None:
      raise Exception('Set messages parameter before adding system message.')
    messages.insert(0, {'role': 'system', 'content': system_message})
    return functools.partial(query_function, messages=messages)

  def _add_json_guidance_to_user_message(
      self,
      query_function: Callable):
    # NOTE: Some API's expects the JSON to be in the user message.
    # This is weird and proxai's workaround to add JSON guidance to the user
    # message.
    messages = query_function.keywords.get('messages')
    if messages is None:
      raise Exception('Set messages parameter before adding system message.')
    for message in messages:
      if message['role'] == 'user':
        if 'json' not in message['content']:
          message['content'] = (
              f'{message["content"]}\n\nYou must respond with valid JSON.')
        break
    return functools.partial(query_function, messages=messages)

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    query_function = self._add_json_guidance_to_user_message(query_function)
    return functools.partial(
        query_function,
        response_format={'type': 'json_object'})

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    return functools.partial(
        query_function,
        response_format=query_record.response_format.value)

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    return functools.partial(
        query_function,
        response_format=query_record.response_format.value.class_value)

  def _feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    # Note: Databricks uses OpenAI-compatible API with 'system', 'user', and
    # 'assistant' as roles.
    # Some parameters may not work as expected for some models. For example,
    # the system instruction doesn't have any effect on the completion for
    # databricks-dbrx-instruct. But the stop parameter works as expected for
    # this model. However, system instruction works for
    # databricks-llama-2-70b-chat.
    query_messages = []
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

    create = self.add_system_and_response_format_params(create, query_record)

    return create

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    return response.choices[0].message.content

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response.choices[0].message.content)

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response.choices[0].message.content)

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> Any:
    return response.choices[0].message.parsed

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self.format_response_from_providers(response, query_record)
