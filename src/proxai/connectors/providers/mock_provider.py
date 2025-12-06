import time
import proxai.types as types
import proxai.connectors.model_connector as model_connector


class MockProviderModelConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return "mock_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    return types.Response(
        value="mock response",
        type=types.ResponseType.TEXT)


class MockFailingProviderModelConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return "mock_failing_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def generate_text_proc(self, query_record: types.QueryRecord):
    raise ValueError('Temp Error')


class MockSlowProviderModelConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return "mock_slow_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    time.sleep(120)
    return types.Response(
        value="mock response",
        type=types.ResponseType.TEXT)
