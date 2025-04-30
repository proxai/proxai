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

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return query_record

  def get_token_count(self, logging_record: types.LoggingRecord):
    return 100

  def get_query_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def get_response_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def generate_text_proc(self, query_record: types.QueryRecord):
    return "mock response"


class MockFailingProviderModelConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return "mock_failing_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return query_record

  def get_token_count(self, logging_record: types.LoggingRecord):
    return 100

  def get_query_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def get_response_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def generate_text_proc(self, query_record: types.QueryRecord):
    raise ValueError('Temp Error')


class MockSlowProviderModelConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return "mock_slow_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return query_record

  def get_token_count(self, logging_record: types.LoggingRecord):
    return 100

  def get_query_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def get_response_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def generate_text_proc(self, query_record: types.QueryRecord):
    time.sleep(120)
    return "mock response"
