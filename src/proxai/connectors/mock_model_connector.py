import proxai.types as types
import proxai.connectors.model_connector as model_connector


class MockModelConnector(model_connector.ModelConnector):
  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return query_record

  def _get_token_count(self, logging_record: types.LoggingRecord):
    return 100

  def _get_query_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def _get_response_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def _get_estimated_cost(self, logging_record: types.LoggingRecord):
    return 0.002

  def generate_text_proc(self, query_record: types.QueryRecord):
    return "mock response"


class MockFailingConnector(model_connector.ModelConnector):
  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return query_record

  def _get_token_count(self, logging_record: types.LoggingRecord):
    return 100

  def _get_query_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def _get_response_token_count(self, logging_record: types.LoggingRecord):
    return 50

  def _get_estimated_cost(self, logging_record: types.LoggingRecord):
    return 0.002

  def generate_text_proc(self, query_record: types.QueryRecord):
    raise ValueError('Temp Error')
