import proxai.types as types


def get_query_record_hash(query_record: types.QueryRecord) -> str:
  _PRIME_1 = 1000000007
  _PRIME_2 = 1000000009
  signature_str = ''
  if query_record.call_type is not None:
    signature_str += query_record.call_type + chr(255)
  if query_record.model is not None:
    provider, provider_model = query_record.model
    signature_str += provider + chr(255)
    signature_str += provider_model + chr(255)
  if query_record.max_tokens is not None:
    signature_str += str(query_record.max_tokens) + chr(255)
  if query_record.prompt is not None:
    signature_str += query_record.prompt + chr(255)
  hash_val = 0
  for char in signature_str:
    hash_val = (hash_val * _PRIME_1 + ord(char)) % _PRIME_2
  return str(hash_val)
