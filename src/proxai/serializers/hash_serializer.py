import proxai.types as types


def get_query_record_hash(query_record: types.QueryRecord) -> str:
  POWER = 65535
  PRIME = 110589769251386569
  signature_str = ''
  if query_record.call_type != None:
    signature_str += query_record.call_type + chr(255)
  if query_record.model != None:
    provider, provider_model = query_record.model
    signature_str += provider + chr(255)
    signature_str += provider_model + chr(255)
  if query_record.prompt != None:
    signature_str += query_record.prompt + chr(255)
  if query_record.system != None:
    signature_str += query_record.system + chr(255)
  if query_record.messages != None:
    for message in query_record.messages:
      signature_str += 'role:'+message['role'] + chr(255)
      signature_str += 'content:'+message['content'] + chr(255)
  if query_record.max_tokens != None:
    signature_str += str(query_record.max_tokens) + chr(255)
  if query_record.temperature != None:
    signature_str += str(query_record.temperature) + chr(255)
  if query_record.stop != None:
    signature_str += str(query_record.stop) + chr(255)
  hash_val = 0
  for char in signature_str:
    hash_val = (hash_val * POWER + ord(char)) % PRIME
  return str(hash_val)
