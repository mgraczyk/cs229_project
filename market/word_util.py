
class Tokenizer(object):
  """Tokenizer automatically assigns new token ids to new tokens"""
  def get_document_token_ids(self, document):
    raise NotImplementedError()

  def get_token_id(self, token):
    raise NotImplementedError()

  def get_token_from_id(self, token_id):
    raise NotImplementedError()

class IDDictionary(dict):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._counter = 0

  def __missing__(self, key):
    new_token_id = self._counter
    self[key] = new_token_id
    self._counter += 1
    return new_token_id

class WordTokenizer(Tokenizer):
  def __init__(self):
    super().__init__()
    self._token_ids = IDDictionary()
    self._tokens_by_id = {}

  def get_document_token_ids(self, document):
    # TODO(mgraczyk): Tokenize more intelligently. This will currently consider
    #                 "word!" different from "word"
    return map(self._get_token_id_impl,
               filter(self.is_valid_token, document.split()))

  def get_token_id(self, token):
    return (self._token_ids[token]
            if self.is_valid_token(token)
            else None)

  def get_token_from_id(self, token_id):
    return self._tokens_by_id[token_id]

  def is_valid_token(self, token):
    return 1 < len(token) < 30

  def _get_token_id_impl(self, token):
    token_id = self._token_ids[token]
    self._tokens_by_id[token_id] = token
    return token_id

class NormalizedWordTokenizer(WordTokenizer):
  def get_document_token_ids(self, document):
    # TODO(mgraczyk): Tokenize more intelligently. This will currently consider
    #                 "word!" different from "word"
    return map(self._get_token_id_impl,
               filter(self.is_valid_token,
                      document.replace(".", "").replace(" 39 ", "'").lower().split()))

  def get_token_id(self, token):
    return (self._token_ids[token.replace(".", "").replace(" 39 ", "'").lower()]
            if self.is_valid_token(token)
            else None)

def create_new_best_tokenizer():
  return NormalizedWordTokenizer()
