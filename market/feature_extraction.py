
import numpy as np
import scipy.sparse

import constants
from choose_title_word_features import choose_tokens

def get_token_totals_from_documents(documents, tokenizer):
  """Returns (token_ids, token_totals).

     Outputs:
      token_ids: list of token ids which correspond to the featuer vector.
      token_totals: NxM matrix of N documents and M tokens, where each entry
                    ij is the number of times token j appeared in message i.
  """
  num_listings = len(documents)

  token_counts = choose_tokens(documents, tokenizer, 200, 0.0001)
  token_ids = [p[0] for p in token_counts]

  # Maps token ids to indices in the token feature vector.
  token_idxs = dict(
      (token_id, idx) for idx, token_id in enumerate(token_ids))

  num_features = len(token_ids)

  token_totals = scipy.sparse.lil_matrix((num_listings, num_features), dtype=np.int16)
  for listing_idx, document in enumerate(documents):
    listing_token_ids = tokenizer.get_document_token_ids(document)
    for token_id in listing_token_ids:
      token_idx = token_idxs.get(token_id)
      if token_idx is not None:
        token_totals[listing_idx, token_idx] += 1

  return token_ids, token_totals
