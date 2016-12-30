#!/usr/bin/env python3

""" Analyze the data to choose features for the learning algorithms.
"""

import sys
from collections import Counter

import constants
from listing_histogram import document_token_frequency
from listing_histogram import plot_histogram
from market_db import MarketDB
from word_util import create_new_best_tokenizer

def choose_tokens(documents, tokenizer, k=1000, max_token_fraction=1.0):
  """Returns a list of tuples [(id, count), ...] corresponding to the k most
     common token ids which make up less than max_token_fraction of the data,
     with their respective counts.
  """
  # Select a random subset of the posts to deterimine common tokens
  token_counts = document_token_frequency(documents, tokenizer)

  # Remove tokens that are too common or only appear once.
  #
  # Completely ignore tokens that are too short

  # Discard tokens that are too common to be useful
  total_tokens = sum(token_counts.values())
  max_count = max_token_fraction * total_tokens
  filtered_token_counts = Counter(dict(
      p for p in token_counts.items() if 1 < p[1] <= max_count)).most_common(k)

  return filtered_token_counts

def main(argv):
  database_path = argv[0] if len(argv) > 0 else constants.DEFAULT_DATABASE_PATH

  with MarketDB(database_path) as db:
    # TODO(mgraczyk): This is now broken.
    tokens = choose_tokens(db.select_listings(), create_new_best_tokenizer())
    for token in tokens.most_common(100):
      print("{},{}".format(token[0], token[1]))
    print("Showing 100 / {} tokens.".format(len(tokens)))

  return 0

if __name__ == "__main__":
  exit(main(sys.argv[1:]))
