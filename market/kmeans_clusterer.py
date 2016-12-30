#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.decomposition import *

import constants
from feature_extraction import get_token_totals_from_listings
from market_db import MarketDB
from word_util import create_new_best_tokenizer
from IPython import embed


def do_kmeans(database_path):
  with MarketDB(database_path) as db:
    listings = db.select_distinct_columns("name", 10000)

  tokenizer = create_new_best_tokenizer()
  (tokens, token_totals) = get_token_totals_from_listings(listings, tokenizer)
  pca = IncrementalPCA(n_components=4, whiten=False, batch_size=100)

  print("num_listings = {}".format(token_totals.shape[0]))
  print("num_features = {}".format(token_totals.shape[1]))

  dense_token_totals = token_totals.toarray()
  pca.fit(dense_token_totals)
  print(pca.components_)
  embed()


def main(argv):
  database_path = argv[1] if len(argv) > 1 else constants.DEFAULT_DATABASE_PATH
  do_pca(database_path)
  return 0

if __name__ == "__main__":
  exit(main(sys.argv))
