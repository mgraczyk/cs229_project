#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython
import sklearn
import sklearn.manifold
import sklearn.decomposition

import constants
from feature_extraction import get_token_totals_from_documents
from market_db import MarketDB
from word_util import create_new_best_tokenizer


def do_tsne(database_path):
  tokenizer = create_new_best_tokenizer()
  with MarketDB(database_path) as db:
    listings = db.select_random_listings("distinct name", 10000)
    documents = [l[0] for l in listings]

  (token_ids, token_totals) = get_token_totals_from_documents(documents, tokenizer)
  num_documents = token_totals.shape[0]
  num_features = token_totals.shape[1]
  print("num_documents = {}".format(num_documents))
  print("num_features = {}".format(num_features))

  # Normalize row values to represent fractions rather than totals.
  row_sums = token_totals.sum(axis=1)
  for i in range(num_documents):
    if row_sums[i] != 0:
      token_totals[i, :] /= row_sums[i]

  dense_token_totals = token_totals.toarray()

  # lsd_id = tokenizer.get_token_id("stimulant")
  # lsd_idx = [i for i in range(len(token_ids)) if token_ids[i] == lsd_id][0]
  # colors = [('b' if token_totals[i, lsd_idx] > 0 else 'r')
            # for i in range(token_totals.shape[0])]
  colors = "b"

  model = sklearn.manifold.TSNE(n_components=2, init='pca', random_state=0)
  Y = model.fit_transform(dense_token_totals.astype(np.float))
  # model = sklearn.manifold.LocallyLinearEmbedding(
      # n_neighbors=10, n_components=2, eigen_solver='auto', method='ltsa')
  # Y = model.fit_transform(dense_token_totals.astype(np.float))
  # model = sklearn.decomposition.MiniBatchSparsePCA(
      # n_components=2)
  # Y = model.fit_transform(dense_token_totals)

  plt.scatter(Y[:, 0], Y[:, 1], c=colors)
  plt.show()

def sklearn_tsne(database_path):
  tokenizer = create_new_best_tokenizer()
  with MarketDB(database_path) as db:
    listings = db.select_random_listings("distinct name", 10000)
    documents = [l[0] for l in listings]

  IPython.embed()


def main(argv):
  database_path = argv[1] if len(argv) > 1 else constants.DEFAULT_DATABASE_PATH
  sklearn_tsne(database_path)
  return 0

if __name__ == "__main__":
  exit(main(sys.argv))
