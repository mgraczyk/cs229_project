#!/usr/bin/env python3

""" Analyze the data to choose features for the learning algorithms.
"""

import sys
from collections import Counter

from message_histogram import message_word_frequency
from message_histogram import plot_histogram
from message_db import MessageDB

DEFAULT_DATABASE_PATH = 'data_backup.db'

def choose_words(db):
  # Select a random subset of the posts to deterimine common words
  posts = db.select_random_posts(2000000)
  word_counts = message_word_frequency(posts, 30)

  # Remove words that are too common or only appear once.
  #
  # Completely ignore words that are too short
  usable_words = dict(p for p in word_counts.items() if len(p[0]) > 1)

  # Discard words that are too common to be useful
  max_word_fraction = 0.001
  total_words = sum(usable_words.values())
  max_count = max_word_fraction * total_words
  filtered_word_counts = Counter(dict(p for p in usable_words.items() if 1 < p[1] <= max_count))

  # Plot the the usable words, indicating which are discarded for being too common.
  plot_histogram(filtered_word_counts)

  for word in filtered_word_counts.most_common(100):
    print("{},{}".format(word[0], word[1]))

  return filtered_word_counts

def main(argv):
  database_path = argv[0] if len(argv) > 0 else DEFAULT_DATABASE_PATH

  with MessageDB(database_path) as db:
    words = choose_words(db)
  return 0

if __name__ == "__main__":
  exit(main(sys.argv[1:]))
