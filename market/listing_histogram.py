#!/usr/bin/env python3

import matplotlib
import collections
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import operator
from word_util import create_new_best_tokenizer


def document_token_frequency(documents, tokenizer):
    counts = collections.Counter()
    for document in documents:
        counts.update(tokenizer.get_document_token_ids(document))

    return counts

def plot_histogram(counter):
    fig = plt.figure()
    ax  = plt.subplot(111)

    values = collections.OrderedDict(
            sorted(counter.items(), key=operator.itemgetter(1), reverse=True))
    y = 10*np.log10(np.fromiter(islice(values.values(), 0, 1000), dtype=np.int64))
    x = range(len(y))
    # xlabels = islice(values, 0, 10000)

    ax.bar(x, y, width=1)
    ax.set_xticks(x)
    # ax.set_xticklabels(xlabels, rotation=70)
    ax.set_xlabel("Word")
    ax.set_ylabel("Total Count (dB)")
    plt.show()

if __name__ == "__main__":
    counts = document_token_frequency([
        { "name": "test test test2" },
        { "name": "test2 test test3" }], create_new_best_tokenizer())

    plot_histogram(counts)
