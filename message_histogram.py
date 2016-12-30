#!/usr/bin/env python3

import matplotlib
matplotlib.use('WebAgg')

import collections
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import operator


def message_word_frequency(posts, cutoff=30):
    counts = collections.Counter()
    for post in posts:
        counts.update([word for word in post["message"].split() if len(word) < cutoff])

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
    counts = message_word_frequency([
        { "message": "test test test2" },
        { "message": "test2 test test3" }])

    plot_histogram(counts)
