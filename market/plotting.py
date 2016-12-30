import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IPython

def plot_confusion_matrix(cm, category_names, cmap=plt.cm.Blues):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    fig.colorbar(res)
    tick_marks = np.arange(len(category_names))
    plt.xticks(tick_marks, category_names, rotation=45)
    plt.yticks(tick_marks, category_names)
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    height = cm.shape[0]
    width = cm.shape[1]
    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    # IPython.embed()

# make some plots
def plot_accuracy(clf_names, scores):
  sns.plt.figure(figsize=(12, 8))
  ax = sns.barplot(clf_names, scores, palette="Blues_d", ci=None)
  for n, score in enumerate(scores):
      ax.annotate(
          s='{:.2f}'.format(abs(score)),
          xy=(n, score),
          ha='center',va='center',
          xytext=(0,10),
          textcoords='offset points',
      )
  ax.set_ylabel("Classification Accuracy")
