import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import IPython

def do_precision_recall(classifier, labels_encoder, X_train, y_train, X_test, y_test):
  class_names = labels_encoder.classes_
  n_classes = len(class_names)	

  # Binarize the labels
  y_train = label_binarize(y_train, classes=np.arange(n_classes))
  y_test = label_binarize(y_test, classes=np.arange(n_classes))

  # Run classifier
  classifier = OneVsRestClassifier(classifier)
  y_score = classifier.fit(X_train, y_train).decision_function(X_test)

  # Compute Precision-Recall and plot curve
  precision = dict()
  recall = dict()
  average_precision = dict()
  for i in range(n_classes):
      precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                          y_score[:, i])
      average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

  # Compute micro-average ROC curve and ROC area
  precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
      y_score.ravel())
  average_precision["micro"] = average_precision_score(y_test, y_score,
                                                      average="micro")

  # Plot Precision-Recall curve
  plt.clf()
  plt.plot(recall[0], precision[0], label='Precision-Recall curve')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision[0]))
  plt.legend(loc="lower left")
  plt.savefig("paper/plots/prc_one.png", format="png", dpi=200)

  # Plot Precision-Recall curve for each class
  plt.clf()
  plt.plot(recall["micro"], precision["micro"],
          label='micro-average Precision-recall curve (AUC = {0:0.2f})'
                ''.format(average_precision["micro"]))
  for i, class_name in enumerate(class_names):
      plt.plot(recall[i], precision[i],
              label='PR of {0} (AUC = {1:0.2f})'
                    ''.format(class_name, average_precision[i]))

  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall For Each Category')
  plt.legend(loc="best")
  plt.savefig("paper/plots/prc_all.png", format="png", dpi=200)
