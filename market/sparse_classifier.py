#!/usr/bin/env python3

import logging
from collections import Counter
from collections import OrderedDict
import numpy as np
from optparse import OptionParser
import operator
import os
import sys
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.svm as svm
from sklearn.decomposition import *
from sklearn.preprocessing import Normalizer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.manifold import *
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import IPython

import constants
from market_db import MarketDB
from classify import ClassifierNoLearning
from precision_recall import do_precision_recall
import plotting
from util import pickle_save
from util import pickle_load

MODEL_PARAMETERS = {
  "max_df": 0.08,
  "r": 350,
  "C": 2.2,
  "alpha": 0.000162975,
  "svm_penalty": "l1",
  "gamma": 0,
}

def load_data(reload_from_db):
    if reload_from_db:
        print("Generating Documents")
        with MarketDB(constants.DEFAULT_DATABASE_PATH) as db:
            labeled_listings = db.select_labeled_listings(
                    "distinct name || ' ' || description as document, category")
            unlabeled_listings = db.select_columns(
                    "distinct name || ' ' || description as document")
        pickle_save(labeled_listings, "labeled_listings")
        pickle_save(unlabeled_listings, "unlabeled_listings")
    else:
        print("Loading Documents")
        labeled_listings = pickle_load("labeled_listings")
        unlabeled_listings = pickle_load("unlabeled_listings")

    return labeled_listings, unlabeled_listings


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--select_chi2",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--stage", type=int, default=0, help="stage to begin")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

# Stage 1, load data.
labeled_listings, unlabeled_listings = load_data(opts.stage <= 1)

print("{} labeled listings loaded".format(len(labeled_listings)))
print("{} unlabeled listings loaded".format(len(unlabeled_listings)))

labeled_documents = [l[0] for l in labeled_listings]
unlabeled_documents = [l[0] for l in unlabeled_listings]
label_names = [l[1] for l in labeled_listings]

labels_encoder = LabelEncoder()
labels = labels_encoder.fit_transform([l[1] for l in labeled_listings])
categories = labels_encoder.classes_

np.random.seed(0)
train_data, test_data, train_labels, test_labels = train_test_split(
    labeled_documents, labels, test_size=0.33, random_state=0)

pickle_save(train_labels, "train_labels")
pickle_save(test_labels, "test_labels")

print("%d documents (training set)" % len(train_data))
print("%d documents (test set)" % len(test_data))
print("%d categories" % len(categories))
print()

# split a training set and a test set

def extract_features(recompute, unlabeled, train, test):
    if recompute:
        print("Extracting features from the training data using a sparse vectorizer")
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=MODEL_PARAMETERS["max_df"],
                                    stop_words='english')
        X_unlabeled = vectorizer.fit_transform(unlabeled)
        X_train = vectorizer.transform(train)
        print()

        print("Extracting features from the test data using the same vectorizer")
        X_test = vectorizer.transform(test)
        print("n_samples: %d, n_features: %d" % X_test.shape)
        print()

        # mapping from integer feature name to original token string
        feature_names = vectorizer.get_feature_names()

        pickle_save(X_unlabeled, "X_unlabeled")
        pickle_save(X_train, "X_train")
        pickle_save(X_test, "X_test")
        pickle_save(feature_names, "vectorizer_feature_names")
    else:
        print("Loading extracted features")
        X_unlabeled = pickle_load("X_unlabeled")
        X_train = pickle_load("X_train")
        X_test = pickle_load("X_test")
        feature_names = pickle_load("vectorizer_feature_names")

    return X_unlabeled, X_train, X_test, feature_names

X_unlabeled, X_train, X_test, feature_names = extract_features(
        opts.stage <= 2, unlabeled_documents, train_data, test_data)

if opts.select_chi2:
    if opts.stage <= 3:
        print("Extracting %d best features by a chi-squared test" %
            opts.select_chi2)
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train_chi = ch2.fit_transform(X_train, train_labels)
        X_unlabeled_chi = ch2.transform(X_unlabeled)
        X_test_chi = ch2.transform(X_test)

        feature_names = [feature_names[i] for i
                        in ch2.get_support(indices=True)]

        pickle_save(X_unlabeled_chi, "X_unlabeled_chi")
        pickle_save(X_train_chi, "X_train_chi")
        pickle_save(X_test_chi, "X_test_chi")
        pickle_save(feature_names, "feature_names")
    else:
        print("Loading chi-squared best features")
        # TODO(mgraczyk): Unpickling changes these values for some reason.
        X_unlabeled_chi = pickle_load("X_unlabeled_chi")
        X_train_chi = pickle_load("X_train_chi")
        X_test_chi = pickle_load("X_test_chi")
        feature_names = pickle_load("feature_names")
else:
    X_train_chi = X_train
    X_unlabeled_chi = X_unlabeled
    X_test_chi = X_test

# embedding = SpectralEmbedding()
# Y = embedding.fit_transform(X_train.toarray())
# plt.subplot(121)
# sns.jointplot(Y[:, 0], Y[:, 1], c=train_labels)
# plt.subplot(122)
# Y = embedding.fit_transform(X_train_chi.toarray())
# sns.jointplot(Y[:, 0], Y[:, 1], c=train_labels)
# sns.plt.show()

def select_unsupervised_features(
        recompute, num_features, unlabeled, train, test):
    if recompute:
        print("Performing dimensionality reduction using LSA")
        dimensionality_reducer = TruncatedSVD(num_features)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(dimensionality_reducer, normalizer)

        # dimensionality_reducer = MDS(opts.r)
        # normalizer = Normalizer(copy=False)
        # X = normalizer.fit_transform(
                # dimensionality_reducer.fit_transform(X.toarray()))

        # explained_variance = dimensionality_reducer.explained_variance_ratio_.sum()
        # print("Explained variance of the dimensionality reduction step: {}%".format(
            # int(explained_variance * 100)))
        # print()

        unlabeled_transformed = lsa.fit_transform(unlabeled)
        train_transformed = lsa.transform(train)
        test_transformed = lsa.transform(test)

        pickle_save(unlabeled_transformed, "unlabeled_transformed")
        pickle_save(train_transformed, "train_transformed")
        pickle_save(test_transformed, "test_transformed")
    else:
        print("Loading Dimensionality Reduction")
        # TODO(mgraczyk): Unpickling changes these values for some reason.
        unlabeled_transformed = pickle_load("unlabeled_transformed")
        train_transformed = pickle_load( "train_transformed")
        test_transformed = pickle_load("test_transformed")

    return (unlabeled_transformed, train_transformed, test_transformed)

(X_unlabeled_unsupervised,
  X_train_unsupervised,
  X_test_unsupervised) = select_unsupervised_features(
    opts.stage <= 4, MODEL_PARAMETERS["r"], X_unlabeled, X_train, X_test)

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


###############################################################################
# Benchmark classifiers
def benchmark(clf, training_data, training_labels, testing_data, testing_labels):
    print(clf)
    clf.fit(training_data, training_labels)

    pred = clf.predict(testing_data)

    try:
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()
    except Exception as e:
        print(e)

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(testing_labels, pred,
                                            target_names=categories))

    return pred


print("Training Models")
results = OrderedDict()

def new_svm_classifier():
    # return svm.SVC(
        # kernel="linear",
        # C=MODEL_PARAMETERS["C"], max_iter=-1, random_state=1,
        # gamma=MODEL_PARAMETERS["gamma"],
        # decision_function_shape="ovr")
    return SGDClassifier(
        alpha=MODEL_PARAMETERS["alpha"], n_iter=100,
        penalty=MODEL_PARAMETERS["svm_penalty"], n_jobs=1, random_state=2)

if opts.stage <= 5:
  param_grid = { "alpha": np.logspace(-5, -3, 20) }
  clf = GridSearchCV(new_svm_classifier(), param_grid, cv=3, n_jobs=-1)
  clf.fit(X_train_unsupervised, train_labels)
  print("Best Parameters")
  print(clf.best_params_)
  MODEL_PARAMETERS["alpha"] = clf.best_params_["alpha"]
  alpha_vals = [params.parameters["alpha"] for params in clf.grid_scores_]
  accuracy_vals = [params.mean_validation_score for params in clf.grid_scores_]
  fig = plt.figure()
  plt.clf()
  ax = fig.add_subplot(111)
  ax.plot(alpha_vals, accuracy_vals)
  plt.ylabel('Cross Validation Accuracy')
  plt.xlabel('C (SVM Penalty Parameter)')
  plt.savefig("paper/plots/cv_accuracy.png", format="png", dpi=200)
  IPython.embed()

results["uSVM"] = (
    "SVM with Unsupervised Feat.",
    benchmark(new_svm_classifier(), X_train_unsupervised, train_labels,
              X_test_unsupervised, test_labels))

# Train SVM model
svm_classifier_supervised = SGDClassifier(
    alpha=0.001, n_iter=100,
    penalty=MODEL_PARAMETERS["svm_penalty"], n_jobs=-1, random_state=2)

results["SVM"] = (
		"SVM with Supervised Feat.",
		benchmark(svm_classifier_supervised, X_train_chi, train_labels,
							X_test_chi, test_labels))

# Train sparse Naive Bayes classifiers
results["NB"] = (
		"Multinomial Naive Bayes",
     benchmark(MultinomialNB(alpha=.01), X_train_chi, train_labels,
     					 X_test_chi, test_labels))

def score_no_learn_classifier(no_learn_classifier, label_encoder, x):
    name = "No Learning Classifier"
    predictions = labels_encoder.transform(
        list(map(no_learn_classifier.classify, x)))
    return name, predictions

no_learn_classifier = ClassifierNoLearning()
results["NL"] = score_no_learn_classifier(
    no_learn_classifier,
    test_data,
    categories[test_labels])

clf_names = [x[0] for x in results.values()]
accuracies = [metrics.accuracy_score(test_labels, x[1]) for x in results.values()]
for name, accuracy in zip(clf_names, accuracies):
    print('=' * 80)
    print(name)
    print("accuracy:   %0.3f" % accuracy)
    print('=' * 80)
  

def plot_confusion_matrices(results, categories, test_labels):
    cm = metrics.confusion_matrix(test_labels, results["uSVM"][1])
    plt.figure()
    plotting.plot_confusion_matrix(cm, categories)

def show_mismatches(results, test_labels, test_data):
  predicted_labels = results["uSVM"][1]
  mismatched_indices = np.argwhere(predicted_labels != test_labels)
  mispredicted_label_names = categories[predicted_labels[mismatched_indices]]
  true_label_names = categories[predicted_labels[mismatched_indices]]
  for mismatched_idx in mismatched_indices:
    print("Mispredicted. Was {}, got {}".format(
      categories[test_labels[mismatched_idx]],
      categories[predicted_labels[mismatched_idx]]))
    print(test_data[mismatched_idx])
    print()

os.makedirs("paper/plots", exist_ok=True)

do_precision_recall(new_svm_classifier(), labels_encoder,
    X_train_unsupervised, train_labels, X_test_unsupervised, test_labels)

# best_guess_labels = svm_classifier_unsupervised.predict(X_unlabeled_unsupervised)
# category_totals = Counter(best_guess_labels)

# f, ax = sns.plt.subplots()
# ax.pie(list(category_totals.values()), labels=list(category_totals.keys()))
# ax.set_title("Fraction of Items Listed By Category")
# ax.legend(loc='best')
# sns.plt.show()

plotting.plot_accuracy(clf_names, accuracies)
sns.plt.savefig("paper/plots/models_accuracy.png", format="png", dpi=200)

plot_confusion_matrices(results, categories, test_labels)
plt.savefig("paper/plots/confusion_matrix.png", format="png", dpi=200)
# show_mismatches(results, test_labels, test_data)
