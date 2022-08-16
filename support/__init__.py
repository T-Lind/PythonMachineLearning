import numpy as np
from numpy import ndarray
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.base import clone

"""
Path to datasets
"""
DATASETS = "C:\\Users\\zenith\\Documents\\MyDatasets\\"


def display_scores(scores):
    """
    Display the scores from an evaluation
    :param scores: the input score objecet to list after evaluation
    """
    print("Scores: ", scores)
    print("Mean:", scores.mean())
    print("Standard deviation", scores.std())


def stratified_shuffle_split(X, y, n_splits=5, random_state=42, train_size=0.8):
    """
    Split input data and labels
    :param train_size: The ratio of data to train
    :param random_state: the random state to use
    :param n_splits: the number of splits to perform
    :param X: The testing data
    :param y: The testing labels
    :return the ndarray of X & y training and testing data
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    sss = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state, train_size=train_size)

    for train_index, test_index in sss.split(X, y):
        X_train.append(X[train_index])
        X_test.append(X[test_index])
        y_train.append(y[train_index])
        y_test.append(y[test_index])

    return ndarray(X_train), ndarray(X_test), ndarray(y_train), ndarray(y_test)


def stratified_k_fold_validation(X, y, clf, n_splits=3):
    """
    Perform validation an test data given a classifier
    :param n_splits: The number of splits to perform on the validation, default 3
    :param X: The testing data
    :param y: The testing labels
    :param clf: The classifier
    """
    skfolds = StratifiedKFold(n_splits=n_splits)  # add shuffle=True if the dataset is not
    # already shuffled
    for train_index, test_index in skfolds.split(X, y):
        clone_clf = clone(clf)
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))
