import numpy as np
from matplotlib import pyplot as plt
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

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

