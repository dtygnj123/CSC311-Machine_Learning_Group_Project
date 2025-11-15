"""
This file perform knn classification
"""
import numpy as np

# Best manhattan : Threshold = 3; k = 49
# Best euclidean : Threshold = 3; k = 45


def euclidean_distance(x_train, x_query):
    """
    Compute pairwise (squared) Euclidean distances between query and train points.

    x_train: (n_train, d)
    x_query: (n_query, d)

    Returns:
        dists: (n_query, n_train) matrix where dists[i, j] is the
               squared Euclidean distance between x_query[i] and x_train[j].
    """
    x_train = np.asarray(x_train, dtype=float)
    x_query = np.asarray(x_query, dtype=float)

    # broadcasting: (n_query, 1, d) - (1, n_train, d) -> (n_query, n_train, d)
    diff = x_query[:, None, :] - x_train[None, :, :]
    dists_sq = np.sum(diff * diff, axis=2)   # (n_query, n_train)

    # For kNN, we don't need the square root; ranking is the same.
    return dists_sq


def manhattan_distance(x_train, x_query):
    """
    Compute pairwise Manhattan (L1) distances between query and train points.

    x_train: (n_train, d)
    x_query: (n_query, d)

    Returns:
        dists: (n_query, n_train) matrix where dists[i, j] is the
               L1 distance between x_query[i] and x_train[j].
    """
    x_train = np.asarray(x_train, dtype=float)
    x_query = np.asarray(x_query, dtype=float)

    diff = np.abs(x_query[:, None, :] - x_train[None, :, :])
    dists = np.sum(diff, axis=2)   # (n_query, n_train)

    return dists


def knn_predict(x_train, y_train, x_query, k, metric="euclidean"):
    """
    k-NN prediction for classification with selectable distance metric.

    Parameters:
        x_train: (n_train, d) training features
        y_train: (n_train,) training labels
        x_query: (n_query, d) query features to classify
        k: number of neighbors
        metric: "euclidean" or "manhattan"

    Returns:
        y_pred: (n_query,) predicted labels
    """
    x_train = np.asarray(x_train, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    y_train = np.asarray(y_train)

    # choose distance metric
    if metric == "euclidean":
        dists = euclidean_distance(x_train, x_query)
    elif metric == "manhattan":
        dists = manhattan_distance(x_train, x_query)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'manhattan'.")

    # indices of k nearest neighbors for each query point
    nn_idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]

    # majority vote
    y_pred = []
    for row in nn_idx:
        labels = y_train[row]
        values, counts = np.unique(labels, return_counts=True)
        y_pred.append(values[np.argmax(counts)])

    return np.array(y_pred)


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Parameters:
        y_true: iterable of true labels
        y_pred: iterable of predicted labels

    Returns:
        float: proportion of correct predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return np.mean(y_true == y_pred)
