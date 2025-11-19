"""
This file perform knn classification
"""
import numpy as np
import pandas as pd
import vectorization
import data_cleaning_and_split_refactored

FILE_NAME = "training_data_clean.csv"
FEATURE_B = "How likely are you to use this model for academic tasks?"
FEATURE_C = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
FEATURE_D = "Based on your experience, how often has this model given you a response that felt suboptimal?"
FEATURE_E = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
FEATURE_G = "How often do you expect this model to provide responses with references or supporting evidence?"
FEATURE_H = "How often do you verify this model's responses?"

THRESHOLD = 6

FEATURE_A = "In your own words, what kinds of tasks would you use this model for?"
FEATURE_F = "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?"
FEATURE_I = "When you verify a response from this model, how do you usually go about it?"

TEXT_COL = [FEATURE_A, FEATURE_F, FEATURE_I]
NUMERIC_COL = [FEATURE_B, FEATURE_D, FEATURE_G, FEATURE_H]

TARGET_TASKS = [
        'math computations',
        'data processing or analysis',
        'explaining complex concepts simply',
        'writing or editing essays/reports',
        'drafting professional text (e.g., emails, résumés)',
        'writing or debugging code',
        'converting content between formats (e.g., latex)',
        'brainstorming or generating creative ideas'
    ]

REMOVE_WORDS = {"a", "an", "and", "or", "do", "does", "be", "so", "by", "as", "if",
                "the", "they", "there", "that", "this", "would", "which", "where", "since", "so",
                "i", "you", "i've", "i'd", "i'm", "me", "my", "it", "it's", "its", "is", "are", "was", "were", "has", "have",
                "of", "for", "to", "in", "on", "at", "about", "into", "from",
                "model", "think"}

# Best euclidean : Threshold = 7; k = 38


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


def main():
    """
    main function
    :return: void
    """
    df = pd.read_csv(FILE_NAME)
    data_cleaning_and_split_refactored.remove_incomplete_row(df)
    df = data_cleaning_and_split_refactored.lower_casing(df)
    df = data_cleaning_and_split_refactored.clean_text_columns(df, TEXT_COL, REMOVE_WORDS)

    df_train, df_val, df_test = data_cleaning_and_split_refactored.data_split(df, 0.5, 0.25, 0.25)  # solve the information leak

    df, df_a, df_f, df_i = data_cleaning_and_split_refactored.clean_text_select_words(
        df, df_train, TEXT_COL, threshold=THRESHOLD
    )

    vectorization.vectorize_B(df)
    vectorization.vectorize_C(df)
    vectorization.vectorize_D(df)
    vectorization.vectorize_E(df)
    vectorization.vectorize_G(df)
    vectorization.vectorize_H(df)

    df = vectorization.vectorize_A(df, df_a)
    df = vectorization.vectorize_F(df, df_f)
    df = vectorization.vectorize_I(df, df_i)
    df_train, df_val, df_test = data_cleaning_and_split_refactored.data_split(df, 0.5, 0.25, 0.25, seed=1)
    data_cleaning_and_split_refactored.remove_student_id(df_train)
    data_cleaning_and_split_refactored.remove_student_id(df_val)
    data_cleaning_and_split_refactored.remove_student_id(df_test)
    x_train, y_train = data_cleaning_and_split_refactored.split_label(df_train)
    x_val, y_val = data_cleaning_and_split_refactored.split_label(df_val)
    x_test, y_test = data_cleaning_and_split_refactored.split_label(df_test)

    for k in range(1, 50):
        train_result = knn_predict(x_train, y_train, x_train, k=k)
        val_result = knn_predict(x_train, y_train, x_val, k=k)
        test_result = knn_predict(x_train, y_train, x_test, k=k)

        print(k)
        print("Train Acc", accuracy(y_train, train_result))
        print("Val Acc", accuracy(y_val, val_result))
        print("Test Acc", accuracy(y_test, test_result))


if __name__ == "__main__":
    main()
