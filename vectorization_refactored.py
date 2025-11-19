"""
This file contains functions to vectorize features.
"""

import numpy as np
import pandas as pd
import re

FILE_NAME = "training_data_clean.csv"
FEATURE_B = "How likely are you to use this model for academic tasks?"
FEATURE_C = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
FEATURE_D = "Based on your experience, how often has this model given you a response that felt suboptimal?"
FEATURE_E = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
FEATURE_G = "How often do you expect this model to provide responses with references or supporting evidence?"
FEATURE_H = "How often do you verify this model's responses?"
FEATURE_A = "In your own words, what kinds of tasks would you use this model for?"
FEATURE_F = "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?"
FEATURE_I = "When you verify a response from this model, how do you usually go about it?"

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


def process_multiselect(series, target_tasks):
    """Convert multiselect strings to lists"""
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            # Check which of the target tasks are present in the response
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing responses
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None
    

def vectorize_B(df):
    """
    Replace df's FEATURE_B column's text entry with scalar in place and normalize by dividing 5
    :param df: pandas df representing training data
    :return: void
    """
    df[FEATURE_B] = df[FEATURE_B].apply(extract_rating)
    df[FEATURE_B] = df[FEATURE_B] / 5
    mean_val = df[FEATURE_B].mean()
    df.fillna({FEATURE_B: mean_val}, inplace=True)


def vectorize_C(df):
    """
    One-hot encode FEATURE_C into separate columns and drop the original.
    Mutates df in place (no return).
    :param df: pandas df representing training data
    :return: void
    """
    # Convert each response into a list of selected target tasks
    best_tasks_lists = process_multiselect(df[FEATURE_C], TARGET_TASKS)

    # Manually one-hot encode
    for task in TARGET_TASKS:
        new_col = f"best_{task}"
        df[new_col] = [1 if task in tasks else 0 for tasks in best_tasks_lists]

    # Drop original text column
    df.drop(columns=[FEATURE_C], inplace=True)
    

def vectorize_D(df):
    """
    Replace df's FEATURE_D column's text entry with scalar in place and normalize it by dividing 5
    :param df: pandas df representing training data
    :return: void
    """
    df[FEATURE_D] = df[FEATURE_D].apply(extract_rating)
    df[FEATURE_D] = df[FEATURE_D] / 5
    mean_val = df[FEATURE_D].mean()
    df.fillna({FEATURE_D: mean_val}, inplace=True)


def vectorize_E(df):
    """
    One-hot encode FEATURE_E into separate columns and drop the original.
    Mutates df in place (no return).
    :param df: pandas df representing training data
    :return: void
    """
    # Convert each response into a list of selected target tasks
    suboptimal_tasks_lists = process_multiselect(df[FEATURE_E], TARGET_TASKS)

    # Manually one-hot encode
    for task in TARGET_TASKS:
        new_col = f"subopt_{task}"
        df[new_col] = [1 if task in tasks else 0 for tasks in suboptimal_tasks_lists]

    # Drop original text column
    df.drop(columns=[FEATURE_E], inplace=True)
    

def vectorize_G(df):
    """
    Replace df's FEATURE_G column's text entry with scalar in place and normalize it by dividing 5
    :param df: pandas df representing training data
    :return: void
    """
    df[FEATURE_G] = df[FEATURE_G].apply(extract_rating)
    df[FEATURE_G] = df[FEATURE_G] / 5
    mean_val = df[FEATURE_G].mean()
    df.fillna({FEATURE_G: mean_val}, inplace=True)


def vectorize_H(df):
    """
    Replace df's FEATURE_H column's text entry with scalar in place and normalize it by dividing 5
    :param df: pandas df representing training data
    :return: void
    """
    df[FEATURE_H] = df[FEATURE_H].apply(extract_rating)
    df[FEATURE_H] = df[FEATURE_H] / 5
    mean_val = df[FEATURE_H].mean()
    df.fillna({FEATURE_H: mean_val}, inplace=True)


def vectorize_F(df, vocab_df):
    """
    Vectorize FEATURE_F using the vocabulary DataFrame
    produced by clean_text_select_words

    This ensures that the Bag-of-Words columns use exactly the same
    word set as the cleaned data pipeline.
    """
    if "word" not in vocab_df.columns:
        raise ValueError("Expected a column named 'word' in vocab_df.")
    vocab = vocab_df["word"].str.lower().tolist()

    if FEATURE_F not in df.columns:
        raise KeyError(f"Column '{FEATURE_F}' not found in DataFrame.")
    texts = df[FEATURE_F].fillna("").astype(str).str.lower()

    N = len(texts)
    V = len(vocab)
    X = np.zeros((N, V), dtype=int)
    vocab_index = {word: j for j, word in enumerate(vocab)}

    for i, text in enumerate(texts):
        for w in text.split():
            if w in vocab_index:
                X[i, vocab_index[w]] = 1

    bow_df = pd.DataFrame(X, columns=[f"F_{w}" for w in vocab], index=df.index)
    df = pd.concat([df.drop(columns=[FEATURE_F]), bow_df], axis=1)
    return df

def vectorize_A(df, vocab_df):
    """Vectorize FEATURE_A using its vocabulary DataFrame."""
    if "word" not in vocab_df.columns:
        raise ValueError("Expected a column named 'word' in vocab_df.")
    vocab = vocab_df["word"].str.lower().tolist()

    if FEATURE_A not in df.columns:
        raise KeyError(f"Column '{FEATURE_A}' not found in DataFrame.")
    texts = df[FEATURE_A].fillna("").astype(str).str.lower()

    N, V = len(texts), len(vocab)
    X = np.zeros((N, V), dtype=int)
    vocab_index = {word: j for j, word in enumerate(vocab)}

    for i, text in enumerate(texts):
        for w in text.split():
            if w in vocab_index:
                X[i, vocab_index[w]] = 1

    bow_df = pd.DataFrame(X, columns=[f"A_{w}" for w in vocab], index=df.index)
    df = pd.concat([df.drop(columns=[FEATURE_A]), bow_df], axis=1)
    return df


def vectorize_I(df, vocab_df):
    """Vectorize FEATURE_I using its vocabulary DataFrame."""
    if "word" not in vocab_df.columns:
        raise ValueError("Expected a column named 'word' in vocab_df.")
    vocab = vocab_df["word"].str.lower().tolist()

    if FEATURE_I not in df.columns:
        raise KeyError(f"Column '{FEATURE_I}' not found in DataFrame.")
    texts = df[FEATURE_I].fillna("").astype(str).str.lower()

    N, V = len(texts), len(vocab)
    X = np.zeros((N, V), dtype=int)
    vocab_index = {word: j for j, word in enumerate(vocab)}

    for i, text in enumerate(texts):
        for w in text.split():
            if w in vocab_index:
                X[i, vocab_index[w]] = 1

    bow_df = pd.DataFrame(X, columns=[f"I_{w}" for w in vocab], index=df.index)
    df = pd.concat([df.drop(columns=[FEATURE_I]), bow_df], axis=1)
    return df