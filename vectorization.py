"""
This file contains functions to vectorize features.
"""
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

FILE_NAME = "training_data_clean.csv"
FEATURE_B = "How likely are you to use this model for academic tasks?"
FEATURE_C = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
FEATURE_D = "Based on your experience, how often has this model given you a response that felt suboptimal?"
FEATURE_E = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
FEATURE_G = "How often do you expect this model to provide responses with references or supporting evidence?"
FEATURE_H = "How often do you verify this model's responses?"

#Text
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
    Replace df's FEATURE_B column's text entry with scalar in place
    :param df: pandas df representing training data
    :return: void
    """
    df[FEATURE_B] = df[FEATURE_B].apply(extract_rating)


def vectorize_C(df):
    """
    One-hot encode FEATURE_C into separate columns and drop the original.
    Mutates df in place (no return).
    :param df: pandas df representing training data
    :return: void
    """
    best_tasks_lists = process_multiselect(df[FEATURE_C], TARGET_TASKS)
    mlb_best = MultiLabelBinarizer(classes=TARGET_TASKS)
    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
    
    # Create a DataFrame with meaningful column names and aligned index
    new_cols = [f"best_{c}" for c in mlb_best.classes_]
    onehot = pd.DataFrame(best_tasks_encoded, columns=new_cols, index=df.index)
    
    # Replace the original column with the new one-hot columns
    df.drop(columns=[FEATURE_C], inplace=True)
    df[new_cols] = onehot.astype(int)


def vectorize_D(df):
    """
    Replace df's FEATURE_D column's text entry with scalar in place
    :param df: pandas df representing training data
    :return: void
    """
    df[FEATURE_D] = df[FEATURE_D].apply(extract_rating)


def vectorize_E(df):
    """
    One-hot encode FEATURE_E into separate columns and drop the original.
    Mutates df in place (no return).
    :param df: pandas df representing training data
    :return: void
    """
    suboptimal_tasks_lists = process_multiselect(df[FEATURE_E], TARGET_TASKS)
    mlb_subopt = MultiLabelBinarizer(classes=TARGET_TASKS)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)
    
    new_cols = [f"subopt_{c}" for c in mlb_subopt.classes_]
    onehot = pd.DataFrame(suboptimal_tasks_encoded, columns=new_cols, index=df.index)
    
    df.drop(columns=[FEATURE_E], inplace=True)
    df[new_cols] = onehot.astype(int)
    

def vectorize_G(df):
    """
    Replace df's FEATURE_G column's text entry with scalar in place
    :param df: pandas df representing training data
    :return: void
    """
    df[FEATURE_G] = df[FEATURE_G].apply(extract_rating)


def vectorize_H(df):
    """
    Replace df's FEATURE_H column's text entry with scalar in place
    :param df: pandas df representing training data
    :return: void
    """
    df[FEATURE_H] = df[FEATURE_H].apply(extract_rating)


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

    bow_df = pd.DataFrame(X, columns=[f"I_{w}" for w in vocab], index=df.index)
    df = pd.concat([df.drop(columns=[FEATURE_I]), bow_df], axis=1)
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

    bow_df = pd.DataFrame(X, columns=[f"I_{w}" for w in vocab], index=df.index)
    df = pd.concat([df.drop(columns=[FEATURE_I]), bow_df], axis=1)
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
