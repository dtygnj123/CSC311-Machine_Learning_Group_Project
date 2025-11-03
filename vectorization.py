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

WORD_COUNT_FILE_G = "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal_word_counts.csv"
FEATURE_realG = "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?"
N_WORDS_AFTER_CODING = 100

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


def vectorize_realG(df):
    """
    Vectorize FEATURE_realG (column G) manually into a bag-of-words matrix.
    df: pandas DataFrame containing FEATURE_realG column.
    N_WORDS_AFTER_CODING: number of lines (of word in WORD_COUNT_FILE_G) to include below the 'coding' anchor word.
    If 'coding' is the first word, this takes the top n words.

    Reads WORD_COUNT_FILE_G for vocabulary selection.
    Finds the row containing 'coding' and includes 'n' words starting from it.
    Builds a bag-of-words binary matrix for FEATURE_realG.
    Replaces the original column with new BoW columns.
    """
    word_counts = pd.read_csv(WORD_COUNT_FILE_G)
    word_counts.columns = [c.lower() for c in word_counts.columns]
    word_col = word_counts.columns[0]

    coding_idx_list = word_counts[word_counts[word_col].str.lower() == "coding"].index
    if len(coding_idx_list) == 0:
        raise ValueError("'coding' not found in word-count file.")
    coding_idx = coding_idx_list[0]

    end_idx = min(coding_idx + N_WORDS_AFTER_CODING, len(word_counts))
    vocab = word_counts[word_col].iloc[coding_idx:end_idx].str.lower().tolist()

    print(f"Vocabulary size (hyperparameter): {len(vocab)} words starting from 'coding' (up to {N_WORDS_AFTER_CODING})")

    if FEATURE_realG not in df.columns:
        raise KeyError(f"Column '{FEATURE_realG}' not found in DataFrame.")
    texts = df[FEATURE_realG].fillna("").astype(str).str.lower()

    N = len(texts)
    V = len(vocab)
    X = np.zeros((N, V), dtype=int)
    vocab_index = {word: j for j, word in enumerate(vocab)}

    for i, text in enumerate(texts):
        words = text.split()
        for w in words:
            if w in vocab_index:
                X[i, vocab_index[w]] = 1

    bow_df = pd.DataFrame(X, columns=[f"G_{w}" for w in vocab], index=df.index)
    df.drop(columns=[FEATURE_realG], inplace=True)
    df[bow_df.columns] = bow_df

