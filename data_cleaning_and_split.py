"""
This file contains functions to clean and split data such that:
    All rows with empty entry or "NAME?" or nan are removed
    Lower casing all text
    Randomly partition data into 2:1:1 ratio corresponds to Training:Validation:Test
"""
from collections import Counter
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FILE_NAME = "training_data_clean.csv"


def remove_incomplete_row(df):
    """
    Remove rows that contain empty entry or "NAME?" or nan in any entry in place
    :param df: pandas df representing training data
    :return: void
    """
    df.replace(['', '#NAME?', np.nan], pd.NA, inplace=True)
    # df.dropna(inplace=True)


def lower_casing(df):
    """
    Lower casing all text entry
    :param df: pandas df representing training data
    :return: mutated df
    """
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
    return df


def data_split(df, train_size, val_size, test_size, seed=42):
    """
    Randomly partition data into specified size
    :param df: pandas df representing training data
    :param train_size: float, represent train_size in percentage
    :param val_size: float, represent val_size in percentage
    :param test_size: float, represent test_size in percentage
    :param seed: random state, default = 42
    :return: partitioned df_train, df_val, df_test in order
    """
    # First, split into train and temp
    df_train, df_temp = train_test_split(df, train_size=train_size, random_state=seed)

    # Then, split temp into validation and test
    df_val, df_test = train_test_split(df_temp, test_size=(test_size / (val_size + test_size)), random_state=seed)

    return df_train, df_val, df_test


def clean_text_columns(dataframe, columns, remove_words):
    """
    This cleaning step removes punctuation, adverbs, leaving only lower cased words separated by space.
    In this step, all the REMOVE_WORDS will be removed, then all the -s in the words will be removed.
    For words like "analysis" and "summaries", will be "analysi" and "summarie", but no negative effect.
    """
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(lambda x:normalize_and_clean_text(x, remove_words))
    return dataframe


def normalize_and_clean_text(s: str, remove_words) -> str:
    """
    Normalize text to lowercase words separated by spaces.
    Keeps hyphens and apostrophes as part of words.
    Preserves accented letters (e.g., é) as single characters.
    """
    if s is None:
        return s
    if not isinstance(s, str):
        s = str(s)

    # Fix mojibake / encoding issues first
    s = try_fix_mojibake(s)

    # clean the dashes, quotes
    s = s.replace('—', '-').replace('–', '-').replace('−', '-')
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = s.replace('“', '"').replace('”', '"')

    # mojibake artifacts
    s = s.replace('‚Äô', "'").replace('‚Äù', '"').replace('‚Äî', '-')

    # Remove any character that is not \w, \s, ', -
    s = re.sub(r"[^\w\s'\-]", " ", s, flags=re.UNICODE)

    # Remove underscores
    s = s.replace('_', ' ')

    # Collapse multiple spaces into single space, strip ends
    s = re.sub(r"\s+", " ", s).strip()



    # remove adverbs
    s = remove_adverbs(s)

    s = remove_selected_words(s, remove_words)

    # remove the plurals
    s = remove_plural_s(s)

    return s


def try_fix_mojibake(s: str) -> str:
    """
    Try to repair common mojibake (e.g., r√©sum√©s, haven‚Äôt).
    Uses ftfy if present; otherwise tries a latin-1 -> utf-8 re-decode trick.
    Returns original if fixes fail.
    """
    if not isinstance(s, str):
        return s

    # try latin-1 -> utf-8 re-decode trick
    try:
        candidate = s.encode('latin-1').decode('utf-8')
        if sum(ch.isalpha() for ch in candidate) >= sum(ch.isalpha() for ch in s):
            return candidate
    except Exception:
        pass

    return s


def remove_adverbs(text): # TODO: revise this adv removing method
    """
    Remove words ending with 'ly' (a simple heuristic for adverbs).
    Keeps the rest of the sentence structure.
    """
    if not isinstance(text, str):
        return text
    words = text.split()
    cleaned = [w for w in words if not w.endswith("ly")]
    return " ".join(cleaned)


def remove_selected_words(text, remove_words):
    """
    Remove any word that appears in the given set `remove_words`.
    Comparison is case-insensitive.
    Keeps the rest of the sentence structure.
    """
    if not isinstance(text, str):
        return text
    words = text.split()
    cleaned = [w for w in words if w.lower() not in remove_words]
    return " ".join(cleaned)


def remove_plural_s(text):
    """
    Remove the trailing 's' from every word that ends with 's'.
    Keeps the rest of the sentence structure intact.
    """
    if not isinstance(text, str):
        return text
    words = text.split()
    cleaned = [w[:-1] if w.endswith("s") else w for w in words] # there is no need to set a set of exceptions!
    return " ".join(cleaned)


def clean_text_select_words(dataframe, dataframe_train, columns, threshold):
    """
    For each column:
    - compute per-class word frequencies
    - compute variance across classes
    - keep top 'threshold' words with highest variance
    - filter dataframe texts to keep only those words
    """

    df_AFI = []

    labels = dataframe_train["label"]  # must contain "Gemini", "ChatGPT", "Claude"

    for col in columns:

        # NEW: get per-class counts + variance
        freq_df = get_word_counts_per_class(dataframe_train[col], labels)

        # Keep the top 'threshold' words with highest variance
        top_words = freq_df.head(threshold)

        # Convert to a set
        word_set = set(top_words["word"])

        # Filter the actual data column
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(
                lambda x: clean_text_select_word_helper(x, word_set)
            )

        df_AFI.append(top_words)

    # return 3 tables instead of 1 (you had df_AFI[0], df_AFI[1], df_AFI[2])
    return dataframe, df_AFI[0], df_AFI[1], df_AFI[2]


def clean_text_select_word_helper(s: str, word_set: set) -> str:
    """ Delete all the words that are not in the word_set (a set of strings). """
    if s is None:
        return s
    if not isinstance(s, str):
        s = str(s)

    s = clean_selected_words(s, word_set)

    return s


def get_word_counts_per_class(text_series, labels):
    """
    text_series : pandas series of text (strings)
    labels      : pandas series of class labels ("Gemini", "ChatGPT", "Claude")

    Returns: DataFrame with columns:
        word | count_gemini | count_chatgpt | count_claude | variance
    """

    # Initialize counters
    counter_gemini = Counter()
    counter_chatgpt = Counter()
    counter_claude = Counter()

    # Iterate through samples
    for text, label in zip(text_series, labels):
        if not isinstance(text, str):
            continue
        words = text.split()

        if label == "gemini":
            counter_gemini.update(words)
        elif label == "chatgpt":
            counter_chatgpt.update(words)
        elif label == "claude":
            counter_claude.update(words)

    # Create a unified vocabulary
    all_words = (
        set(counter_gemini.keys()) |
        set(counter_chatgpt.keys()) |
        set(counter_claude.keys())
    )

    rows = []
    for word in all_words:
        g = counter_gemini.get(word, 0)
        c = counter_chatgpt.get(word, 0)
        a = counter_claude.get(word, 0)

        var = np.var([g, c, a])   # compute variance

        rows.append({
            "word": word,
            "count_gemini": g,
            "count_chatgpt": c,
            "count_claude": a,
            "variance": var
        })

    df = pd.DataFrame(rows)

    # Sort by variance descending
    df = df.sort_values(by="variance", ascending=False).reset_index(drop=True)

    return df

def clean_selected_words(text, selected_words):
    """ Keep only the selected words (a set of strings)."""
    selected_words = selected_words
    text = ' '.join(
        word for word in text.split()
        if word.lower() in selected_words
    )

    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    return text

def split_label(df):
    """
    ...
    :param df:
    :return:
    """
    df_label = df['label']
    del df['label']
    return df, df_label


def remove_student_id(df):
    """
    remove student id colume in given df
    :param df:
    :return: vodi
    """
    del df['student_id']
