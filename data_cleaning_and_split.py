"""
This file contains functions to clean and split data such that:
    All rows with empty entry or "NAME?" or nan are removed
    Lower casing all text
    Randomly partition data into 2:1:1 ratio corresponds to Training:Validation:Test
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re

FILE_NAME = "training_data_clean.csv"


def remove_incomplete_row(df):
    """
    Remove rows that contain empty entry or "NAME?" or nan in any entry in place
    :param df: pandas df representing training data
    :return: void
    """
    df.replace(['', '#NAME?'], pd.NA, inplace=True)
    df.dropna(inplace=True)


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


def clean_text_select_words(dataframe, columns, threshold):
    """
    Keep only the 'threshold' most frequent words and has less frequency than the word 'code'.
    Select the words with a hyperparameter. Not including the words in remove_words.
    """

    df_AFI = []

    for col in columns:
        counts = get_word_counts(dataframe[col])

        freq_df = pd.DataFrame(counts.items(), columns=["word", "count"]).sort_values(by="count", ascending=False)

        freq_df.to_csv(f"csv_files/{col}_word_counts.csv", index=False)

        code_count = counts.get("code", 0)
        less_than_code = freq_df[freq_df["count"] <= code_count] # less frequent than "code", including "code"
        top_less_than_code = less_than_code.head(threshold) # Take the top 'threshold' words

        # Convert to a set of words
        word_set = set(top_less_than_code["word"]) # TODO: we should save these sets of words for the model

        if col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(lambda x: clean_text_select_word_helper(x, word_set))

        # Save to CSV
        top_less_than_code.to_csv(f"csv_files/{col}_rare_words.csv", index=False)
        df_AFI.append(top_less_than_code)
        print(f"Saved {col}_rare_words.csv with {len(word_set)} words (less frequent than 'code')")

    return dataframe, df_AFI[0], df_AFI[1], df_AFI[2]


def clean_text_select_word_helper(s: str, word_set: set) -> str:
    """ Delete all the words that are not in the word_set (a set of strings). """
    if s is None:
        return s
    if not isinstance(s, str):
        s = str(s)

    s = clean_selected_words(s, word_set)

    return s


def get_word_counts(text_series):
    """ Get a word to frequency count table."""
    # Combine all text into one big string
    text = ' '.join(text_series.dropna().astype(str))
    # Split into words
    words = text.split()
    # Count frequency
    return Counter(words) # TODO: is it ok to import Counter?? probably ok, since we won't be applying this to test data


def clean_selected_words(text, selected_words):
    """ Keep only the selected words (a set of strings)."""
    selected_words = selected_words
    text = ' '.join(
        word for word in text.split()
        if word.lower() in selected_words
    )

    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    return text