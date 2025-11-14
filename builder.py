"""
This file is a builder script that performs data pre-processing in order.
"""

import pandas as pd
import vectorization
import data_cleaning_and_split

FILE_NAME = "training_data_clean.csv"
FEATURE_B = "How likely are you to use this model for academic tasks?"
FEATURE_C = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
FEATURE_D = "Based on your experience, how often has this model given you a response that felt suboptimal?"
FEATURE_E = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
FEATURE_G = "How often do you expect this model to provide responses with references or supporting evidence?"
FEATURE_H = "How often do you verify this model's responses?"

THRESHOLD = 200  # hyperparameter for top words below 'code'

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


def main():
    """
    main function
    :return: void
    """
    df = pd.read_csv(FILE_NAME)
    data_cleaning_and_split.remove_incomplete_row(df)
    df = data_cleaning_and_split.lower_casing(df)
    df = data_cleaning_and_split.clean_text_columns(df, TEXT_COL, REMOVE_WORDS)

    df_train, df_val, df_test = data_cleaning_and_split.data_split(df, 0.5, 0.25, 0.25) # solve the information leak

    df, df_a, df_f, df_i = data_cleaning_and_split.clean_text_select_words(
        df, df_train, TEXT_COL, threshold=THRESHOLD
    )
    # df.to_csv("csv_files/only_selected_words.csv", index=False)
    vectorization.vectorize_B(df)
    vectorization.vectorize_C(df)
    vectorization.vectorize_D(df)
    vectorization.vectorize_E(df)
    vectorization.vectorize_G(df)
    vectorization.vectorize_H(df)

    df = vectorization.vectorize_A(df, df_a)
    df = vectorization.vectorize_F(df, df_f)
    df = vectorization.vectorize_I(df, df_i)
    df_train, df_val, df_test = data_cleaning_and_split.data_split(df, 0.5, 0.25, 0.25)
    print("Training data:\n", df_train)
    print("Validation data:\n", df_val)
    print("Test data:\n", df_test)

    return df_train, df_val, df_test


if __name__ == "__main__":
    main()
##