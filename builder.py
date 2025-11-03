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

WORD_COUNT_FILE_G = "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal_word_counts.csv"
FEATURE_realG = "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?"

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


def main():
    """
    main function
    :return: void
    """
    df = pd.read_csv(FILE_NAME)
    data_cleaning_and_split.remove_incomplete_row(df)
    df = data_cleaning_and_split.lower_casing(df)
    vectorization.vectorize_B(df)
    vectorization.vectorize_C(df)
    vectorization.vectorize_D(df)
    vectorization.vectorize_E(df)
    vectorization.vectorize_G(df)
    vectorization.vectorize_H(df)

    vectorization.vectorize_realG(df, 50)

    df_train, df_val, df_test = data_cleaning_and_split.data_split(df, 0.5, 0.25, 0.25)
    print("Training data:\n", df_train)
    print("Validation data:\n", df_val)
    print("Test data:\n", df_test)


if __name__ == "__main__":
    main()
