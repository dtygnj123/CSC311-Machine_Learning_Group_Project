"""
This file is a builder script that perform data pre=process in order
"""

import pandas
import vectorize_scalar
import data_cleaning_n_split

FILE_NAME = "training_data_clean.csv"
FEATURE_B = "How likely are you to use this model for academic tasks?"
FEATURE_D = "Based on your experience, how often has this model given you a response that felt suboptimal?"
FEATURE_G = "How often do you expect this model to provide responses with references or supporting evidence?"
FEATURE_H = "How often do you verify this model's responses?"
TEXT_COL = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
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
    df = pandas.read_csv(FILE_NAME)
    data_cleaning_n_split.remove_incomplete_row(df)
    df = data_cleaning_n_split.lower_casing(df)
    df = data_cleaning_n_split.clean_text_columns(df, TEXT_COL, REMOVE_WORDS)
    df = data_cleaning_n_split.clean_text_select_words(df, TEXT_COL, threshold=200)
    df.to_csv("csv_files/only_selected_words.csv", index=False) # for debug
    vectorize_scalar.vectorize_B(df)
    vectorize_scalar.vectorize_D(df)
    vectorize_scalar.vectorize_G(df)
    vectorize_scalar.vectorize_H(df)
    df_train, df_val, df_test = data_cleaning_n_split.data_split(df, 0.5, 0.25, 0.25)
    print("Training data\n", df_train)
    print("Validation data\n", df_val)
    print("Test data\n", df_test)


if __name__ == "__main__":
    main()
