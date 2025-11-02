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

def main():
    """
    main function
    :return: void
    """
    df = pandas.read_csv(FILE_NAME)
    data_cleaning_n_split.remove_incomplete_row(df)
    df = data_cleaning_n_split.lower_casing(df)
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
