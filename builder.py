"""
This file is a builder script that performs data pre-processing in order.
"""

import pandas as pd
import vectorization
import data_cleaning_and_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

FILE_NAME = "training_data_clean.csv"
FEATURE_B = "How likely are you to use this model for academic tasks?"
FEATURE_C = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
FEATURE_D = "Based on your experience, how often has this model given you a response that felt suboptimal?"
FEATURE_E = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
FEATURE_G = "How often do you expect this model to provide responses with references or supporting evidence?"
FEATURE_H = "How often do you verify this model's responses?"
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
    data_cleaning_and_split.remove_student_id(df)
    del df['In your own words, what kinds of tasks would you use this model for?']
    del df['Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?']
    del df['When you verify a response from this model, how do you usually go about it?']
    vectorization.vectorize_B(df)
    vectorization.vectorize_C(df)
    vectorization.vectorize_D(df)
    vectorization.vectorize_E(df)
    vectorization.vectorize_G(df)
    vectorization.vectorize_H(df)
    df_train, df_val, df_test = data_cleaning_and_split.data_split(df, 0.5, 0.25, 0.25)
    x_train, y_train = data_cleaning_and_split.split_label(df_train)
    x_val, y_val = data_cleaning_and_split.split_label(df_val)
    x_test, y_test = data_cleaning_and_split.split_label(df_test)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train, y_train)
    train_acc = knn.score(x_train, y_train)
    test_acc = knn.score(x_test, y_test)
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

if __name__ == "__main__":
    main()
