"""
This file contains functions to clean and split data such that:
    All rows with empty entry or "NAME?" or nan are removed
    Lower casing all text
    Randomly partition data into 2:1:1 ratio corresponds to Training:Validation:Test
"""

import pandas as pd
from sklearn.model_selection import train_test_split

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