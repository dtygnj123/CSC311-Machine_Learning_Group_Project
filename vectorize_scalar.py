"""
This file contain functions vectorize features B, D, G, H into scalar
"""

import re

FILE_NAME = "training_data_clean.csv"
FEATURE_B = "How likely are you to use this model for academic tasks?"
FEATURE_D = "Based on your experience, how often has this model given you a response that felt suboptimal?"
FEATURE_G = "How often do you expect this model to provide responses with references or supporting evidence?"
FEATURE_H = "How often do you verify this model's responses?"


def extract_rating(response):
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing responses
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def vectorize_B(dataframe):
    """
    Replace dataframe's FEATURE_B column's text entry with scalar in place
    :param dataframe: panda dataframe representing training data
    :return: void
    """
    dataframe[FEATURE_B] = dataframe[FEATURE_B].apply(extract_rating)


def vectorize_D(dataframe):
    """
    Replace dataframe's FEATURE_D column's text entry with scalar in place
    :param dataframe: panda dataframe representing training data
    :return: void
    """
    dataframe[FEATURE_D] = dataframe[FEATURE_D].apply(extract_rating)


def vectorize_G(dataframe):
    """
    Replace dataframe's FEATURE_G column's text entry with scalar in place
    :param dataframe: panda dataframe representing training data
    :return: void
    """
    dataframe[FEATURE_G] = dataframe[FEATURE_G].apply(extract_rating)


def vectorize_H(dataframe):
    """
    Replace dataframe's FEATURE_H column's text entry with scalar in place
    :param dataframe: panda dataframe representing training data
    :return: void
    """
    dataframe[FEATURE_H] = dataframe[FEATURE_H].apply(extract_rating)
