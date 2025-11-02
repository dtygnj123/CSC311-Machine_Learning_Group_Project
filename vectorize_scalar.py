"""
This file contain function vectorize features B, D, G, H into scalar
"""

import numpy as np
import pandas as pd
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
    dataframe[FEATURE_B] = dataframe[FEATURE_B].apply(extract_rating)


def vectorize_D(dataframe):
    dataframe[FEATURE_D] = dataframe[FEATURE_D].apply(extract_rating)


def vectorize_G(dataframe):
    dataframe[FEATURE_G] = dataframe[FEATURE_G].apply(extract_rating)


def vectorize_H(dataframe):
    dataframe[FEATURE_H] = dataframe[FEATURE_H].apply(extract_rating)
