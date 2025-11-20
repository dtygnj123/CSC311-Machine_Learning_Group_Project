import numpy as np
import pandas as pd
import data_cleaning_and_split_refactored
import vectorization_refactored 
import random_forest_refactored


FEATURE_B = "How likely are you to use this model for academic tasks?"
FEATURE_C = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
FEATURE_D = "Based on your experience, how often has this model given you a response that felt suboptimal?"
FEATURE_E = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
FEATURE_G = "How often do you expect this model to provide responses with references or supporting evidence?"
FEATURE_H = "How often do you verify this model's responses?"
FEATURE_A = "In your own words, what kinds of tasks would you use this model for?"
FEATURE_F = "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?"
FEATURE_I = "When you verify a response from this model, how do you usually go about it?"

TEXT_COL = [FEATURE_A, FEATURE_F, FEATURE_I]

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


def preprocess_for_rf(df, feature_names):
    df = df.copy()

    data_cleaning_and_split_refactored.remove_incomplete_row(df)

    if "label" in df.columns:
        df = df.drop(columns=["label"])

    df = data_cleaning_and_split_refactored.lower_casing(df)
    df = data_cleaning_and_split_refactored.clean_text_columns(df, TEXT_COL, REMOVE_WORDS)

    vectorization_refactored.vectorize_B(df)
    vectorization_refactored.vectorize_D(df)
    vectorization_refactored.vectorize_G(df)
    vectorization_refactored.vectorize_H(df)
    vectorization_refactored.vectorize_C(df)
    vectorization_refactored.vectorize_E(df)

    # Bag-of-words for A/F/I:
    # The vocabulary was implicitly defined by training and is encoded
    # in the feature_names array as column names like "A_<word>".
    feature_names = np.asarray(feature_names)

    vocab_A = [name[len("A_"):] for name in feature_names if name.startswith("A_")]
    vocab_F = [name[len("F_"):] for name in feature_names if name.startswith("F_")]
    vocab_I = [name[len("I_"):] for name in feature_names if name.startswith("I_")]

    if vocab_A:
        df_a = pd.DataFrame({"word": vocab_A})
        df = vectorization_refactored.vectorize_A(df, df_a)
    if vocab_F:
        df_f = pd.DataFrame({"word": vocab_F})
        df = vectorization_refactored.vectorize_F(df, df_f)
    if vocab_I:
        df_i = pd.DataFrame({"word": vocab_I})
        df = vectorization_refactored.vectorize_I(df, df_i)

    # At this point df may have extra columns (e.g., original text cols
    # that are not part of the model). We align strictly to feature_names.
    df_features = df.reindex(columns=feature_names, fill_value=0.0)

    return df_features


def predict_all(file_name):
    # 1. Load forest parameters and metadata
    forest, classes, feature_names = random_forest_refactored.load_forest("rf_model_params.npz")
    classes = np.asarray(classes)
    feature_names = np.asarray(feature_names)

    # 2. Load CSV
    df = pd.read_csv(file_name)

    # 3. Preprocess to match feature space
    df_features = preprocess_for_rf(df, feature_names)
    X = df_features.to_numpy(dtype=float)

    # 4. Random forest prediction (indices)
    pred_indices = random_forest_refactored.predict_forest(forest, classes, X)

    # 5. Map to label strings
    raw_preds = classes[pred_indices]
    
    # 6. Convert to required case: ChatGPT / Claude / Gemini
    label_map = {
        "chatgpt": "ChatGPT",
        "claude": "Claude",
        "gemini": "Gemini",
    }
    preds = [label_map.get(str(label), str(label)) for label in raw_preds]
    
    return preds


# Optional local test (won't be used by the autograder)
if __name__ == "__main__":
    print(predict_all("training_data_clean.csv")[:10])




