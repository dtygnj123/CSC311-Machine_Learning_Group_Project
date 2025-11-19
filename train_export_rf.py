import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import data_cleaning_and_split_refactored 
import vectorization_refactored


FILE_NAME = "training_data_clean.csv"
FEATURE_B = "How likely are you to use this model for academic tasks?"
FEATURE_C = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
FEATURE_D = "Based on your experience, how often has this model given you a response that felt suboptimal?"
FEATURE_E = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
FEATURE_G = "How often do you expect this model to provide responses with references or supporting evidence?"
FEATURE_H = "How often do you verify this model's responses?"

THRESHOLD = 3

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


def build_full_feature_matrix():
    df = pd.read_csv(FILE_NAME)
    data_cleaning_and_split_refactored.remove_incomplete_row(df)
    df = data_cleaning_and_split_refactored.lower_casing(df)
    data_cleaning_and_split_refactored.remove_student_id(df)
    df = data_cleaning_and_split_refactored.clean_text_columns(df, TEXT_COL, REMOVE_WORDS)

    df_train, df_val, df_test = data_cleaning_and_split_refactored.data_split(
        df, 0.5, 0.25, 0.25)

    df, df_a, df_f, df_i = data_cleaning_and_split_refactored.clean_text_select_words(
        df, df_train, TEXT_COL, threshold=THRESHOLD
    )

    vectorization_refactored.vectorize_B(df)
    vectorization_refactored.vectorize_C(df)
    vectorization_refactored.vectorize_D(df)
    vectorization_refactored.vectorize_E(df)
    vectorization_refactored.vectorize_G(df)
    vectorization_refactored.vectorize_H(df)

    df = vectorization_refactored.vectorize_A(df, df_a)
    df = vectorization_refactored.vectorize_F(df, df_f)
    df = vectorization_refactored.vectorize_I(df, df_i)
    
    df_train, df_val, df_test = data_cleaning_and_split_refactored.data_split(
        df, 0.5, 0.25, 0.25)
    
    X_train, y_train = data_cleaning_and_split_refactored.split_label(df_train)
    X_val, y_val = data_cleaning_and_split_refactored.split_label(df_val)
    X_test, y_test = data_cleaning_and_split_refactored.split_label(df_test)

    # get full feature matrix for final training
    df_all = df.copy()
    X_all, y_all = data_cleaning_and_split_refactored.split_label(df_all)

    return X_train, y_train, X_val, y_val, X_test, y_test, X_all, y_all


def train_acc(model, X_train, y_train):
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_acc:.3f}")


def val_acc(model, X_val, y_val):
    y_val_pred = model.predict(X_val)
    val_acc_ = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc_:.3f}")


def test_acc(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    test_acc_ = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc_:.3f}")


def evaluate_split(model, x, y, split_name="Validation"):
    y_pred = model.predict(x)
    
    acc = accuracy_score(y, y_pred)
    # For multi-class, 'macro' treats all classes equally
    prec = precision_score(y, y_pred, average="macro", zero_division=0)
    rec = recall_score(y, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y, y_pred)

    print(f"\n===== {split_name} metrics =====")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y, y_pred))
    

def train_random_forest():
    X_train, y_train, X_val, y_val, X_test, y_test, X_all, y_all = build_full_feature_matrix()

    # manual grid search
    n_estimators_list = [50, 100, 200, 300, 400, 500]     
    max_depth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                      12, 13, 14, 15, 16, 17, 18, 19, 20]  
    min_samples_leaf_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    max_features_list = ["sqrt", 0.5]   
    criterion_list = ["gini", "entropy"]

    best_model = None
    best_acc = 0.0
    best_params = None

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_leaf in min_samples_leaf_list:
                for max_features in max_features_list:
                    for criterion in criterion_list:
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            criterion=criterion,
                            random_state=0,
                            n_jobs=-1,
                        )
                        model.fit(X_train, y_train)
                        y_val_pred = model.predict(X_val)
                        acc = accuracy_score(y_val, y_val_pred)
                        
                        if acc > best_acc:
                            best_acc = acc
                            best_params = (n_estimators, max_depth,
                                           min_samples_leaf, max_features,
                                           criterion)
                            best_model = model

    print(
        "Best RF params: "
        f"n_estimators={best_params[0]}, "
        f"max_depth={best_params[1]}, "
        f"min_samples_leaf={best_params[2]}, "
        f"max_features={best_params[3]}, "
        f"criterion={best_params[4]}"
    )
    
    train_acc(best_model, X_train, y_train)
    val_acc(best_model, X_val, y_val)
    test_acc(best_model, X_test, y_test)
    
    evaluate_split(best_model, X_train, y_train, split_name="Train")
    evaluate_split(best_model, X_val, y_val, split_name="Validation")
    evaluate_split(best_model, X_test, y_test, split_name="Test")
    
    # re-train best RF on all data (train+val+test) for final submission
    final_rf = RandomForestClassifier(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        min_samples_leaf=best_params[2],
        max_features=best_params[3],
        criterion=best_params[4],
        random_state=0,
        n_jobs=-1,
    )
    final_rf.fit(X_all, y_all)

    feature_names = np.array(X_all.columns)
    classes = final_rf.classes_
    
    return final_rf, feature_names, classes


def export_forest(rf: RandomForestClassifier, feature_names: np.ndarray, 
                  classes: np.ndarray, out_path: str = "rf_model_params.npz"):
    """
    Export all trees from a trained RandomForestClassifier into an .npz file
    that random_forest_refactored.py can load and use.
    """
    children_left_list = []
    children_right_list = []
    feature_list = []
    threshold_list = []
    leaf_class_list = []

    for est in rf.estimators_:
        tree = est.tree_
        children_left_list.append(tree.children_left.copy())
        children_right_list.append(tree.children_right.copy())
        feature_list.append(tree.feature.copy())
        threshold_list.append(tree.threshold.copy())

        # tree.value: shape (n_nodes, 1, n_classes)
        values = tree.value.squeeze(axis=1) # (n_nodes, n_classes)
        leaf_class = values.argmax(axis=1) # predicted class index at each node
        leaf_class_list.append(leaf_class)

    np.savez(
        out_path,
        children_left_list=np.array(children_left_list, dtype=object),
        children_right_list=np.array(children_right_list, dtype=object),
        feature_list=np.array(feature_list, dtype=object),
        threshold_list=np.array(threshold_list, dtype=object),
        leaf_class_list=np.array(leaf_class_list, dtype=object),
        classes=classes,
        feature_names=feature_names,
    )
    print(f"Saved forest parameters to {out_path}")


if __name__ == "__main__":
    rf_model, feature_names, classes = train_random_forest()
    export_forest(rf_model, feature_names, classes, out_path="rf_model_params.npz")



