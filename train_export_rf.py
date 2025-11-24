import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.util.hashing import hash_pandas_object
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score, ConfusionMatrixDisplay,
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


# def build_full_feature_matrix():
#     df = pd.read_csv(FILE_NAME)
#     data_cleaning_and_split_refactored.remove_incomplete_row(df)
#     df = data_cleaning_and_split_refactored.lower_casing(df)
#     df = data_cleaning_and_split_refactored.clean_text_columns(df, TEXT_COL, REMOVE_WORDS)
#
#     df_train, df_val, df_test = data_cleaning_and_split_refactored.data_split(df, 0.5, 0.25, 0.25)  # solve the information leak
#
#     df, df_a, df_f, df_i = data_cleaning_and_split_refactored.clean_text_select_words(
#         df, df_train, TEXT_COL, threshold=THRESHOLD
#     )
#
#     vectorization_refactored.vectorize_B(df)
#     vectorization_refactored.vectorize_C(df)
#     vectorization_refactored.vectorize_D(df)
#     vectorization_refactored.vectorize_E(df)
#     vectorization_refactored.vectorize_G(df)
#     vectorization_refactored.vectorize_H(df)
#
#     df = vectorization_refactored.vectorize_A(df, df_a)
#     df = vectorization_refactored.vectorize_F(df, df_f)
#     df = vectorization_refactored.vectorize_I(df, df_i)
#     df_train, df_val, df_test = data_cleaning_and_split_refactored.data_split(df, 0.5, 0.25, 0.25)
#     data_cleaning_and_split_refactored.remove_student_id(df_train)
#     data_cleaning_and_split_refactored.remove_student_id(df_val)
#     data_cleaning_and_split_refactored.remove_student_id(df_test)
#     X_train, y_train = data_cleaning_and_split_refactored.split_label(df_train)
#     X_val, y_val = data_cleaning_and_split_refactored.split_label(df_val)
#     X_test, y_test = data_cleaning_and_split_refactored.split_label(df_test)
#
#     # get full feature matrix for final training
#     df_all = df.copy()
#     X_all, y_all = data_cleaning_and_split_refactored.split_label(df_all)
#
#     return X_train, y_train, X_val, y_val, X_test, y_test, X_all, y_all


def build_matrix_5_folds(threshold):
    df = pd.read_csv(FILE_NAME)
    data_cleaning_and_split_refactored.remove_incomplete_row(df)
    df = data_cleaning_and_split_refactored.lower_casing(df)
    df = data_cleaning_and_split_refactored.clean_text_columns(df, TEXT_COL, REMOVE_WORDS)

    df_trainval, df_test = data_cleaning_and_split_refactored.split_train_test(df, 0.15)  # solve the information leak

    df, df_a, df_f, df_i = data_cleaning_and_split_refactored.clean_text_select_words(
        df, df_trainval, TEXT_COL, threshold
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
    df_trainval, df_test = data_cleaning_and_split_refactored.split_train_test(df, 0.15)

    folds = data_cleaning_and_split_refactored.split_into_folds(df_trainval) # folds = [folds[1], folds[2], folds[3], folds[4]]

    for fold in folds:
        data_cleaning_and_split_refactored.remove_student_id(fold)
    data_cleaning_and_split_refactored.remove_student_id(df_test)

    X_train_folds = []
    y_train_folds = []
    for fold in folds:
        X_train, y_train = data_cleaning_and_split_refactored.split_label(fold)
        X_train_folds.append(X_train)
        y_train_folds.append(y_train)

    X_test, y_test = data_cleaning_and_split_refactored.split_label(df_test)

    # get full feature matrix for final training
    df_all = df.copy()
    data_cleaning_and_split_refactored.remove_student_id(df_all)
    X_all, y_all = data_cleaning_and_split_refactored.split_label(df_all)

    return X_train_folds, y_train_folds, X_test, y_test, X_all, y_all


def train_acc(model, X_train, y_train):
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_acc:.3f}")


def val_acc(model, X_val, y_val):
    y_val_pred = model.predict(X_val)
    val_acc_ = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc_:.3f}")


def acc_test_acc(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    test_acc_ = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc_:.3f}")


def evaluate_split(model, x, y, split_name="Validation"):
    y_pred = model.predict(x)

    # print("Train Acc", accuracy(y_train, train_result))
    # print("Val Acc", accuracy(y_val, val_result))
    # print("Test Acc", accuracy(y_test, test_result))
    print(classification_report(y, y_pred))
    # compute confusion matrix
    cm = confusion_matrix(y, y_pred, labels=["chatgpt", "claude", "gemini"])
    print(cm)
    # plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["chatgpt", "claude", "gemini"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

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
    best_model = None
    best_acc = 0.0
    best_acc_test_acc = 0.0
    best_acc_train_acc = 0.0
    best_params = None
    best_X_trainval = None
    best_y_trainval = None
    best_X_test = None
    best_y_test = None
    for threshold in range(78, 80, 2):
        print(f"start running threshold {threshold}")
        X_train_folds, y_train_folds, X_test, y_test, X_all, y_all = build_matrix_5_folds(threshold)

        X_train1 = pd.concat([X_train_folds[1], X_train_folds[2], X_train_folds[3], X_train_folds[4]], axis=0)
        y_train1 = pd.concat([y_train_folds[1], y_train_folds[2], y_train_folds[3], y_train_folds[4]], axis=0)

        X_train2 = pd.concat([X_train_folds[0], X_train_folds[2], X_train_folds[3], X_train_folds[4]], axis=0)
        y_train2 = pd.concat([y_train_folds[0], y_train_folds[2], y_train_folds[3], y_train_folds[4]], axis=0)

        X_train3 = pd.concat([X_train_folds[0], X_train_folds[1], X_train_folds[3], X_train_folds[4]], axis=0)
        y_train3 = pd.concat([y_train_folds[0], y_train_folds[1], y_train_folds[3], y_train_folds[4]], axis=0)

        X_train4 = pd.concat([X_train_folds[0], X_train_folds[1], X_train_folds[2], X_train_folds[4]], axis=0)
        y_train4 = pd.concat([y_train_folds[0], y_train_folds[1], y_train_folds[2], y_train_folds[4]], axis=0)

        X_train5 = pd.concat([X_train_folds[0], X_train_folds[1], X_train_folds[2], X_train_folds[3]], axis=0)
        y_train5 = pd.concat([y_train_folds[0], y_train_folds[1], y_train_folds[2], y_train_folds[3]], axis=0)

        X_trainval = pd.concat([X_train_folds[0], X_train_folds[1], X_train_folds[2], X_train_folds[3], X_train_folds[4]], axis=0)
        y_trainval = pd.concat([y_train_folds[0], y_train_folds[1], y_train_folds[2], y_train_folds[3], y_train_folds[4]], axis=0)

        five_fold_list = [(X_train1, y_train1, X_train_folds[0], y_train_folds[0], X_test, y_test),
                          (X_train2, y_train2, X_train_folds[1], y_train_folds[1], X_test, y_test),
                          (X_train3, y_train3, X_train_folds[2], y_train_folds[2], X_test, y_test),
                          (X_train4, y_train4, X_train_folds[3], y_train_folds[3], X_test, y_test),
                          (X_train5, y_train5, X_train_folds[4], y_train_folds[4], X_test, y_test)]

        # manual grid search
        n_estimators_list = [300]
        # n_estimators_list = [500]
        max_depth_list = [3]
        # max_depth_list = [6]
        min_samples_leaf_list = [4]
        # min_samples_leaf_list = [6]
        max_features_list = ["sqrt"]
        # max_features_list = ["sqrt", 0.3]
        criterion_list = ["entropy"]
        # criterion_list = ["gini"]

        ########################################
        depths = []
        train_accs = []
        val_accs = []
        test_accs = []
        ########################################

        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                for min_samples_leaf in min_samples_leaf_list:
                    for max_features in max_features_list:
                        for criterion in criterion_list:
                            train_accs_avg = []
                            val_accs_avg = []
                            test_accs_avg = []
                            acc_avg = []
                            models = []

                            for X_train, y_train, X_val, y_val, X_test, y_test in five_fold_list:
                                model = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features,
                                    criterion=criterion,
                                    random_state=42,
                                    n_jobs=-1,
                                )
                                model.fit(X_train, y_train)
                                y_train_pred = model.predict(X_train)
                                y_val_pred = model.predict(X_val)
                                y_test_pred = model.predict(X_test)

                                train_accuracy = accuracy_score(y_train, y_train_pred)
                                train_accs_avg.append(train_accuracy)

                                val_accuracy = accuracy_score(y_val, y_val_pred)
                                val_accs_avg.append(val_accuracy)

                                test_accuracy = accuracy_score(y_test, y_test_pred)
                                test_accs_avg.append(test_accuracy)

                                acc = (val_accuracy + test_accuracy) / 2
                                acc_avg.append(acc)

                                models.append(model)

                            ########################################
                            depths.append(min_samples_leaf)
                            train_accs.append(sum(train_accs_avg)/len(train_accs_avg))
                            val_accs.append(sum(val_accs_avg)/len(val_accs_avg))
                            test_accs.append(sum(test_accs_avg)/len(test_accs_avg))
                            ########################################

                            avg_val_acc = sum(val_accs_avg) / len(val_accs_avg)
                            avg_test_acc = sum(test_accs_avg) / len(test_accs_avg)
                            avg_train_acc = sum(train_accs_avg) / len(train_accs_avg)
                            # changed the best hyperparameter choosing metric
                            # if avg_val_acc > best_acc - 0 and avg_test_acc > best_acc_test_acc - 0.03 :
                            if avg_val_acc > best_acc:
                                best_acc = avg_val_acc
                                best_acc_test_acc = avg_test_acc
                                best_acc_train_acc = avg_train_acc

                                best_X_trainval = X_trainval
                                best_y_trainval = y_trainval
                                best_X_test = X_test
                                best_y_test = y_test

                                best_params = (n_estimators, max_depth,
                                                min_samples_leaf, max_features,
                                                criterion, threshold)

                                best_val_acc = max(val_accs_avg)
                                best_id = val_accs_avg.index(best_val_acc)
                                # the best model is the one with the highest validation accuracy
                                best_model = models[best_id]
        print(f"finishing on threshold {threshold}")

    # plt.figure(figsize=(8, 5))
    # plt.plot(depths, train_accs, marker='o', label="Train Accuracy")
    # plt.plot(depths, val_accs, marker='o', label="Validation Accuracy")
    # plt.plot(depths, test_accs, marker='o', label="Test Accuracy")
    #
    # plt.xlabel("min_samples_leaf")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy vs. max_depth for Random Forest")
    # plt.xticks(max_depth_list)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    print(
        "Best RF params: "
        f"n_estimators={best_params[0]}, "
        f"max_depth={best_params[1]}, "
        f"min_samples_leaf={best_params[2]}, "
        f"max_features={best_params[3]}, "
        f"criterion={best_params[4]}"
        f"threshold={best_params[5]}"
    )

    # train_acc(best_model, X_train, y_train)
    # val_acc(best_model, X_val, y_val)
    # acc_test_acc(best_model, X_test, y_test)
    print(f"Training accuracy: {best_acc_train_acc:.3f}")
    print(f"Validation accuracy: {best_acc:.3f}")
    print(f"Test accuracy: {best_acc_test_acc:.3f}")

    feature_names = np.array(X_all.columns)
    classes = best_model.classes_

    evaluate_split(best_model, best_X_trainval, best_y_trainval, split_name="Train+Validation")
    evaluate_split(best_model, best_X_test, best_y_test, split_name="Test")

    return best_model, feature_names, classes


def export_forest(rf, feature_name, classes, out_path="rf_model_params.npz"):
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
        values = tree.value.squeeze(axis=1)  # (n_nodes, n_classes)
        leaf_class = values.argmax(axis=1)  # predicted class index at each node
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