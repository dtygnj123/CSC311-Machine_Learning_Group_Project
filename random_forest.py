"""
This file is a python script that training a random forest model.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

def train_acc(model, x_train, y_train):
    y_train_pred = model.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_acc:.3f}")

def val_acc(model, x_val, y_val):
    y_val_pred = model.predict(x_val)
    val_acc_ = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc_:.3f}")

def test_acc(model, x_test, y_test):
    y_test_pred = model.predict(x_test)
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

def rf(x_train, y_train, x_val, y_val):
    best_model = None
    best_acc = 0.0
    best_params = None

    n_estimators_list = [50, 100, 200, 500]     # number of trees
    max_depth_list = [3, 5, 7, 9]   # shallower than before
    min_samples_leaf_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    # >= 5 or 10 = stronger regularization
    max_features_list = ["sqrt", 0.5]     # try using 50% features per split
    criterion_list = ["gini", "entropy", "log_loss"]
    
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
                        model.fit(x_train, y_train)
                        y_val_pred = model.predict(x_val)
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
    return best_model

def gb(x_train, y_train, x_val, y_val):
    best_model = None
    best_acc = 0.0
    best_params = None

    for n_estimators in [50, 100, 200, 500]:
        for learning_rate in [0.01, 0.03, 0.05, 0.1]:
            for max_depth in [3, 5, 7, 9]:
                gb = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=0
                )
                gb.fit(x_train, y_train)
                y_val_pred = gb.predict(x_val)
                acc = accuracy_score(y_val, y_val_pred)

                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_estimators, learning_rate, max_depth)
                    best_model = gb

    print(
        f"Best GB params: n_estimators={best_params[0]}, "
        f"learning_rate={best_params[1]}, max_depth={best_params[2]}, "
        f"val_acc={best_acc:.3f}"
    )
    return best_model

        
    
    
    
    