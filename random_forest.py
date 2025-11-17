"""
This file is a python script that training a decision tree model.
"""

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def train_acc(model, x_train, y_train):
    X_train = x_train.to_numpy(dtype=float)
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_acc:.3f}")

def val_acc(model, x_val, y_val):
    X_val = x_val.to_numpy(dtype=float)
    y_val_pred = model.predict(X_val)
    val_acc_ = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc_:.3f}")

def test_acc(model, x_test, y_test):
    X_test = x_test.to_numpy(dtype=float)
    y_test_pred = model.predict(X_test)
    test_acc_ = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc_:.3f}")

def rf(x_train, y_train, x_val, y_val):
    best_model = None
    best_acc = 0.0
    best_params = None

    n_estimators_list = [50, 100, 200, 500]     # number of trees
    max_depth_list = [3, 5, 7, 9]   # shallower than before
    min_samples_leaf_list = [1, 5, 10]    # >= 5 or 10 = stronger regularization
    max_features_list = ["sqrt", 0.5]     # try using 50% features per split

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_leaf in min_samples_leaf_list:
                for max_features in max_features_list:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=0,
                        n_jobs=-1,
                    )
                    model.fit(x_train, y_train)
                    y_val_pred = model.predict(x_val)
                    acc = accuracy_score(y_val, y_val_pred)

                    if acc > best_acc:
                        best_acc = acc
                        best_params = (n_estimators, max_depth,
                                       min_samples_leaf, max_features)
                        best_model = model

    print(
        "Best RF params: "
        f"n_estimators={best_params[0]}, "
        f"max_depth={best_params[1]}, "
        f"min_samples_leaf={best_params[2]}, "
        f"max_features={best_params[3]}, "
        f"val_acc={best_acc:.3f}"
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

        
    
    
    
    