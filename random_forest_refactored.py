import numpy as np


def load_forest(path = "rf_model_params.npz"):
    """
    Load pre-trained random forest parameters from an .npz file.

    Returns:
        forest: dict with keys
            - 'children_left'
            - 'children_right'
            - 'feature'
            - 'threshold'
            - 'leaf_class'
        classes: np.ndarray of label strings
        feature_names: np.ndarray of feature names (strings)
    """
    data = np.load(path, allow_pickle=True)

    forest = {
        "children_left": data["children_left_list"],
        "children_right": data["children_right_list"],
        "feature": data["feature_list"],
        "threshold": data["threshold_list"],
        "leaf_class": data["leaf_class_list"],
    }
    classes = data["classes"]
    feature_names = data["feature_names"]
    return forest, classes, feature_names


def predict_forest(forest, classes, X):
    """
    Predict class indices for each row of X using a pre-trained random forest.

    Parameters
    ----------
    forest : dict
        Dictionary returned by load_forest, containing per-tree parameters.
    classes : np.ndarray
        Array of class labels, length n_classes.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    pred_indices : np.ndarray
        Array of predicted class indices (0 .. n_classes-1), shape (n_samples,).
    """
    n_samples, _ = X.shape
    n_classes = len(classes)

    children_left_list = forest["children_left"]
    children_right_list = forest["children_right"]
    feature_list = forest["feature"]
    threshold_list = forest["threshold"]
    leaf_class_list = forest["leaf_class"]

    n_trees = len(children_left_list)

    # votes[sample, class] = number of trees voting for that class
    votes = np.zeros((n_samples, n_classes), dtype=int)

    for t in range(n_trees):
        cl = children_left_list[t]
        cr = children_right_list[t]
        feat = feature_list[t]
        thr = threshold_list[t]
        leaf_cls = leaf_class_list[t]

        # Traverse this tree for each sample
        for i in range(n_samples):
            node = 0
            while True:
                left = cl[node]
                right = cr[node]

                # leaf: by convention, left == right == -1
                if left == -1 and right == -1:
                    c = leaf_cls[node]
                    votes[i, c] += 1
                    break

                f = feat[node]
                if f < 0:
                    # safety: treat unknown feature index as leaf
                    c = leaf_cls[node]
                    votes[i, c] += 1
                    break

                if X[i, f] <= thr[node]:
                    node = left
                else:
                    node = right

    pred_indices = np.argmax(votes, axis=1)
    return pred_indices

        
    
    
    
    