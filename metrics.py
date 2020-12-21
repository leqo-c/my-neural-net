import numpy as np
from sklearn.metrics import accuracy_score

def accuracy(Y_true, Y_pred):
    """Computes the accuracy measure (sum of the true positives and the
    true negatives over the total number of samples) by comparing the
    ground truth values with the predicted values.

    Args:
        Y_true (numpy-array): The ground truth values.
        Y_pred (numpy-array): The predicted values.

    Returns:
        float: The accuracy measure achieved by the predictions `Y_pred`
            w.r.t. the ground truths `Y_true`. 
    """
    Y_pred_bin = (Y_pred > 0.5).astype(float)
    return np.mean(Y_true == Y_pred_bin)

def accuracy_sklearn(Y_true, Y_pred):
    """Computes the accuracy measure (sum of the true positives and the
    true negatives over the total number of samples) by comparing the
    ground truth values with the predicted values. `Note`: This is the
    scikit-learn package's implementation.

    Args:
        Y_true (numpy-array): The ground truth values.
        Y_pred (numpy-array): The predicted values.

    Returns:
        float: The accuracy measure achieved by the predictions `Y_pred`
            w.r.t. the ground truths `Y_true`. 
    """
    Y_pred_bin = (Y_pred > 0.5).astype(float)
    return accuracy_score(np.squeeze(Y_true), np.squeeze(Y_pred_bin))