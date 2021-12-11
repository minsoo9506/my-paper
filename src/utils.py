from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score
import numpy as np


def get_score(y_true: np.ndarray, y_pred: np.ndarray, anomaly_score: np.ndarray):
    """print score of inference result

    Parameters
    ----------
    y_true : np.darray
        [description]
    y_pred : np.darray
        [description]
    anomaly_score : np.darray
        [description]
    """
    print(f"accuracy_score = {accuracy_score(y_true, y_pred):.3f}")
    print(f"precision_score = {precision_score(y_true, y_pred):.3f}")
    print(f"recall_score = {recall_score(y_true, y_pred):.3f}")
    print(f"roc_auc_score = {roc_auc_score(y_true, anomaly_score):.3f}")
