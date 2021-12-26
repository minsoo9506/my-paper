from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
)
import numpy as np
import torch


def print_score(y_true: np.ndarray, y_pred: np.ndarray, anomaly_score: np.ndarray):
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
    print(f"f1_score = {f1_score(y_true, y_pred):.3f}")
    print(f"roc_auc_score = {roc_auc_score(y_true, anomaly_score):.3f}")


def get_total_anomaly_score(total_dataloader, best_model, window_anomaly_score_result):
    idx = 0
    # train, val, test 포함: total
    best_model.eval()
    with torch.no_grad():
        for input_x, _ in total_dataloader:
            input_x = input_x
            input_x_batch_size = input_x.shape[0]
            y_hat = best_model(input_x)
            anomaly_score = abs(input_x - y_hat)
            window_anomaly_score = torch.mean(anomaly_score, 1).numpy()
            window_anomaly_score_result[
                idx : idx + input_x_batch_size
            ] += window_anomaly_score
            idx += input_x_batch_size

    return window_anomaly_score_result


def get_true_anomaly_info(window_anomaly_score_result, total_y):

    len_wasr = len(window_anomaly_score_result)
    only_anomaly_score = np.where(
        total_y[:len_wasr] == 1, window_anomaly_score_result, np.nan
    )

    only_anomaly_score = only_anomaly_score[~np.isnan(only_anomaly_score)]

    avg_true_anomaly_score = np.mean(only_anomaly_score)
    std_true_anomaly_score = np.std(only_anomaly_score)

    return avg_true_anomaly_score, std_true_anomaly_score


def get_score(window_anomaly_score_result, total_y, threshold, config):
    test_idx = int(config.data_name.split("_")[-3])
    test_anomaly_score = window_anomaly_score_result[test_idx:]
    test_y_true = total_y[test_idx : len(window_anomaly_score_result)]
    test_y_pred = np.where(test_anomaly_score > threshold, 1, 0)

    accuracy = accuracy_score(test_y_true, test_y_pred)
    precision = precision_score(test_y_true, test_y_pred)
    recall = recall_score(test_y_true, test_y_pred)
    f1 = f1_score(test_y_true, test_y_pred)
    roc_auc = roc_auc_score(test_y_true, test_anomaly_score)

    return accuracy, precision, recall, f1, roc_auc
