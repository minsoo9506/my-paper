from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    auc,
)
import numpy as np
import pandas as pd
import torch
import datetime
import os


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
    precision, recall, _ = precision_recall_curve(y_true, anomaly_score)
    print(f"pr_auc = {auc(recall, precision):.3f}")


def _get_total_anomaly_score(total_dataloader, best_model, window_anomaly_score_result):
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


def _get_true_anomaly_info(window_anomaly_score_result, total_y):

    len_wasr = len(window_anomaly_score_result)
    only_anomaly_score = np.where(
        total_y[:len_wasr] == 1, window_anomaly_score_result, np.nan
    )

    only_anomaly_score = only_anomaly_score[~np.isnan(only_anomaly_score)]

    avg_true_anomaly_score = np.mean(only_anomaly_score)
    std_true_anomaly_score = np.std(only_anomaly_score)

    return avg_true_anomaly_score, std_true_anomaly_score


def _get_score(window_anomaly_score_result, total_y, tst_start_idx, config):
    tst_ano_scr = window_anomaly_score_result[tst_start_idx:]
    if config.data == "tabular":
        tst_y_true = total_y[tst_start_idx:]
    else:
        tst_y_true = total_y[tst_start_idx : len(window_anomaly_score_result)]
    roc_auc = roc_auc_score(tst_y_true, tst_ano_scr)
    _precision, _recall, _ = precision_recall_curve(tst_y_true, tst_ano_scr)
    pr_auc = auc(_recall, _precision)

    return roc_auc, pr_auc


def _get_anomaly_score_result(anomaly_score, trn_end_idx, val_end_idx):
    trn_ano_scr = anomaly_score[:trn_end_idx]
    val_ano_scr = anomaly_score[trn_end_idx:val_end_idx]
    tst_ano_scr = anomaly_score[val_end_idx:]
    return trn_ano_scr, val_ano_scr, tst_ano_scr


def _save_final_result(
    config,
    return_epoch,
    hidden_size,
    train_loss,
    val_loss,
    avg_trn_ano_scr,
    std_trn_ano_scr,
    avg_val_ano_scr,
    std_val_ano_scr,
    avg_tst_ano_scr,
    std_tst_ano_scr,
    avg_true_ano_scr,
    std_true_ano_scr,
    IR,
    roc_auc,
    pr_auc,
    sampling_term,
    PATH,
):
    cols = [
        "trainer_name",
        "now",
        "return_epoch",
        "early_stop_round",
        "hidden_size",
        "trn_loss",
        "val_loss",
        "avg_trn_ano_scr",
        "std_trn_ano_scr",
        "avg_val_ano_scr",
        "std_val_ano_scr",
        "avg_tst_ano_scr",
        "std_tst_ano_scr",
        "avg_true_ano_scr",
        "std_true_ano_scr",
        "IR",
        "roc_auc",
        "pr_auc",
        "sampling_term",
        "config",
    ]
    if PATH is None:
        if config.data == "tabular":
            PATH = "../run_results_tabular/"
        else:
            PATH = "../run_results_time/"
    now = datetime.datetime.now()
    file_list = os.listdir(PATH)
    file_name = "result_" + config.data_name + ".csv"
    if file_name not in file_list:
        print("New file generated!")
        df = pd.DataFrame(columns=cols)
        df.to_csv(PATH + "result_" + config.data_name + ".csv", index=False)
    df = pd.read_csv(PATH + "result_" + config.data_name + ".csv")
    df = df.append(
        {
            "trainer_name": config.trainer_name,
            "now": now,
            "return_epoch": return_epoch,
            "early_stop_round": config.early_stop_round,
            "hidden_size": hidden_size,
            "trn_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "avg_trn_ano_scr": round(avg_trn_ano_scr, 4),
            "std_trn_ano_scr": round(std_trn_ano_scr, 4),
            "avg_val_ano_scr": round(avg_val_ano_scr, 4),
            "std_val_ano_scr": round(std_val_ano_scr, 4),
            "avg_tst_ano_scr": round(avg_tst_ano_scr, 4),
            "std_tst_ano_scr": round(std_tst_ano_scr, 4),
            "avg_true_ano_scr": round(avg_true_ano_scr, 4),
            "std_true_ano_scr": round(std_true_ano_scr, 4),
            "IR": IR,
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "sampling_term": sampling_term,
            "config": config,
        },
        ignore_index=True,
    )
    return df


def inference(
    config,
    total_dataloader,
    best_model,
    train_x,
    valid_x,
    total_x,
    total_y,
    return_epoch,
    hidden_size,
    train_loss,
    val_loss,
    IR,
    sampling_term,
    PATH=None,
):
    if config.data == "tabular":
        window_anomaly_score_result = np.zeros(len(total_x))
        trn_end_idx = len(train_x)
        val_end_idx = len(train_x) + len(valid_x)
    else:
        window_anomaly_score_result = np.zeros(len(total_x) - config.window_size + 1)
        trn_end_idx = len(train_x) - config.window_size + 1
        val_end_idx = len(train_x) + len(valid_x) - config.window_size + 1

    window_anomaly_score_result = _get_total_anomaly_score(
        total_dataloader, best_model, window_anomaly_score_result
    )

    avg_true_ano_scr, std_true_ano_scr = _get_true_anomaly_info(
        window_anomaly_score_result, total_y
    )

    trn_ano_scr, val_ano_scr, tst_ano_scr = _get_anomaly_score_result(
        window_anomaly_score_result, trn_end_idx, val_end_idx
    )

    avg_trn_ano_scr, std_trn_ano_scr = np.mean(trn_ano_scr), np.std(trn_ano_scr)
    avg_val_ano_scr, std_val_ano_scr = np.mean(val_ano_scr), np.std(val_ano_scr)
    avg_tst_ano_scr, std_tst_ano_scr = np.mean(tst_ano_scr), np.std(tst_ano_scr)

    roc_auc, pr_auc = _get_score(
        window_anomaly_score_result, total_y, val_end_idx, config
    )

    df = _save_final_result(
        config,
        return_epoch,
        hidden_size,
        train_loss,
        val_loss,
        avg_trn_ano_scr,
        std_trn_ano_scr,
        avg_val_ano_scr,
        std_val_ano_scr,
        avg_tst_ano_scr,
        std_tst_ano_scr,
        avg_true_ano_scr,
        std_true_ano_scr,
        IR,
        roc_auc,
        pr_auc,
        sampling_term,
        PATH,
    )
    return df
