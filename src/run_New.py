import argparse

from models.BaseAutoEncoder import BaseSeq2Seq
from preprocess import split_train_valid_test
from dataload.window_based import WindowBasedDataset
from trainer import NewTrainer, BaseTrainer
from utils import get_score, get_total_anomaly_score, get_true_anomaly_info

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import random
import pandas as pd

from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
)

import datetime
import os


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--total_iter", type=int, default=1)
    # set up
    p.add_argument("--data_name", type=str)
    p.add_argument("--trainer_name", type=str, default="NewTrainer")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    # model
    p.add_argument("--hidden_size", type=list, default=[2, 4, 8])
    # data
    p.add_argument("--train_ratio", type=int, default=0.7)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--window_size", type=int, default=60)
    # experiment
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--early_stop_round", type=int, default=10)
    p.add_argument("--initial_epochs", type=int, default=10)
    p.add_argument("--sampling_term", type=list, default=[2, 4, 8])

    config = p.parse_args()
    device = "cpu" if config.gpu_id < 0 else "cuda:" + str(config.gpu_id)
    config.device = device

    return config


def main(config):
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    PATH = "../UCR_Anomaly_FullData/"

    train_x, valid_x, test_x, train_y, valid_y, test_y = split_train_valid_test(
        PATH, config.data_name, config.train_ratio
    )

    print(f"train_x.shape : {train_x.shape}")
    print(f"train_y.shape : {train_y.shape}")
    print(f"valid_x.shape : {valid_x.shape}")
    print(f"valid_y.shape : {valid_y.shape}")
    print(f"test_x.shape  : {test_x.shape}")
    print(f"test_y.shape  : {test_y.shape}")

    # data setting
    train_dataset = WindowBasedDataset(train_x, train_y, config.window_size)
    valid_dataset = WindowBasedDataset(valid_x, valid_y, config.window_size)
    # test_dataset = WindowBasedDataset(test_x, test_y, config.window_size)

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=config.batch_size
    )
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, batch_size=config.batch_size
    )
    # test_dataloader = DataLoader(
    #     test_dataset, shuffle=False, batch_size=config.batch_size
    # )

    total_x = np.concatenate([train_x, valid_x, test_x])
    total_y = np.concatenate([train_y, valid_y, test_y])
    splited_data_name = config.data_name.split("_")
    abnormal_start_idx, abnormal_end_idx = int(splited_data_name[-2]), int(
        splited_data_name[-1]
    )
    total_y[abnormal_start_idx - config.window_size : abnormal_start_idx] = 1
    total_y[abnormal_end_idx : abnormal_end_idx + config.window_size] = 1

    for iteration in range(config.total_iter):
        for hidden_size in config.hidden_size:
            for sampling_term in config.sampling_term:
                print(f"-----iteration {iteration} starts with hidden_size={hidden_size}, sampling_term={sampling_term}-----")
                # model setting
                model = BaseSeq2Seq(
                    input_size=config.window_size,
                    hidden_size=hidden_size,
                    output_size=config.window_size,
                    dropout_p=0.2,
                ).to(config.device)

                optimizer = optim.Adam(model.parameters())
                criterion = nn.MSELoss()

                # train
                trainer = NewTrainer(model=model, optimizer=optimizer, crit=criterion)

                train_loss, val_loss, return_epoch, best_model = trainer.train(
                    train_x=train_x,
                    train_y=train_y,
                    val_loader=valid_dataloader,
                    config=config,
                    sampling_term=sampling_term,
                    use_wandb=False,
                )

                # test
                total_dataset = WindowBasedDataset(total_x, total_y, config.window_size)
                total_dataloader = DataLoader(
                    total_dataset, shuffle=False, batch_size=config.batch_size
                )
                best_model.to("cpu")
                # get anomaly score of all data
                window_anomaly_score_result = np.zeros(len(total_x) - config.window_size + 1)
                window_anomaly_score_result = get_total_anomaly_score(
                    total_dataloader, best_model, window_anomaly_score_result
                )

                stat_true_anomaly_scores = get_true_anomaly_info(
                    window_anomaly_score_result, total_y
                )

                train_anomaly_score = window_anomaly_score_result[
                    : len(train_x) - config.window_size + 1
                ]

                val_anomaly_score = window_anomaly_score_result[
                    : len(train_x) - config.window_size + 1
                ]

                test_anomaly_score = window_anomaly_score_result[
                    : len(train_x) - config.window_size + 1
                ]

                avg_train_anomaly_score, std_train_anomaly_score = np.mean(
                    train_anomaly_score
                ), np.std(train_anomaly_score)
                avg_val_anomaly_score, std_val_anomaly_score = np.mean(
                    val_anomaly_score
                ), np.std(val_anomaly_score)
                avg_test_anomaly_score, std_test_anomaly_score = np.mean(
                    test_anomaly_score
                ), np.std(test_anomaly_score)

                threshold_list = avg_train_anomaly_score + np.arange(0, 3.5, 0.5) * std_train_anomaly_score

                scores = []
                for threshold in threshold_list:
                    score = get_score(window_anomaly_score_result, total_y, threshold, config)
                    scores.append(score)

                # save result
                cols = [
                    "trainer_name",
                    "now",
                    "return_epoch",
                    "hidden_size",
                    "train_loss",
                    "val_loss",
                    "avg_train_anomaly_score",
                    "std_train_anomaly_score",
                    "avg_val_anomaly_score",
                    "std_val_anomaly_score",
                    "avg_test_anomaly_score",
                    "std_test_anomaly_score",
                    "avg_true_anomaly_score",
                    "std_true_anomaly_score",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "pr_auc",
                    "threshold",
                    "config"
                ]

                PATH = "../run_results/"
                now = datetime.datetime.now()

                file_list = os.listdir(PATH)
                file_name = "result_" + config.data_name + ".csv"
                if file_name not in file_list:
                    print("New file generated!")
                    df = pd.DataFrame(columns=cols)
                    df.to_csv(PATH + "result_" + config.data_name + ".csv", index=False)

                df = pd.read_csv(PATH + "result_" + config.data_name + ".csv")

                for idx, threshold in enumerate(threshold_list):
                    df = df.append(
                        {
                            "trainer_name": config.trainer_name,
                            "now": now,
                            "return_epoch": return_epoch,
                            "hidden_size": hidden_size,
                            "train_loss": round(train_loss, 4),
                            "val_loss": round(val_loss, 4),
                            "avg_train_anomaly_score": round(avg_train_anomaly_score, 4),
                            "std_train_anomaly_score": round(std_train_anomaly_score, 4),
                            "avg_val_anomaly_score": round(avg_val_anomaly_score, 4),
                            "std_val_anomaly_score": round(std_val_anomaly_score, 4),
                            "avg_test_anomaly_score": round(avg_test_anomaly_score, 4),
                            "std_test_anomaly_score": round(std_test_anomaly_score, 4),
                            "avg_true_anomaly_score": round(stat_true_anomaly_scores[0], 4),
                            "std_true_anomaly_score": round(stat_true_anomaly_scores[1], 4),
                            "accuracy": round(scores[idx][0], 4),
                            "precision": round(scores[idx][1], 4),
                            "recall": round(scores[idx][2], 4),
                            "f1_score": round(scores[idx][3], 4),
                            "roc_auc": round(scores[idx][4], 4),
                            "pr_auc": round(scores[idx][5], 4),
                            "threshold": round(threshold, 4),
                            "config": config
                        },
                        ignore_index=True,
                    )

                df.to_csv(PATH + "result_" + config.data_name + ".csv", index=False)

    # # save best model
    # from datetime import datetime

    # today = datetime.now()
    # time_chk = (
    #     str(today).split(" ")[0]
    #     + "-"
    #     + "-".join(str(today).split(" ")[1].split(":")[:2])
    # )
    # PATH = (
    #     "./saved_models/"
    #     + time_chk
    #     + "-"
    #     + config.train_data_name.split(".")[0]
    #     + ".pt"
    # )
    # torch.save(
    #     {
    #         "model": best_model.state_dict(),
    #         "config": config,
    #     },
    #     PATH,
    # )


if __name__ == "__main__":
    config = define_argparser()
    main(config)
