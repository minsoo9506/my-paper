import argparse

from models.BaseAutoEncoder import BaseSeq2Seq
from preprocess import split_train_valid_test
from dataload.window_based import WindowBasedDataset
from utils import inference
from trainer import NewTrainer, BaseTrainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import random
import os


def define_argparser():
    p = argparse.ArgumentParser()
    # set up
    p.add_argument("--data", type=str, default="time")
    p.add_argument("--data_name", type=str, default="all")
    p.add_argument("--trainer_name", type=str, default="BaseTrainer")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    # model
    p.add_argument("--hidden_size", type=list, default=[2, 4, 8])
    # data
    p.add_argument("--train_ratio", type=int, default=0.7)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--window_size", type=int, default=60)
    # experiment
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--early_stop_round", type=int, default=1000)
    p.add_argument("--initial_epochs", type=int, default=20)
    p.add_argument("--sampling_term", type=list, default=[1, 4, 16])

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

    PATH = "../time_series_data/"
    file_list = os.listdir(PATH)
    file_list_txt = [file for file in file_list if file.endswith(".txt")]

    for file_name in file_list_txt:
        PATH = "../time_series_data/"
        config.data_name = file_name.split(".")[0]

        train_x, valid_x, test_x, train_y, valid_y, test_y = split_train_valid_test(
            PATH, config.data_name, config.train_ratio
        )
        train_dataset = WindowBasedDataset(train_x, train_y, config.window_size)
        valid_dataset = WindowBasedDataset(valid_x, valid_y, config.window_size)

        train_dataloader = DataLoader(
            train_dataset, shuffle=False, batch_size=config.batch_size
        )
        valid_dataloader = DataLoader(
            valid_dataset, shuffle=False, batch_size=config.batch_size
        )

        total_x = np.concatenate([train_x, valid_x, test_x])
        total_y = np.concatenate([train_y, valid_y, test_y])
        IR = round((len(total_y) - np.sum(total_y)) / np.sum(total_y), 4)
        # fix y label (because window_base approach)
        splited_data_name = config.data_name.split("_")
        abnormal_start_idx = int(splited_data_name[-2])
        abnormal_end_idx = int(splited_data_name[-1])
        total_y[abnormal_start_idx - config.window_size : abnormal_start_idx] = 1
        total_y[abnormal_end_idx : abnormal_end_idx + config.window_size] = 1
        # for inference
        total_dataset = WindowBasedDataset(total_x, total_y, config.window_size)
        total_dataloader = DataLoader(
            total_dataset, shuffle=False, batch_size=config.batch_size
        )

        for hidden_size in config.hidden_size:
            print(f"-----BaseTrainer starts with hidden_size={hidden_size}-----")
            config.trainer_name = "BaseTrainer"

            model = BaseSeq2Seq(
                input_size=config.window_size,
                hidden_size=hidden_size,
                output_size=config.window_size,
                dropout_p=0.2,
            ).to(config.device)

            optimizer = optim.Adam(model.parameters())
            criterion = nn.MSELoss()

            # train
            trainer = BaseTrainer(model=model, optimizer=optimizer, crit=criterion)

            train_loss, val_loss, return_epoch, best_model = trainer.train(
                train_loader=train_dataloader,
                val_loader=valid_dataloader,
                config=config,
                use_wandb=False,
            )

            best_model.to("cpu")
            sampling_term = 0

            df = inference(
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
            )

            PATH = "../run_results_time/"
            df.to_csv(PATH + "result_" + config.data_name + ".csv", index=False)

        for hidden_size in config.hidden_size:
            for sampling_term in config.sampling_term:
                print(
                    f"-----NewTrainer starts with hidden_size={hidden_size}, sampling_term={sampling_term}-----"
                )
                config.trainer_name = "NewTrainer"

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
                    train_loader=train_dataloader,
                    val_loader=valid_dataloader,
                    config=config,
                    sampling_term=sampling_term,
                    use_wandb=False,
                )

                best_model.to("cpu")

                df = inference(
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
                )

                PATH = "../run_results_time/"
                df.to_csv(PATH + "result_" + config.data_name + ".csv", index=False)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
