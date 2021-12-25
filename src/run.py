import argparse

from src.models.BaseAutoEncoder import BaseSeq2Seq
from src.preprocess import split_train_valid_test
from src.dataload.window_based import WindowBasedDataset
from src.trainer import NewTrainer, BaseTrainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import random
import pandas as pd


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--total_iter", type=int, default=5)
    # set up
    p.add_argument("--data_name", type=str)
    p.add_argument("--trainer_name", type=str, default="BaseTrainer")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    # model
    p.add_argument("--hidden_size", type=int, default=4)
    # data
    p.add_argument("--train_ratio", type=int, default=0.7)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--window_size", type=int, default=60)
    # experiment
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--early_stop_round", type=int, default=10)
    p.add_argument("--initial_epochs", type=int, default=10)
    p.add_argument("--sampling_term", type=int, default=5)

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

    train_dataset = WindowBasedDataset(train_x, train_y, config.window_size)
    valid_dataset = WindowBasedDataset(valid_x, valid_y, config.window_size)
    test_dataset = WindowBasedDataset(test_x, test_y, config.window_size)

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=config.batch_size
    )
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, batch_size=config.batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=config.batch_size
    )

    model = BaseSeq2Seq(
        input_size=config.window_size,
        hidden_size=config.hidden_size,
        output_size=config.window_size,
        dropout_p=0.2,
    ).to(config.device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    if config.trainer_name == "NewTrainer":
        trainer = NewTrainer(model=model, optimizer=optimizer, crit=criterion)

        best_model = trainer.train(
            train_x=train_x,
            train_y=train_y,
            val_loader=valid_dataloader,
            config=config,
            use_wandb=False,
        )
    else:
        trainer = BaseTrainer(model=model, optimizer=optimizer, crit=criterion)

        best_model = trainer.train(
            train_loader=train_dataloader,
            val_loader=valid_dataloader,
            config=config,
            use_wandb=False,
        )

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
