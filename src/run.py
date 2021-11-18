import argparse
import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.seq2seq import Seq2Seq
from dataload.seq2seqLoad import Seq2SeqDataset
from trainer import Seq2SeqTrainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--train_data_name", type=str)
    p.add_argument("--valid_data_name", type=str)
    p.add_argument("--model_name", type=str, default="seq2seq")
    p.add_argument("--project", type=str, default="base_seq2seq")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden_size", type=int, default=32)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--early_stop_round", type=int, default=10)

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

    # data
    TRAIN_DATA_PATH = "./Mydata/" + config.train_data_name
    VALID_DATA_PATH = "./Mydata/" + config.valid_data_name

    # data_loader
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    FEATURE_DIM = train_data.shape[1]

    if config.model_name == "seq2seq":
        print(f"Train Data Shape = {train_data.shape}")
        train_dataset = Seq2SeqDataset(
            data=np.array(train_data),
            seq_len=config.seq_len,
            feature_dim=FEATURE_DIM,
        )
        valid_data = pd.read_csv(VALID_DATA_PATH)
        print(f"Valid Data Shape = {valid_data.shape}")
        valid_dataset = Seq2SeqDataset(
            data=np.array(valid_data),
            seq_len=config.seq_len,
            feature_dim=FEATURE_DIM,
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=config.batch_size
    )
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, batch_size=config.batch_size
    )

    if config.model_name == "seq2seq":
        # model, optimizer, crit
        model = Seq2Seq(
            input_size=FEATURE_DIM,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout_p=0.2,
            seq_len=config.seq_len,
        ).to(config.device)
        optimizer = optim.Adam(model.parameters())
        crit = nn.L1Loss()

        # train
        trainer = Seq2SeqTrainer(model, optimizer, crit)
        best_model = trainer.train(train_dataloader, valid_dataloader, config)

    # save best model
    from datetime import datetime

    today = datetime.now()
    time_chk = (
        str(today).split(" ")[0]
        + "-"
        + "-".join(str(today).split(" ")[1].split(":")[:2])
    )
    PATH = (
        "./saved_models/"
        + time_chk
        + "-"
        + config.train_data_name.split(".")[0]
        + ".pt"
    )
    torch.save(
        {
            "model": best_model.state_dict(),
            "config": config,
        },
        PATH,
    )


if __name__ == "__main__":
    config = define_argparser()
    main(config)
