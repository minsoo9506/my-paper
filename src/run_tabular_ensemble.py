import argparse

from models.BaseAutoEncoder import BaseSeq2Seq
from dataload.tabular import tabularDataset
from preprocess import split_train_valid_test_tabular
from utils import inference, ensemble_inference
from simulation_trainer import NewTrainer, BaseTrainer

import winsound
import pickle

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
    p.add_argument("--data", type=str, default="tabular")
    p.add_argument("--data_name", type=str, default="all")
    p.add_argument("--trainer_name", type=str, default="BaseTrainer")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    # model
    p.add_argument("--hidden_size", type=list, default=[2])
    # data
    p.add_argument("--train_ratio", type=int, default=0.7)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--window_size", type=int, default=60)
    # experiment
    p.add_argument("--n_epochs", type=int, default=1000)
    p.add_argument("--early_stop_round", type=int, default=50)
    p.add_argument("--initial_epochs", type=list, default=[5, 20])
    p.add_argument("--sampling_term", type=list, default=[1, 5])
    p.add_argument("--sampling_ratio", type=list, default=[0.01, 0.1])
    
    config = p.parse_args()
    device = "cpu" if config.gpu_id < 0 else "cuda:" + str(config.gpu_id)
    config.device = device

    return config


def main(config):
    # torch.backends.cudnn.deterministic = True
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    is_debug = False
    PATH = "../tabular_data/"
    file_list = os.listdir(PATH)
    file_list_csv = [file for file in file_list if file.endswith(".csv")]

    for file_name in file_list_csv:
        PATH = "../tabular_data/"
        config.data_name = file_name.split(".")[0]

        (
            train_x,
            valid_x,
            test_x,
            train_y,
            valid_y,
            test_y,
        ) = split_train_valid_test_tabular(PATH, config.data_name, config.train_ratio)
        # resize 'window_size' = 'col_len'
        config.window_size = train_x.shape[1]

        train_dataset = tabularDataset(train_x, train_y)
        valid_dataset = tabularDataset(valid_x, valid_y)

        train_dataloader = DataLoader(
            train_dataset, shuffle=False, batch_size=config.batch_size
        )
        valid_dataloader = DataLoader(
            valid_dataset, shuffle=False, batch_size=config.batch_size
        )

        total_x = np.concatenate([train_x, valid_x, test_x])
        total_y = np.concatenate([train_y, valid_y, test_y])
        IR = round((len(total_y) - np.sum(total_y)) / np.sum(total_y), 4)
        # for inference
        total_dataset = tabularDataset(total_x, total_y)
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
            sampling_ratio = 0
            initial_epoch = 0
            PATH = "../run_results_tabular/"
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
                sampling_ratio,
                initial_epoch,
                PATH
            )

            df.to_csv(PATH + "result_" + config.data_name + ".csv", index=False)

        for hidden_size in config.hidden_size:
            for sampling_ratio in config.sampling_ratio:
                for initial_epoch in config.initial_epochs: 
                    for sampling_term in config.sampling_term:
                        print(
                            f"-----NewTrainer -----"
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

                        train_loss, val_loss, return_epoch, best_model, _, _, _ = trainer.train(
                            train_x=train_x,
                            train_y=train_y,
                            train_loader=train_dataloader,
                            val_loader=valid_dataloader,
                            sampling_term=sampling_term,
                            initial_epoch=initial_epoch,
                            sampling_ratio=sampling_ratio,
                            config=config,
                            use_wandb=False,
                            is_debug=is_debug
                        )

                        best_model.to("cpu")
                        PATH = "../run_results_tabular/"
                        df, tst_ano_score = ensemble_inference(
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
                            sampling_ratio,
                            initial_epoch,
                            PATH
                        )
                        df.to_csv(PATH + "result_" + config.data_name + ".csv", index=False)
            
                        hp = '_hs' + str(hidden_size) + '_st' + str(sampling_term) + '_sr' + str(sampling_ratio) + '_ie' + str(initial_epoch)
                        with open('./ensemble_tab;ular_1/' + config.data_name + hp + '.pickle', 'wb') as f:
                            pickle.dump(tst_ano_score, f, pickle.HIGHEST_PROTOCOL)
                        
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    config = define_argparser()
    main(config)
    frequency = 800
    duration = 2000
    winsound.Beep(frequency, duration)
