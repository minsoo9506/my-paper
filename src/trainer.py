from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataload.window_based import WindowBasedDataset, WeightedWindowBasedDataset


class BaseTrainer:
    def __init__(self, model, optimizer, crit):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

    def _train(self, train_loader, device):
        self.model.train()
        total_loss = 0
        for input_x, _ in train_loader:
            if device != "cpu":
                input_x = input_x.to(device)
                output_x = input_x.to(device)
            y_hat = self.model(input_x)
            loss = self.crit(y_hat, output_x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # prevent memory leak
            total_loss += float(loss)
        return total_loss / len(train_loader)

    def _validate(self, val_loader, device):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for input_x, _ in val_loader:
                if device != "cpu":
                    input_x = input_x.to(device)
                    output_x = input_x.to(device)
                y_hat = self.model(input_x)
                loss = self.crit(y_hat, output_x)
                # prevent memory leak
                total_loss += float(loss)
            return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, config, use_wandb):
        lowest_loss = np.inf
        best_model = None
        early_stop_round = 0

        if use_wandb:
            import wandb

            wandb.login()
            wandb.init(project=config.project, config=config)
            wandb.watch(self.model, self.crit, log="gradients", log_freq=100)

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_loader, config.device)
            valid_loss = self._validate(val_loader, config.device)

            if use_wandb:
                wandb.log({"train_loss": train_loss})
                wandb.log({"valid_loss": valid_loss})

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                early_stop_round = 0
            else:
                early_stop_round += 1
            if early_stop_round == config.early_stop_round:
                print(f"Early Stopped!")
                print(
                    f"Epoch{epoch_index+1}/{config.n_epochs}: train_loss={train_loss:.3f}, valid_loss={valid_loss:.3f}"
                )
                break
            if (epoch_index + 1) % 10 == 0:
                print(
                    f"Epoch{epoch_index+1}/{config.n_epochs}: train_loss={train_loss:.3f}, valid_loss={valid_loss:.3f}"
                )
        self.model.load_state_dict(best_model)
        return self.model


class NewTrainer:
    def __init__(self, model, optimizer, crit):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

    def _train(self, train_loader, cal_train_recon_error, train_recon_error, device):
        self.model.train()
        total_loss = 0
        idx = 0
        for input_x, _ in train_loader:
            input_x_batch_size = input_x.shape[0]
            if device != "cpu":
                input_x = input_x.to(device)
                output_x = input_x.to(device)
            y_hat = self.model(input_x)
            # calculate train recon error
            if cal_train_recon_error:
                anomaly_score = abs(input_x - y_hat)
                mean_anomaly_score = torch.mean(anomaly_score, 1).numpy()
                train_recon_error[idx:idx+input_x_batch_size] = mean_anomaly_score
                idx += input_x_batch_size
            loss = self.crit(y_hat, output_x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # prevent memory leak
            total_loss += float(loss)
        return total_loss / len(train_loader), train_recon_error

    def _validate(self, val_loader, device):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for input_x, _ in val_loader:
                if device != "cpu":
                    input_x = input_x.to(device)
                    output_x = input_x.to(device)
                y_hat = self.model(input_x)
                loss = self.crit(y_hat, output_x)
                # prevent memory leak
                total_loss += float(loss)
            return total_loss / len(val_loader)

    def train(self, train_x, train_y, val_loader, config, use_wandb):
        lowest_loss = np.inf
        best_model = None
        early_stop_round = 0
        
        # 일단 처음 train loader 만들기
        train_dataset = WindowBasedDataset(train_x, train_y, config.window_size)
        train_loader = DataLoader(
            train_dataset, shuffle=False, batch_size=config.batch_size
        )
        
        # train reconstruction error
        data_len = len(train_x) - config.window_size + 1
        train_recon_error = np.zeros(data_len)
        
        # 특정 에포크 지나면 

        if use_wandb:
            import wandb
            wandb.login()
            wandb.init(project=config.project, config=config)
            wandb.watch(self.model, self.crit, log="gradients", log_freq=100)

        for epoch_index in range(config.n_epochs):
            if (epoch_index + 1) >= config.initial_epochs:
                train_loss, train_recon_error = self._train(train_loader, True, train_recon_error,  config.device)
                valid_loss = self._validate(val_loader, config.device)
                # calculate weight
                sample_weight = 1 - train_recon_error / np.max(train_recon_error)
                # sampling with weight
                train_dataset = WeightedWindowBasedDataset(train_x, train_y, config.window_size, sample_weight)
                train_loader = DataLoader(
                    train_dataset, shuffle=False, batch_size=config.batch_size
                )
                
            # 나중에 debugging해서 어떤 데이터들이 sample에서 빠지는지 확인 필요 #
                
            else:
                train_loss = self._train(train_loader, False, train_recon_error,  config.device)
                valid_loss = self._validate(val_loader, config.device)

            if use_wandb:
                wandb.log({"train_loss": train_loss})
                wandb.log({"valid_loss": valid_loss})

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                early_stop_round = 0
            else:
                early_stop_round += 1
            if early_stop_round == config.early_stop_round:
                print(f"Early Stopped!")
                print(
                    f"Epoch{epoch_index+1}/{config.n_epochs}: train_loss={train_loss:.3f}, valid_loss={valid_loss:.3f}"
                )
                break
            if (epoch_index + 1) % 10 == 0:
                print(
                    f"Epoch{epoch_index+1}/{config.n_epochs}: train_loss={train_loss:.3f}, valid_loss={valid_loss:.3f}"
                )
        self.model.load_state_dict(best_model)
        return self.model



# class Seq2SeqTrainer:
#     def __init__(self, model, optimizer, crit):
#         super().__init__()
#         self.model = model
#         self.optimizer = optimizer
#         self.crit = crit

#     def _train(self, train_loader, device):
#         self.model.train()
#         total_loss = 0
#         for rnn_enc_x, dec_x, target in train_loader:
#             if device != "cpu":
#                 rnn_enc_x = rnn_enc_x.to(device)
#                 dec_x = dec_x.to(device)
#                 target = target.to(device)
#             y_hat = self.model(rnn_enc_x, dec_x)
#             loss = self.crit(y_hat, target)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             # prevent memory leak
#             total_loss += float(loss)
#         return total_loss / len(train_loader)

#     def _validate(self, val_loader, device):
#         self.model.eval()
#         with torch.no_grad():
#             total_loss = 0
#             for rnn_enc_x, dec_x, target in val_loader:
#                 if device != "cpu":
#                     rnn_enc_x = rnn_enc_x.to(device)
#                     dec_x = dec_x.to(device)
#                     target = target.to(device)
#                 y_hat = self.model(rnn_enc_x, dec_x)
#                 loss = self.crit(y_hat, target)
#                 total_loss += float(loss)
#             return total_loss / len(val_loader)

#     def train(self, train_loader, val_loader, config):
#         lowest_loss = np.inf
#         best_model = None
#         early_stop_round = 0

#         import wandb

#         wandb.login()
#         wandb.init(project=config.project, config=config)
#         # wandb.watch(self.model, self.crit, log="gradients", log_freq=100)

#         for epoch_index in range(config.n_epochs):
#             train_loss = self._train(train_loader, config.device)
#             valid_loss = self._validate(val_loader, config.device)

#             wandb.log({"train_loss": train_loss})
#             wandb.log({"valid_loss": valid_loss})

#             if valid_loss < lowest_loss:
#                 lowest_loss = valid_loss
#                 best_model = deepcopy(self.model.state_dict())
#             else:
#                 early_stop_round += 1
#             if early_stop_round == config.early_stop_round:
#                 print(f"Early Stopped!")
#                 print(
#                     f"Epoch{epoch_index+1}/{config.n_epochs}: train_loss={train_loss:.3f}, valid_loss={valid_loss:.3f}"
#                 )
#                 break
#             if (epoch_index + 1) % 10 == 0:
#                 print(
#                     f"Epoch{epoch_index+1}/{config.n_epochs}: train_loss={train_loss:.3f}, valid_loss={valid_loss:.3f}"
#                 )
#         self.model.load_state_dict(best_model)
#         return self.model