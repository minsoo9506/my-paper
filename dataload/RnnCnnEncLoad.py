import numpy as np
import torch
from torch.utils.data import Dataset

class ReconstructionBasedDataset(Dataset):
    '''
    Dataset

    Args:
        data (int): np.array
        seq_len (int): time series window
        feature_dim (int)
    '''
    def __init__(self,
        data,
        seq_len: int,
        feature_dim: int,
        ):
        super().__init__()
        # data preprocessing
        _mean = np.mean(data, axis=0)
        _std = np.std(data, axis=0)
        data = (data - _mean) / _std
        
        x = data[1:seq_len+1, :].reshape(1, seq_len, feature_dim)
        for start_idx in range(2, len(data)):
            end_idx = start_idx + seq_len - 1
            if end_idx > len(data)-2:
                break
            seq_x = data[start_idx:end_idx+1, :].reshape(1, seq_len, feature_dim)
            x = np.vstack([x, seq_x])
        self.rnn_enc_x = torch.tensor(x, dtype=torch.float32)
        self.target = torch.tensor(x, dtype=torch.float32)
        
        x = data[1:seq_len+1, :].reshape(1, feature_dim, seq_len)
        for start_idx in range(2, len(data)):
            end_idx = start_idx + seq_len - 1
            if end_idx > len(data)-2:
                break
            seq_x = data[start_idx:end_idx+1, :].reshape(1, feature_dim, seq_len)
            x = np.vstack([x, seq_x]) 
        self.cnn_enc_x = torch.tensor(x, dtype=torch.float32)

        # x for decoder
        x = data[0:seq_len, :].reshape(1, seq_len, feature_dim)
        for start_idx in range(1, len(data)):
            end_idx = start_idx + seq_len - 1
            if end_idx > len(data)-2:
                break
            seq_x = data[start_idx:end_idx+1, :].reshape(1, seq_len, feature_dim)
            x = np.vstack([x, seq_x])
        self.dec_x = torch.tensor(x, dtype=torch.float32)
    
    def __len__(self):
        return self.rnn_enc_x.shape[0]
    
    def __getitem__(self, idx):
        rnn_enc_x = self.rnn_enc_x[idx]
        cnn_enc_x = self.cnn_enc_x[idx]
        dec_x = self.dec_x[idx]
        target = self.target[idx]
        # |rnn_enc_x| = |target| = |dec_x| = (seq_len, feature_dim)
        # |cnn_x| = (feature_dim, seq_len)
        return rnn_enc_x, cnn_enc_x, dec_x, target

class InferenceDataset(Dataset):
    '''
    Dataset

    Args:
        data (int): np.array
        seq_len (int): time series window
        feature_dim (int)
    '''
    def __init__(self,
        data,
        seq_len: int,
        feature_dim: int,
        ):
        super().__init__()
        # data preprocessing
        _mean = np.mean(data, axis=0)
        _std = np.std(data, axis=0)
        data = (data - _mean) / _std
        
        x = data[1:seq_len+1, :].reshape(1, seq_len, feature_dim)
        for start_idx in range(seq_len + 1, len(data), seq_len):
            end_idx = start_idx + seq_len - 1
            if end_idx > len(data)-2:
                break
            seq_x = data[start_idx:end_idx+1, :].reshape(1, seq_len, feature_dim)
            x = np.vstack([x, seq_x])
        self.rnn_enc_x = torch.tensor(x, dtype=torch.float32)
        self.target = torch.tensor(x, dtype=torch.float32)
        
        x = data[1:seq_len+1, :].reshape(1, feature_dim, seq_len)
        for start_idx in range(seq_len + 1, len(data), seq_len):
            end_idx = start_idx + seq_len - 1
            if end_idx > len(data)-2:
                break
            seq_x = data[start_idx:end_idx+1, :].reshape(1, feature_dim, seq_len)
            x = np.vstack([x, seq_x]) 
        self.cnn_enc_x = torch.tensor(x, dtype=torch.float32)

        # x for decoder
        x = data[0:seq_len, :].reshape(1, seq_len, feature_dim)
        for start_idx in range(seq_len, len(data), seq_len):
            end_idx = start_idx + seq_len - 1
            if end_idx > len(data)-2:
                break
            seq_x = data[start_idx:end_idx+1, :].reshape(1, seq_len, feature_dim)
            x = np.vstack([x, seq_x])
        self.dec_x = torch.tensor(x, dtype=torch.float32)
    
    def __len__(self):
        return self.rnn_enc_x.shape[0]
    
    def __getitem__(self, idx):
        rnn_enc_x = self.rnn_enc_x[idx]
        cnn_enc_x = self.cnn_enc_x[idx]
        dec_x = self.dec_x[idx]
        target = self.target[idx]
        # |rnn_enc_x| = |target| = |dec_x| = (seq_len, feature_dim)
        # |cnn_x| = (feature_dim, seq_len)
        return rnn_enc_x, cnn_enc_x, dec_x, target