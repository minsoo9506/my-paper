import torch
import torch.nn as nn
from models.basicModel import RnnEncoder, CnnEncoder, Decoder

class MyModel(nn.Module):
    def __init__(
        self, 
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 3,
        dropout_p: float = 0.2,
        seq_len: int = 16
    ):
        super().__init__()

        self.rnn_encoder = RnnEncoder(
            input_size,
            hidden_size,
            num_layers,
            dropout_p
        )

        self.cnn_encoder = CnnEncoder(
            input_size
        )

        self.decoder = Decoder(
            input_size,
            seq_len,
            hidden_size,
            num_layers,
            dropout_p
        )

        self.seq_len = seq_len

    def forward(self, rnn_enc_x, cnn_enc_x, dec_x):
        rnn_h = self.rnn_encoder(rnn_enc_x)
        cnn_emb = self.cnn_encoder(cnn_enc_x)
        _shape = rnn_enc_x.shape
        preds = rnn_enc_x.new(_shape[0], _shape[1], _shape[2]).zero_()
        # teacher-forcing
        x = torch.cat([dec_x[:, 0, :].unsqueeze(dim=1), cnn_emb], dim=2)
        y_t_1 = None
        h_t_1 = rnn_h
        pred, y_t, h_t = self.decoder(x, y_t_1, h_t_1)
        preds[:, 0, :] = pred
        for i in range(1, self.seq_len):
            x = torch.cat([dec_x[:, i, :].unsqueeze(dim=1), cnn_emb], dim=2)
            pred, y_t, h_t = self.decoder(x, y_t, h_t)
            preds[:, i, :] = pred
        # |preds| = (batch_size, seq_len, feature_dim)
        return preds