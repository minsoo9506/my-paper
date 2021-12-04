import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 16,
        num_layers: int = 3,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p,
            batch_first=True,
        )

    def forward(self, x):
        # |x| = (batch_size, seq_len, input_size)
        _, h = self.rnn(x)
        # |h[0]| = (num_layers, batch_size, hidden_size)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 16,
        num_layers: int = 3,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size

        self.rnn = nn.LSTM(
            input_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout_p,
            batch_first=True,
        )

        self.last_fully = nn.Linear(hidden_size, input_size)

    def forward(self, x, y_t_1, h_t_1):
        # y_t_1 : 이전 time-step에서의 output
        # h_t_1 = (h,c) : 이전 time-step의 hidden, cell state

        # |x| = (batch_size, 1, input_size + hidden_size)
        # |y_t_1| = (batch_size, 1, hidden_size)
        # |h_t_1[0]| = (num_layers, batch_size, hidden_size)
        batch_size = x.size(0)
        hidden_size = h_t_1[0].size(-1)

        if y_t_1 is None:
            y_t_1 = x.new(batch_size, 1, hidden_size).zero_()

        x = torch.cat([x, y_t_1], dim=-1)

        y_t, h_t = self.rnn(x, h_t_1)
        pred = self.last_fully(y_t.squeeze(1))
        # |pred| = (batch_size, input_size)
        # |y_t| = (batch_size, 1, hidden_size)
        # |h_t[0]| = (num_layers, batch_size, hidden_size)

        return pred, y_t, h_t


class Seq2Seq(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 16,
        num_layers: int = 3,
        dropout_p: float = 0.2,
        seq_len: int = 16,
    ):
        super().__init__()

        self.rnn_encoder = Encoder(input_size, hidden_size, num_layers, dropout_p)

        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout_p)

        self.seq_len = seq_len

    def forward(self, enc_x, dec_x):
        enc_h = self.rnn_encoder(enc_x)
        _shape = enc_x.shape
        preds = enc_x.new(_shape[0], _shape[1], _shape[2]).zero_()
        # teacher-forcing
        x = dec_x[:, 0, :].unsqueeze(dim=1)
        y_t_1 = None
        h_t_1 = enc_h
        pred, y_t, h_t = self.decoder(x, y_t_1, h_t_1)
        preds[:, 0, :] = pred
        for i in range(1, self.seq_len):
            x = dec_x[:, i, :].unsqueeze(dim=1)
            pred, y_t, h_t = self.decoder(x, y_t, h_t)
            preds[:, i, :] = pred
        # |preds| = (batch_size, seq_len, input_size)
        return preds
