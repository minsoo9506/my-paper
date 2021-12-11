import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 3,
        dropout_p: float = 0.2,
    ):
        """make LSTMEncoder

        Parameters
        ----------
        input_size : int, optional
            feature dimension of data, by default 1
        hidden_size : int, optional
            [description], by default 32
        num_layers : int, optional
            [description], by default 3
        dropout_p : float, optional
            dropout rate, by default 0.2
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
        """[summary]

        Parameters
        ----------
        x : torch.Tensor
            [description]

        Returns
        -------
        h : torch.Tensor
            [description]
        """
        # |x| = (batch_size, seq_len, feature_dim)
        _, h = self.lstm(x)
        # |h[0]| = (num_layers, batch_size, hidden_size) : hidden state
        # |h[1]| = (num_layers, batch_size, hidden_size) : cell state
        return h


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 3,
        dropout_p: float = 0.2,
    ):
        """make LSTMDecoder

        Parameters
        ----------
        input_size : int
            [description], by default 1
        hidden_size : int, optional
            [description], by default 32
        num_layers : int, optional
            [description], by default 3
        dropout_p : float, optional
            [description], by default 0.2
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size + hidden_size,  # 이전 input data와 hidden state
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p,
            batch_first=True,
        )

        self.last_fully = nn.Linear(in_features=hidden_size, out_features=input_size)

    def forward(self, x: torch.Tensor, y_t_1: torch.Tensor, h_t_1: torch.Tensor):
        """[summary]

        Parameters
        ----------
        x : torch.Tensor
            직전 time-step에서의 input data
            |x| = (batch_size, 1,  input_size)
        y_t_1 : torch.Tensor
            이전 time-step에서의 output
            |y_t_1| = (batch_size, 1, hidden_size)
        h_t_1 : torch.Tensor
            이전 time-step에서의 (hidden state, cell state)
            |h_t_1[0]| = (num_layers, batch_size, hidden_size)
            |h_t_1[1]| = (num_layers, batch_size, hidden_size)

        Returns
        -------
        pred, y_t, h_t
            [description]
        """

        batch_size = x.size(0)
        hidden_size = h_t_1[0].size(-1)

        if y_t_1 is None:
            y_t_1 = x.new(batch_size, 1, hidden_size).zero_()

        x = torch.cat([x, y_t_1], dim=-1)

        y_t, h_t = self.lstm(x, h_t_1)
        # |y_t| = (batch_size, 1, hidden_size)
        # |h_t[0]| = (num_layers, batch_size, hidden_size) : hidden state
        # |h_t[1]| = (num_layers, batch_size, hidden_size) : cell state
        pred = self.last_fully(y_t.squeeze(1))
        # |pred| = (batch_size, input_size)

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
        """[summary]

        Parameters
        ----------
        input_size : int, optional
            [description], by default 1
        hidden_size : int, optional
            [description], by default 16
        num_layers : int, optional
            [description], by default 3
        dropout_p : float, optional
            [description], by default 0.2
        seq_len : int, optional
            [description], by default 16
        """
        super().__init__()

        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout_p)
        self.decoder = LSTMDecoder(input_size, hidden_size, num_layers, dropout_p)
        self.seq_len = seq_len

    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor):
        """[summary]

        Parameters
        ----------
        enc_x : torch.Tensor
            input data of encoder
            |enc_x| = (batch_size, seq_len, input_size)
        dec_x : torch.Tensor
            input data of decoder
            |enc_x| = (batch_size, seq_len, input_size)

        Returns
        -------
        preds : torch.Tensor
            prediction result
            |preds| = (batch_size, seq_len, input_size)
        """
        enc_h = self.encoder(enc_x)
        _shape = enc_x.shape
        # |_shape| = (batch_size, seq_len, input_size)
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


# class CNNEncoder(nn.Module):
#     def __init__(self, in_channels: int = 1):
#         """make CNNEncoder

#         Parameters
#         ----------
#         in_channels : int, optional
#             feature dimension of data, by default 1
#         """
#         super().__init__()

#         # Conv1d에서 in_channels는 feature_dim
#         # stride=1(default)로 하면
#         # |conv1d output| = seq_len + 2 * pad - kernel_size + 1

#         self.conv1d_1 = nn.Conv1d(
#             in_channels=in_channels, out_channels=8, kernel_size=3, padding=1
#         )
#         self.conv1d_2 = nn.Conv1d(
#             in_channels=8, out_channels=4, kernel_size=3, padding=1
#         )
#         self.conv1d_3 = nn.Conv1d(
#             in_channels=4, out_channels=1, kernel_size=3, padding=1
#         )

#         def weights_init(m):
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight.data)
#                 nn.init.zeros_(m.bias.data)

#         self.conv1d_1.apply(weights_init)
#         self.conv1d_2.apply(weights_init)
#         self.conv1d_3.apply(weights_init)

#         self.maxpool1d = nn.MaxPool1d(kernel_size=3, padding=1)

#     def forward(self, x: torch.Tensor):
#         """[summary]

#         Parameters
#         ----------
#         x : torch.Tensor
#             [description]

#         Returns
#         -------
#         x : torch.Tensor
#             [description]
#         """
#         x = self.maxpool1d(self.conv1d_1(x))
#         x = self.maxpool1d(self.conv1d_2(x))
#         x = self.conv1d_3(x)
#         # |x| = (batch_size, 1, seq_len)
#         return x
