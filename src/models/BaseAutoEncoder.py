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
        # |h[0]| = (num_layers, batch_size, hidden_size)
        return h

class LSTMDecoder(nn.Module):
    def __init__(
        self,
        original_feature_dim: int,
        seq_len: int,
        hidden_size: int = 32,
        num_layers: int = 3,
        dropout_p: float = 0.2,
    ):
        """make LSTMDecoder

        Parameters
        ----------
        original_feature_dim : int
            [description]
        seq_len : int
            [description]
        hidden_size : int, optional
            [description], by default 32
        num_layers : int, optional
            [description], by default 3
        dropout_p : float, optional
            [description], by default 0.2
        """
        super().__init__()

        self.original_feature_dim = original_feature_dim

        self.lstm = nn.LSTM(
            input_size=original_feature_dim + seq_len + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p,
            batch_first=True,
        )

        self.last_fully = nn.Linear(
            in_features=hidden_size, out_features=original_feature_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        y_t_1: torch.Tensor,
        h_t_1: torch.Tensor
        ):
        """[summary]

        Parameters
        ----------
        x : torch.Tensor
            [description]
        y_t_1 : torch.Tensor
            이전 time-step에서의 output
        h_t_1 : torch.Tensor
            (h,c) : 이전 time-step의 (hidden, cell state)

        Returns
        -------
        pred, y_t, h_t
            [description]
        """

        # |x| = (batch_size, 1, original_input_dim + seq_len from CnnEncoder)
        # |y_t_1| = (batch_size, 1, hidden_size)
        # |h_t_1[0]| = (num_layers, batch_size, hidden_size)

        batch_size = x.size(0)
        hidden_size = h_t_1[0].size(-1)

        if y_t_1 is None:
            y_t_1 = x.new(batch_size, 1, hidden_size).zero_()

        x = torch.cat([x, y_t_1], dim=-1)

        y_t, h_t = self.lstm(x, h_t_1)
        pred = self.last_fully(y_t.squeeze(1))
        
        # |pred| = (batch_size, original_feature_dim)
        # |y_t| = (batch_size, 1, hidden_size)
        # |h_t[0]| = (num_layers, batch_size, hidden_size)

        return pred, y_t, h_t
    
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