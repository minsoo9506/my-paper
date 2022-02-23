# window-based fully connected AE 구현
import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 2,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor):
        z = self.model(x)
        return z


class BaseDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2,
        output_size: int = 3,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(output_size, output_size),
        )

    def forward(
        self,
        z: torch.Tensor,
    ):
        x = self.model(z)
        return x


class BaseSeq2Seq(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 2,
        output_size: int = 3,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        self.encoder = BaseEncoder(input_size, hidden_size, dropout_p)
        self.decoder = BaseDecoder(hidden_size, output_size, dropout_p)

    def forward(
        self,
        x: torch.Tensor,
    ):
        z = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x
