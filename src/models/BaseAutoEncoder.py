# window-based fully connected AE 구현
import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 2,
        dropout_p: float = 0.2,
    ):
        """Encoder with 3 fully-connected layers

        Parameters
        ----------
        input_size : int, optional
            [description], by default 16
        hidden_size : int, optional
            output size of encoder, by default 2
        dropout_p : float, optional
            [description], by default 0.2
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(8, hidden_size),
        )

    def forward(self, x: torch.Tensor):
        """[summary]

        Parameters
        ----------
        x : torch.Tensor
            [description]

        Returns
        -------
        z
            output of encoder
        """
        z = self.model(x)
        # |z| = (batch_size, hidden_size)
        return z


class BaseDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2,
        output_size: int = 16,
        dropout_p: float = 0.2,
    ):
        """[summary]

        Parameters
        ----------
        hidden_size : int, optional
            input size of decoder, by default 2
        output_size : int, optional
            output size of decoder, by default 16
        dropout_p : float, optional
            [description], by default 0.2
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hidden_size, 8),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(16, output_size),
        )

    def forward(
        self,
        z: torch.Tensor,
    ):
        """[summary]

        Parameters
        ----------
        z : torch.Tensor
            input of decoder (= output of encoder)

        Returns
        -------
        x : torch.Tensor
            Reconstruction result of original data
        """

        x = self.model(z)
        # |x| = (batch_size, output_size)
        return x


class BaseSeq2Seq(nn.Module):
    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 2,
        output_size: int = 16,
        dropout_p: float = 0.2,
    ):
        """[summary]

        Parameters
        ----------
        input_size : int, optional
            [description], by default 16
        hidden_size : int, optional
            [description], by default 2
        output_size : int, optional
            [description], by default 16
        dropout_p : float, optional
            [description], by default 0.2
        """
        super().__init__()

        self.encoder = BaseEncoder(input_size, hidden_size, dropout_p)
        self.decoder = BaseDecoder(hidden_size, output_size, dropout_p)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """[summary]

        Parameters
        ----------
        x : torch.Tensor
            [description]

        Returns
        -------
        recon_x
            reconstuction result of input data
        """

        z = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x
