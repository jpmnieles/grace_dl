import torch
from torch import nn


class FKNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 12,
        hidden_size1: list = [8],
        latent_size: int = 6,
        hidden_size2: list = [512,512],
        output_size: int = 12,
        dropout = 0.5
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        # Encoder1
        layers = []
        layers.append(nn.Linear(input_size, hidden_size1[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        if len(hidden_size1) > 1:
            for i in range(len(hidden_size1)-1):
                layers.append(nn.Linear(hidden_size1[i], hidden_size1[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size1[-1], latent_size))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        # Decoder2
        layers = []
        layers.append(nn.Linear(latent_size, hidden_size2[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        if len(hidden_size2) > 1:
            for i in range(len(hidden_size2)-1):
                layers.append(nn.Linear(hidden_size2[i], hidden_size2[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size2[-1], output_size))
        self.decoder = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # Encoder
        latent = self.encoder(x)

        # Decoder
        pred = self.decoder(latent)  

        return pred,latent


if __name__ == "__main__":
    _ = FKNet()
