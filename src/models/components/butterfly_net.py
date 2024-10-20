import torch
from torch import nn


class ButterflyNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size1: int = 12,
        hidden_size1: list = [128,64],
        output_size1: int = 12,
        latent_size: int = 6,
        input_size2: int = 12,
        hidden_size2: list = [128,64],
        output_size2: int = 12,
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
        layers.append(nn.Linear(input_size1, hidden_size1[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        if len(hidden_size1) > 1:
            for i in range(len(hidden_size1)-1):
                layers.append(nn.Linear(hidden_size1[i], hidden_size1[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size1[-1], latent_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        self.encoder1 = nn.Sequential(*layers)

        # Encoder2
        layers = []
        layers.append(nn.Linear(input_size2, hidden_size2[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        if len(hidden_size2) > 1:
            for i in range(len(hidden_size2)-1):
                layers.append(nn.Linear(hidden_size2[i], hidden_size2[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size2[-1], latent_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        self.encoder2 = nn.Sequential(*layers)

        # Decoder1
        layers = []
        layers.append(nn.Linear(latent_size, hidden_size1[-1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        if len(hidden_size1) > 1:
            for i in range(-len(hidden_size1)+1,0):
                layers.append(nn.Linear(hidden_size1[i], hidden_size1[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size1[0], output_size1))
        self.decoder1 = nn.Sequential(*layers)

        # Decoder2
        layers = []
        layers.append(nn.Linear(latent_size, hidden_size2[-1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        if len(hidden_size1) > 1:
            for i in range(-len(hidden_size2)+1,0):
                layers.append(nn.Linear(hidden_size2[i], hidden_size2[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size2[0], output_size2))
        self.decoder2 = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # batch_size, features = x.size()
        x1 = x[:,:12]
        x2 = x[:,12:]

        # Encoder
        latent1 = self.encoder1(x1)
        latent2 = self.encoder2(x2)

        # Latent Combination
        shared_latent = (latent1 + latent2) / 2 

        # Decoder
        output1 = self.decoder1(shared_latent)  
        output2 = self.decoder2(shared_latent)

        return output1, output2, latent1, latent2


if __name__ == "__main__":
    _ = ButterflyNet()
