import torch
from torch import nn


class FKNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: list = [1024,1024],
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
        layers.append(nn.Linear(input_size, hidden_size[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        if len(hidden_size) > 1:
            for i in range(len(hidden_size)-1):
                layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        pred = self.model(x)  

        return pred


if __name__ == "__main__":
    _ = FKNet()
