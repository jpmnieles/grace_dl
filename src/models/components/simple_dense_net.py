import torch
from torch import nn


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 6,
        output_size: int = 2,
        hidden_size: int = 3,
        hidden_nodes: int = 256
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        # Create the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_nodes))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        if hidden_size > 1:
            for i in range(1, hidden_size):
                layers.append(nn.Linear(hidden_nodes, hidden_nodes))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_nodes, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, features = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
