import math
import torch
from torch import nn
from torch.nn.functional import mse_loss


# torch.log  and math.log is e based
class ReconLatentLoss(nn.Module):
    def __init__(self, epsilon=0.5):
        super(ReconLatentLoss, self).__init__()
        self.epsilon = epsilon
        self.mse = mse_loss

    def forward(self, output1, output2, latent1, latent2, target1, target2):
        output_loss1 = self.mse(output1, target1)
        output_loss2 = self.mse(output2, target2)
        
        # Latent loss with emphasis
        latent_loss = self.mse(latent1, latent2)
        
        # Combine losses with heavier emphasis on latent loss
        total_loss = output_loss1 + output_loss2 + self.epsilon*latent_loss
        return total_loss


if __name__ == "__main__":
    _ = ReconLatentLoss(epsilon=0.5)