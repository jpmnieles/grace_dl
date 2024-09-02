import torch
from torch import nn

from torch.nn.functional import mse_loss


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = mse_loss
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss