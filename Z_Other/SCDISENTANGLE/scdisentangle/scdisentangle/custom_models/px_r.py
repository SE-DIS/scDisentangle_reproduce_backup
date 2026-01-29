import torch.nn as nn
import  torch

class PXR(nn.Module):
    def __init__(self, n_genes):
        super(PXR, self).__init__()
        self.px_r = nn.Parameter(torch.randn(n_genes))

    def forward(self, x):
        return self.px_r