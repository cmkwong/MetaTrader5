import torch
from torch import nn


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tt:torch.tensor, dim=1):
        # magnitude of embedding vectors, |b|
        magnitudes = tt.pow(2).sum(dim=dim).sqrt().unsqueeze(0)
        similarities = torch.mm(tt, tt.t()) / magnitudes
        return similarities

