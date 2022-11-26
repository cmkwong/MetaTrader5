import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, lstmLayer, actions_n):
        """
        :param inputSize: int, 6, eg: ohlcvs
        :param hiddenSize: int, eg: 32
        :param lstmLayer: int, eg: 2
        :param actions_n: int, 3 = (B, S, N)
        """
        super(SimpleLSTM, self).__init__()
        self.device = torch.device("cuda")

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.lstmLayer = lstmLayer

        self.lstm = nn.LSTM(self.inputSize, self.hiddenSize, self.lstmLayer, batch_first=True)

        self.fc_val = nn.Sequential(
            nn.Linear(self.hiddenSize, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        ).to(self.device)

        self.fc_adv = nn.Sequential(
            nn.Linear(self.hiddenSize, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, actions_n),
        ).to(self.device)

    def init_hidden(self, batchSize):
        weight = next(self.parameters()).data
        hidden = (weight.new(batchSize, self.lstmLayer, self.hiddenSize).zero_().cuda(),
                  weight.new(batchSize, self.lstmLayer, self.hiddenSize).zero_().cuda())
        return hidden

    def forward(self, x, hidden):
        """
        :param x: input, eg ohlcvs
        :param hidden: always init, mean=0, std=1
        :return: action vector
        """
        x = x.to(self.device)
        out, hidden = self.lstm(x, hidden)
        val = self.fc_val(out)
        adv = self.fc_adv(out)
        return val + adv - adv.mean(dim=1, keepdim=True)
