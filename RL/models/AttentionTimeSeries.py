import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Encoder(nn.Module):
    device = torch.device("cuda")

    def __init__(self, inputSize, hiddenSize):
        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.gru = nn.GRU(inputSize, self.hiddenSize, batch_first=True).to(self.device)

    def forward(self, input, h0):
        output, hn = self.gru(input.to(self.device), h0)
        return output, hn

    def initHidden(self, batchSize):
        self.batchSize = batchSize
        return torch.zeros(1, batchSize, self.hiddenSize, device=self.device)


class DecoderAttn(nn.Module):
    device = torch.device("cuda")

    def __init__(self, maxSeqLen, hiddenSize, outputSize, dropout_p):
        super(DecoderAttn, self).__init__()
        self.maxSeqLen = maxSeqLen
        self.dropout_p = dropout_p
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.attn = nn.Linear(self.hiddenSize * 2, self.maxSeqLen - 1).to(self.device)
        self.attn_combine = nn.Linear(self.hiddenSize * 2, self.hiddenSize).to(self.device)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize, batch_first=True).to(self.device)
        self.out = nn.Linear(self.hiddenSize, self.outputSize).to(self.device)

    def forward(self, input, h0, encoderOutputs):
        input = input.to(self.device)
        # concat the hidden
        concatHidden = torch.cat((input, h0.squeeze(0)), dim=1).squeeze(-1)
        attn_weights = F.softmax(self.attn(concatHidden), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoderOutputs).permute(0, 2, 1)

        concatAttnApplied = torch.cat((input, attn_applied.squeeze(-1)), dim=1)
        output = self.attn_combine(concatAttnApplied)

        output = F.relu(output).unsqueeze(1)
        output, hn = self.gru(output, h0)

        output = F.log_softmax(self.out(output.squeeze(1)), dim=1)
        return output, hn, attn_weights

    def initHidden(self, batchSize):
        self.batchSize = batchSize
        return torch.zeros(1, batchSize, self.hiddenSize, device=self.device)


class DecoderAttnFull(nn.Module):
    device = torch.device("cuda")

    def __init__(self, maxSeqLen, hiddenSize, outputSize, dropout_p):
        super(DecoderAttnFull, self).__init__()
        self.maxSeqLen = maxSeqLen
        self.dropout_p = dropout_p
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.attn = nn.Sequential(
            nn.Linear(self.hiddenSize * self.maxSeqLen + self.hiddenSize, 1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.maxSeqLen)
        ).to(self.device)
        self.attn_combine = nn.Sequential(
            nn.Linear(self.hiddenSize * self.maxSeqLen + self.hiddenSize, 1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.hiddenSize)
        ).to(self.device)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize, batch_first=True).to(self.device)
        self.out = nn.Linear(self.hiddenSize, self.outputSize).to(self.device)

    def forward(self, encoderOutputs, hn):
        input = encoderOutputs.reshape(encoderOutputs.shape[0], -1)
        # concat the hidden
        concatHidden = torch.cat((input, hn.squeeze(0)), dim=1).squeeze(-1)
        attn_weights = F.softmax(self.attn(concatHidden), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoderOutputs).permute(0, 2, 1)

        concatAttnApplied = torch.cat((input, attn_applied.squeeze(-1)), dim=1)
        output = self.attn_combine(concatAttnApplied)

        output = F.relu(output).unsqueeze(1)
        output, hn = self.gru(output, hn)  # (N, L, H_in)

        output = F.softmax(self.out(output).squeeze(1), dim=1)
        return output, hn, attn_weights

    def initHidden(self, batchSize):
        self.batchSize = batchSize
        return torch.zeros(1, batchSize, self.hiddenSize, device=self.device)


hiddenSize = 128
inputSize = 53
# seqLen = 301 # for last day not good
seqLen = 30  # for full
batchSize = 32
outputSize = 3

encdoerInputs = []
for i in range(seqLen):
    encdoerInput = torch.randn(batchSize, seqLen, inputSize)
    encdoerInputs.append(encdoerInput)


def runNetwork(encdoerInputs):
    """
    :param encdoerInputs: torch tensor with shape (N, L, H_in)
    :return:
    """
    start = time.time()
    encoder = Encoder(inputSize, hiddenSize)
    encoderHn = encoder.initHidden(batchSize)
    encoderOutputs = torch.zeros(batchSize, seqLen, hiddenSize, device=encoder.device)
    for i, encoderInput in enumerate(encdoerInputs):
        encoderOutput, encoderHn = encoder(encoderInput, encoderHn)
        encoderOutputs[:, i, :] = encoderOutput[:, -1, :]

    # decoderInput = torch.randn(batchSize, inputSize, 1)
    decoder = DecoderAttnFull(seqLen, hiddenSize, outputSize, 0.1)
    decoderHn = encoderHn
    # decoderOutput, hidden, attn_weights = decoder(encoderOutput[:, -1, :], hidden, encoderOutput[:, 0:-1, :])
    decoderOutput, decoderHn, attn_weights = decoder(encoderOutputs, decoderHn)
    print(time.time() - start)
    return decoderOutput


output = runNetwork(encdoerInputs)
print(output)
