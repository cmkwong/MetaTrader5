import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy


class Encoder(nn.Module):
    device = torch.device("cuda")

    def __init__(self, inputSize, hiddenSize):
        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.gru = nn.GRU(inputSize, self.hiddenSize, batch_first=True).to(self.device)

    def forward(self, input, h0):
        self.gru.flatten_parameters()
        output, hn = self.gru(input, h0)
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

    def __init__(self, maxSeqLen, statusSize, hiddenSize, outputSize, dropout_p):
        super(DecoderAttnFull, self).__init__()
        self.maxSeqLen = maxSeqLen
        self.dropout_p = dropout_p
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.attn = nn.Sequential(
            nn.Linear(self.hiddenSize * self.maxSeqLen + self.hiddenSize + statusSize, 1024),
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

    def forward(self, encoderOutputs, hn, status):
        input = encoderOutputs.reshape(encoderOutputs.shape[0], -1)
        # concat the hidden
        concatVec = torch.cat((input, status, hn.squeeze(0)), dim=1).squeeze(-1)
        attn_weights = F.softmax(self.attn(concatVec), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoderOutputs).permute(0, 2, 1)

        concatAttnApplied = torch.cat((input, attn_applied.squeeze(-1)), dim=1)
        output = self.attn_combine(concatAttnApplied)

        output = F.relu(output).unsqueeze(1)
        self.gru.flatten_parameters()
        output, hn = self.gru(output, hn)  # (N, L, H_in)

        output = self.out(output).squeeze(1)
        return output, hn, attn_weights

    def initHidden(self, batchSize):
        self.batchSize = batchSize
        return torch.zeros(1, batchSize, self.hiddenSize, device=self.device)

class AttentionTimeSeries(nn.Module):
    def __init__(self, hiddenSize, inputSize, seqLen, batchSize, outputSize, statusSize, pdrop):
        super(AttentionTimeSeries, self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.outputSize = outputSize
        self.encoder = Encoder(inputSize, hiddenSize)
        self.decoder = DecoderAttnFull(seqLen, statusSize, hiddenSize, outputSize, pdrop)

    def forward(self, state):
        """
        :param state(dict):     'encoderInput': torch tensor (N, L * 2, H_in)
                                'status': torch tensor (N, 2) which are earning and havePosition
        :return: action array
        """
        # start = time.time()
        cur_batchSize = state['encoderInput'].shape[0]
        encoderHn = self.encoder.initHidden(cur_batchSize)
        encoderOutputs = torch.zeros(cur_batchSize, self.seqLen, self.hiddenSize, device=self.encoder.device)
        for i in range(self.seqLen):
            input = state['encoderInput'][:, 0 + i:0 + i + self.seqLen, :]  # taking part of series data
            encoderOutput, encoderHn = self.encoder(input, encoderHn)
            encoderOutputs[:, i, :] = encoderOutput[:, -1, :]  # taking the last occurred vector to form the encoderOutputs, then feed into decoder

        # assign encoder hidden to decoder hidden
        decoderHn = encoderHn
        decoderOutput, decoderHn, attn_weights = self.decoder(encoderOutputs, decoderHn, state['status'].to(self.encoder.device))
        # print(time.time() - start)
        return decoderOutput

# hiddenSize = 128
# inputSize = 57
# seqLen = 30
# batchSize = 32
# outputSize = 3
# statusSize = 2
#
# encoderInput = torch.randn(batchSize, seqLen * 2, inputSize)
#
# attentionTimeSeries = AttentionTimeSeries(hiddenSize=128, inputSize=57, seqLen=30, batchSize=128, outputSize=3, statusSize=2, pdrop=0.1)
# outputAction = attentionTimeSeries({'encoderInput': encoderInput,
#                                     'status': torch.randn(batchSize, 2)})
#
# target_model = copy.deepcopy(attentionTimeSeries)
# target_model.load_state_dict(attentionTimeSeries.state_dict())
# print()

# def runNetwork(encdoerInputs):
#     """
#     :param encdoerInputs: torch tensor with shape (N, L, H_in)
#     :return:
#     """
#     start = time.time()
#     encoder = Encoder(inputSize, hiddenSize)
#     encoderHn = encoder.initHidden(batchSize)
#     encoderOutputs = torch.zeros(batchSize, seqLen, hiddenSize, device=encoder.device)
#     for i, encoderInput in enumerate(encdoerInputs):
#         encoderOutput, encoderHn = encoder(encoderInput, encoderHn)
#         encoderOutputs[:, i, :] = encoderOutput[:, -1, :]
#
#     # decoderInput = torch.randn(batchSize, inputSize, 1)
#     decoder = DecoderAttnFull(seqLen, hiddenSize, outputSize, 0.1)
#     decoderHn = encoderHn
#     # decoderOutput, hidden, attn_weights = decoder(encoderOutput[:, -1, :], hidden, encoderOutput[:, 0:-1, :])
#     decoderOutput, decoderHn, attn_weights = decoder(encoderOutputs, decoderHn)
#     print(time.time() - start)
#     return decoderOutput

# output = runNetwork(encdoerInputs)
# print(output)
