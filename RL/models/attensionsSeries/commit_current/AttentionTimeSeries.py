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


class DecoderAttnFull(nn.Module):
    device = torch.device("cuda")

    def __init__(self, maxSeqLen, statusSize, hiddenSize, outputSize, dropout_p):
        super(DecoderAttnFull, self).__init__()
        self.maxSeqLen = maxSeqLen
        self.dropout_p = dropout_p
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.attn = nn.Sequential(
            nn.Linear(self.hiddenSize * 2 + statusSize, 1024),
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
        self.attn_combine = nn.Sequential(
            nn.Linear(self.hiddenSize * 2, 1024),
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

    def forward(self, encoderOutput, hn, status):
        # concat the hidden
        concatVec = torch.cat((encoderOutput, hn.squeeze(0), status), dim=1)
        attn_weights = F.softmax(self.attn(concatVec), dim=1)
        attn_applied = torch.mul(attn_weights, encoderOutput)

        concatAttnApplied = torch.cat((encoderOutput, attn_applied), dim=1)
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
        encoderOutput, encoderHn = self.encoder(state['encoderInput'], encoderHn)

        # assign encoder hidden to decoder hidden
        decoderHn = encoderHn
        decoderOutput, decoderHn, attn_weights = self.decoder(encoderOutput[:, -1, :], decoderHn, state['status'].to(self.encoder.device))
        # print(time.time() - start)
        return decoderOutput


# ===================== TEST =====================

# hiddenSize = 128
# inputSize = 57
# seqLen = 30
# batchSize = 32
# outputSize = 3
# statusSize = 2
#
# encoderInput = torch.randn(batchSize, seqLen, inputSize)
#
# attentionTimeSeries = AttentionTimeSeries(hiddenSize=hiddenSize, inputSize=inputSize, seqLen=seqLen, batchSize=batchSize, outputSize=outputSize, statusSize=statusSize, pdrop=0.1)
# outputAction = attentionTimeSeries({'encoderInput': encoderInput.to(torch.device("cuda")),
#                                     'status': torch.randn(batchSize, 2).to(torch.device("cuda"))})

# ===================== TEST =====================

# target_model = copy.deepcopy(attentionTimeSeries)
# target_model.load_state_dict(attentionTimeSeries.state_dict())
# print()
