import torch
from torch import nn
import numpy as np
from production.codes.models import criterionModel

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda:0")

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first).to(self.device)
        self.linear = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, input, hiddens):
        """
        :param input: (batch, seq_len, input_size)
        :param hiddens: ( (batch, num_layers, hidden_size), (batch, num_layers, hidden_size) )
        :LSTM return: output (batch, seq_len, hidden_size)
        :return: output
        """
        input = input.to(self.device)
        lstm_output, _ = self.rnn(input, (hiddens[0], hiddens[1]))
        output = nn.ReLU()(lstm_output[:,-1,:])
        output = self.linear(output)
        return output.squeeze(-1)

    def init_hiddens(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
        return (h0, c0)

    def get_predicted_arr(self, input_arr, seq_len):
        """
        :param input: array, size = (total_len, input_size)
        :param model: torch model
        :param seq_len: int, number of days input to LSTM
        :return: array, size = (total_len, 1)
        """
        input = torch.from_numpy(input_arr)
        self.eval()
        predict_arr = np.zeros((len(input_arr), 1), dtype=np.float)
        for i in range(seq_len, len(input_arr)):
            x = input[i-seq_len:i,:].unsqueeze(0) # size = (batch_size, seq_len, 1)
            hiddens = self.init_hiddens(1)
            predict = self(x, hiddens)
            predict_arr[i, 0] = predict
        return predict_arr

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def _model_mode(self, train_mode):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

    # def _dataset(self, train_mode):
    #     if train_mode:
    #         inputs = self.train_batch.input
    #         targets = self.train_batch.target
    #     else:
    #         inputs = self.test_batch.input
    #         targets = self.test_batch.target
    #     return inputs, targets

    def _learn(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def run(self, inputs, targets, batch_size, train_mode=True):
        total_loss = 0
        steps = 0
        for input, target in zip(inputs, targets):
            self._model_mode(train_mode)
            input, target = torch.FloatTensor(input).requires_grad_(), torch.FloatTensor(target).requires_grad_()
            hiddens = self.model.init_hiddens(batch_size)
            output = self.model(input, hiddens)
            loss = criterionModel.get_mse_loss(output, target.to(torch.device("cuda:0")))
            if train_mode: self._learn(loss)
            total_loss += loss.detach().cpu().item()
            steps += 1
        mean_loss = total_loss / steps
        return mean_loss