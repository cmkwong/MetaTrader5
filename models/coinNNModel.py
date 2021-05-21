import torch
from torch import nn
import numpy as np
import pandas as pd
from production.codes.models import criterionModel
from production.codes.utils import maths

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, batch_first=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len  # int, number of days input to LSTM
        self.device = torch.device("cuda:0")

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first).to(self.device)
        self.linear = nn.Linear(hidden_size, input_size).to(self.device)
        self.weights_layer = nn.Linear(input_size, 1).to(self.device)

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
        output = self.weights_layer(output)
        return output.squeeze(-1)

    def init_hiddens(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
        return (h0, c0)

    def get_predicted_arr(self, input_arr):
        """
        :param input: array, size = (total_len, input_size)
        :param model: torch model
        :return: array, size = (total_len, 1)
        """
        input = torch.FloatTensor(input_arr)
        self.eval()
        predict_arr = np.zeros((len(input_arr), ), dtype=np.float)
        for i in range(self.seq_len, len(input_arr)):
            x = input[i-self.seq_len:i,:].unsqueeze(0) # size = (batch_size, seq_len, 1)
            hiddens = self.init_hiddens(1)
            predict = self(x, hiddens)
            predict_arr[i,] = predict
        return predict_arr

    def get_coefficient_vector(self):
        weights = self.weights_layer.bias.detach().cpu().numpy().reshape(-1,)
        bias = self.weights_layer.weight.detach().cpu().numpy().reshape(-1,)
        coefficient_vector = np.append(bias, weights)
        return coefficient_vector

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

def get_coinNN_data(close_prices, model, window):
    """
    :param prices_matrix: accept the train and test prices in array format
    :param model: torch model to get the predicted value
    :param data_options: dict
    :return: array for plotting
    """
    coinNN_data = pd.DataFrame(index=close_prices.index)
    coinNN_data['real'] = close_prices.iloc[:, -1]
    coinNN_data['predict'] = model.get_predicted_arr(close_prices.iloc[:,:-1].values)
    spread = coinNN_data['real'] - coinNN_data['predict']
    coinNN_data['spread'] = spread
    coinNN_data['z_score'] = maths.z_score_with_rolling_mean(spread.values, window)
    return coinNN_data