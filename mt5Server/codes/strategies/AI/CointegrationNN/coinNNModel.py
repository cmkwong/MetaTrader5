import torch
import numpy as np
from torch import nn
from strategies.AI import criterionModel
from strategies.Cointegration import coinModel


def get_coefficient_vector(model):
    """
    :param weights_layer: nn fully connected layer
    :return: nnp.array
    """
    weights = model.weights_layer.weight.detach().cpu().numpy().reshape(-1, )
    bias = model.weights_layer.bias.detach().cpu().numpy().reshape(-1, )
    coefficient_vector = np.append(bias, weights)
    return coefficient_vector

def get_predicted_arr(model, input_arr, seq_len):
    """
    :param model: nn.Model
    :param input: array, size = (total_len, input_size)
    :param seq_len: int, the loader length needed to feed into model
    :return: array, size = (total_len, 1)
    """
    input = torch.FloatTensor(input_arr)
    model.eval()
    predict_arr = np.zeros((len(input_arr),), dtype=np.float)
    for i in range(seq_len, len(input_arr)):
        x = input[i - seq_len:i, :].unsqueeze(0)  # size = (batch_size, seq_len, features_len)
        predict = model(x)
        predict_arr[i,] = predict
    return predict_arr

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda:0")

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first).to(self.device)
        self.linear = nn.Linear(hidden_size, input_size).to(self.device)
        self.weights_layer = nn.Linear(input_size, 1).to(self.device)

    def forward(self, input):
        """
        :param input: (batch, seq_len, input_size)
        :return: output
        """
        # get the batch size
        batch_size = len(input)
        hiddens = self.init_hiddens(batch_size) # size =  ( (batch, num_layers, hidden_size), (batch, num_layers, hidden_size) )

        # forward the model
        input = input.to(self.device)
        lstm_output, _ = self.rnn(input, (hiddens[0], hiddens[1]))
        output = nn.ReLU()(lstm_output[:,-1,:])
        output = self.linear(output)
        output = self.weights_layer(output)
        return output.view(-1,)

    def init_hiddens(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
        return (h0, c0)

class FC_NN(nn.Module):
    def __init__(self, input_size):
        super(FC_NN, self).__init__()
        self.device = torch.device("cuda:0")
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, 8),
            # nn.ReLU(),
            nn.Linear(8,16),
            # nn.ReLU(),
            nn.Linear(16, input_size),
            # nn.ReLU()
        ).to(self.device)
        self.weights_layer = nn.Linear(input_size, 1).to(self.device)

    def forward(self, input):
        input = input.to(self.device)
        raw_output = self.hidden_layer(input)
        output = self.weights_layer(raw_output)
        return output.view(-1,)

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def _model_mode(self, train_mode):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

    def _learn(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def run(self, inputs, targets, train_mode=True):
        total_loss = 0
        steps = 0
        for input, target in zip(inputs, targets):
            self._model_mode(train_mode)
            input, target = torch.FloatTensor(input).requires_grad_(), torch.FloatTensor(target).requires_grad_()
            output = self.model(input)
            loss = criterionModel.get_mse_loss(output, target.to(torch.device("cuda:0")))
            if train_mode: self._learn(loss)
            total_loss += loss.detach().cpu().item()
            steps += 1
        mean_loss = total_loss / steps
        return mean_loss

def get_modify_coefficient_vector(coefficient_vector, long_mode, lot_times=1):
    return coinModel.get_modified_coefficient_vector(coefficient_vector, long_mode, lot_times) # note 57e

# def get_coinNN_data(close_prices, model, mean_window, std_window):
#     """
#     :param prices_matrix: accept the train and test prices in array format
#     :param model: torch model to get the predicted value
#     :param data_options: dict
#     :return: array for plotting
#     """
#     coinNN_data = pd.DataFrame(index=close_prices.index)
#     coinNN_data['real'] = close_prices.iloc[:, -1]
#     coinNN_data['predict'] = model.get_predicted_arr(close_prices.iloc[:,:-1].values)
#     spread = coinNN_data['real'] - coinNN_data['predict']
#     coinNN_data['spread'] = spread
#     coinNN_data['z_score'] = maths.z_score_with_rolling_mean(spread.values, mean_window, std_window)
#     return coinNN_data