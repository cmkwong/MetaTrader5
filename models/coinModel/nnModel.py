import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input, hiddens):
        """
        :param input: (batch, seq_len, input_size)
        :param hiddens: ( (batch, num_layers, hidden_size), (batch, num_layers, hidden_size) )
        :LSTM return: output (batch, seq_len, hidden_size)
        :return:
        """
        lstm_output, _ = self.rnn(input, (hiddens[0], hiddens[1]))
        output = self.linear(lstm_output)
        return output

    def init_hiddens(self, batch_szie):
        h0 = torch.zeros((self.num_layers, batch_szie, self.hidden_size), dtype=torch.double)
        c0 = torch.zeros((self.num_layers, batch_szie, self.hidden_size), dtype=torch.double)
        return (h0, c0)