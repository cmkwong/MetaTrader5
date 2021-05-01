import torch
from torch import nn

class lstm_price(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(lstm_price).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input, hiddens):
        """
        :param input: (seq_len, batch, input_size)
        :param hiddens: ( (num_layers, batch, hidden_size), (num_layers, batch, hidden_size) )
        :return: output (seq_len, batch, hidden_size)
        """
        output, (hn, cn) = self.rnn(input, (hiddens[0], hiddens[1]))
        return output, (hn, cn)

    def init_hiddens(self, batch_szie):
        h0 = torch.zeros(self.num_layers, batch_szie, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_szie, self.hidden_size)
        return (h0, c0)