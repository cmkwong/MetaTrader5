import torch
from torch import nn

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
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.double).to(self.device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.double).to(self.device)
        return (h0, c0)