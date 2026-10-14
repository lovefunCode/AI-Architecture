import torch
import torch.nn as nn

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # compute all 4 gates together
        self.x_to_gates = nn.Linear(input_size, 4 * hidden_size)
        self.h_to_gates = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, states = None):
        # x: (batch_size, input_size)
        batch_size = x.size(0)

        # initialize hidden state and cell state
        if states is None:
            h = x.new_zeros(batch_size, self.hidden_size)
            c = x.new_zeros(batch_size, self.hidden_size)
        else:
            h, c = states
        
        # combine input and previous hidden state
        gates = self.x_to_gates(x) + self.h_to_gates(h)

        # split into 4 vectors
        f, i, g, o = gates.chunk(4, dim = -1)

        #  forget gate, input gate, candidate memory, output gate
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g

        h = o * torch.tanh(c)

        return h, c
