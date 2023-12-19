import torch
from .FC import build_fc_layers


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_lstm, num_layer_lstm, hidden_fc, output_size, bi=False, dropout=0.0, **kwargs):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_lstm, num_layer_lstm, batch_first=True, bidirectional=bi, dropout=dropout)
        if bi:
            hidden_lstm *= 2
        if hidden_fc is not None and len(hidden_fc) > 0:
            self.fc = build_fc_layers(hidden_lstm, hidden_fc)
            hidden_size = hidden_fc[-1]
        else:
            self.fc = None
            hidden_size = hidden_lstm
        self.out_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y, _ = self.lstm(x)
        y = torch.relu(y[:, -1, :])
        if self.fc is not None:
            y = self.fc(y)
        y = self.out_layer(y).unsqueeze(1) # (batchsize, outsize)->(batchsize, 1, outsize)
        return y
