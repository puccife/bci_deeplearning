import math
import torch.nn.functional as F
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, init_weights=True):
        super(LSTM, self).__init__()
        self.dlstm = nn.LSTM(4, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        debugging = False
        x = x.unsqueeze(1)
        print(x.shape) if debugging else ...
        x = x.float()
        print(x.shape) if debugging else ...
        x, _ = self.dlstm(x)
        print(x.shape) if debugging else ...
        x = F.leaky_relu(x)
        print(x.shape) if debugging else ...
        x = x.view(x.size(0), -1)
        print(x.shape) if debugging else ...
        x = self.fc(x)
        print('fc', x.shape) if debugging else ...
        return x

