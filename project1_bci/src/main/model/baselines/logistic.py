import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        out = self.linear(x)
        return out