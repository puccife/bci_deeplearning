import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, kernel_size=5, padding=2):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(28, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.MaxPool1d(2))
        self.fc = nn.Linear(4000, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

