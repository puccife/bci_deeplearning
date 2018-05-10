import math
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, kernel_size=4, padding=2, init_weights=False):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Dropout2d(0.9),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(0.9),
            nn.MaxPool1d(2)
            )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout2d(0.9),
            nn.MaxPool1d(2)
            )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout2d(0.9),
            nn.MaxPool1d(2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout2d(0.9),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Linear(128, 2)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        debugging = False
        x = x.unsqueeze(1).float()
        print(x.shape) if debugging else ...
        x = self.layer1(x)
        print('layer 1') if debugging else ...
        print(x.shape) if debugging else ...
        x = self.layer2(x)
        print('layer 2') if debugging else ...
        print(x.shape) if debugging else ...
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        print(x.shape) if debugging else ...
        x = x.view(x.size(0), -1)
        print(x.shape) if debugging else ...
        out = self.fc(x)
        print(x.shape) if debugging else ...
        return out

