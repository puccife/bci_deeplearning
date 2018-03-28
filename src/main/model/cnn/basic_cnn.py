import math
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, kernel_size=5, padding=2, init_weights=True):
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

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            '''
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            '''

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

