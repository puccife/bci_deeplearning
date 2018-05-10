import torch.nn as nn
import torch.nn.functional as F

class TheNet(nn.Module):
    def __init__(self):
        super(TheNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 14, (1, 20), padding=(0))
        self.batchnorm1 = nn.BatchNorm2d(14, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((14, 2, 0, 0))
        self.conv2 = nn.Conv2d(31, 4, (1, 5))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(1, 3)

        # FC Layer
        self.fc1 = nn.Linear(280, 2)

    def forward(self, x):
        # Layer 1

        x = x.unsqueeze(1)
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.90)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.90)
        x = self.pooling2(x)

        x = x.view(x.shape[0], -1, 280)
        x = self.fc1(x)
        x = x.squeeze(1)
        return x