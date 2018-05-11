import torch.nn as nn
import torch.nn.functional as F
import torch

class TSnet(nn.Module):
    def __init__(self):
        super(TSnet, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(28, 14, (1, 10), padding=(0))
        self.batchnorm1 = nn.BatchNorm2d(14)
        # Layer 2
        self.padding1 = nn.ZeroPad2d((10, 10, 0, 0))
        self.conv2 = nn.Conv2d(14, 4, (1, 3))
        self.batchnorm2 = nn.BatchNorm2d(4)
        self.pooling2 = nn.MaxPool2d(1, 4)
        # Layer 2
        self.padding3 = nn.ZeroPad2d((4, 4, 0, 0))
        self.conv3 = nn.Conv2d(4, 4, (1, 3))
        self.batchnorm3 = nn.BatchNorm2d(4)
        self.pooling3 = nn.MaxPool2d(1, 4)
        # FC Layer
        self.fc1 = nn.Linear(24, 2)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.50)
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.50)
        x = self.pooling2(x)
        x = self.padding3(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.50)
        x = self.pooling3(x)
        x = x.view(x.shape[0], -1, 24)
        x = self.fc1(x)
        x = x.squeeze(1)
        return x