import torch.nn as nn
import torch.nn.functional as F
import torch

"""
Time series network module
"""
class TSnet(nn.Module):

    """
    Initialization
    """
    def __init__(self):
        super(TSnet, self).__init__()

        # First Convolutional Layer + BatchNorm
        self.conv1 = nn.Conv2d(28, 14, (1, 10), padding=(0))
        self.batchnorm1 = nn.BatchNorm2d(14)

        # Padding the input that goes into the second Conv layer
        self.padding1 = nn.ZeroPad2d((10, 10, 0, 0))

        # Second Convolutional Layer + BatchNorm and MaxPooling
        self.conv2 = nn.Conv2d(14, 4, (1, 3))
        self.batchnorm2 = nn.BatchNorm2d(4)
        self.pooling2 = nn.MaxPool2d(1, 4)

        # Padding the input that goes into the third Conv layer
        self.padding3 = nn.ZeroPad2d((4, 4, 0, 0))

        # Third Convolutional Layer + BatchNorm and MaxPooling
        self.conv3 = nn.Conv2d(4, 4, (1, 3))
        self.batchnorm3 = nn.BatchNorm2d(4)
        self.pooling3 = nn.MaxPool2d(1, 4)

        # Linear layer, output on two classes
        self.fc1 = nn.Linear(24, 2)

        # Applying weights initialization
        self.apply(self.init_weights)


    """
    Weight initialization
    """
    def init_weights(self, m):

        # Xavier initialization for Convolutional Layers
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal(m.weight)


    """
    Forward propagation
    """
    def forward(self, x):

        # Unsqueezing the input from (B, C, T) to (B, C, 1, T)
        x = x.unsqueeze(2)

        # Forward propagation on the layers
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

        # Shaping output for the dense layer
        x = x.view(x.shape[0], -1, 24)
        x = self.fc1(x)

        # Unsqueezing the output
        x = x.squeeze(1)
        return x