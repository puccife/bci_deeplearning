import torch
import torch.nn as nn

import torch.utils.data as dt
from torch.autograd import Variable

import utils.dlc_bci as bci

learning_rate = 0.001
batch_size = 1
num_epochs = 20

train_input , train_target = bci.load(root='../../data_bci', one_khz = True)
test_input , test_target = bci.load ( root = '../../data_bci', train = False, one_khz = True)

print(train_input.size())
print(train_target.size())

train_dataset = dt.TensorDataset(train_input, train_target)
test_dataset = dt.TensorDataset(train_input, train_target)

train_loader = dt.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = dt.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

class CNN(nn.Module):
    def __init__(self):
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

cnn = CNN()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), '../../model/cnn.pkl')
