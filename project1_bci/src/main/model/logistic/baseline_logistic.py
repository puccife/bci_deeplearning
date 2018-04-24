import torch
import torch.nn as nn
from torch.autograd import Variable
import project1_bci.src.main.utils.dlc_bci as bci
import torch.utils.data as dt
import numpy as np
from EegDataset import EegDataset

# Hyper Parameters

num_classes = 2
num_epochs = 300
batch_size = 2
learning_rate = 0.001

train_dataset = EegDataset(data_path='/media/sf_semester_project/bci_deeplearning/project1_bci/data_bci',
                           model='log', training=True)
test_dataset = EegDataset(data_path='/media/sf_semester_project/bci_deeplearning/project1_bci/data_bci', model='log',
                          training=False)


'''
# flatten the values of our inputs for the SVM
train_input = dataset.train_inputs
test_input = dataset.test_inputs

train_input = train_input.view((train_input.size()[0], -1))
test_input = test_input.view((test_input.size()[0], -1))

train_dataset = dt.TensorDataset(train_input, train_target)
test_dataset = dt.TensorDataset(test_input, test_target)
'''


train_loader = dt.DataLoader(dataset=train_dataset,
                             batch_size=batch_size,
                             shuffle=True)

test_loader = dt.DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=True)


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

shape = train_dataset.train_inputs.shape[1]

model = LogisticRegression(train_dataset.train_inputs.shape[1], num_classes)

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training the Model
for epoch in range(num_epochs):
    av_losses = []
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images, requires_grad=True)
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        av_losses.append(loss.data[0])
        loss.backward()
        optimizer.step()
    print('Epoch: [%d/%d], Loss: %.8f'
          % (epoch + 1, num_epochs, np.mean(av_losses)))

# Test the Model
correct = 0
total = 0
pred = []
gt = []
for images, labels in test_loader:
    images = Variable(images)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    pred.append(predicted)
    gt.append(labels)

print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

for pr, lb in zip(pred, gt):
    print('predicted: %d, label: %d' % (pr.numpy()[0], lb.numpy()[0]))

