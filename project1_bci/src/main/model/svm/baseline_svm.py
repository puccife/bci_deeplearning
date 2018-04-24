import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import project1_bci.src.main.utils.dlc_bci as bci
import torch.utils.data as dt
import numpy as np

# Hyper Parameters

num_classes = 2
num_epochs = 300
batch_size = 2
learning_rate = 0.0001

train_input, train_target = bci.load(root='/media/sf_semester_project/bci_deeplearning/project1_bci/data_bci',
                                     one_khz=True)
test_input, test_target = bci.load(root='/media/sf_semester_project/bci_deeplearning/project1_bci/data_bci',
                                   train=False, one_khz=True)

#normalize input
mean = torch.mean(train_input)
std = torch.std(train_input)
train_input = (train_input - mean) / std

mean = torch.mean(test_input)
std = torch.std(test_input)
test_input = (test_input - mean) / std


# flatten the values of our inputs for the SVM
train_input = train_input.view((train_input.size()[0], -1))
test_input = test_input.view((test_input.size()[0], -1))

train_target = torch.FloatTensor([elem if elem == 1 else -1 for elem in train_target])
test_target = torch.FloatTensor([elem if elem == 1 else -1 for elem in test_target])


train_dataset = dt.TensorDataset(train_input, train_target)
test_dataset = dt.TensorDataset(test_input, test_target)


train_loader = dt.DataLoader(dataset=train_dataset,
                             batch_size=batch_size,
                             shuffle=True)

test_loader = dt.DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=True)


# Model
class SVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


model = SVM(train_input.shape[1], num_classes)

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
# criterion = nn.CrossEntropyLoss()
criterion = nn.MarginRankingLoss()
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
        loss = criterion(outputs[:][0].float(), outputs[:][1].float(), target=labels.float())
        # loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))
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
    labels = [int(elem) for elem in labels.numpy()]

    '''
    in pytorch documentation of the loss function it is written 
    if y == 1 then it assumed the first input should be ranked higher 
    (have a larger value) than the second input, and vice-versa for y == -1.
    Therefore, the real labels should follow this pattern for the 
    computation of the accuracy
    '''
    labels = [1 if elem == -1 else 0 for elem in labels]

    correct += (predicted == torch.LongTensor(labels)).sum()
    pred.append(predicted)
    gt.append(labels)

print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

for pr, lb in zip(pred, gt):
    if(int(pr.numpy()[0]) == 0):
        print('predicted: %d, label: %d' % (pr.numpy()[0], lb[0]))

