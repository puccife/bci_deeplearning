from scipy import signal

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import project1_bci.src.main.utils.dlc_bci as bci
import torch.utils.data as dt
import numpy as np
from eegtools import spatfilt
from sklearn import base
from project1_bci.src.main.preprocessing.preprocessing import Preprocess


# Create sklearn-compatible feature extraction and classification pipeline:
class CSP(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y):
        class_covs = []

        # calculate per-class covariance
        for ci in np.unique(y):
            class_mask = y == ci
            num_ones = class_mask.sum()
            x_filtered = X[class_mask]
            # to_cov = np.hstack(X[class_mask])
            to_cov = np.concatenate(x_filtered, axis=1)
            class_covs.append(np.cov(to_cov))
        assert len(class_covs) == 2

        # calculate CSP spatial filters, the third argument is the number of filters to extract
        self.W = spatfilt.csp(class_covs[0], class_covs[1], 8)
        print("csp concluded, spatial filter computed!")
        return self

    def transform(self, X):
        # Note that the projection on the spatial filter expects zero-mean data.
        print("projection on the spatial filter started!")
        projection = np.asarray([np.dot(self.W, trial) for trial in X])
        return projection


class ChanVar(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y): return self

    def transform(self, X):
        print("variance on channels started!")
        result = np.var(X, axis=2)  # X.shape = (trials, channels, time)
        return result

#################################################


# Hyper Parameters

num_classes = 2
num_epochs = 300
batch_size = 2
learning_rate = 0.0001
data_path = '../../data_bci'
train_inputs, train_targets = bci.load(root=data_path, one_khz=False)
test_inputs, test_targets = bci.load(root=data_path, train=False, one_khz=False)

# normalize inputs
pp = Preprocess()
train_inputs, test_inputs = pp.transform(train_inputs, test_inputs, permutate=False, model=None)

train_targets = torch.Tensor([elem if elem == 1 else -1 for elem in train_targets])
test_targets = torch.Tensor([elem if elem == 1 else -1 for elem in test_targets])

train_inputs = train_inputs.numpy()
train_targets = train_targets.numpy()
test_inputs = test_inputs.numpy()
test_targets = test_targets.numpy()

count = 0
id_elem_removed = []
for index, target in enumerate(train_targets):
    if target == 0:
        count += 1
        if count > 2:
            break
        id_elem_removed.append(index)
train_inputs = np.delete(train_inputs, id_elem_removed, axis=0)
train_targets = np.delete(train_targets, id_elem_removed, axis=0)

count = 0
id_test_removed = []
for index, target in enumerate(test_targets):
    if target == 0:
        count += 1
        if count > 2:
            break
        id_test_removed.append(index)

test_inputs = np.delete(test_inputs, id_test_removed, axis=0)
test_targets = np.delete(test_targets, id_test_removed, axis=0)

print(train_inputs.shape)
print(train_targets.shape)
print(test_inputs.shape)
print(test_targets.shape)

# create band-pass filter for the  8--30 Hz where the power change is expected
sample_rate = 100
(b, a) = signal.butter(6, np.array([8, 30]) / (sample_rate / 2), btype='bandpass')

# band-pass filter the EEG
train_inputs = signal.lfilter(b, a, train_inputs, 2)
test_inputs = signal.lfilter(b, a, test_inputs, 2)

###############################################



# projecting on spacial filters
csp = CSP()
csp.fit(train_inputs, train_targets)
train_inputs = csp.transform(train_inputs)
train_inputs = ChanVar().transform(train_inputs)
train_inputs = torch.from_numpy(train_inputs)
train_targets = torch.from_numpy(train_targets)

csp_test = CSP().fit(test_inputs, test_targets)
test_inputs = csp_test.transform(test_inputs)
test_inputs = ChanVar().transform(test_inputs)
test_inputs = torch.from_numpy(test_inputs)
test_targets = torch.from_numpy(test_targets)

train_dataset = dt.TensorDataset(train_inputs, train_targets)
test_dataset = dt.TensorDataset(test_inputs, test_targets)


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


model = SVM(train_inputs.shape[1], num_classes)

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
        images = Variable(images.float(), requires_grad=True)
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
    images = Variable(images.float())
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

