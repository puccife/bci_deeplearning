import torch
import torch.utils.data as dt
from torch.autograd import Variable

import utils.dlc_bci as bci

batch_size = 10

train_input , train_target = bci.load(root='../../data_bci', one_khz = True)
test_input , test_target = bci.load ( root = '../../data_bci', train = False, one_khz = True)

print(train_input.size())
print(train_target.size())

dataset = dt.TensorDataset(train_input, train_target)

train_loader = dt.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

for i, (images, labels) in enumerate(train_loader):
    images = Variable(images)
    labels = Variable(labels)
    print(images.size())
    print(labels.size())
