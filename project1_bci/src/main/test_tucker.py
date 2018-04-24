import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import utils.dlc_bci as bci
import torch.utils.data as dt
from IPython.display import clear_output
import time
import tensorly as tl
from tensorly.decomposition import parafac, tucker
tl.set_backend('pytorch')

# Hyperparams
batch_size = 1
num_epochs = 20

# Loading data
train_input , train_target = bci.load(root='../../data_bci', one_khz = True)
test_input , test_target = bci.load ( root = '../../data_bci', train = False, one_khz = True)
train_input = train_input.permute(0,2,1)
test_input = test_input.permute(0,2,1)
#train_input = train_input.view(train_input.shape[0], 1, 28, 500).permute(0,1,3,2)
#test_input = test_input.view(test_input.shape[0], 1, 28, 500).permute(0,1,3,2)
normalized_input = (train_input - train_input.mean(dim=1).mean(dim=0)) / train_input.std(dim=1).mean(dim=0)
# Rank of the Tucker decomposition
tucker_rank = [315, 500, 3]
core, tucker_factors = tucker(normalized_input, ranks=tucker_rank, n_iter_max=100, init="svd", tol=1e-06, random_state=None, verbose=True)