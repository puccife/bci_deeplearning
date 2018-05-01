import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import collections
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as dt

from preprocessing.preprocessing import Preprocess
import utils.dlc_bci as bci
import utils.hyperparams as hyperparams
from model.eegnet.trainer import EEGNetTrainer

DATA_PATH = '../../data_bci'

def main(model, epochs, batch_size):

    train_inputs, train_targets = bci.load(root=DATA_PATH, one_khz=True)
    test_inputs, test_targets = bci.load(root=DATA_PATH, train=False, one_khz=True)

    train_inputs, test_inputs = Preprocess().transform(train_inputs, test_inputs)

    # Datasets
    train_dataset = dt.TensorDataset(train_inputs, train_targets)
    test_dataset = dt.TensorDataset(test_inputs, test_targets)
    train_loader = dt.DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_loader = dt.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    tr = EEGNetTrainer(num_epochs=epochs, batch_size=batch_size)

    tr.train(train_loader=train_loader, test_loader=test_loader)

    tr.create_graph()

    return 0

if __name__ == '__main__':
    model, \
    epochs, \
    batch_size, \
    learning_rate = hyperparams.load_config()

    parser = argparse.ArgumentParser(description='Select ECOBICI dataset to load.')
    parser.add_argument("-m", "--model", help="Model for training", type=str, default=model)
    parser.add_argument("-e", "--epochs", help="Number of training epochs", type=int, default=epochs)
    parser.add_argument("-b", "--batch_size", help="Batch size for training", type=int, default=batch_size)
    args = parser.parse_args()
    main(args.model, args.epochs, args.batch_size)