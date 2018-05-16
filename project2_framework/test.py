from utils.data_generator import generate_data
from utils.loader import DataLoader
import math
import torch
from torch import Tensor
from utils.loader import DataLoader
from utils.data_generator import generate_data
from nn.module import Module
from nn.layers.linear import Linear
from nn.activations.relu import Relu
from nn.activations.tanh import Tanh
from nn.activations.softmax import Softmax
from nn.losses.mse import LossMSE
from nn.losses.cross_entropy import CrossEntropy
from nn.network.sequential import Sequential
from trainer import train

def main():

    batch_size = 10

    train_inputs, train_targets = generate_data(1000, 2)
    test_inputs, test_targets = generate_data(1000, 2)

    train_loader = DataLoader(train_inputs, train_targets, batch_size)
    test_loader = DataLoader(test_inputs, test_targets, batch_size)

    layers = [Linear(input_dim=train_inputs[0].shape[0], output_dim=25), Relu(),
              Linear(input_dim=25, output_dim=25), Relu(),
              Linear(input_dim=25, output_dim=2), Softmax()]

    model = Sequential(layers)

    train(model=model, epochs=500, train_loader=train_loader, test_loader=test_loader, loss=CrossEntropy(), lr=0.01)

    return 0


if __name__=='__main__':
    main()