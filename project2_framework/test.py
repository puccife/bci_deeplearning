from utils.loader import DataLoader
from utils.data_generator import generate_data
from nn.layers.linear import Linear
from nn.activations.relu import Relu
from nn.activations.tanh import Tanh
from nn.activations.softmax import Softmax
from nn.losses.mse import LossMSE
from nn.losses.cross_entropy import CrossEntropy
from nn.network.sequential import Sequential

from nn.optimizers.SGD import SGD
from trainer import Trainer


def main():

    batch_size = 10

    # generating our data
    train_inputs, train_targets = generate_data(1000, 2)
    test_inputs, test_targets = generate_data(1000, 2)

    # creating our loaders for training and test sets
    train_loader = DataLoader(train_inputs, train_targets, batch_size)
    test_loader = DataLoader(test_inputs, test_targets, batch_size)

    # defining our layers
    layers = [Linear(input_dim=train_inputs[0].shape[0], output_dim=25), Relu(),
              Linear(input_dim=25, output_dim=25), Relu(),
              Linear(input_dim=25, output_dim=2), Softmax()]

    # creating our model
    model = Sequential(layers)

    # init our optimizer
    optimizer = SGD(model.get_params(), lr=0.01)

    # init our trainer
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      epochs=500,
                      loss=CrossEntropy(),
                      train_loader=train_loader,
                      test_loader=test_loader)

    # starting the training session
    trainer.train()

    return 0


if __name__=='__main__':
    main()