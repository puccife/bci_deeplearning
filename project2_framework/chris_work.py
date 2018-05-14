import math
import torch
from torch import Tensor
from project2_framework.loader import *
import numpy as np


class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Linear(Module):

    def __init__(self, input_dim, output_dim, bias=True):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.weights = Tensor(output_dim, input_dim).normal_(mean=0, std=0.01)
        if self.bias:
            self.bias = Tensor(output_dim).normal_(mean=0, std=0.01)

        # initialize the tensors to accumulate the gradients during backprop
        # remember to initialize to zero at the beginning of every mini-batch step
        self.dl_dw = Tensor(self.weights.size()).zero_()
        self.dl_db = Tensor(self.bias.size()).zero_()

    def forward(self, *input):
        x = input[0]

        # saving the input of the previous layer (x_{l-1}) for the backprop algorithm
        self.input_prec_layer = input[0]

        # saving the output of this layer (s_{l}) for the backprop algorithm
        self.output_non_activated = self.weights.mv(x) + self.bias

        return self.output_non_activated

    def backward(self, *gradwrtoutput):

        # for a linear layer l, the gradwrtoutput will be the grad output
        # from the activation module, that is the product of dsigma(s_{l})
        # and the grad wrt the output of the activation function
        #
        #  #
        grad_wrt_s_l = gradwrtoutput[0]

        # compute the grad wrt the input of previous layer (x_{l-1})
        grad_wrt_input_prev_layer = self.weights.t().mv(grad_wrt_s_l)

        # compute the grad wrt the weights of this layer
        # accumulate the grad in our specific tensor
        self.dl_dw.add_(grad_wrt_s_l.view(-1, 1).mm(self.input_prec_layer.view(1, -1)))

        # compute grad wrt the bias term
        self.dl_db.add_(grad_wrt_s_l)

        return grad_wrt_input_prev_layer

    def param(self):
        return [(self.weights, self.dl_dw), (self.bias, self.dl_db)]

    def update_weights(self, lr):
        self.weights -= lr*self.dl_dw
        self.bias -= lr * self.dl_db

        #reset accumulators
        self.dl_dw.zero_()
        self.dl_db.zero_()


class Relu(Module):

    def forward(self, *input):
        self.input_non_activated = input[0]

        # TODO: risolvere questione input come liste, assert?

        # apply torch clamp(min) to obtain the Relu function
        return input[0].clamp(min=0)

    def backward(self, *gradwrtoutput):

        # computing the derivative of the input of the previous layer
        derivative = self.derivative(self.input_non_activated)
        return derivative * gradwrtoutput[0]


    def derivative(self, inputs):
        """
        compute the relu derivative wrt inputs, if
        input < 0 then dRelu = 0, else dRelu = 1
        :param inputs: torch tensor
        """

        inputs[inputs <= 0] = 0
        inputs[inputs > 0] = 1
        return inputs


class Tanh(Module):

    def forward(self, *input):
        self.input_non_activated = input[0]
        return (2. / (1. + torch.exp(-2 * self.input_non_activated))) - 1


    def backward(self, *gradwrtoutput):
        derivative = self.derivative(self.input_non_activated)
        return derivative * gradwrtoutput[0]

    def derivative(self, inputs):
        return 4 * (inputs.exp() + inputs.mul(-1).exp()).pow(-2)


class LossMSE:
    def compute(self, predicted, target):
        return (predicted - target).pow(2).sum()

    def derivative(self, predicted, target):
        return 2 * (predicted - target)

class Sequential:

    def __init__(self, modules):
        self.modules = modules

    def forward(self, inputs):

        temp_out = inputs
        for index, module in enumerate(self.modules):
            temp_out = module.forward(temp_out)

        return temp_out

    def backward(self, gradwrtoutput):
        temp_back = gradwrtoutput

        for module in self.modules[::-1]:
            temp_back = module.backward(temp_back)

    def update_weights(self, lr):
        for module in self.modules:
            if not isinstance(module, Relu) and not isinstance(module, Tanh):
                module.update_weights(lr)


def sgd(model, x_batch, y_batch, loss, lr):
    batch_losses = []
    for x, y in zip(x_batch, y_batch):
        predicted = model.forward(x)
        # print(predicted)
        loss_value = loss.compute(predicted, y)
        batch_losses.append(loss_value)

        derivative_loss = loss.derivative(predicted, y)
        model.backward(derivative_loss)

    # TODO: temporaneo sgd update fatto cosi
    model.update_weights(lr)

    # return the updated model and the average loss of the batch
    return model, sum(batch_losses) / len(batch_losses)



def generate_data(row_dimension, col_dimension):
    data_input = np.random.uniform(0,1,(row_dimension,col_dimension))
    indices = np.arange(0,row_dimension)
    # apply a function on indices if input belongs to circle assign 1 otherwise assign 0
    data_output = np.asarray(list(map(lambda index: inCircle(data_input[index]),indices)))
    input_tensor = torch.from_numpy(data_input)
    target_tensor = torch.from_numpy(data_output)
    new_target = []
    for e in target_tensor.float():
        t = [0,0]
        t[int(e)] = 1
        new_target.append(t)
    target_tensor = torch.Tensor(new_target)
    return input_tensor.float(), target_tensor.float()

# check if (x,y) point is inside circle or not
def inCircle(values):
    radius = 1 / (math.sqrt(math.pi))
    area = math.pi * math.pow(radius, 2) / 2
    x_co = math.pow(values[0] - 0.5, 2)
    y_co = math.pow(values[1] - 0.5, 2)
    return 1 if x_co + y_co < (math.pow(radius,2) / 2) else 0

################################################################

def train(model, epochs, train_loader, test_loader, loss, lr=0.01):

    for epoch in range(epochs):
        epoch_train_losses = []
        for x_batch, y_batch in train_loader.get_loader():

            model, batch_loss = sgd(model=model, x_batch=x_batch, y_batch=y_batch, loss=loss, lr=lr)
            epoch_train_losses.append(batch_loss)
            # print('Epoch [%d/%d], batch Loss: %.9f' % (epoch + 1, epochs, batch_loss))

        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)

        # validation
        epoch_val_losses = []
        for x_batch, y_batch in test_loader.get_loader():
            for x, y in zip(x_batch, y_batch):
                predicted = model.forward(x)
                loss_value = loss.compute(predicted, y)
                epoch_val_losses.append(loss_value)

        epoch_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)

        print('Epoch [%d/%d], train Loss: %.6f, val loss: %.6f'% (epoch + 1, epochs,
                 epoch_train_loss, epoch_val_loss))





###################### run #######################

inputs, targets = generate_data(1000, 2)

targets = targets * 0.9 # for tanh range

# TODO: splitting in train test, this is temporary
train_inputs = inputs[:800]
train_targets = targets[:800]

test_inputs = inputs[800:]
test_targets = targets[800:]

batch_size = 10

train_loader = DataLoader(train_inputs, train_targets, batch_size)

test_loader = DataLoader(test_inputs, test_targets, batch_size)

# defining layers

shape = train_inputs[0].shape

layers = [Linear(input_dim=train_inputs[0].shape[0], output_dim=25), Relu(), Linear(input_dim=25, output_dim=2), Tanh()]

model = Sequential(layers)

train(model=model, epochs=500, train_loader=train_loader, test_loader=test_loader, loss=LossMSE(), lr=0.001)








