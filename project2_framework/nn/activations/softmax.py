from ..module import Module
import torch


class Softmax(Module):
    """
    implementation of softmax with forward,
    bacward and derivative methods
    """

    def forward(self, *input):

        # saving the input from previous layer for backprop
        self.input_non_activated = input[0]
        return self._softmax(input[0])

    def backward(self, *gradwrtoutput):
        derivative = self.derivative(self.input_non_activated)
        return derivative * gradwrtoutput[0]

    def _softmax(self, inputs):
        inputs = inputs.view(-1, 1)

        # getting the max value
        max_row = inputs.view(-1,1).max(0)[0].view(-1, 1)

        # using the max value to make the computation more stable
        exps = torch.exp(inputs - max_row)
        sum_exp = torch.sum(exps, 0).view(-1,1)
        sm = exps / sum_exp
        return sm.view(-1)

    def derivative(self, inputs):
        sm = self._softmax(inputs)
        return sm.mul(1 - sm)
