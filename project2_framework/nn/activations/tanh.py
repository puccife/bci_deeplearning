from ..module import Module
import torch

class Tanh(Module):
    """
    Tanh activation implementation
    """

    def forward(self, *input):

        # saving input from previous layer used for backprop
        self.input_non_activated = input[0]

        # computing the tanh value
        return (2. / (1. + torch.exp(-2 * self.input_non_activated))) - 1


    def backward(self, *gradwrtoutput):
        derivative = self.derivative(self.input_non_activated)
        return derivative * gradwrtoutput[0]

    def derivative(self, inputs):
        return 4 * (inputs.exp() + inputs.mul(-1).exp()).pow(-2)

