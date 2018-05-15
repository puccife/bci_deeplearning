from ..module import Module
import torch

class Softmax(Module):

    def forward(self, *input):
        self.input_non_activated = input[0]
        return self._softmax(input[0])

    def backward(self, *gradwrtoutput):
        derivative = self.derivative(self.input_non_activated)
        return derivative * gradwrtoutput[0]

    def _diagflat(self, s):
        size = len(s)
        diag = torch.zeros(size, size)
        for index, elem in enumerate(s):
            diag[index][index] = elem[0]
        return diag

    def _softmax(self, inputs):
        exps = torch.exp(inputs - (inputs.max(1)[0].view(-1,1)))
        return exps / torch.sum(exps, 1).view(-1,1)

    def derivative(self, inputs):
        sm = self._softmax(inputs)
        s = sm.view(-1, 1)
        return self._diagflat(s) - torch.mm(s, s.t())