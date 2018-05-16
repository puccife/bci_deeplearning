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
        inputs = inputs.view(-1, 1)
        max_row = inputs.view(-1,1).max(0)[0].view(-1, 1)
        exps = torch.exp(inputs - max_row)
        sum_exp = torch.sum(exps, 0).view(-1,1)
        sm = exps / sum_exp
        return sm.view(-1)

    def derivative(self, inputs):
        sm = self._softmax(inputs)
        s = sm.view(-1, 1)
        der = self._diagflat(s) - torch.mm(s, s.t())
        return der