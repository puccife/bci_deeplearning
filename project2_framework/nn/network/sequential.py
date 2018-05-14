from ..activations.relu import Relu
from ..activations.tanh import Tanh

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