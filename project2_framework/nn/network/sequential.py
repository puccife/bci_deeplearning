from ..activations.relu import Relu
from ..activations.tanh import Tanh
from ..activations.softmax import Softmax


class Sequential:
    """
    class to collect different layers of
    a network and compute forward and backward
    pass on those layers
    """

    def __init__(self, modules):

        # saving the layers
        self.modules = modules

    def forward(self, inputs):
        """
        calls the forward function of each
        layer starting from the first one
        """

        temp_out = inputs
        for index, module in enumerate(self.modules):
            temp_out = module.forward(temp_out)

        return temp_out

    def backward(self, gradwrtoutput):
        """
        calls the backward function on each
        layer starting from the last one
        """
        temp_back = gradwrtoutput

        for module in self.modules[::-1]:
            temp_back = module.backward(temp_back)

    def get_params(self):
        """
        returns all the parameters of each layer
        """
        model_params = []
        for module in self.modules:
            model_params.append(module.param())

        return model_params
