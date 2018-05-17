from ..module import Module


class Relu(Module):
    """
    implementation of the Relu layer with forward,
    backward and derivative functions
    """

    def forward(self, *input):
        self.input_non_activated = input[0]

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