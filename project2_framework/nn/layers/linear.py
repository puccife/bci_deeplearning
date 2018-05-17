from ..module import Module
from torch import Tensor


class Linear(Module):
    """
    Linear layer implementation
    """

    def __init__(self, input_dim, output_dim, bias=True):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        # init layer weights using xavier initialization
        self.weights = Tensor(output_dim, input_dim).normal_(mean=0, std=1 / self.input_dim)
        if self.bias:
            self.bias = Tensor(output_dim).zero_()

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
        """
        returns pair of tensors: first is a parameter tensor,
        the second is the gradient accumulator for this parameter tensor
        :return:
        """
        return [(self.weights, self.dl_dw), (self.bias, self.dl_db)]
