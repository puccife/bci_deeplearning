class SGD:
    """
    Stochastic Gradient Descent optimizer
    """

    def __init__(self, model_params, lr=0.01):
        """
        saves the learning rate  and all the parameters
        and gradient accumulators of the network to optimize
        """
        self.model_params = model_params
        self.lr = lr

    def step(self):
        """
        updates all the parameters of the models
        using the respective gradient accumulator
        """
        for layers_params in self.model_params:
            for param_update in layers_params:
                param = param_update[0]
                update = param_update[1]

                # updating the parameter
                param -= self.lr * update

                # initialize to zero the accumulator
                update.zero_()
