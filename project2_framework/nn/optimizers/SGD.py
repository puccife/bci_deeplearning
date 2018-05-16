class SGD:

    def __init__(self, model_params, lr=0.01):
        self.model_params = model_params
        self.lr = lr

    def step(self):
        for layers_params in self.model_params:
            for param_update in layers_params:
                param = param_update[0]
                update = param_update[1]
                param -= self.lr * update
                update.zero_()
