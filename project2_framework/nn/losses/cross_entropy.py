import torch


class CrossEntropy():
    """Cross Entropy Loss implementation"""

    def compute(self, predicted, target):
        target = target.view(-1,1).max(0)[1]
        predicted = predicted.view(1, -1)

        # computing the negative log likelihood
        log_likelihood = -torch.log(predicted[range(predicted.shape[0]), target])
        loss = torch.sum(log_likelihood) / predicted.shape[0]
        return loss

    def derivative(self, predicted, target):

        # adding 1e-6 in the division in order to avoid division by zero
        der = (predicted.sub(target)).div(predicted*(1-predicted) + 1e-6)
        return der
