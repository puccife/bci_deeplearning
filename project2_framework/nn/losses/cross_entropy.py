import torch

class CrossEntropy():

    def compute(self, predicted, target):
        target = target.max(1)[1]
        log_likelihood = -torch.log(predicted[range(predicted.shape[0]), target])
        loss = torch.sum(log_likelihood) / predicted.shape[0]
        return loss

    def derivative(self, predicted, target):
        return (predicted.sub(target)).div(predicted*(1-predicted))