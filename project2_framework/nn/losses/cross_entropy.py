import torch

class CrossEntropy():

    def compute(self, predicted, target):
        target = target.view(-1,1).max(0)[1]
        predicted = predicted.view(1, -1)
        log_likelihood = -torch.log(predicted[range(predicted.shape[0]), target])
        loss = torch.sum(log_likelihood) / predicted.shape[0]
        return loss

    def derivative(self, predicted, target):
        der = (predicted.sub(target)).div(predicted*(1-predicted))
        print("cross entropy")
        return der