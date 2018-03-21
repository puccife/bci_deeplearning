import torch

class Preprocessing:

    def __init__(self):
        return None

    def PCA(self, data, k=3):
        # preprocess the data
        X = data
        X_mean = torch.mean(X, 0)
        X = X - X_mean.expand_as(X)

        # svd
        U, S, V = torch.svd(torch.t(X))
        return torch.mm(X, U[:, :k])