from .filtering import *
import torch
"""
This class is used to preprocess the data.
@authors:
    - Federico Pucci
    - Christian Sciuto
"""
class Preprocess:

    def __init__(self):
        return None


    """
    Function used to transform the raw data
    @:parameters
        - train_inputs: raw data for training
        - test_inputs: raw data for testing
        - extend: flag used to reshape the data based on the model used for training
    @:returns
        - train_inputs: preprocessed data for training
        - test_inputs: preprocessed data for testing
    """
    def transform(self, train_inputs, test_inputs, train_targets, test_targets):
        train_inputs, test_inputs = self.__normalize(train_inputs, test_inputs)
        train_inputs, test_inputs = self.__filter(train_inputs, test_inputs, train_targets, test_targets)
        return train_inputs, test_inputs


    """
    Function used to normalized the data using Z-score normalization.
    @:parameters
        - train_inputs: training data not normalized
        - test_inputs: testing data not normalized
    @:returns
        - train_inputs: training data normalized using Z-score
        - test_inputs: testing data normalized using Z-score
    """
    def __normalize(selfs, train_inputs, test_inputs):
        means = train_inputs.mean(dim=1).mean(dim=0)
        stds = train_inputs.std(dim=1).mean(dim=0)
        train_inputs = train_inputs.sub(means).div(stds)
        test_inputs = test_inputs.sub(means).div(stds)
        return train_inputs, test_inputs


    """
    Function used to reshape the data. (S = samples, T = timeseries, C = channels)
    @:parameters
        - train_inputs: training data with shape (S, T, C)
        - test_inputs: testing data with shape (S, T, C)
    @:returns
        - train_inputs: training data with shape (S, 1,  T, C)
        - test_inputs: testing data with shape (S, 1, T, C)
    """
    def __shape2D(self, train_inputs, test_inputs):
        train_inputs = train_inputs\
            .contiguous()\
            .view(train_inputs.shape[0],
                  1,
                  train_inputs.shape[1],
                  train_inputs.shape[2])
        test_inputs = test_inputs\
            .contiguous()\
            .view(test_inputs.shape[0],
                  1,
                  test_inputs.shape[1],
                  test_inputs.shape[2])
        return train_inputs, test_inputs


    """
    Filtering signal
    """
    def __filter(self, train_inputs, test_inputs, train_targets, test_targets):

        train_inputs = train_inputs.numpy()
        train_targets = train_targets.numpy()
        test_inputs = test_inputs.numpy()
        test_targets = test_targets.numpy()

        train_inputs, test_inputs = ButterFilter().apply_filter(train_inputs, test_inputs)

        # projecting on spacial filters
        csp = CSP()
        csp.fit(train_inputs, train_targets)
        train_inputs = csp.transform(train_inputs)
        train_inputs = ChanVar().transform(train_inputs)

        csp_test = CSP().fit(test_inputs, test_targets)
        test_inputs = csp_test.transform(test_inputs)
        test_inputs = ChanVar().transform(test_inputs)

        train_inputs = torch.from_numpy(train_inputs)
        test_inputs = torch.from_numpy(test_inputs)

        return train_inputs, test_inputs


    """
    Extracting channels
    """
    def __extract_channels(self, train_inputs, test_inputs):
        return train_inputs, test_inputs

    def PCA(self, X, k=3):
        # preprocess the data
        X = X - torch.mean(X, 0).expand_as(X)
        # svd
        U, S, V = torch.svd(torch.t(X))
        return torch.mm(X, U[:, :k])