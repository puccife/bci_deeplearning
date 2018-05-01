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
    def transform(self, train_inputs, test_inputs, extend=True):
        train_inputs, test_inputs = self.__permutate(train_inputs, test_inputs)
        train_inputs, test_inputs = self.__normalize(train_inputs, test_inputs)
        if extend:
            train_inputs, test_inputs = self.__shape2D(train_inputs, test_inputs)
        return train_inputs, test_inputs


    """
    Function used to permutate the shape of the data. (S = samples, T = timeseries, C = channels)
    @:parameters
        - train_inputs: training data with shape (S, C, T)
        - test_inputs: testing data with shape (S, C, T)
    @:returns
        - train_inputs: training data with shape (S, T, C)
        - test_inputs: testing data with shape (S, T, C)
    """
    def __permutate(self, train_inputs, test_inputs):
        return train_inputs.permute(0, 2, 1), test_inputs.permute(0, 2, 1)


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
    def __filter(self, train_inputs, test_inputs):
        return train_inputs, test_inputs


    """
    Extracting channels
    """
    def __extract_channels(self, train_inputs, test_inputs):
        return train_inputs, test_inputs