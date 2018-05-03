import numpy as np
from eegtools import spatfilt
from sklearn import base
from scipy import signal


class CSP(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y):
        class_covs = []

        # calculate per-class covariance
        for ci in np.unique(y):
            class_mask = y == ci
            num_ones = class_mask.sum()
            x_filtered = X[class_mask]
            # to_cov = np.hstack(X[class_mask])
            to_cov = np.concatenate(x_filtered, axis=1)
            class_covs.append(np.cov(to_cov))
        assert len(class_covs) == 2

        # calculate CSP spatial filters, the third argument is the number of filters to extract
        self.W = spatfilt.csp(class_covs[0], class_covs[1], 8)
        print("csp concluded, spatial filter computed!")
        return self

    def transform(self, X):
        # Note that the projection on the spatial filter expects zero-mean data.
        print("projection on the spatial filter started!")
        projection = np.asarray([np.dot(self.W, trial) for trial in X])
        return projection


class ChanVar(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y): return self

    def transform(self, X):
        print("variance on channels started!")
        result = np.var(X, axis=2)  # X.shape = (trials, channels, time)
        return result

class ButterFilter:
    def apply_filter(self, train_inputs, test_inputs):
        # create band-pass filter for the  8--30 Hz where the power change is expected
        sample_rate = 100
        (b, a) = signal.butter(6, np.array([8, 30]) / (sample_rate / 2), btype='bandpass')

        # band-pass filter the EEG
        train_inputs = signal.lfilter(b, a, train_inputs, 2)
        test_inputs = signal.lfilter(b, a, test_inputs, 2)
        return train_inputs, test_inputs