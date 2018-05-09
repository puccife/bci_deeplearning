import numpy as np
from sklearn import base
from scipy import signal


def whitening_transform(cov_matrix, cut_off=1e-15):
    """
    Calculate the whitening transform for signals in order
    to remove covariance among them

    :param cov_matrix: covariance matrix of signals
    :param cut_off: fraction of the largest eigenvalue of the matrix C
    """
    eigen_values, eigenvectors = np.linalg.eigh(cov_matrix)

    return np.linalg.multi_dot([eigenvectors,
                                np.diag(np.where(eigen_values > np.max(eigen_values) * cut_off,
                                                 eigen_values,
                                                 np.inf) ** -.5),
                                eigenvectors.T])


def csp(cov_first_class, cov_second_class, num_of_filters):
    """
    implementation of the common spatial patterns (CSP).

    :param cov_first_class: signals covariance for class 1
    :param cov_second_class: signal covariance for class 2
    :param num_of_filters: number of CSP filters to return, this number must be even
    """
    assert num_of_filters % 2 == 0

    whitened_matrix = whitening_transform(cov_first_class + cov_second_class)
    P_C_b = np.linalg.multi_dot([whitened_matrix, cov_second_class, whitened_matrix.T])
    _, _, B = np.linalg.svd((P_C_b))
    csp_matrix_full = np.dot(B, whitened_matrix.T)
    assert csp_matrix_full.shape[1] >= num_of_filters

    half_num_filters = int(num_of_filters / 2)

    # selecting the indices of the wanted filters from begin and the end of the rows
    # for num_filters = 4 --> indices [0,1,-2,-1]
    indices = np.roll(np.arange(num_of_filters) - half_num_filters, half_num_filters)

    # returning the selected filters extracted from the full csp matrix
    return csp_matrix_full[indices]

#########################


class CSP(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y):
        class_covs = []

        # calculate per-class covariance
        for ci in np.unique(y):
            class_mask = y == ci
            x_filtered = X[class_mask]
            to_cov = np.concatenate(x_filtered, axis=1)
            class_covs.append(np.cov(to_cov))
        assert len(class_covs) == 2

        # calculate CSP spatial filters, the third argument is the number of filters to extract
        self.W = csp(class_covs[0], class_covs[1], 8)
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