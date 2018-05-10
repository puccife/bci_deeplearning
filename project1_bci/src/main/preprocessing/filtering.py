import numpy as np
from scipy import signal


def whitening_transform(cov_matrix):
    """
    Calculate the whitening transform for signals in order
    to remove covariance among them

    :param cov_matrix: covariance matrix of signals
    """
    u, s, v = np.linalg.svd(cov_matrix)

    # returning the whitening transormation
    return np.dot(u, np.dot(np.sqrt(np.linalg.inv(np.diag(s))), v))


def csp(cov_first_class, cov_second_class, num_of_filters):
    """
    implementation of the common spatial patterns (CSP).

    :param cov_first_class: signals covariance for class 1
    :param cov_second_class: signal covariance for class 2
    :param num_of_filters: number of CSP filters to return, this number must be even
    """
    assert num_of_filters % 2 == 0

    whitened_matrix = whitening_transform(cov_first_class + cov_second_class)

    whitened_cov_second_class = np.linalg.multi_dot([whitened_matrix, cov_second_class, whitened_matrix.T])
    u, s, v = np.linalg.svd(whitened_cov_second_class)
    csp_matrix_full = np.dot(v, whitened_matrix)
    assert csp_matrix_full.shape[1] >= num_of_filters

    half_num_filters = int(num_of_filters / 2)

    # selecting the indices of the wanted filters from begin and the end of the rows
    # for num_filters = 4 --> indices [0,1,-2,-1]
    indices = np.roll(np.arange(num_of_filters) - half_num_filters, half_num_filters)

    # returning the selected filters extracted from the full csp matrix
    return csp_matrix_full[indices]


def apply_csp(X, y, on, filters=4):
    cov_matrices_list = []
    # calculate per-class covariance
    for class_ in np.unique(y):
        class_mask = y == class_
        x_filtered = X[class_mask]
        to_cov = np.concatenate(x_filtered, axis=1)
        cov_matrices_list.append(np.cov(to_cov))
    assert len(cov_matrices_list) == 2
    # calculate CSP spatial filters, the third argument is the number of filters to extract
    W = csp(cov_matrices_list[0], cov_matrices_list[1], filters)
    print("projection on the spatial filter started!")
    projection = np.asarray([np.dot(W, trial) for trial in on])
    print("applied csp")
    return projection


def apply_channel_variance(X):
    print("variance on channels started!")
    result = np.var(X, axis=2)
    return result


class ButterFilter:
    def apply_filter(self, train_inputs, test_inputs, sample_rate=100):
        # create band-pass filter for the  8--30 Hz where the power change is expected
        (b, a) = signal.butter(6, np.array([8, 30]) / (sample_rate / 2), btype='bandpass')

        # band-pass filter the EEG
        train_inputs = signal.lfilter(b, a, train_inputs, 2)
        test_inputs = signal.lfilter(b, a, test_inputs, 2)
        return train_inputs, test_inputs
