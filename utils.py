import numpy as np


def labelled_unlabbeled_split(x_data, c_data, y_data, n_labelled=100, n_unlabelled=200):
    '''
    Split data into those with labels, and those without labels
    :param x_data: Input data, numpy array of shape (n_samples, ...)
    :param c_data: Concept data, numpy array of shape (n_samples, n_concepts)
    :param y_data: Label data, numpy array of shape (n_samples,)
    :param n_labelled: Number of labelled samples to return
    :param n_unlabelled: Number of unlabelled samples to return
    :return: Returns six numpy arrays: x_data_l, c_data_l, y_data_l, x_data_u, c_data_u, y_data_u
             Corresponding to labelled, and unlabelled data subsets
    '''

    # Ensure you don't request more data than possible
    assert (n_labelled + n_unlabelled) <= x_data.shape[0]

    # Generate indices for labelled and unlabelled datasets
    idx = np.random.choice(x_data.shape[0], n_labelled + n_unlabelled, replace=False)
    idx_l = idx[:n_labelled]
    idx_u = idx[n_labelled:]

    # Extract the labelled datasets
    x_data_l, c_data_l, y_data_l = x_data[idx_l, :], c_data[idx_l, :], y_data[idx_l, :]

    # Extract the unlabelled datasets
    x_data_u, c_data_u, y_data_u = x_data[idx_u, :], c_data[idx_u, :], y_data[idx_u, :]

    return x_data_l, c_data_l, y_data_l, x_data_u, c_data_u, y_data_u



