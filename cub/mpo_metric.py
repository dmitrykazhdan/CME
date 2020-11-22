import numpy as np


def get_mpo_vals(c_test, c_pred):
    '''
    :param c_test: ground truth concept data
    :param c_pred: predicted concept data
    :return: mpo metric values computed from c_test and c_pred, with m ranging from 0 to the number of concepts
    '''

    n_concepts = c_test.shape[1]

    # Set entries where concepts predicted correctly to 0, set to 1 otherwise
    eq = (c_test != c_pred)

    # Compute, for each sample, the number of concepts guessed incorrectly
    eq = np.sum(eq, axis=-1)

    y_vals = []

    for i in range(n_concepts):

        # Compute number of samples with at least i incorrect concept predictions
        n_incorrect = (eq >= i).astype(np.int)

        # Compute % of these samples from total
        metric = (np.sum(n_incorrect) / c_test.shape[0])
        y_vals.append(metric)

    return np.array(y_vals)
