import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from collections import defaultdict

from utils import flatten_activations, compute_activation_per_layer


class Net2Vec(object):

    def __init__(self, model, **params):

        # Create copy of passed-in parameters
        self.params = params

        self.layer_id   = self.params["layer_id"]
        self.n_concepts = params['n_concepts']
        self.layer_ids  = [self.layer_id]

        # Retrieve the concept names
        if "concept_names" in self.params:
            self.concept_names = self.params["concept_names"]
        else:
            self.concept_names = ["Concept " + str(i) for i in range(self.n_concepts)]

        # Batch size to use during computation
        self.batch_size = params.get("batch_size", 128)

        self.model = model


    def _get_layer_concept_predictor_model(self, h_data, c_data):

        # Create safe copies of data
        x_data = np.copy(h_data)
        c_data = np.copy(c_data)
        x_train, x_test, y_train, y_test = train_test_split(x_data, c_data, test_size=0.2)

        # If there is only 1 value, then return simple classifier
        unique_vals = np.unique(c_data)

        if len(unique_vals) == 1:
            clf = DummyClassifier(strategy="constant", constant=c_data[0])
        else:
            assert len(unique_vals) == 2, "Defined only for binary classification"
            clf = LogisticRegression(max_iter=200)

        clf.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(x_test))

        return clf, accuracy


    def train(self, x_data_l, c_data_l, x_data_u):

        del x_data_u

        self.vlc_list = defaultdict(lambda: defaultdict(list))
        self.model_ensemble = {}
        x_data = x_data_l
        c_data = c_data_l

        hidden_features = compute_activation_per_layer(x_data, self.layer_ids, self.model,
                                                       self.batch_size, aggregation_function=flatten_activations)[0]

        # Generate predictive models for every concept
        for c in range(self.n_concepts):

            c_vector = c_data[:, c]
            unique = np.unique(c_vector)

            if len(unique) > 1:
                for c_val in unique:
                    # Get training data for a one-vs-all predictor
                    h_l_sample, c_target = one_vs_all_concept_values([hidden_features, c_vector], c_val=c_val, balance=True)
                    clf, desc = self._get_layer_concept_predictor_model(h_l_sample, c_target)
                    self.vlc_list[c].append(clf)

                concept_predictor = MaxEstimatorClassifier(self.vlc_list[c])
                concept_predictor.fit(hidden_features, c_vector)

            else:
                concept_predictor = DummyClassifier(strategy="constant", constant=c_vector[0])
                concept_predictor.fit(hidden_features, c_vector)

            self.model_ensemble[c] = concept_predictor


    def predict_concepts(self, x_data):
        '''
        Given the same model and new x_data, predict their concept values
        '''

        n_samples = x_data.shape[0]
        concept_vals = np.zeros((n_samples, self.n_concepts), dtype=float)

        hidden_features = compute_activation_per_layer(x_data, self.layer_ids, self.model,
                                                       self.batch_size, aggregation_function=flatten_activations)[0]

        for c in range(self.n_concepts):
            clf = self.model_ensemble[c]
            concept_vals[:, c] = clf.predict(hidden_features)

        return concept_vals


class MaxEstimatorClassifier(ClassifierMixin, BaseEstimator):
    '''
    sklearn classifier for getting selecting the concept with the maximum predicted probability
    '''

    def __init__(self, estimators):
        '''
        Set the concept predictor models
        '''
        self.estimators = estimators

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        y_list = []

        # Retrieve probability of all values for the concept (assumed to be binary)
        for clf in self.estimators:
            y_ = clf.predict_proba(X)
            y_ = y_[:, 1]
            y_list.append(y_)
        y_hat = np.vstack(y_list).transpose()

        # Return the value with the largest probability
        return np.argmax(y_hat, axis=-1)


def one_vs_all_concept_values(data_list, c_val, balance=True):
    '''
    :param data_list: Tuple of (activations, concept_values)
    :param c_val: One of the values in concept_values to use as the one, vs all
    :param balance: Whether to return a balanced dataset
    :return: Tuple (activations, concept_values) of the samples, with c_val being 1, and other values being 0
    '''
    c_data = data_list[-1]
    idx_ = one_vs_all_concept_values_idx(c_data, c_val)
    return sample_data_(data_list, idx_, balance=balance)


def one_vs_all_concept_values_idx(c_data, c_val):
    '''
    :param c_data: Concept values for a set of samples (n_samples, 1)
    :param c_val: Concept value to use as 1-vs-all
    :return: Return all sample indices in c_data where the concept value is c_val
    '''
    c_shape = c_data.shape
    assert c_shape[-1] == 1 or len(c_shape) == 1, "Not defined for multiple concepts"
    idx_ = np.where(c_data == c_val)[0]
    return idx_


def sample_data_(data_list, idx_, balance=True):
    '''
    :param data_list: Tuple of (activations, concept_values)
    :param idx_: All sample indices in c_data where the concept value is c_val
    :param idx_neg:
    :param balance: Whether to balance the dataset or not
    :return:
    '''

    # Set all values in idx_ indices to 1, and 0 otherwise
    c_data = data_list[-1]
    assert idx_.shape[0] < c_data.shape[0], "The entire batch has the same concepts"
    mask = np.zeros(c_data.shape, dtype='int32')
    mask[idx_] = 1
    idx_neg = np.where(mask == 0)[0]
    y_data = mask
    data_list[-1] = y_data

    # Balance 0s and 1s via sampling
    data_output = sample_idx(data_list, idx_, idx_neg, balance)
    return tuple(data_output)


def sample_idx(data_list, idx_, idx_neg, balance=True):
    '''
    :param data_list:  Tuple of (activations, concept_values), where concept_values are either 0, or 1
    :param idx_: List of indices of 1
    :param idx_neg: List of indices of 0
    :param balance: Whether to balance them out or not
    :return:
    '''

    # Retrieve the subsampled indices for both positive and negative samples
    if balance:
        idx_, idx_neg = balance_index(idx_, idx_neg)

    # Extract the samples from the dataset
    data_output = []
    for _data in data_list:
        x_data = sample_(_data, idx_, idx_neg)
        data_output.append(x_data)
    return tuple(data_output)


def balance_index(idx_, idx_neg):
    '''
    :param idx_: Positive indices
    :param idx_neg: Negative indices
    :return: Balance via downsampling (radomly subsample the bigger index set)
    '''
    n_points = idx_.shape[0]
    n_points_neg = idx_neg.shape[0]
    if n_points > n_points_neg:
        idx_ = np.random.choice(idx_.shape[0], size=n_points_neg)
    else:
        idx_neg = np.random.choice(idx_neg.shape[0], size=n_points)
    return idx_, idx_neg


def sample_(x_data, idx_, idx_neg):
    idx = np.concatenate((idx_, idx_neg))
    _data = x_data[idx]
    return _data

