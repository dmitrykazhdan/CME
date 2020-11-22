import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math

from cub.utils import compute_activation_per_layer, flatten_activations
from cub.cub_loader import load_batch

'''
Standard (recommended) implementation of the CME concept extractor (located in ItCModel.py) works with:
    (i)  Raw numpy
    (ii) Tensorflow-based generators

The default CUB loader is written in PyTorch, and uses PyTorch-based pre-processing.

Unfortunately, at the time of development, it was not possible to link up that specific PyTorch loader+preprocessing to a 
TF loader in any intuitive way.

Thus, this class serves as an ad-hoc extension of the standard ItCModel, which has an extra batch-based loader
'''

class ItCModel_CUB(object):

    def __init__(self, model, **params):

        # Create copy of passed-in parameters
        self.params = params

        # Retrieve the layers to use for hidden representations
        if "layer_ids" in self.params:
            self.layer_ids = self.params["layer_ids"]
        else:
            self.layer_ids = [i for i in range(len(model.layers))]

        # Retrieve the corresponding layer names
        if "layer_names" in self.params:
            self.layer_names = self.params["layer_names"]
        else:
            self.layer_names = ["Layer " + str(i) for i in range(len(model.layers))]

        self.n_concepts = params['n_concepts']

        # Retrieve the concept names
        if "concept_names" in self.params:
            self.concept_names = self.params["concept_names"]
        else:
            self.concept_names = ["Concept " + str(i) for i in range(self.n_concepts)]

        # Retrieve the concepts to use at each layer
        self.layer_concepts = []

        # Batch size to use during computation
        self.batch_size = params.get("batch_size", 128)

        # Set classifier to use for concept value prediction
        self.clf_method = params.get("method", "LR")

        self.model = model


    def get_clf(self):
        if self.clf_method == 'LR':
            semi_supervised = False
            clf = LogisticRegression(max_iter=200)
        elif self.clf_method == 'LP':
            semi_supervised = True
            clf = LabelSpreading()
        else:
            raise ValueError("Non-implemented method")

        return clf, semi_supervised


    def _get_layer_concept_predictor_model(self, h_data_l, c_data_l, h_data_u):
        '''
        Train cme concept extractor
        '''

        # Create safe copies of data
        x_data_l = np.copy(h_data_l)
        y_data_l = np.copy(c_data_l)
        x_data_u = np.copy(h_data_u)
        y_data_u = np.ones((x_data_u.shape[0])) * -1    # Here, label is 'undefined'

        # Specify whether to use the unlabelled data at all
        semi_supervised = False

        # If there is only 1 value, then return simple classifier
        unique_vals = np.unique(y_data_l)

        if len(unique_vals) == 1:
            clf = DummyClassifier(strategy="constant", constant=c_data_l[0])
        else:
            # Otherwise, train classifier model
            clf, semi_supervised = self.get_clf()

        # Split the labelled data for train/test
        x_train, x_test, y_train, y_test = train_test_split(x_data_l, y_data_l, test_size=0.15)

        # Combine with unlabelled data, if using a semi-supervised method
        if semi_supervised:
            x_train = np.concatenate([x_train, x_data_u])
            y_train = np.concatenate([y_train, y_data_u])

        # Train classifier
        clf.fit(x_train, y_train)

        # Retrieve predictive accuracy of classifier
        pred_acc = accuracy_score(y_test, clf.predict(x_test))

        return clf, pred_acc


    def train(self, x_data_l_paths, c_data_l, x_data_u_paths):
        '''
        Compute a dictionary with structure: layer --> concept --> model
        i.e., the dictionary returns which clf to use to predict a given concept, from a given layer
        '''

        self.model_ensemble = {}
        self.model_accuracies = {}

        # Compute activations for specified layers
        h_data_ls = compute_activation_per_layer(x_data_l_paths, self.layer_ids, self.model, aggregation_function=flatten_activations)
        h_data_us = compute_activation_per_layer(x_data_u_paths, self.layer_ids, self.model, aggregation_function=flatten_activations)

        for i, layer_id in enumerate(self.layer_ids):

            # Retrieve activations for next layer
            activations_l = h_data_ls[i]
            activations_u = h_data_us[i]

            output_layer = self.model.layers[layer_id]
            self.model_ensemble[output_layer] = []
            self.model_accuracies[output_layer] = []

            # Generate predictive models for every concept
            for c in range(self.n_concepts):
                clf, pred_acc = self._get_layer_concept_predictor_model(activations_l, c_data_l[:, c], activations_u)
                self.model_ensemble[output_layer].append(clf)
                self.model_accuracies[output_layer].append(pred_acc)

            print("Processed layer ", str(i + 1), " of ", str(len(self.layer_ids)))

        # Initialise concept predictors with models in the first layer
        # Consists of an array of size |concepts|, in which
        # The element arr[i] is the Layer Id of the layer to use when predicting that concept
        self.concept_predictor_layer_ids = [self.layer_ids[0] for _ in range(self.n_concepts)]

        # For every concept, identify the layer with the best clf predictive accuracy
        for c in range(self.n_concepts):
            max_acc = 0
            for i, layer_id in enumerate(self.layer_ids):
                layer = self.model.layers[layer_id]
                acc = self.model_accuracies[layer][c]

                if acc > max_acc:
                    max_acc = acc
                    self.concept_predictor_layer_ids[c] = layer_id


    def predict_concepts(self, x_data_paths):
        '''
        Given the same model and new x_data, predict their concept values
        '''

        n_samples = len(x_data_paths)
        n_epochs = math.ceil(n_samples / self.batch_size)
        concept_vals = np.zeros((n_samples, self.n_concepts), dtype=float)

        for c in range(self.n_concepts):
            # Retrieve clf corresponding to concept c
            layer_id = self.concept_predictor_layer_ids[c]
            output_layer = self.model.layers[layer_id]
            clf = self.model_ensemble[output_layer][c]

            # Compute activations for that layer
            output_layer = self.model.layers[layer_id]
            reduced_model = tf.keras.Model(inputs=self.model.inputs, outputs=[output_layer.output])
            hidden_features = []

            for i in range(n_epochs):
                start = self.batch_size * i
                end = min(n_samples, self.batch_size * (i + 1))
                paths = x_data_paths[start:end]
                x_data = load_batch(paths)
                batch_preds = reduced_model.predict(x_data)
                hidden_features.append(batch_preds)

                print("Processing epoch ", str(i + 1), " of ", str(n_epochs))

            hidden_features = np.concatenate(hidden_features)
            clf_data = flatten_activations(hidden_features)
            concept_vals[:, c] = clf.predict(clf_data)

        return concept_vals






