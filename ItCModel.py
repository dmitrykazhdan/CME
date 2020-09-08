from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading

from utils import aggregate_activations, compute_activation_per_layer


class ItCModel(ABC):

    @abstractmethod
    def __init__(self, model, **params):
        '''
        :param model:    DNN model concepts are extracted from
        :param c_data_l: Concept data
        :param params:   Extra parameters
        '''

        # Create copy of passed-in parameters
        self.params = params

        # Specify name of concept extractor
        self.name = self.params.get("name", "ConceptExtractor")

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

        # Specify number of workers to use, for parallelism
        workers = os.cpu_count()
        workers = workers // 2
        workers = max(1, workers)
        self.workers = workers

        # Set classifier to use for concept value prediction
        self.clf_method = params.get("method", "LR")

        self.model = model


    def get_clf(self):
        '''
        Specify which classifier to use for concept value prediction
        :return:
        '''

        if self.clf_method == 'LR':
            clf = LogisticRegression(max_iter=200, n_jobs=self.workers)
        elif self.clf_method == 'LP':
            clf = LabelSpreading()
        else:
            raise ValueError("Non-implemented method")

        return clf


    @abstractmethod
    def predict_concepts(self, x_data):
        '''
        Predict concept values from input data
        :return:
        '''
        pass

    @abstractmethod
    def _extract_concepts(self, x_data_l, c_data_l, x_data_u, y_data_l):
        """
        Train the concept extraction model

        :param x_data_l: labelled input data
        :param c_data_l: labelled concept data
        :param x_data_u: unlabelled input data
        :return:
        """
        pass


    def compute_tsne_embedding(self, x_data):
        '''
        Compute tSNE latent space embeddings for specified layers of the DNN model
        :param x_data:
        :return:
        '''

        h_l_list_agg = compute_activation_per_layer(x_data, self.layer_ids, self.model,
                                                    self.batch_size,
                                                    aggregation_function=aggregate_activations)
        h_l_embedding_list = []

        for i, h_l in enumerate(h_l_list_agg):
            h_embedded = TSNE(n_components=2, n_jobs=4).fit_transform(h_l)
            h_l_embedding_list.append(h_embedded)
            print(self.layer_names[i])
        return h_l_embedding_list

    def visualise_hidden_space(self, x_data, c_data):

        # Compute tSNE embeddings
        h_l_embedding_list = self.compute_tsne_embedding(x_data)

        # Create figure of size |n_concepts| * |n_layers|
        n_concepts = len(self.concept_names)
        n_rows = n_concepts
        n_cols = len(h_l_embedding_list)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 4 * n_rows))

        # Plot the embeddings of every layer, highlighting concept values
        for i, h_2 in enumerate(h_l_embedding_list):
            for j in range(1, n_concepts):
                ax = axes[j-1, i]
                ax.scatter(h_2[:, 0], h_2[:, 1], c=c_data[:, j])
                ax.set_title(self.layer_names[i], fontsize=20)
                ax.set_xticks([])
                ax.set_yticks([])

                if i == 0:
                    ax.set_ylabel(self.concept_names[j], fontsize=20)

        return fig





