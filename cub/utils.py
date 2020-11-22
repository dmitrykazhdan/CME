import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import flatten_activations, aggregate_activations
from cub.cub_loader import load_batch


def labelled_unlabbeled_split_fpaths(x_train_path, c_train, n_labelled=100, n_unlabelled=None):
    '''
    Perform labelled/unlabelled split, whilst maintaining x_train represented as filepaths
    '''

    if n_unlabelled is None:
        # Labelled are first n_labelled points
        x_train_l, c_train_l = x_train_path[:n_labelled], c_train[:n_labelled]

        # Non-labelled are all the data-points
        x_train_u, c_train_u = x_train_path[n_labelled:], c_train[n_labelled:]
    else:
        # Otherwise, select randomly
        from random import randrange
        id_ub = len(x_train_path) - (n_labelled + n_unlabelled)
        id = randrange(id_ub)

        x_train_l, c_train_l = x_train_path[id : id+n_labelled], c_train[id : id+n_labelled]

        x_train_u, c_train_u = x_train_path[id+n_labelled : id+n_labelled+n_unlabelled], \
                               c_train[id+n_labelled : id+n_labelled+n_unlabelled]

    print('x_train_l length:', len(x_train_l))
    print('c_train_l shape:', c_train_l.shape)
    print('x_train_u length:', len(x_train_u))
    print('c_train_u shape:', c_train_u.shape)

    return x_train_l, c_train_l, x_train_u, c_train_u


def compute_activations_from_paths(model, x_data_paths, batch_size):

    batch_size = batch_size
    n_samples = len(x_data_paths)
    n_epochs = math.ceil(n_samples / batch_size)
    hidden_features = []

    for i in range(n_epochs):
        start = batch_size * i
        end = min(n_samples, batch_size * (i + 1))
        paths = x_data_paths[start:end]
        x_data = load_batch(paths)

        batch_hidden_features = model.predict(x_data)
        hidden_features.append(batch_hidden_features)

        print("Processing epoch ", str(i), " of ", str(n_epochs))

    hidden_features = np.concatenate(hidden_features)

    return hidden_features


def compute_activation_per_layer(x_data_paths, layer_ids, model, batch_size=128,
                                 aggregation_function=flatten_activations):
    '''
    Compute activations of x_data for 'layer_ids' layers
    For every layer, aggregate values using 'aggregation_function'

    Returns a list of size |layer_ids|, in which element L[i] is the activations
    computed from the model layer model.layers[layer_ids[i]]
    '''

    hidden_features_list = []

    for layer_id in layer_ids:
        # Compute and aggregate hidden activtions
        output_layer = model.layers[layer_id]
        reduced_model = tf.keras.Model(inputs=model.inputs, outputs=[output_layer.output])
        hidden_features = compute_activations_from_paths(reduced_model, x_data_paths, batch_size)

        flattened = aggregation_function(hidden_features)

        hidden_features_list.append(flattened)

    return hidden_features_list




def compute_tsne_embedding(x_data_paths, model, layer_ids, layer_names, batch_size=256):
    '''
    Compute tSNE latent space embeddings for specified layers of the DNN model
    '''

    h_l_list_agg = compute_activation_per_layer(x_data_paths, layer_ids, model,
                                                batch_size,
                                                aggregation_function=aggregate_activations)
    h_l_embedding_list = []

    for i, h_l in enumerate(h_l_list_agg):
        h_embedded = TSNE(n_components=2, n_jobs=4).fit_transform(h_l)
        h_l_embedding_list.append(h_embedded)
        print(layer_names[i])
    return h_l_embedding_list


def visualise_hidden_space(x_data_paths, c_data, c_names, layer_names, layer_ids, model, batch_size=256):

    # Compute tSNE embeddings
    h_l_embedding_list = compute_tsne_embedding(x_data_paths, model, layer_ids, layer_names, batch_size)

    # Create figure of size |n_concepts| * |n_layers|
    n_concepts = len(c_names)
    n_rows = n_concepts
    n_cols = len(h_l_embedding_list)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 4 * n_rows))

    # Plot the embeddings of every layer, highlighting concept values
    for i, h_2 in enumerate(h_l_embedding_list):
        for j in range(1, n_concepts):
            ax = axes[j-1, i]
            ax.scatter(h_2[:, 0], h_2[:, 1], c=c_data[:, j])
            ax.set_title(layer_names[i], fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_ylabel(c_names[j], fontsize=20)

    return fig

