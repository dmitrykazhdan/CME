import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf



def labelled_unlabelled_split(x_data, c_data, y_data, n_labelled=100, n_unlabelled=200):
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


def flatten_activations(x_data):
    '''
    Flatten all axes except the first one
    '''

    if len(x_data.shape) > 2:
        n_samples = x_data.shape[0]
        shape = x_data.shape[1:]
        flattened = np.reshape(x_data, (n_samples, np.prod(shape)))
    else:
        flattened = x_data

    return flattened


def aggregate_activations(activations):
    if len(activations.shape) == 4:
        score_val = np.mean(activations, axis=(1, 2))
    elif len(activations.shape) == 3:
        score_val = np.mean(activations, axis=(1))
    elif len(activations.shape) == 2:
        score_val = activations
    else:
        raise ValueError("Unexpected data dimensionality")

    return score_val


def compute_activation_per_layer(x_data, layer_ids, model, batch_size=128,
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
        hidden_features = reduced_model.predict(x_data, batch_size=batch_size)
        flattened = aggregation_function(hidden_features)

        hidden_features_list.append(flattened)

    return hidden_features_list



def plot_summary(concept_model):

    # For decision trees, also save their plots
    if concept_model.clf_type == "DT":

        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        dt = concept_model.clf
        fig, ax = plt.subplots(figsize=(10, 10))  # whatever size you want

        plot_tree(dt,
                  ax=ax,
                  feature_names=concept_model.concept_names,
                  filled=True,
                  rounded=True,
                  proportion=True,
                  precision=2,
                  class_names=concept_model.class_names,
                  impurity=False)

        plt.show()

    elif concept_model.clf_type == 'LR':
        coeffs = concept_model.clf.coef_
        print("LR Coefficients: ", coeffs)




def compute_tsne_embedding(x_data, model, layer_ids, layer_names, batch_size=256):
    '''
    Compute tSNE latent space embeddings for specified layers of the DNN model
    '''

    h_l_list_agg = compute_activation_per_layer(x_data, layer_ids, model,
                                                batch_size,
                                                aggregation_function=aggregate_activations)
    h_l_embedding_list = []

    for i, h_l in enumerate(h_l_list_agg):
        h_embedded = TSNE(n_components=2, n_jobs=4).fit_transform(h_l)
        h_l_embedding_list.append(h_embedded)
        print(layer_names[i])
    return h_l_embedding_list


def visualise_hidden_space(x_data, c_data, c_names, layer_names, layer_ids, model, batch_size=256):

    # Compute tSNE embeddings
    h_l_embedding_list = compute_tsne_embedding(x_data, model, layer_ids, layer_names, batch_size)

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

