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
