import torch
import numpy as np
import math

from cub.cbm_model_loader import load_model
from cub.cub_loader import load_batch

'''
Note: this is NOT a standard Pytorch model, but rather the implementation used in https://github.com/yewsiang/ConceptBottleneck
See "models", and "template_model" for more
This is a wrapper for the StandardNB model type

'''


class BottleneckModel:

    def __init__(self, model_path, **params):

        # Create copy of passed-in parameters
        self.params = params

        # Create copy of model
        self.model = load_model(model_path, self.use_gpu)

        # Whether running on GPU, or CPU
        self.use_gpu = params.get("use_gpu")

        if "n_layers" in params:
            self.n_layers = params["n_layers"]
        else:
            self.n_layers = 20

        self.layer_names = [        "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "max_pool2d_1",
                                    "Conv2d_3b_1x1", "Conv2d_4a_3x3", "max_pool2d_2", "Mixed_5b",
                                    "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b",
                                    "Mixed_6c", "Mixed_6d", "Mixed_6e", "Mixed_7a",
                                    "Mixed_7b", "Mixed_7c", "adaptive_avg_pool2d", "linear"]

        self.model.out_layer = -1
        self.model.training = False


    def predict_batched(self, x_data_paths):

        if self.use_gpu:
            batch_size = 512
        else:
            batch_size = 64

        n_samples = len(x_data_paths)
        n_epochs = math.ceil(n_samples / batch_size)
        preds = []

        for i in range(n_epochs):
            start = batch_size*i
            end = min(n_samples, batch_size*(i+1))
            paths = x_data_paths[start:end]
            x_data = load_batch(paths)

            batch_preds = self.predict(x_data)
            preds.append(batch_preds)

            print("Processing epoch ", str(i+1), " of ", str(n_epochs))

        preds = np.concatenate(preds)

        return preds


    def predict(self, x_data):

        with torch.no_grad():

            self.model.eval()

            x_data_t = torch.from_numpy(x_data)

            if self.use_gpu:
                input_var = torch.autograd.Variable(x_data_t).cuda()
            else:
                input_var = torch.autograd.Variable(x_data_t).to("cpu")

            y_pred = self.model(input_var)[0]

            if self.use_gpu:
                y_pred = y_pred.cpu().numpy()
            else:
                y_pred = y_pred.numpy()

            y_pred = np.argmax(y_pred, axis=-1)
            return y_pred


    def get_layer_activations(self, x_data, layer_id):

        if layer_id >= 0:
            out_layer = -(self.n_layers - layer_id)
        else:
            out_layer = layer_id

        with torch.no_grad():

            self.model.eval()

            x_data_t = torch.from_numpy(x_data)

            # Assumes you need to set a property variable
            if self.use_gpu:
                input_var = torch.autograd.Variable(x_data_t).cuda()
            else:
                input_var = torch.autograd.Variable(x_data_t).to("cpu")

            self.model.out_layer = out_layer

            if out_layer == -1:
                activations = self.model(input_var)[0]
            else:
                activations = self.model(input_var)

            if self.use_gpu:
                activations = activations.cpu().numpy()
            else:
                activations = activations.numpy()

        return activations
