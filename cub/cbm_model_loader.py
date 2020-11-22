import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def load_model(model_path, use_gpu=True):

    if use_gpu:
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location='cpu')

    return model
