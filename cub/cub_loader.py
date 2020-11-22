import pickle
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os

from cub.bottleneck_model import BottleneckModel

'''
See the Google Collab provided in https://github.com/yewsiang/ConceptBottleneck/tree/master/CUB
In order to download pre-processed data, and pre-trained models
'''


def load_img(img_path, is_train=False):
    '''
    Function for loading CUB images
    :param img_path: path to the image file
    :param is_train: whether this is an image from the training set, or not
    Note: function copied from 'https://github.com/yewsiang/ConceptBottleneck/tree/master/CUB'
    '''

    if is_train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(299),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])

    x_data = Image.open(img_path).convert('RGB')
    x_data = transform(x_data).unsqueeze(0)
    x_data = x_data.numpy()

    return x_data



def load_batch(img_paths_and_train_flag):
    '''
    Load a batch of images using load_img()
    :param img_paths_and_train_flag: list of pairs of (img_path, train_flag)
    '''

    x_data = []

    for img_path, is_train in img_paths_and_train_flag:
        x_data.append(load_img(img_path, is_train)[0])

    x_data = np.array(x_data)

    return x_data



def load_data_from_pkl(metadata_path, img_dir_path, is_train=False, n_samples_per_cls=None):
    '''
    :param metadata_path: Path to metadata .pkl files holding sample information
    :param img_dir_path: Path to CUB_200_2011 directory
    :param is_train: whether this is a training, or inference dataset
    :param n_samples_per_cls: how many samples to extract for every class. 'None' - if extracting all of them
    '''

    data = pickle.load(open(metadata_path, 'rb'))

    if n_samples_per_cls is None:
        n_samples_per_cls = len(data)

    c_datas = []
    y_datas = []
    x_paths = []
    n_classes = 200

    # Create dictionary, holding the number of samples collected for each class
    sample_counts = {}
    for i in range(n_classes):
        sample_counts[i] = 0

    for i, d in enumerate(data):

        # Create local path to image
        img_path = d["img_path"]
        idx = img_path.split('/').index('CUB_200_2011')
        img_path = os.path.join(img_dir_path, '/'.join(img_path.split('/')[idx:]))

        # Extract class and concept labels
        y_data = d['class_label']
        c_data = np.array(d['attribute_label'])

        # Add new sample information, provided the maximum n_samples for that class is not reached
        if sample_counts[y_data] < n_samples_per_cls:
            sample_counts[y_data] += 1
            x_paths.append((img_path, is_train))
            y_datas.append(y_data)
            c_datas.append(c_data)

    y_datas = np.array(y_datas)
    c_datas = np.array(c_datas)

    return x_paths, y_datas, c_datas





def load_cub_data(model_path, metadata_dir_path, img_dir_path, use_gpu=True, n_samples_per_cls=None):
    '''
    :param model_path:          path to saved CUB model .pth file
    :param metadata_dir_path:   path to the "class_attr_data_10" directory
    :param img_dir_path:        path to the "CUB_200_2011" (most outer one, containing the other "CUB_200_2011" one inside)
    :param n_samples_per_cls:   how many samples to extract for every class. 'None' - if extracting all of them
    :param use_gpu:             whether using a gpu or not
    :return:
    '''

    # Load original saved model
    model_params = {"use_gpu" : use_gpu}
    btl_model = BottleneckModel(model_path, **model_params)
    print("Model loaded")

    # Load training data
    pkl_filepath = os.path.join(metadata_dir_path, "train.pkl")
    is_train = True
    x_train_paths, y_train, c_train = load_data_from_pkl(pkl_filepath, img_dir_path, is_train, n_samples_per_cls)

    # Load test data
    pkl_filepath = os.path.join(metadata_dir_path, "test.pkl")
    is_train = False
    x_test_paths, y_test, c_test = load_data_from_pkl(pkl_filepath, img_dir_path, is_train)

    c_names = [str(i) for i in range(c_train.shape[1])]

    return btl_model, x_train_paths, y_train, x_test_paths, y_test, c_train, c_test, c_names
