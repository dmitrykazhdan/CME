
from CtLModel import CtLModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


def get_top_concepts(c_data, y_data):

    c_train, c_test, y_train, y_test = train_test_split(c_data, y_data, test_size=0.2)

    # Train model on ground-truth concept data
    params = {"method": "LR"}
    conc_model = CtLModel(c_train, y_train, **params)

    # Evaluate predictive accuracy
    y_test_extr = conc_model.predict(c_test)
    acc = accuracy_score(y_test, y_test_extr)
    print("Accuracy of q trained on ground-truth concepts, using all concepts: ", acc)

    # Retrieve LR coefficient magnitudes (array of size 200 classes by 112 concepts)
    coeffs = np.abs(conc_model.clf.coef_)
    norm_coeffs = coeffs / np.sum(coeffs)

    top_weights = np.sort(norm_coeffs, axis=0)[-1]

    top_inds = np.argsort(top_weights)
    print("Top concept indices: ", top_inds)
