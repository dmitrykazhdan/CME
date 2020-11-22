from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

'''
CtL : Concept-to-Label

Class for the transparent model representing a function from concepts to task labels.
Represents the decision-making process of a given black-box model, in the concept representation
'''


class CtLModel:

    def __init__(self, c_data, y_data, **params):

        # Create copy of passed-in parameters
        self.params = params

        if 'method' in self.params:
            method = self.params["method"]
        else:
            method = 'DT'

        # Retrieve the classifier type
        self.clf_type = method

        # Retrieve total number of concepts
        self.n_concepts = c_data.shape[1]

        # Retrieve the concept names
        if "concept_names" in self.params:
            self.concept_names = self.params["concept_names"]
        else:
            self.concept_names = ["Concept " + str(i) for i in range(self.n_concepts)]

        # Retrieve the class names
        if "class_names" in self.params:
            self.class_names = self.params["class_names"]
        else:
            n_classes = np.max(y_data) + 1
            self.class_names = [str(i) for i in range(n_classes)]

        # Train classifier for predicting the output labels from concept data
        self.clf = self._train_label_classifier(c_data, y_data, self.clf_type)


    def _train_label_classifier(self, c_data, y_data, method='DT'):

        if method == 'DT':
            clf = DecisionTreeClassifier(class_weight='balanced')
        elif method == 'LR':
            clf = LogisticRegression(max_iter=200, multi_class='auto', solver='lbfgs')
        elif method == 'LinearRegression':
            clf = LinearRegression()
        else:
            raise ValueError("Unrecognised model type...")

        clf.fit(c_data, y_data)

        return clf


    def predict(self, c_data):
        return self.clf.predict(c_data)


