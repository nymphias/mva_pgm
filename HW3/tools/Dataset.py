import os
import pandas as pd
import numpy as np


class Dataset():

    """
    Contains the dataset and prediction results
    """

    def __init__(self):
        trainFile = os.path.join('classification_data_HWK2', 'EMGaussian.data')
        testFile = os.path.join('classification_data_HWK2', 'EMGaussian.test')
        self.train = pd.read_csv(trainFile, sep=' ', header=None, names=['x1', 'x2'])
        self.test = pd.read_csv(testFile, sep=' ', header=None, names=['x1', 'x2'])

        self.X_train = self.train.values
        self.X_test = self.test.values
