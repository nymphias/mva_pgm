import os
import pandas as pd
import numpy as np

class Dataset():
    """
    Contains the dataset and prediction results
    """
    def __init__(self, dataset):
        assert dataset in ['A', 'B', 'C']
        self.dataset = dataset
        trainFile = os.path.join('classification_data_HWK1',f'classification{dataset}.train')
        testFile = os.path.join('classification_data_HWK1',f'classification{dataset}.test')
        self.train = pd.read_csv(trainFile, sep='\t', header=None, names=['x1', 'x2', 'y'])
        self.test = pd.read_csv(testFile, sep='\t', header=None, names=['x1', 'x2', 'y'])
        
        self.X_train = self.train.values[:, :2]        
        self.y_train = self.train.values[:, 2]
        self.X_test = self.test.values[:, :2]        
        self.y_test = self.test.values[:, 2]
        
        self.y_train_pred = {'LDA': None, 'Logistic': None, 'LinReg': None, 'QDA': None}
        self.y_test_pred = {'LDA': None, 'Logistic': None, 'LinReg': None, 'QDA': None}
        self.error_train = {'LDA': None, 'Logistic': None, 'LinReg': None, 'QDA': None}
        self.error_test = {'LDA': None, 'Logistic': None, 'LinReg': None, 'QDA': None}
        
def misclassification_error(y_true, y_pred):
    """
    Fraction of the data misclassified
    """
    assert y_true.size == y_pred.size
    return np.sum(y_true != y_pred) / y_true.size

