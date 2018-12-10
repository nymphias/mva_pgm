import numpy as np

class LinReg():
    # Class for linear regression solving least-squares:
    name = 'Least-squares'

    def __init__(self,):
        self.coef_ = None
        
    def fit(self, X, y):
        """ Fit the data (X, y).
    
        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            Design matrix
        y: (num_sampes, ) np.array
            Output vector
        
        Note:
        -----
        Updates self.coef_
        """
        # Create a (num_samples, num_features+1) np.array X_aug whose first column 
        # is a column of all ones (so as to fit an intercept).
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0],1)
        X_aug = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        # Update self.coef_
        self.coef_ = (np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T) @ y


    def predict(self, X):
        """ Make predictions for data X.
    
        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            Design matrix
        
        Returns:
        -----
        y_pred: (num_samples, ) np.array
            Predictions
        """
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0],1)
        X_aug = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        return ((X_aug @ self.coef_) > 0.5).astype(int)
    
    def _compute_contour(self, X):
        
        X_aug = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X_aug @ self.coef_ - 0.5
