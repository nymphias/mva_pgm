import numpy as np

class QDA():

    name = 'QDA'

    def __init__(self):
        self.mu_ = None
        self.sigmas_ = None
        self.H_ = None
        self.n_ = None
        self.nlabels = 0
        self.pi_ = 0
        self.w_ = None
        self.C_ = None

    def _get_label_obs(self, X, y,  label):
        return X[np.where(y==label)]

    def _cov_matrix(self, X, y, label):
        obs = self._get_label_obs(X, y, label)
        cov = (obs-self.mu_[label]).T @ (obs-self.mu_[label])/self.n_[label]
        return cov

    def _compute_sqnorm(self, x, dp_matrix):
        return x.T @ dp_matrix @ x

    def fit(self, X, y):
        """ Fit the data (X,Y).

        Parameters:
        -----------
        X: (number_sample, num_features) np.array
            Design matrix
        Y: (num_samples, ) np.array
            Output vector
        -----------
        """

        if len(X.shape) == 1:
            X = X.reshape(X.shape[0],1)

        labels = np.sort(np.unique(y))
        self.nlabels = len(labels)
        self.n_ = dict(zip(labels, [np.sum(y == label) for label in labels]))
        self.pi_ = np.mean(y)

        self.mu_ = dict(zip(labels, [np.mean(self._get_label_obs(X,y, label), axis = 0) for label in labels]))
        self.sigmas_ = dict(zip(labels, [self._cov_matrix(X,y, label)  for label in labels] ))

        self.H_ = dict(zip(labels, [np.linalg.inv(self.sigmas_[label]) for label in labels]))
        self.C_ = np.log(-1+1/self.pi_) - 0.5*(self._compute_sqnorm(self.mu_[0], self.H_[0]) - self._compute_sqnorm(self.mu_[1], self.H_[1]))

        
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0],1)


        quadr = -0.5 * np.diag( X @ (self.H_[0] - self.H_[1]) @ X.T ) 
        lin = 2 * X @ (self.H_[0] @ self.mu_[0] - self.H_[1] @ self.mu_[1])
        
        y_pred = (quadr + lin + self.C_) < 0
        return y_pred


    def _compute_contour(self, X):
        
        quadr = -0.5 * np.diag( X @ (self.H_[0] - self.H_[1]) @ X.T ) 
        lin = 2 * X @ (self.H_[0] @ self.mu_[0] - self.H_[1] @ self.mu_[1])

        return quadr + lin + self.C_
        
