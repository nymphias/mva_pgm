import numpy as np

class LDA():

    name = 'LDA'

    def __init__(self):
        self.mu_ = None
        self.sigma_ = None
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
        X: (number_samples, num_features) np.array
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
        sigmas_ = dict(zip(labels, [self._cov_matrix(X,y, label)  for label in labels] ))

        self.sigma_ = np.average(list(sigmas_.values()), axis = 0, weights = list(self.n_.values()))

        H = np.linalg.inv(self.sigma_)
        self.w_ = H @ (self.mu_[0] - self.mu_[1])
        self.C_ = np.log(-1+1/self.pi_) - 0.5*(self._compute_sqnorm(self.mu_[0], H) - self._compute_sqnorm(self.mu_[1], H))

        
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0],1)

        y_pred = (X @ self.w_  + self.C_) < 0
        return y_pred


    def _compute_contour(self, X):
        return X @ self.w_ + self.C_


        

   
