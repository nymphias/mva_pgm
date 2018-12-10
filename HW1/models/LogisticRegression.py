import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression():

    name = 'Logistic'

    def __init__(self, tolerance=1e-4):
        self.w_ = None
        self.tolerance_ = tolerance
        self.f_history = None
        self.w_history = None

    def fit(self, X, y, plot=True):
        """ Fit the data (X, Y).

        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            Design matrix
        Y: (num_samples, ) np.array
            Output vector

        Note:
        -----
        Updates self.w_
        """

        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        # add constant
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # initialize weights
        #         self.w_ = np.random.rand(X.shape[1])
        self.w_ = np.zeros((X.shape[1],))

        self.w_history = []
        self.f_history = []

        lossp = 0
        lossp1 = np.inf

        while np.abs(lossp - lossp1) > self.tolerance_:
            lossp = lossp1
            self.update_w(X=X, y=y)
            lossp1 = self.loss(X, y)
            self.w_history.append(self.w_)
            self.f_history.append(lossp1)

        if plot:
            plt.plot(np.arange(len(self.f_history)), self.f_history)
            plt.title("Evolution of loss function across iterations")
            plt.show()

    def predict(self, X):
        """ Make binary predictions for data X.

        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            Design matrix

        Returns:
        -----
        y_pred: (num_samples, ) np.array
            Predictions (0 or 1)
        """
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y_pred = (X @ self.w_) > 0  # because X @ coef == log(P(Y=1|X=x)/P(Y=0|X=x))
        return y_pred

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def eta(self, X):
        return self.sigmoid(X @ self.w_)

    def gradient_loss(self, X, y):
        return X.T @ (y - self.eta(X))

    def hessian_loss(self, X):
        eta_value = self.eta(X)
        diag = np.diagflat(eta_value * (1 - eta_value))
        return -X.T @ diag @ X

    def update_w(self, X, y):
        grad = self.gradient_loss(X, y)
        hess = self.hessian_loss(X)
        self.w_ += (np.linalg.inv(-hess) @ grad)

    def loss(self, X, y):
        #     return y.T @ np.log(sigmoid(X @ w)) + (1-y).T @ np.log(sigmoid(-X @ w))
        return y.T @ (X @ self.w_) + np.sum(np.log(self.sigmoid(-X @ self.w_)))  # same thing...\

    def _compute_contour(self, X):

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X @ self.w_
