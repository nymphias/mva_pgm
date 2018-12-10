import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_context('poster')
# sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
from sklearn.cluster import KMeans


class my_GMM():

    def __init__(self, k, initialization="kmeans"):
        '''
        Attributes:

        k_: integer
            number of components
        initialization_: {"kmeans", "random"}
            type of initialization
        mu_: np.array
            array containing means
        Sigma_: np.array
            array cointaining covariance matrix
        cond_prob_: (n, K) np.array
            conditional probabilities for all data points
        labels_: (n, ) np.array
            labels for data points
        '''
        self.k_ = k
        self.initialization_ = initialization
        self.mu_ = None
        self.Sigma_ = None
        self.cond_prob_ = None
        self.labels_ = None
        self.p_k = None
        self.n_iter_ = 0

    def E_step(self, X):
        '''Compute the conditional probability matrix
        shape: (n, K)
        '''

        px_given_z_p_z = [
            multivariate_normal.pdf(X, mean=self.mu_[j, :], cov=self.Sigma_[j, :, :]) * self.p_k[j] for j in range(self.k_)
        ]
        px_z = np.stack(px_given_z_p_z, axis=-1)
        normalize_px_z = np.sum(px_z, axis=-1)
        return px_z / normalize_px_z[:, None]

    def M_step(self, X, cond_prob):
        '''Compute the expectation to check increment'''
        store_parameters = self.mu_
        self.p_k = np.mean(cond_prob, axis=0)

        self.mu_ = (cond_prob.T @ X) / np.sum(cond_prob, axis=0)[:, None]

        # self.Sigma_ = np.empty((, self.mu_.shape[1], self.mu_.shape[1]))
        div = np.sum(cond_prob, axis=0)
        for k in range(X.shape[1]):
            diff = X - self.mu_[k, :]

            a = diff.T * cond_prob[:, k]

            b = a @ diff

            self.Sigma_[k] = b / div[k]

        parameters = self.mu_
        stop_array = parameters - store_parameters
        criteria = np.linalg.norm(stop_array, ord=2)
        return criteria

    def fit(self, X, eps=1e-3):
        """ Find the parameters mu_ and Sigma_
        that better fit the data

        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix

        Returns:
        -----
        self
        """
        # initialization
        if self.initialization_ == 'kmeans':
            self.kmeans_initiliazation(X)
        elif self.initialization_ == 'random':
            self.random_initialization(X)

        convergence = False

        while not convergence:
            cond_prob = self.E_step(X)
            criteria = self.M_step(X, cond_prob)
            convergence = (criteria < eps)
            self. n_iter_ += 1

        self.labels_ = self.predict(X)

    def predict(self, X):
        """ Predict labels for X

        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix

        Returns:
        -----
        label assigment
        """
        cond_prob = self.E_step(X)
        return np.argmax(cond_prob, axis=1)

    def kmeans_initiliazation(self, X):
        import pandas as pd

        n_data, n_features = X.shape

        kmeans = KMeans(n_clusters=self.k_, max_iter=200)
        kmeans.fit(X)

        self.mu_ = kmeans.cluster_centers_  # center of clusters
        # p_k is the proportion of each cluster
        #         self.p_k = pd.Series(kmeans.labels_).value_counts(normalize=True, ascending=True).values
        self.p_k = pd.Series(kmeans.labels_).value_counts(normalize=True).sort_index().values
        self.Sigma_ = np.zeros((self.k_, n_features, n_features))
        # covariance inside each cluster
        for i in range(self.k_):
            cluster_elements = kmeans.labels_ == i
            self.Sigma_[i] = np.cov(X[cluster_elements].T)

    def random_initialization(self, X):
        from sklearn.datasets import make_spd_matrix

        n_data, n_features = X.shape

        self.mu_ = np.random.rand(self.k_, n_features)

        self.p_k = np.random.rand(self.k_)  # all positive
        self.p_k /= np.sum(self.p_k)  # normalize

        self.Sigma_ = np.zeros((self.k_, n_features, n_features))
        for i in range(self.k_):
            self.Sigma_[i] = make_spd_matrix(n_features)  # semi-pos-def
