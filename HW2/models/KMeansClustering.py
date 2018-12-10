import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt


class KMeansClustering(object):

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-5, init_mode = "random"):
        """

        :param n_clusters: number of clusters
        :param max_iter: maximum number of iterations
        :param tol: minimum difference of loss function between each iteration
        :param init_mode: initialization of clusters centers ("random" or "kmpp")

        Attributes
             X_ : data used (n_samples, n_features)
             cluster_centers_ : center of clusters (n_clusters_, n_features)
             initial_centers_ : center of clusters after initialization
             labels_ : labels of each point
             dist_ : distance matrix between each point (i-th row) and each cluster (j-th column)
             n_iter_ : number of iterations ran
             loss_history_ : evolution of loss function
        """
        self.n_clusters_ = n_clusters
        self.max_iter_ = max_iter
        self.tol_ = tol
        assert init_mode in ['random', 'kmpp']
        self.init_mode = init_mode

        self.X_ = None
        self.n_samples = None
        self.n_features = None
        self.cluster_centers_ = None
        self.initial_centers_ = None
        self.labels_ = None
        self.dist_ = None
        self.n_iter_ = 0
        self.loss_history_ = np.full(self.max_iter_, np.nan)

    def fit(self, X):
        self.X_ = X
        self.n_samples, self.n_features = self.X_.shape

        # random initialization using n_clusters points or KMeans++ initialization
        self._calculate_initial_centers()
        self.cluster_centers_ = self.initial_centers_

        lossp1 = np.inf
        lossp = 0

        while (self.n_iter_ < self.max_iter_) and (np.abs(lossp1 - lossp) > self.tol_):

            self._calculate_labels()
            self._calculate_centers()

            lossp = lossp1
            lossp1 = self._calculate_loss()
            self.loss_history_[self.n_iter_] = lossp1
            self.n_iter_ += 1

    def _calculate_initial_centers(self):
        """
        :return: Compute initial cluster centers using random assignment or KMeans++ algorithm
        """
        if self.init_mode == "random":
            self.initial_centers_ = self.X_[np.random.choice(self.n_samples, self.n_clusters_, replace=False)]

        if self.init_mode == "kmpp":
            self.initial_centers_ = np.zeros((self.n_clusters_, self.n_features))
            self.initial_centers_[0] = self.X_[np.random.choice(self.n_samples, 1)]
            for k in np.arange(1, self.n_clusters_):
                dist_ = distance.cdist(self.X_, self.initial_centers_[:k], metric='euclidean')
                labels_ = np.argmin(dist_, axis = 1)

                dist_ = dist_[np.arange(self.n_samples), labels_]
                p_ = dist_**2 / np.sum(dist_**2)
                self.initial_centers_[k] = self.X_[np.random.choice(self.n_samples, 1, p = p_)]

    def _calculate_labels(self):
        """
        :return: Assign each point to closest cluster
        """

        self.dist_ = distance.cdist(self.X_, self.cluster_centers_, metric='euclidean')
        self.labels_ = np.argmin(self.dist_, axis=1)

    def _calculate_centers(self):
        """
        :return:  Compute centroid as center of mass of the cluster
        """
        centers = pd.DataFrame(self.X_)
        centers.loc[:, 'label'] = self.labels_
        self.cluster_centers_ = centers.groupby('label').mean().values

    def _calculate_loss(self):
        """
        :return: Actual value of distorsion
        """
        return np.sum(self.dist_[np.arange(self.n_samples), self.labels_] ** 2)

    def plot_loss(self):
        """
        :return: Evolution of loss function
        """
        plt.plot(self.loss_history_)
        plt.title('Evolution of distorsion')
        plt.ylabel('Distorsion')
        plt.xlabel('Number of iterations')
        plt.show()

    def plot_clusters(self):
        """
        :return: Scatter plot of each the clustering results, initial and final cluster centers
        """
        plt.figure(figsize=(10, 10))
        plt.ylim(-11, 11)
        plt.xlim(-11, 11)
        for label in np.unique(self.labels_):
            plt.scatter(self.X_[self.labels_ == label, 0], self.X_[self.labels_ == label, 1], label=f'cluster {label}')
        plt.scatter(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1], c='b', label='Final Centers', marker='v')
        plt.scatter(self.initial_centers_[:, 0], self.initial_centers_[:, 1], c='r', label='Initial Centers', marker='^')
        plt.suptitle(f'[n_clusters={self.n_clusters_}] KMeans Clustering Training Data', size=14)
        plt.title(f'Distorsion = {self.loss_history_[self.n_iter_ - 1]}')
        plt.legend()
        plt.show();