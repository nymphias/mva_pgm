import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.stats as ss
import models.KMeansClustering as km
from matplotlib.patches import Ellipse
from scipy.stats import chi2


class EMGaussian(object):

    def __init__(self, n_clusters=4, max_iter=300, tol=1e-5, init_mode="random", ellipse_level=0.9):
        """

        :param n_clusters: number of Gaussian clusters
        :param max_iter: maximum number of iterations
        :param tol: minimum difference of loss function between each iteration

        Attributes
             X_ : data used (n_samples, n_features)
             cluster_centers_ : center of gaussian clusters (n_clusters_, n_features)
             cluster_cov_ : covariance matrix of gaussian clusters (n_clusters_, n_features, n_features)
             tau_: probability that datapoint i belongs to cluster j conditioned wrt its position (n_clusters, n_samples)
             pi_: probability of drawing of each cluster following a multinomial distribution (n_clusters)

             initial_centers_ : center of clusters after initialization
             initial_cov_ : (empirical) covariance matrix of clusters after initialization

             labels_ : labels of each point
             dist_ : distance matrix between each point (i-th row) and each cluster (j-th column)
             n_iter_ : number of iterations ran
             loss_history_ : evolution of loss function
        """
        self.n_clusters_ = n_clusters
        self.max_iter_ = max_iter
        self.tol_ = tol
        self.init_mode = init_mode
        self.nstd_ = np.sqrt(chi2.ppf(ellipse_level, 2))

        self.X_ = None
        self.n_samples = None
        self.n_features = None
        self.cluster_centers_ = None
        self.cluster_cov_ = None
        self.initial_centers_ = None
        self.initial_cov_ = None
        self.tau_ = None
        self.pi_ = None

        self.labels_ = None
        self.dist_ = None
        self.n_iter_ = 0
        self.loss_history_ = np.full(self.max_iter_, np.nan)

    def _calculate_cluster_init(self):
        """
        :return: Initiate clusters with KMeans clustering
        """

        model = km.KMeansClustering(n_clusters = self.n_clusters_, max_iter = 300, tol=1e-5, init_mode = self.init_mode)
        model.fit(self.X_)
        model.plot_clusters()

        self.initial_centers_ = model.cluster_centers_
        self.cluster_centers_ = self.initial_centers_

        centers = pd.DataFrame(self.X_)
        centers.loc[:, 'label'] = model.labels_
        self.initial_cov_ = centers.groupby('label').cov().values.reshape((self.n_clusters_, self.n_features, self.n_features))
        self.cluster_cov_ = self.initial_cov_

        self.labels_ = model.labels_
        self.dist_ = model.dist_
        self.pi_ = pd.Series(self.labels_).value_counts(normalize=True).sort_index().values


    def _calculate_EStep(self):
        """
        :return: Computes tau_: condtional probability of datapoint i belonging to class j / (n_clusters, n_samples)
        """

        pdf = np.asarray([ss.multivariate_normal.pdf(self.X_, self.cluster_centers_[k], self.cluster_cov_[k]) for k in range(self.n_clusters_)])
        buf = self.pi_.reshape((len(self.pi_), 1)) * pdf

        self.tau_ = buf / buf.sum(axis=0)

    def _calculate_MStep(self):
        """
        :return: Computes and updates multinomial parameters and each Gaussian parameters (mean and covariance matrix)
        """

        self.pi_ = self.tau_.sum(axis = 1) / self.n_samples
        self.cluster_centers_ = (self.tau_ @ self.X_)/(self.tau_.sum(axis = 1).reshape(self.n_clusters_, 1))

        for k in range(self.n_clusters_):
            self.cluster_cov_[k] = (self.X_ - self.cluster_centers_[k]).T @ (self.tau_[k].reshape(len(self.tau_[k]), 1) * (self.X_ - self.cluster_centers_[k])) / self.tau_[k].sum()

    def fit(self, X):
        self.X_ = X
        self.n_samples, self.n_features = self.X_.shape

        # initialization using kmeans method
        self._calculate_cluster_init()

        lossp1 = np.inf
        lossp = 0

        while (self.n_iter_ < self.max_iter_) and (np.abs(lossp1 - lossp) > self.tol_):

            # computes E step and M step
            self._calculate_EStep()
            self._calculate_MStep()

            # we keep track of the prediction to compute the loss history
            self._calculate_labels()

            lossp = lossp1
            lossp1 = self._calculate_loss()
            self.loss_history_[self.n_iter_] = lossp1
            self.n_iter_ += 1

    def predict(self, X):
        """
        :return: Computes tau_: condtional probability of datapoint i belonging to class j / (n_clusters, n_samples)
        """

        pdf = np.asarray([
            ss.multivariate_normal.pdf(X, self.cluster_centers_[k], self.cluster_cov_[k])
            for k in range(self.n_clusters_)
        ])
        buf = self.pi_.reshape((len(self.pi_), 1)) * pdf
        tau = buf / buf.sum(axis=0)
        labels = np.argmax(tau, axis=0)
        return labels

    def _calculate_labels(self):
        """
        :return: Assign each point to closest cluster
        """
        self.dist_ = distance.cdist(self.X_, self.cluster_centers_, metric='euclidean')
        self.labels_ = np.argmax(self.tau_, axis=0)

    def _compute_sqnorm(self, x, dp_matrix):
        x = x.T
        return x.T @ dp_matrix @ x

    def _calculate_loss(self):
        """
        :return: Actual value of negative likelihood
        """
        loss = 0
        for k in range(self.n_clusters_):
            sigma_inv = np.linalg.inv(self.cluster_cov_[k])
            for i in range(self.n_samples):
                loss1 = self.tau_[k,i]*np.log(self.pi_[k])

                ecart = (self.X_[i]-self.cluster_centers_[k])
                loss2 = - self.n_clusters_/2 * np.log(2*np.pi) + 0.5*np.linalg.det(sigma_inv) - 0.5 * ecart.T @ sigma_inv @ ecart

                loss = loss + loss1 + self.tau_[k, i]*loss2

        return -loss

    def plot_loss(self):
        """
        :return: Evolution of loss function
        """
        plt.plot(self.loss_history_)
        plt.title('Evolution of negative loglikelihood')
        plt.ylabel('-loglikelihood')
        plt.xlabel('Number of iterations')
        plt.show()

    def plot_clusters(self):
        """
        :return: Scatter plot of each the clustering results, initial and final cluster centers
        """
        plt.scatter(self.X_[:, 0], self.X_[:, 1], c=self.labels_, cmap=plt.get_cmap('viridis'))
        plt.scatter(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1], c='b', label='Final Centers', marker='v')
        plt.scatter(self.initial_centers_[:, 0], self.initial_centers_[:, 1], c='r', label='Initial Centers', marker='^')

        for k in range(self.n_clusters_):
            self.plot_cov_ellipse(self.cluster_cov_[k], self.cluster_centers_[k], nstd=self.nstd_, alpha=0.1)

        plt.suptitle(f'[n_clusters={self.n_clusters_}] GMM', size=14)
        plt.title(f'-LogLikelihood = {self.loss_history_[self.n_iter_ - 1]}')
        plt.legend()
        plt.show();

    def plot_predicted(self, X):
        labels = self.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.get_cmap('viridis'))
        plt.title("Results on test data - GMM")
        plt.show()


    @staticmethod
    def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Function from :
        https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
        Theory :
        http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/

        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the
        ellipse patch artist.

        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        """

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        if ax is None:
            ax = plt.gca()

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        ax.add_artist(ellip)
        return ellip