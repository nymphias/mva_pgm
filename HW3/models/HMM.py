import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib.patches import Ellipse
from scipy.stats import chi2


class HiddenMarkovModel(object):

    """
    Hidden Markov Model implementation
    """

    def __init__(self, pi, cluster_centers, cluster_cov, A, max_iter=300, tol=1e-5, ellipse_level=0.9):
        assert A.shape[0] == A.shape[1]
        self.params_init = {"pi": pi, "cluster_centers": cluster_centers, "cluster_cov": cluster_cov, "A": A}
        self.pi = pi
        self.cluster_centers = cluster_centers
        self.cluster_cov = cluster_cov
        self.A_ = A
        self.K_ = A.shape[0]
        self.max_iter_ = max_iter
        self.tol_ = tol
        self.n_iter_ = 0
        self.alpha = None
        self.beta = None
        self.alpha_log = None
        self.beta_log = None
        self.gamma = None
        self.ksi = None

        self.X_ = None
        self.X_test_ = None
        self.labels_ = None
        self.T_ = None
        self.n_features = None
        self.loss_history_ = np.full(self.max_iter_, np.nan)
        self.test_history_ = np.full(self.max_iter_, np.nan)
        self.nstd_ = np.sqrt(chi2.ppf(ellipse_level, 2))

    def fit(self, X):
        self.X_ = X
        self.T_, self.n_features = self.X_.shape

        lossp1, lossp = np.inf, 0

        while (self.n_iter_ < self.max_iter_) and (np.abs(lossp1 - lossp) > self.tol_):
            print(self.n_iter_)
            # computes E step and M step
            self._calculate_e_step()
            self._calculate_m_step()

            lossp = lossp1
            lossp1 = self._calculate_loss()  # can be calculated before M step
            self.loss_history_[self.n_iter_] = lossp1

            self.n_iter_ += 1
        self._calculate_labels()

    def _calculate_e_step(self):
        # update alpha_log and beta_log using new parameters of cluster_cov / cluster_centers / A / pi
        self.log_alpha_recursion(self.X_)
        self.log_beta_recursion(self.X_)

        # update gamma
        self.gamma = np.zeros((self.T_, self.K_))
        for t in range(self.T_):
            self.gamma[t] = np.exp(self.smoothing_log_vect(t))
        # self.gamma = self.smoothing_log_entire()  # approximately same results

        # update ksi
        self.ksi = np.zeros((self.T_ - 1, self.K_, self.K_))  # not until T-1 but T-2
        for t in range(self.T_ - 1):
            self.ksi[t] = self.compute_joint_proba_log(y=self.X_, t=t)

    def _calculate_m_step(self):
        # update pi
        self.pi = self.gamma[0]

        # update A
        self.A_ = np.sum(self.ksi, axis=0)
        self.A_ /= np.sum(self.gamma, axis=0).reshape(-1, 1)
        # approximately same results
        # self.A_ = np.sum(self.ksi, axis=0) / np.sum(self.ksi, axis=(0, 2)).reshape(-1, 1)

        gamma_sum = np.sum(self.gamma, axis=0)

        # update cluster centers
        self.cluster_centers = np.zeros_like(self.cluster_centers)
        for cluster in range(self.K_):
            self.cluster_centers[cluster] = np.sum(self.X_ * self.gamma[:, cluster].reshape(-1, 1), axis=0)
            self.cluster_centers[cluster] /= gamma_sum[cluster]

        # update cluster covariance
        self.cluster_cov = np.zeros_like(self.cluster_cov)
        for cluster in range(self.K_):
            arr = (self.X_ - self.cluster_centers[cluster]) * np.sqrt(self.gamma[:, cluster].reshape(-1, 1))
            self.cluster_cov[cluster] = arr.T @ arr
            self.cluster_cov[cluster] /= gamma_sum[cluster]

    def _calculate_loss(self, X=None):
        """
        Value of loglikelihood
        """
        if X is None:
            return HiddenMarkovModel.log_sum(self.alpha_log[0] + self.beta_log[0])
        else:
            alpha_log = self.log_alpha_recursion(X, output=True)
            beta_log = self.log_beta_recursion(X, output=True)
            return HiddenMarkovModel.log_sum(alpha_log[0] + beta_log[0])

    def _calculate_labels(self, X=None):
        """
        Implement the viterbi decoding algorithm for finding the most likely state.
        """
        output = True
        if X is None:
            X = self.X_
            output = False
        viterbi = np.zeros((self.T_, self.K_))
        backpointer = np.zeros((self.T_, self.K_)).astype(int)

        # initializing
        # backpointer[0] = 0
        viterbi[0] = np.log(self.pi * self.emission_prob_vect(X[0]))

        # recursion
        for t in range(1, self.T_):
            b_obs = np.log(self.emission_prob_vect(X[t]))
            viterbi[t] = np.max(np.log(self.A_) + viterbi[t - 1] + b_obs.reshape(-1, 1), axis=1)
            backpointer[t] = np.argmax(np.log(self.A_) + viterbi[t - 1] + b_obs.reshape(-1, 1), axis=1)
            # backpointer[t] = np.argmax(np.log(self.A_) + viterbi[t - 1], axis=1)

        # decoding
        backtrace = np.zeros(self.T_).astype(int)
        backtrace[-1] = np.argmax(viterbi[-1])
        for t in range(self.T_ - 2, -1, -1):
            backtrace[t] = backpointer[t + 1, backtrace[t + 1]]

        if not output:
            self.labels_ = backtrace
        else:
            return backtrace

    def emission_probability(self, y_value, z):
        return ss.multivariate_normal.pdf(y_value, self.cluster_centers[z], self.cluster_cov[z])

    def emission_prob_vect(self, y_value):
        """
        Calculate the gaussian emission probability of P(y_value | cluster) for all clusters
        """
        return np.array([self.emission_probability(y_value, z) for z in range(self.K_)])

    def log_alpha_recursion(self, y, output=False):
        """
        Vectorized alpha recursion algorithm using log for minimizing rounding errors
        """
        T = len(y)
        alpha_log = np.zeros((T, self.K_))

        # initialization
        alpha_log[0] = np.log(self.emission_prob_vect(y[0]) * self.pi)

        # propagation
        for t in range(1, T):
            # update alpha_log t : sum is over columns of A for each term alpha_log[t]
            alpha_log[t] = np.log(self.emission_prob_vect(y[t]))
            arr = np.log(self.A_) + alpha_log[t - 1]
            alpha_log[t] += HiddenMarkovModel.log_sum(arr)

        if not output:
            self.alpha_log = alpha_log
        else:
            return alpha_log

    def log_beta_recursion(self, y, output=False):
        """
        Vectorized beta recursion algorithm using log for minimizing rounding errors
        """
        T = len(y)
        beta_log = np.zeros((T, self.K_))

        # propagation
        for t in np.arange(T - 2, -1, -1):
            # update beta t : sum is over lines of A for each term of beta[t]
            arr = np.log(self.emission_prob_vect(y[t + 1])) + beta_log[t + 1] + np.log(self.A_).T
            beta_log[t] = HiddenMarkovModel.log_sum(arr)

        if not output:
            self.beta_log = beta_log
        else:
            return beta_log

    def smoothing_log_vect(self, t):
        """
        Calculate from alpha_log and beta_log for all z_t at time t:
        p(q_t | u_1, \dots, u_T)
        = \alpha_t(z_t) * \beta_t(z_t) / (\sum_{i=1}{K} \alpha_t(i) * \beta_t(i))
        """
        log_p_y = HiddenMarkovModel.log_sum(self.alpha_log[t] + self.beta_log[t])
        return self.alpha_log[t] + self.beta_log[t] - log_p_y

    def compute_joint_proba_log(self, y, t):
        # i-th row j-th col corresponds to log(p(z_t1=i, z_t=j | y))
        p_log = np.log(self.A_)
        # add same value for each row, which corresponds to a fixed z_t1
        p_log += np.log(self.emission_prob_vect(y[t + 1]).reshape(-1, 1))
        p_log += self.beta_log[t + 1].reshape(-1, 1)
        p_log += self.alpha_log[t]
        p_log -= HiddenMarkovModel.log_sum(self.alpha_log[t] + self.beta_log[t])
        return np.exp(p_log)
        
    @staticmethod
    def log_sum(log_arr):
        """
        Summing over axis 1 of log_array in order to prevent numerical errors
        Equals to np.sum(np.exp(log_arr))
        """
        if len(log_arr.shape) == 2:
            log_arr_max = np.max(log_arr, axis=1)
            return np.log(np.sum(np.exp(log_arr - log_arr_max.reshape(-1, 1)), axis=1)) + log_arr_max
        elif len(log_arr.shape) == 1:
            log_arr_max = np.max(log_arr)
            return np.log(np.sum(np.exp(log_arr - log_arr_max))) + log_arr_max

    def plot_clusters(self):
        """
        :return: Scatter plot of each the clustering results, initial and final cluster centers
        """
        plt.figure(figsize=(10, 10))
        plt.ylim(-11, 11)
        plt.xlim(-11, 11)

        for label in np.unique(self.labels_):
            plt.scatter(self.X_[self.labels_ == label, 0], self.X_[self.labels_ == label, 1], label=f'cluster {label}')

        # add centers
        plt.scatter(
            self.cluster_centers[:, 0],
            self.cluster_centers[:, 1],
            c='b', label='Final Centers', marker='v', s=100)
        plt.scatter(
            self.params_init['cluster_centers'][:, 0],
            self.params_init['cluster_centers'][:, 1],
            c='r', label='Initial Centers', marker='^', s=100
        )

        for k in range(self.K_):
            self.plot_cov_ellipse(self.cluster_cov[k], self.cluster_centers[k], nstd=self.nstd_, alpha=0.1)

        plt.suptitle(f'HMM [n_clusters={self.K_}] Training Data', size=14)
        plt.title(f'-LogLikelihood = {self.loss_history_[self.n_iter_ - 1]}')
        plt.legend()
        plt.show();

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

    def test_results(self, X_test):
        labels_test = self._calculate_labels(X_test)

        loss_test = self._calculate_loss(X_test)
        plt.figure(figsize=(10, 10))
        plt.ylim(-11, 11)
        plt.xlim(-11, 11)

        for label in np.unique(labels_test):
            plt.scatter(X_test[labels_test == label, 0], X_test[labels_test == label, 1], label=f'cluster {label}')

        # # add centers
        # plt.scatter(
        #     self.cluster_centers[:, 0],
        #     self.cluster_centers[:, 1],
        #     c='b', label='Final Centers', marker='v', s=100)
        # plt.scatter(
        #     self.params_init['cluster_centers'][:, 0],
        #     self.params_init['cluster_centers'][:, 1],
        #     c='r', label='Initial Centers', marker='^', s=100
        # )

        for k in range(self.K_):
            self.plot_cov_ellipse(self.cluster_cov[k], self.cluster_centers[k], nstd=self.nstd_, alpha=0.1)

        plt.suptitle(f'HMM [n_clusters={self.K_}] Testing Data', size=14)
        plt.title(f'-LogLikelihood = {loss_test}')
        plt.legend()
        plt.show();

        print(f"Test loglikelihood = {loss_test}, Test loglikelihood per point = {loss_test / len(X_test)}")