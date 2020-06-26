
import numpy as np


def gaussian_pdf(x, mu, sigma):
    # x vector should be in column, but numpy treat 1-d array as vector so can save the extra step
    n = len(x)
    return (2*np.pi)**(-n/2) * (np.linalg.det(sigma) ** (-1/2) ) * np.exp(-1/2*((x-mu).T@np.linalg.inv(sigma)@(x-mu)))


def component_pdfs(x, mus, sigmas):
    """
    The component pdf p_k(x; mu_k, sigma_k)
        :param mus: Kxn array (n be the dimension of x vector and K be the num of components), mean of k n-dim gaussian distr.
        :param sigmas: Knxn array covariance of k n-dim gaussian distr.
        :return: K-dim array contain probability of each component pdf
    """
    n_components = mus.shape[0]
    return np.array([gaussian_pdf(x, mus[k,:], sigmas[k, :, :]) for k in range(n_components)])


def e_step(X, taus, mus, sigmas):
    """
        E step of the EM algorithm, caculates the posterior T_{k, i}=P(z_i=k|x_i)
        it returns T_{k,i} in the form of a KxN T matrix where the i, j th component is the T_{k, i}
        :param X: Nxn matrix represents N number of n-dim data points
        :param taus: K-dim vector, the weight of each component, or the prior of the hidden stats z
        :param mus: Kxn matrix (n be the dimension of x vector and K be the num of components), mean of k n-dim gaussian distr.
        :param sigmas: Kxnxn matrix covariance of k n-dim gaussian distr.
    """
    K, N = mus.shape[0], X.shape[0] # dimensions, K: num of hidden component, N: number of data points
    get_component_prob = lambda x: component_pdfs(x, mus, sigmas)
    T = np.apply_along_axis(arr=X, func1d=get_component_prob, axis=1) # gaussian component probabilities in row format (NxK)
    taus_rep = np.tile(taus, reps=(N, 1)) # repeat tau along N-axis so elementwise product can work

    norm_const = np.sum(T*taus_rep, axis=1) # the normalisation factor \sum_{k=1}^K p_k * tau_k
    norm_const_rep = np.tile(norm_const, reps=(K, 1)).T # repeat normalisation constant along K-axis

    T = T*taus_rep/norm_const_rep # calculate the posterior 
    return T.T #return the transposed matrix so that the index is matched

def m_step(X, T):
    """
        M step of the EM algorithm, caculates the MLE of taus, mus and sigmas
        :param X: Nxn matrix, the dataset, N number of n-dim data points
        :param T: KxN matrix, the T matrix is the posterior matrix where the i, j th component is the T_{k, i}

    """
    N, n = X.shape #  N: number of data points, n: dimension of the data point
    K = T.shape[0] # num of hidden component
    T_sum = np.sum(T, axis=1) # caculate the common term sum of T_{k, i} over all i, this is a k-dim vector

    taus = T_sum / N # average over i  for T_{k, i} gives MLE for all tau_k

    T_sum_rep = np.tile(T_sum, reps=(n, 1)).T # repeat T_sum n times in column
    mus = T@X/T_sum_rep # T@X gives a Kxn matrix with it's k, i th component be \sum_{i=1}^NT_{k, i}x_i then each row is divided by T_sum, gives MLE for all mu_k

    def get_sigma(X, muk, Tk):
        """
            function that calculate the covariance of the k-th component
            :param muk: n-dim vector, the k-th component's mean
            :param Tk: N-dim vector, the k-th component posterior of hidden state z_i, for each x_i
        """
        X_centred = X - muk
        X_weighted = X_centred * np.tile(Tk, reps=(X.shape[1],1)).T # repeat Tk in N-direction to match X's shape and weigh it
        return X_weighted.T@X_centred/np.sum(Tk) # weighted and centred are exchangable: we only need to weigh it by T_k once

    sigmas = np.array([get_sigma(X, mus[k, :], T[k, :]) for k in range(K)])
    return taus, mus, sigmas


class GaussianMixture():
    def __init__(self, n_hidden=None):
        self.n_hidden = n_hidden or 2

    def fit(self, X):
        """
            :param X: Nxn matrix (N be the num of data points and n be the dimension of each data point)
        """
        n_hid = self.n_hidden
        n_var = X.shape[1]
        mus = np.random.randn(n_hid, n_var) # means of k components
        sigmas = np.random.randn(n_hid, n_var, n_var) # sigmas of k components
        taus = np.random.randn(n_hid) # component multipliers, or hidden states priors

        T = e_step(X, taus, mus, sigmas)
        taus, mus, sigmas = m_step(X, T)

