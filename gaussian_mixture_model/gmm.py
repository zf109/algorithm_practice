
import numpy as np

import logging
logging.basicConfig(format='[%(asctime)s]-[%(name)-1s-%(levelname)2s]: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

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


def likelihood_function(X, taus, mus, sigmas):
    """
    The component pdf p_k(x; mu_k, sigma_k)
        :param taus: K-dim array contains the weight (or prior of hidden var) of each gaussian component
        :param mus: Kxn array (n be the dimension of x vector and K be the num of components), mean of k n-dim gaussian distr.
        :param sigmas: Knxn array covariance of k n-dim gaussian distr.
        :return: numeric between 0 and 1, the likelihood function value
    """
    N = X.shape[0] # number of data points
    get_component_prob = lambda x: component_pdfs(x, mus, sigmas)
    T = np.apply_along_axis(arr=X, func1d=get_component_prob, axis=1) # gaussian component probabilities in row format (NxK)
    taus_rep = np.tile(taus, reps=(N, 1)) # repeat tau along N-axis so elementwise product can work

    return np.sum(T*taus_rep, axis=1)


def e_step(X, taus, mus, sigmas):
    """
        E step of the EM algorithm, caculates the posterior T_{k, i}=P(z_i=k|x_i)
        it returns T_{k,i} in the form of a KxN T matrix where each element is T_{k, i}
        :param X: Nxn matrix represents N number of n-dim data points
        :param taus: K-dim vector, the weight of each component, or the prior of the hidden stats z
        :param mus: Kxn matrix (n be the dimension of x vector and K be the num of components), mean of k n-dim gaussian distr.
        :param sigmas: Kxnxn matrix covariance of k n-dim gaussian distr.
        :return: T_{k,i} in the form of a KxN T matrix where each element is T_{k, i}
    """
    K, N = mus.shape[0], X.shape[0] # dimensions, K: num of hidden component, N: number of data points
    get_component_prob = lambda x: component_pdfs(x, mus, sigmas)
    T = np.apply_along_axis(arr=X, func1d=get_component_prob, axis=1) # gaussian component probabilities in row format (NxK)
    taus_rep = np.tile(taus, reps=(N, 1)) # repeat tau along N-axis so elementwise product can work

    norm_const = np.sum(T*taus_rep, axis=1) # the normalisation factor \sum_{k=1}^K p_k * tau_k， and is currently estimated likelihood
    norm_const_rep = np.tile(norm_const, reps=(K, 1)).T # repeat normalisation constant along K-axis

    T = T*taus_rep/norm_const_rep # calculate the posterior 
    return T.T #return the transposed matrix so that the index is matched

def m_step(X, T):
    """
        M step of the EM algorithm, caculates the MLE of taus, mus and sigmas
        :param X: Nxn matrix, the dataset, N number of n-dim data points
        :param T: KxN matrix, the T matrix is the posterior matrix where the i, j th component is the T_{k, i}
        :return: a 3-tuple:
            - taus: K-dim array, the estimated prior probability for each hidden variable z
            - mus: Kxn matrix, the estimated mean of the n-dim gaussian component, for each of the k component
            - sigmas: Kxnxn matrix, the covariance matrix of the n-dim gaussian component, for each of the k component
    """
    def get_sigma(X, muk, Tk):
        """
            function that calculate the covariance of the k-th component
            :param muk: n-dim vector, the k-th component's mean
            :param Tk: N-dim vector, the k-th component posterior of hidden state z_i, for each x_i
        """
        X_centred = X - muk
        X_weighted = X_centred * np.tile(Tk, reps=(X.shape[1],1)).T # repeat Tk in N-direction to match X's shape and weigh it
        return X_weighted.T@X_centred/np.sum(Tk) # weighted and centred are exchangable: we only need to weigh it by T_k once

    N, n = X.shape #  N: number of data points, n: dimension of the data point
    K = T.shape[0] # num of hidden component
    T_sum = np.sum(T, axis=1) # caculate the common term sum of T_{k, i} over all i, this is a k-dim vector

    taus = T_sum / N # average over i  for T_{k, i} gives MLE for all tau_k

    T_sum_rep = np.tile(T_sum, reps=(n, 1)).T # repeat T_sum n times in column
    mus = T@X/T_sum_rep # T@X gives a Kxn matrix with it's k, i th component be \sum_{i=1}^NT_{k, i}x_i then each row is divided by T_sum, gives MLE for all mu_k

    sigmas = np.array([get_sigma(X, mus[k, :], T[k, :]) for k in range(K)])
    return taus, mus, sigmas


class GaussianMixture():
    def __init__(self, n_hidden=2, max_iter=100, seed=None):
        """
            :param n_hidden: number of hidden variables (or the number of components)
            :param max_iter: maximum EM iteration allowed, default to 100
            :param seed: the random seed for initialisation
        """
        self.n_hidden = n_hidden
        self.max_iter = max_iter
        self.seed = seed
        self.taus, self.mus, self.sigmas = None, None, None

    def fit(self, X):
        """
            :param X: Nxn matrix (N be the num of data points and n be the dimension of each data point)
        """
        n_hid = self.n_hidden
        n_var = X.shape[1]

        np.random.seed(self.seed) # setup seed, if None means no seed
        mus = np.random.randn(n_hid, n_var)*10 # initialise means of k components
        np.random.seed(self.seed)
        # initialise  sigmas of k components with identity
        sigmas = np.array([np.eye(n_var) for _ in range(n_hid)])
        taus = np.ones(n_hid)/n_hid # assume uninformative prior

        for i in range(self.max_iter):
            _logger.debug(f"iter: {i}, objective: {np.sum(np.log(likelihood_function(X, taus, mus, sigmas)))}") # only show this in debug mode to reduce unnecessary evaluation
            T = e_step(X, taus, mus, sigmas)
            sigmas_prev = sigmas
            taus, mus, sigmas = m_step(X, T)
            if np.min(abs(sigmas - sigmas_prev) < 0.1):
                _logger.debug(f"break after iteration {i+1}")
                break
        self.taus, self.mus, self.sigmas = taus, mus, sigmas
        return self

    def predict_proba(self, X):
        T = e_step(X, self.taus, self.mus, self.sigmas)
        return T.T # transpose, so it's 1st dimension matches the X's dimension N.

    def predict(self, X):
        T = e_step(X, self.taus, self.mus, self.sigmas)
        return np.argmax(T.T, axis=1)


if __name__ == "__main__":

    # generate a dataset
    mean1, mean2 = np.array([0, 0]), np.array([10, 20])
    sigma1, sigma2 = np.array([[1, 0], [0, 1]]), np.array([[5, -5], [-5, 10]])

    np.random.seed(42)
    X1 = np.random.multivariate_normal(mean1, sigma1, 1000)
    np.random.seed(42)
    X2 = np.random.multivariate_normal(mean2, sigma2, 200)
    X = np.vstack([X1, X2])

    print("="*20, "data info", "="*20)
    print(f"cluster 1: mean={np.mean(X1, axis=0)}\nsigma={np.cov(X1.T)}")
    print('-'*8)
    print(f"cluster 2: mean={np.mean(X2, axis=0)}\nsigma={np.cov(X2.T)}")
    print("="*50)
    print("\n\n")

    n_components = 2
    gmm = GaussianMixture(n_hidden=n_components, seed=0)
    print("="*20, f"fitting Gaussian mixture model with {n_components} components", "="*20)
    gmm.fit(X)

    print(f"{'='*20}finished fitting{'='*20}")
    print(f"{'-'*5}estimated taus:\n{gmm.taus}")
    print(f"{'-'*5}estimated mus:\n{gmm.mus}")
    print(f"{'-'*5}estimated sigmas:\n{gmm.sigmas}")

    print("\n\n")
    print(f"{'='*20}test predictive power{'='*20}")
    # construct a new dataset for test, but with the same distribution 
    np.random.seed(0)
    X1 = np.random.multivariate_normal(mean1, sigma1, 1000)
    np.random.seed(0)
    X2 = np.random.multivariate_normal(mean2, sigma2, 200)
    X_test = np.vstack([X1, X2])
    y_true = np.hstack([np.zeros(1000), np.ones(200)])

    print("-"*20, "test data info", "-"*20)
    print(f"cluster 1: mean={np.mean(X1, axis=0)}\nsigma={np.cov(X1.T)}")
    print('-'*8)
    print(f"cluster 2: mean={np.mean(X2, axis=0)}\nsigma={np.cov(X2.T)}")
    print("-"*8)

    # predict use the model trained
    y_pred = gmm.predict(X_test)
    accuracy = np.mean(y_true==y_pred)
    print(f"accuracy of the model on test is {accuracy}")



