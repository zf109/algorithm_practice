import numpy as np
from sklearn.mixture import GaussianMixture

inv = np.linalg.inv
det = np.linalg.det
matmul = np.matmul


def quadratic_form(x, A):
    assert len(x.shape) == 1, "shape of x is {}, requires shape of the form (n,)".format(x.shape)
    return x.T.dot(A).dot(x)


def gaussian(x, mean=None, covariance=None):
    dim = len(mean)
    const = 1/np.sqrt(2*np.pi**dim * np.linalg.det(covariance))
    exponent = -quadratic_form(x - mean, covariance)
    return const * np.exp(exponent)


class ConfidenceEstimator():
    """
        This confidence estimator is based on the paper:
            Confidence Estimation in Classification Decision: A Method for Detecting Unseen Patterns
            Pandu R Devarakota, Bruno Mirbach, Bjorn Ottersten, 2007
            https://pdfs.semanticscholar.org/be73/b307de1cea0426f2111f081d298bd1486d20.pdf
        the variable names are also based on this paper
    """
    def __init__(self, radius=None, n_components=None):
        self.gm = GaussianMixture(n_components=n_components or 5)
        self.radius = radius or .1
        self.data_covariance = None
        self.gm_covariances: list= None
        # self.T: list= None      # T matrices (tmp matrices)

    def fit(self, X):
        self.gm.fit(X)

        C = np.cov(X.T)     # covariance matrix of the training dataset
        Nw = self.gm.weights_ * len(X)   # normalisation factors of the gaussian mixture, multiplied by total number of data points
        S = self.gm.covariances_    # covariance matrices of each gaussian component
        r = self.radius

        # compute equation 9 in the paper
        get_Tk = lambda r, Sk, C: inv( inv(C)/(r**2) + inv(Sk) )
        T = [get_Tk(r, Sk, C) for Sk in S]

        get_Nw_primek = lambda Nk, Tk, Sk: Nk * np.sqrt(det( matmul( Tk, inv(Sk))) ) 
        Nw_prime = [get_Nw_primek(Nk, Tk, Sk) for Nk, Tk, Sk in zip(Nw, S, T)]

        get_S_primek = lambda Tk, Sk: matmul( inv( 1 - matmul(Tk, inv(Sk)) ), Sk)
        S_prime = [get_S_primek(Tk, Sk) for Tk, Sk in zip(T, S)]

        self.Np = Nw_prime
        self.Sp = S_prime
        self.mu = self.gm.means_

    def calc_neighbour(self, x):
        get_Nrk = lambda x, Npk, Spk, muk: Npk * quadratic_form(x - muk, inv(Spk)) / 2
        return sum([get_Nrk(x, Npk, Spk, muk) for Npk, Spk, muk in zip(self.Np, self.Sp, self.mu)])


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from matplotlib import pyplot as plt

    # generate dummy dataset, y in {0, 1} are labels, y = 2 is unseen pattern
    _X, _y = make_classification(
        n_classes=3,
        n_samples=600,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.01, class_sep=1.0
    )

    is_label = np.vectorize(lambda x: x in {0 ,1})
    is_unseen = np.vectorize(lambda x: x == 2)

    X, y = _X[is_label(_y)], _y[is_label(_y)]
    X_unseen, y_unseen = _X[is_unseen(_y)], _y[is_unseen(_y)]

    assert X.shape[0] + X_unseen.shape[0] == _X.shape[0], "shape mismatch"

    distribution_model = GaussianMixture(n_components=2).fit(X)
    
    
    conf_model = ConfidenceEstimator()
    conf_model.fit(X)

    plt.scatter(_X[:, 0], _X[:, 1],c=_y)
    plt.colorbar()
    plt.show()
