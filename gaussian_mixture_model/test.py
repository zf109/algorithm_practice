from gmm import gaussian_pdf, e_step, m_step, likelihood_function
import numpy as np
from scipy.stats import multivariate_normal

def test_gaussian_dist():
    x = np.array([3, 2, 0])
    mu = np.array([3, 2, 0])
    sigma = np.array([
        [1, .5,  0],
        [.5, 1, .5],
        [0, .5,  1],
        ])
    y = gaussian_pdf(x, mu, sigma)
    y_expected = multivariate_normal.pdf(x, mean=mu, cov=sigma)
    np.testing.assert_almost_equal(y, y_expected)

def test_likelihood_function():
    X = np.array([
        [3,     2,     0],
        [30,    20,   10],
        [10,    15,   20],
        [35,    25,   10],
        [35,    21,   12],
    ])
    mus = np.array([[3, 2, 0], [30, 20, 10]])
    sigmas = np.array([
        [[1, .5,  0],
        [.5, 1, .5],
        [0, .5,  1]],
        [[1, .5,  0],
        [.5, 1, .5],
        [0, .5,  1]],
    ])
    taus = np.array([.5, .5])
    likelihood = likelihood_function(X, taus, mus, sigmas)

    np.testing.assert_almost_equal(likelihood, np.array([4.48967805e-02, 4.48967805e-02, 1.10797610e-99, 3.22993412e-10,
       4.37124049e-11]))
    

def test_e_step():
    X = np.array([
        [3,     2,     0],
        [30,    20,   10],
        [10,    15,   20],
        [35,    25,   10],
        [35,    21,   12],
    ])
    mus = np.array([[3, 2, 0], [30, 20, 10]])
    sigmas = np.array([
        [[1, .5,  0],
        [.5, 1, .5],
        [0, .5,  1]],
        [[1, .5,  0],
        [.5, 1, .5],
        [0, .5,  1]],
    ])
    taus = np.array([.5, .5])
    T = e_step(X, taus, mus, sigmas)

    T_expected = np.array([
        [1.00000000e+000, 7.52252484e-181, 1.00000000e+000, 2.14953627e-238, 2.98526391e-249],
        [7.52252484e-181, 1.00000000e+000, 1.08159416e-011, 1.00000000e+000, 1.00000000e+000]
    ])
    np.testing.assert_almost_equal(T, T_expected)


def test_em():
    mean1, mean2 = np.array([0, 0]), np.array([10, 20])
    sigma1, sigma2 = np.array([[1, 0], [0, 1]]), np.array([[5, -5], [-5, 10]])

    np.random.seed(42)
    X1 = np.random.multivariate_normal(mean1, sigma1, 1000)
    np.random.seed(42)
    X2 = np.random.multivariate_normal(mean2, sigma2, 200)
    X = np.vstack([X1, X2])
    mus = np.array([[0, 0], [2, 2]])
    sigmas = np.array([
        [[1, .5],
        [.5, 1,]],
        [[1, .5],
        [.5, 1]],
    ])
    taus = np.array([.5, .5])
    for i in range(100):
        T = e_step(X, taus, mus, sigmas)
        sigmas_prev = sigmas
        taus, mus, sigmas = m_step(X, T)
        if np.min(abs(sigmas - sigmas_prev) < 0.1):
            print(f"break after {i}th iteration")
            break
    mu1, mu2 = mus[0, :], mus[1, :]
    sig1, sig2 = sigmas[0,:,:], sigmas[1,:,:]
    assert np.min(abs(mu1 - mean1) < 0.1)
    assert np.min(abs(mu2 - mean2) < 0.5)
    assert np.min(abs(sigma1 - sig1) < 0.1)
    assert np.min(abs(sigma2 - sig2) < 1)
