import numpy as np

gaussian = lambda x, sigma, mu: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-1/(2*sigma**2) * (x - mu)**2)

class NaiveBayes():
    """
        practice implementation for gaussian naive bayes,
        take some inspiration from sklearn
    """

    def __init__(self, class_prior=None):
        self.n_classes = None # number of classes
        self.classes_ = None # list of classes with their symbols
        self.class_priors = class_prior # class prior, if not provided, then use equalproba

    def fit(self, X, y):
        self.classes_ = list(set(y))
        self.n_classes = len(self.classes_)

        if not self.class_priors:
            self.class_priors = 1/self.n_classes * np.ones(self.n_classes)

        for c in self.classes_:
            _X = X[y == c]
            

    def calculate_posterior(self, X):
        pass

    def predict_proba(self, X):
        return self.calculate_posterior(X)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification()
