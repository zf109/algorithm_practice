import numpy as np


def viterbi(y, A, B, pi):
    """
        viterbi algorithm
        :param y: observation sequence
        :param A: the transition matrix
        :param B: the emission matrix
        :param pi: the initial probability distribution
    """
    N = B.shape[0]
    x_seq = np.zeros([N, 0])
    V = B[:, y[0]] * pi

    # forward to compute the optimal value function V
    for y_ in y[1:]:
        _V = np.tile(B[:, y_], reps=[N, 1]).T * A.T * np.tile(V, reps=[N, 1])
        x_ind = np.argmax(_V, axis=1)
        x_seq = np.hstack([x_seq, np.c_[x_ind]])
        V = _V[np.arange(N), x_ind]
    x_T = np.argmax(V)

    # backward to fetch optimal sequence
    x_seq_opt, i = np.zeros(x_seq.shape[1]+1), x_seq.shape[1]-1
    prev_ind = x_T
    while i >= 0:
        x_seq_opt[i] = prev_ind
        i -= 1
        prev_ind = x_seq[int(prev_ind), i]
    return x_seq_opt


class HiddenMarkovModel():
    def __init__(self, A=None, B=None, pi=None):
        self.pi = pi.ravel()
        self.A = A
        self.B = B

    def decode(self, y):
        """
        implementation of viterbi algorithm
        y: the observed states range from {0, 1,2,3,...,M-1}
        """
        A, B, pi = self.A, self.B, self.pi
        x_seq_opt = viterbi(y, A, B, pi)
        return x_seq_opt
