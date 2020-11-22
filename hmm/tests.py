import numpy as np
from hmm import viterbi
def test_multi():
    """
    To test, A & B are designed in a specific way so we can be more assured 
    the result is correct.
    in this test scenario:
        - observation 0 can only be produced by state 0
        - observation 1 can be produced by state 1 and 2
        - state 0 is highly likely to stay within state 0
        - state 1 is highly likely to jump to state 2
        - state 2 cannot transition back to state 0
    with the above in mind, so given the particular observation in the test y:
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    we can expect a state sequence like 
        [0, 0, 0, 0, 2, 2, 2, 2, 1, 0, 0, 0]
    """
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.2, 0.1, 0.7],
        [0.0, 0.3, 0.7],
    ])

    B = np.array([
        [1., 0.],
        [0., 1.],
        [0., 1.]]
    )

    pi = np.array([0.3, 0.3, 0.4])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0])

    x_seq_opt = viterbi(y, A, B, pi)
    np.testing.assert_array_equal(x_seq_opt, np.array([0, 0, 0, 0, 2, 2, 2, 2, 1, 0, 0, 0]))
