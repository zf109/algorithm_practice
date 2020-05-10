import numpy as np

def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig

def d_sigmoid(m):
    return sigmoid(m)*(1-sigmoid(m))

sigmoid_vec = np.vectorize(sigmoid)
d_sigmoid_vec = np.vectorize(d_sigmoid)

def cost(y_pred, y):
    norm = np.linalg.norm
    return 1/len(y)*sum([1/2*norm(y_pred_ - y_)**2 for y_pred_, y_ in zip(y_pred, y)])

def cost_gradient(y_pred, y):
    return y_pred - y


class SingleLayerSigmoidMLP():
    """
        Simple implemnetation of single hidden layer perceptrons.
        this can be easily generalised to multi-layer perceptrions.
    """
    def __init__(self, n_in=784, n_hid=100, n_out=10):
        self.W_hid = np.random.rand(n_in, n_hid)
        self.b_hid = np.zeros((n_hid, 1))
        self.W_out = np.random.rand(n_hid, n_out)
        self.b_out = np.zeros((n_out, 1))

    def predict(self, X):
        sigmoid_vec = np.vectorize(sigmoid)
        netin_hid = np.matmul(X, self.W_hid) + self.b_hid.T
        netout_hid = sigmoid_vec(netin_hid)
        netin_out = np.matmul(netout_hid, self.W_out) + self.b_out.T
        netout_out = sigmoid_vec(netin_out)
        return netout_out

    def backpropagate(self, x, y):
        """
            Backpropagate a single data point (x, y)
        """
        # feedforward
        x = x.reshape(len(x), 1) #shape to proper vector shape
        y = y.reshape(len(y), 1)

        m_h = np.matmul(self.W_hid.T, x) + self.b_hid
        z_h = sigmoid_vec(m_h)
        m_o = np.matmul(self.W_out.T, z_h) + self.b_out
        z_o = sigmoid_vec(m_o)

        #back propagate to calculate deltas
        delta_o = cost_gradient(z_o, y) * d_sigmoid_vec(m_o)
        delta_h = np.matmul(self.W_out, delta_o) * d_sigmoid_vec(m_h)
        delta_W_o = np.outer(z_h, delta_o)
        delta_W_h = np.outer(x, delta_h)

        delta_b_o = delta_o
        delta_b_h = delta_h

        return delta_W_h, delta_b_h, delta_W_o, delta_b_o

    def fit(self, X, Y, eta=.8, eps=1e-5, max_epoch=1000):
        """
            Obviously this is not practical at all :p
        """
        for epoch in range(max_epoch):
            Y_pred = self.predict(X)
            error = cost(Y_pred, Y)
            if epoch % int(max_epoch/10) == 0:
                print("Epoch: {}, error function value: {}".format(epoch, error))
            if  error < eps:
                return
            for x, y in zip(X, Y):
                delta_W_h, delta_b_h, delta_W_o, delta_b_o = self.backpropagate(x,  y)
                self.W_hid -= eta * delta_W_h
                self.b_hid -= eta * delta_b_h
                self.W_out -= eta * delta_W_o
                self.b_out -= eta * delta_b_o

def show(y, y_pred):
    for y_, pred_ in zip(y, y_pred):
        print("true value: {}, predicted: {}".format(y_, pred_))


if __name__ == "__main__":

    """
    Generate some random classification problem:
        the output class y_1 = x_1 and x_2, y_2 = x_1 or x_2 
    """
    binarize = np.vectorize(lambda x: 1 if 0.5 <= x else 0)
    X = binarize(np.random.rand(10, 2))
    Y = np.array([np.logical_and(X[:,0], X[:,1]), np.logical_or(X[:,0], 2*X[:,1])]).T
    print(X)
    print("===============")
    print(Y)

    mlp = SingleLayerSigmoidMLP(n_in=2, n_hid=4, n_out=2)
    mlp.fit(X, Y)

    y_pred = mlp.predict(X[:10])
    y = Y[:10]
    show(y=y, y_pred=y_pred)
