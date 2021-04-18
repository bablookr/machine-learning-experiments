import numbers
import numpy as np
from sklearn.utils import gen_even_slices

n_iter = 10
learning_rate = 0.1
batch_size = 10


def get_rng(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class BernoulliRBM():
    def __init__(self, random_state=None):
        self.random_state = random_state

    def get_hidden(self, v):
        p = np.dot(v, self.W.T) + self.c
        return sigmoid(p)

    def get_visible(self, h):
        p = np.dot(h, self.W) + self.b
        return sigmoid(p)

    def sample_hidden(self, v, rng):
        p = self.get_hidden(v)
        return (p > rng.random_sample(p.shape)), p

    def sample_visible(self, h, rng):
        p = self.get_visible(h)
        return (p > rng.random_sample(p.shape)), p

    def fit(self, x_train, n_hidden):
        n_train = x_train.shape[0]
        n_visible = x_train.shape[1]

        rng = get_rng(self.random_state)
        self.W = rng.normal(0, 0.01, size=(n_hidden, n_visible))
        self.b = np.zeros(shape=(n_visible,))
        self.c = np.zeros(shape=(n_hidden,))

        n_batches = int(np.ceil(n_train / batch_size))
        batch_slices = list(gen_even_slices(n_batches * batch_size, n_batches, n_samples=n_train))

        h_samples = np.zeros(shape=(batch_size, n_hidden))
        for i in range(n_iter):
            for batch_slice in batch_slices:
                v = x_train[batch_slice]
                h = self.get_hidden(v)

                v_samples, _ = self.sample_visible(h_samples, rng)
                h_samples, h_prime = self.sample_hidden(v_samples, rng)

                dW = np.dot(h.T, v) - np.dot(h_prime.T, v_samples)
                db = v.sum(axis=0) - v_samples.sum(axis=0)
                dc = h.sum(axis=0) - h_prime.sum(axis=0)

                alpha = learning_rate/v.shape[0]
                self.W += (alpha * dW)
                self.b += (alpha * db)
                self.c += (alpha * dc)


if __name__ == '__main__':
    model = BernoulliRBM(random_state=0)
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    model.fit(X, n_hidden=2)
    print(model.get_hidden(X))

