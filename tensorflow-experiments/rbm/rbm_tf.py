import tensorflow as tf
import numpy as np
import numbers
import math

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


class BernoulliRBM():
    def __init__(self, random_state=None):
        self.random_state = random_state

    def get_hidden(self, v):
        p = tf.matmul(v, tf.transpose(self.W)) + self.c
        return tf.sigmoid(p)

    def get_visible(self, h):
        p = tf.matmul(h, self.W) + self.b
        return tf.sigmoid(p)

    def sample_hidden(self, v, rng):
        p = self.get_hidden(v)
        return tf.cast(p > tf.constant(rng.random_sample(p.shape), dtype='float32'), dtype='float32'), p

    def sample_visible(self, h, rng):
        p = self.get_visible(h)
        return tf.cast(p > tf.constant(rng.random_sample(p.shape), dtype='float32'), dtype='float32'), p

    def fit(self, x_train, n_hidden):
        n_train = x_train.shape[0]
        n_visible = x_train.shape[1]

        rng = get_rng(self.random_state)
        self.W = tf.Variable(rng.normal(0, 0.01, size=(n_hidden, n_visible)), dtype='float32')
        self.b = tf.Variable(tf.zeros(shape=(n_visible,)))
        self.c = tf.Variable(tf.zeros(shape=(n_hidden,)))

        n_batches = math.ceil(n_train / batch_size)
        batch_slices = list(gen_even_slices(n_batches * batch_size, n_batches, n_samples=n_train))

        h_samples = tf.zeros(shape=(batch_size, n_hidden))
        for i in range(n_iter):
            for batch_slice in batch_slices:
                v = x_train[batch_slice]
                h = self.get_hidden(v)

                v_samples, _ = self.sample_visible(h_samples, rng)
                h_samples, h_prime = self.sample_hidden(v_samples, rng)

                dW = tf.matmul(tf.transpose(h), v) - tf.matmul(tf.transpose(h_prime), v_samples)
                db = tf.reduce_sum(v, axis=0) - tf.reduce_sum(v_samples, axis=0)
                dc = tf.reduce_sum(h, axis=0) - tf.reduce_sum(h_prime, axis=0)

                alpha = learning_rate / v.shape[0]
                self.W.assign_add(alpha * dW)
                self.b.assign_add(alpha * db)
                self.c.assign_add(alpha * dc)



if __name__ == '__main__':
    model = BernoulliRBM(random_state=0)
    X = tf.Variable([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype='float32')
    model.fit(X, n_hidden=2)
    print(model.get_hidden(X))