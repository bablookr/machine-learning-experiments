import tensorflow as tf
import numpy as np

epochs = 10000
alpha = 0.001


def get_data():
    x_train = np.array([x for x in range(10)])
    y_train = np.array([2*x**2 + 3*x + 5 for x in range(10)])
    return x_train, y_train


def initialize_parameters():
    w_2 = tf.Variable(0.)
    w_1 = tf.Variable(0.)
    b = tf.Variable(0.)
    return w_2, w_1, b


def get_gradients(x, y, w_2, w_1, b):
    with tf.GradientTape() as tape:
        h_x = w_2*x**2 + w_1*x + b
        loss = tf.abs(y - h_x)

    dw_2, dw_1, db = tape.gradient(loss, (w_2, w_1, b))
    return dw_2, dw_1, db


def update_parameters(w_2, w_1, b, dw_2, dw_1, db):
    w_2.assign(tf.subtract(w_2, alpha * dw_2))
    w_1.assign(tf.subtract(w_1, alpha * dw_1))
    b.assign(tf.subtract(b, alpha * db))
    return w_2, w_1, b


def train_model(x_train, y_train):
    w_2, w_1, b = initialize_parameters()

    for _ in range(epochs):
        dw_2, dw_1, db = get_gradients(x_train, y_train, w_2, w_1, b)
        w_2, w_1, b = update_parameters(w_2, w_1, b, dw_2, dw_1, db)

    return w_2, w_1, b


if __name__ == '__main__':
    x_train, y_train = get_data()
    w_2, w_1, b = train_model(x_train, y_train)

    print(w_2)
    print(w_1)
    print(b)

