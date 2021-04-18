import tensorflow as tf
import numpy as np

epochs = 10000
alpha = 0.001


def get_data():
    x_train = np.array([x for x in range(10)])
    y_train = np.array([2*x + 5 for x in range(10)])
    return x_train, y_train


def initialize_parameters():
    w = tf.Variable(0.)
    b = tf.Variable(0.)
    return w, b


def get_gradients(x, y, w, b):
    with tf.GradientTape() as tape:
        h_x = w * x + b
        loss = tf.abs(y - h_x)

    dw, db = tape.gradient(loss, (w, b))
    return dw, db


def update_parameters(w, b, dw, db):
    w.assign(tf.subtract(w, alpha * dw))
    b.assign(tf.subtract(b, alpha * db))
    return w, b


def train_model(x_train, y_train):
    w, b = initialize_parameters()

    for _ in range(epochs):
        dw, db = get_gradients(x_train, y_train, w, b)
        w, b = update_parameters(w, b, dw, db)

    return w, b


if __name__ == '__main__':
    x_train, y_train = get_data()
    w, b = train_model(x_train, y_train)

    print(w)
    print(b)