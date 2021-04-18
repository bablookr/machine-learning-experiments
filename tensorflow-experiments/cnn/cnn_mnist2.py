from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

batch_size = 128
epochs = 2

input_shape = (28, 28, 1)
num_categories = 10

default_kernel_size = (3, 3)
default_pool_size = (2, 2)

optimizer = tf.keras.optimizers.Adam(lr=0.001)
loss_function = tf.keras.losses.categorical_crossentropy


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
    x_train = x_train.astype('float32') / 255.

    x_test = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1], input_shape[2])
    x_test = x_test.astype('float32') / 255.

    y_train = to_categorical(y_train, num_categories)
    y_test = to_categorical(y_test, num_categories)

    return (x_train, y_train), (x_test, y_test)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=default_kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=default_kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=default_pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_categories, activation='softmax'))

    return model


def get_gradients(x_train, y_train, model):
    with tf.GradientTape() as tape:
        loss = loss_function(y_train, model(x_train))

    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients


def update_parameters(parameters, gradients):
    zipped_tuples = zip(gradients, parameters)
    optimizer.apply_gradients(zipped_tuples)


def print_util(completed, total):
    num_equals = int(30 * completed/total) + 1
    num_dots = 30 - num_equals
    print(str(completed) + '/' + str(total) + ' [' + '=' * num_equals + '.' * num_dots + ']')


def train_model(model, x_train, y_train):
    batch_per_epoch = int(x_train.shape[0] / batch_size)
    for epoch in range(epochs):
        print('\nEpoch ' + str(epoch + 1) + '/' + str(epochs) + '\n')
        for i in range(batch_per_epoch):
            n = i * batch_size
            print_util(n + batch_size, x_train.shape[0])

            gradients = get_gradients(x_train[n:n + batch_size], y_train[n:n + batch_size], model)
            update_parameters(model.trainable_variables, gradients)


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = load_and_preprocess_data()
    cnn_model = create_model()
    train_model(cnn_model, X_train, Y_train)


