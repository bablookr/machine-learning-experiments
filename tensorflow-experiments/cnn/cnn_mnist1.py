import tensorflow as tf

batch_size = 128
epochs = 2

input_shape = (28, 28, 1)
num_categories = 10

kernel_size = 3
pool_size = 2

initializer = tf.initializers.glorot_normal()
optimizer = tf.optimizers.Adam(lr=0.001)
loss_function = tf.losses.categorical_crossentropy


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
    x_train = x_train.astype('float32') / 255.

    x_test = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1], input_shape[2])
    x_test = x_test.astype('float32') / 255.

    y_train = tf.one_hot(y_train, num_categories)
    y_test = tf.one_hot(y_test, num_categories)

    return (x_train, y_train), (x_test, y_test)


def initialize_parameters():
    input_channels = input_shape[2]

    conv_kernel_1 = tf.Variable(initializer(shape=[kernel_size, kernel_size, input_channels, 32]))
    conv_kernel_2 = tf.Variable(initializer(shape=[kernel_size, kernel_size, 32, 64]))
    dense_kernel_1 = tf.Variable(initializer(shape=[9216, 128]))
    dense_kernel_2 = tf.Variable(initializer(shape=[128, num_categories]))

    return conv_kernel_1, conv_kernel_2, dense_kernel_1, dense_kernel_2


def model(x_input, parameters):
    conv_kernel_1, conv_kernel_2, dense_kernel_1, dense_kernel_2 = parameters

    # Convolution 1
    x = tf.nn.conv2d(x_input, conv_kernel_1, strides=[1, 1], padding='VALID')
    x = tf.nn.relu(x)

    # Convolution 2
    x = tf.nn.conv2d(x, conv_kernel_2, strides=[1, 1], padding='VALID')
    x = tf.nn.relu(x)

    # Pooling
    x = tf.nn.max_pool2d(x, ksize=[pool_size, pool_size], strides=[pool_size, pool_size], padding='VALID')

    # Dropout 1
    x = tf.nn.dropout(x, 0.25)

    # Flatten
    x = tf.reshape(x, (batch_size, -1))

    # Dense 1
    x = tf.matmul(x, dense_kernel_1)
    x = tf.nn.relu(x)

    # Dropout 2
    x = tf.nn.dropout(x, 0.5)

    # Dense 2
    x = tf.matmul(x, dense_kernel_2)
    x = tf.nn.softmax(x)

    return x


def get_gradients(x_train, y_train, parameters):
    with tf.GradientTape() as tape:
        loss = loss_function(y_train, model(x_train, parameters))

    gradients = tape.gradient(loss, parameters)
    return gradients


def update_parameters(parameters, gradients):
    zipped_tuples = zip(gradients, parameters)
    optimizer.apply_gradients(zipped_tuples)
    return parameters


def print_util(completed, total):
    num_equals = int(30 * completed/total) + 1
    num_dots = 30 - num_equals
    print(str(completed) + '/' + str(total) + ' [' + '=' * num_equals + '.' * num_dots + ']')


def train_model(x_train, y_train):
    parameters = initialize_parameters()

    batch_per_epoch = int(x_train.shape[0] / batch_size)
    for epoch in range(epochs):
        print('\nEpoch ' + str(epoch + 1) + '/' + str(epochs) + '\n')
        for i in range(batch_per_epoch):
            n = i * batch_size
            print_util(n + batch_size, x_train.shape[0])

            gradients = get_gradients(x_train[n:n + batch_size], y_train[n:n + batch_size], parameters)
            parameters = update_parameters(parameters, gradients)

    return parameters


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = load_and_preprocess_data()
    train_model(X_train, Y_train)
