from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.utils import to_categorical

batch_size = 128
epochs = 2

input_shape = (28, 28, 1)
num_categories = 10

default_kernel_size = (3, 3)
default_pool_size = (2, 2)


def preprocess_data(train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data

    x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
    x_train = x_train.astype('float32') / 255.

    x_test = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1], input_shape[2])
    x_test = x_test.astype('float32') / 255.

    y_train = to_categorical(y_train, num_categories)
    y_test = to_categorical(y_test, num_categories)

    return (x_train, y_train), (x_test, y_test)


def run(model, train_data, test_data):
    (x_train, y_train), (x_test, y_test) = preprocess_data(train_data, test_data)

    model.compile(Adadelta(),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    scores = model.evaluate(x_test, y_test)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


(X_train, Y_train), (X_test, Y_test) = preprocess_data(mnist.load_data())
X_train_lt5 = X_train[Y_train < 5]
Y_train_lt5 = Y_train[Y_train < 5]
X_test_lt5 = X_test[Y_test < 5]
Y_test_lt5 = Y_test[Y_test < 5]

X_train_gte5 = X_train[Y_train >= 5]
Y_train_gte5 = Y_train[Y_train >= 5] - 5
X_test_gte5 = X_test[Y_test >= 5]
Y_test_gte5 = Y_test[Y_test >= 5] - 5

train_data_1, test_data_1 = (X_train_lt5, Y_train_lt5), (X_test_lt5, Y_test_lt5)
train_data_2, test_data_2 = (X_train_gte5, Y_train_gte5), (X_test_gte5, Y_test_gte5)


feature_layers =[
        Conv2D(32, kernel_size=default_kernel_size, activation='relu', input_shape=input_shape),
        Conv2D(64, kernel_size=default_kernel_size, activation='relu'),
        MaxPooling2D(pool_size=default_pool_size),
        Dropout(0.25),
        Flatten()
    ]

classification_layers = [
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_categories, activation='softmax')
]

model = Sequential(feature_layers + classification_layers)
run(model, train_data_1, test_data_1)

for layer in feature_layers:
    layer.trainable = False

run(model, test_data_1, test_data_2)