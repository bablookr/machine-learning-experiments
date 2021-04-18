import os
from sklearn.neural_network import BernoulliRBM
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

batch_size = 128
epochs = 2

n_train = 60000
n_test = 10000

#input_shape = (784,)
num_categories = 10

#default_kernel_size = (3, 3)
#default_pool_size = (2, 2)


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(n_train, -1)
    x_train = x_train.astype('float32') / 255.

    x_test = x_test.reshape(n_test, -1)
    x_test = x_test.astype('float32') / 255.

    y_train = to_categorical(y_train, num_categories)
    y_test = to_categorical(y_test, num_categories)

    return (x_train, y_train), (x_test, y_test)



def transform(x, rbm):
    return np.around(rbm.transform(x))

def train_rbm(x, nh):
    rbm = BernoulliRBM(n_components=nh, verbose=True)
    rbm.fit(x)
    xh = transform(x, rbm)
    return xh, rbm


def train_dense_model(x, y):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(128,)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_categories, activation='softmax'))

    model.compile(Adadelta(),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs)

    return model


(X_train, Y_train), (X_test, Y_test) = load_and_preprocess_data()
Xh_1, rbm_1 = train_rbm(X_train, 256)
Xh_2, rbm_2 = train_rbm(Xh_1, 128)
model= train_dense_model(tf.convert_to_tensor(Xh_2), Y_train)


'''cnn_model = create_model(X_train)
cnn_model.compile(Adadelta(),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

cnn_model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, Y_test))

scores = cnn_model.evaluate(X_test, Y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#model_path = os.path.join(os.getcwd(), 'saved_models\rbm_mnist')
#cnn_model.save(model_path)'''



