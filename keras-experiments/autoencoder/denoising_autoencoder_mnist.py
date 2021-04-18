import os
import numpy as np
import keras.backend as K

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from keras.losses import mse
from keras.optimizers import adam

batch_size = 128
epochs = 2

input_shape = (28, 28, 1)
default_kernel_size = (3, 3)
latent_dim = 16


def load_and_preprocess_data():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    x_train = x_train.astype('float32') / 255.

    x_test = x_test.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    x_test = x_test.astype('float32') / 255.

    return x_train, x_test


def noise_data(x_train, x_test):
    noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
    x_train_noisy = x_train + noise
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)

    noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
    x_test_noisy = x_test + noise
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_train_noisy, x_test_noisy


def create_model():
    #Encoder
    x_input = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=default_kernel_size, padding='same', activation='relu', strides=2)(x_input)
    x = Conv2D(64, kernel_size=default_kernel_size, padding='same', activation='relu', strides=2)(x)
    shape_before_flattening = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(latent_dim)(x)
    encoder = Model(inputs=x_input, outputs=x)

    #Decoder
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(np.prod(shape_before_flattening[1:]))(decoder_input)
    x = Reshape(shape_before_flattening[1:])(x)
    x = Conv2DTranspose(64, kernel_size=default_kernel_size, padding='same', activation='relu', strides=2)(x)
    x = Conv2DTranspose(32, kernel_size=default_kernel_size, padding='same', activation='relu', strides=2)(x)
    x = Conv2DTranspose(1, kernel_size=default_kernel_size, padding='same', activation='sigmoid')(x)
    decoder = Model(inputs=decoder_input, outputs=x)

    #Autoencoder
    autoencoder = Model(inputs=x_input, outputs=decoder(encoder(x_input)))
    return autoencoder


X_train, X_test = load_and_preprocess_data()
X_train_noisy, X_test_noisy = noise_data(X_train, X_test)
ae_model = create_model()

ae_model.compile(loss=mse,
                 optimizer=adam,
                 metrics=['accuracy'])

ae_model.fit(X_train_noisy, X_train,
             batch_size=batch_size,
             epochs=epochs,
             validation_data=(X_test_noisy, X_test))

scores = ae_model.evaluate(X_test_noisy, X_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model_path = os.path.join(os.getcwd(), 'saved_models\denoising_autoencoder_mnist')
ae_model.save(model_path)
