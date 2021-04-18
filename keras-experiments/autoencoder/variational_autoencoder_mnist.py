import os
import numpy as np
import keras.backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Layer, Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from keras.losses import binary_crossentropy
from keras.optimizers import rmsprop

batch_size = 16
epochs = 2

input_shape = (28, 28, 1)
default_kernel_size = (3, 3)
latent_dim = 2


def load_and_preprocess_data():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    x_train = x_train.astype('float32') / 255.

    x_test = x_test.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    x_test = x_test.astype('float32') / 255.

    return x_train, x_test


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal((K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon


def create_model():
    # Encoder
    x_input = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=default_kernel_size, padding='same', activation='relu')(x_input)
    x = Conv2D(64, kernel_size=default_kernel_size, padding='same', activation='relu', strides=2)(x)
    x = Conv2D(64, kernel_size=default_kernel_size, padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=default_kernel_size, padding='same', activation='relu')(x)
    shape_before_flattening = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling)([z_mean, z_log_var])
    encoder = Model(inputs=x_input, outputs=z)

    # Decoder
    decoder_input = Input(K.int_shape(z)[1:])
    x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
    x = Reshape(shape_before_flattening[1:])(x)
    x = Conv2DTranspose(32, kernel_size=default_kernel_size, padding='same', activation='relu', strides=2)(x)
    x = Conv2D(1, kernel_size=default_kernel_size, padding='same', activation='sigmoid')(x)
    decoder = Model(inputs=decoder_input, outputs=x)

    #VariationalLayer
    class VariationalLayer(Layer):
        def call(self, inputs):
            x, z_decoded = K.flatten(inputs[0]), K.flatten(inputs[1])
            xent_loss = binary_crossentropy(x, z_decoded)
            kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            loss = K.mean(xent_loss + kl_loss)
            self.add_loss(loss, inputs=inputs)
            return x

    #VAE
    variational_layer_output = VariationalLayer()([x_input, decoder(encoder(x_input))])
    vae = Model(inputs=x_input, outputs=variational_layer_output)
    return vae


X_train, X_test = load_and_preprocess_data()
vae_model = create_model()
vae_model.compile(loss=None,
                  optimizer=rmsprop)

vae_model.fit(X_train, None,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, None))

scores = vae_model.evaluate(X_test, None)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model_path = os.path.join(os.getcwd(), 'saved_models\variational_autoencoder_mnist')
vae_model.save(model_path)
