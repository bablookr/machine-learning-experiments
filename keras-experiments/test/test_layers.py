import keras.backend as K
from unittest import TestCase
from unittest import main
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Lambda, Embedding
from keras.layers import Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose
from keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
from keras.layers import Reshape, Flatten, UpSampling1D, UpSampling2D, UpSampling3D
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Attention

from keras.layers import Dropout, MaxPooling2D


class TestLayersAPI(TestCase):
    # Core Layers
    def test_Dense_layer(self):
        model = Sequential()
        model.add(Dense(2, activation='relu', input_shape=(4,)))
        model.add(Dense(1, activation='sigmoid'))
        self.assertEqual(model.get_layer(index=0).input_shape, (None, 4))
        self.assertEqual(model.get_layer(index=0).output_shape, (None, 2))
        self.assertEqual(model.get_layer(index=0).activation.__name__, 'relu')
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 2))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 1))
        self.assertEqual(model.get_layer(index=1).activation.__name__, 'sigmoid')

    def test_Activation_layer(self):
        x_input = Input(shape=(4,))
        x = Activation(activation='relu')(x_input)
        x = Dense(2)(x)
        x = Activation(activation='sigmoid')(x)
        x = Dense(1)(x)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 4))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 4))
        self.assertEqual(model.get_layer(index=1).activation.__name__, 'relu')
        self.assertEqual(model.get_layer(index=3).input_shape, (None, 2))
        self.assertEqual(model.get_layer(index=3).output_shape, (None, 2))
        self.assertEqual(model.get_layer(index=3).activation.__name__, 'sigmoid')

    def test_Lambda_layer_1(self):
        def fun(z):
            return z ** 2

        x_input = Input(shape=(784,))
        x = Lambda(fun)(x_input)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 784))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 784))

    def test_Lambda_layer_2(self):
        def fun(args):
            z_1, z_2 = args
            return z_1 + K.exp(z_2)

        x_input = Input(shape=(784,))
        x_1 = Dense(10)(x_input)
        x_2 = Dense(10)(x_input)
        x = Lambda(fun)([x_1, x_2])
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 784))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 10))

    def test_Embedding_layer(self):
        model = Sequential()
        model.add(Embedding(1000, 64, input_length=10))
        self.assertEqual(model.get_layer(index=0).input_dim, 1000)
        self.assertEqual(model.get_layer(index=0).output_dim, 64)
        self.assertEqual(model.get_layer(index=0).embeddings.shape, (1000, 64))
        self.assertEqual(model.get_layer(index=0).input_shape, (None, 10))
        self.assertEqual(model.get_layer(index=0).output_shape, (None, 10, 64))

    # Convolutional Layers
    def test_Conv1D_layer(self):
        model = Sequential()
        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(10, 64,)))
        self.assertEqual(model.get_layer(index=0).filters, 32)
        self.assertEqual(model.get_layer(index=0).kernel_size, (3,))
        self.assertEqual(model.get_layer(index=0).activation.__name__, 'relu')
        self.assertEqual(model.get_layer(index=0).input_shape, (None, 10, 64))
        self.assertEqual(model.get_layer(index=0).output_shape, (None, 8, 32))

    def test_Conv2D_layer(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
        self.assertEqual(model.get_layer(index=0).filters, 32)
        self.assertEqual(model.get_layer(index=0).kernel_size, (3, 3))
        self.assertEqual(model.get_layer(index=0).activation.__name__, 'relu')
        self.assertEqual(model.get_layer(index=0).input_shape, (None, 28, 28, 1))
        self.assertEqual(model.get_layer(index=0).output_shape, (None, 26, 26, 32))

    def test_Conv3D_layer(self):
        model = Sequential()
        model.add(Conv3D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 10, 1)))
        self.assertEqual(model.get_layer(index=0).filters, 32)
        self.assertEqual(model.get_layer(index=0).kernel_size, (3, 3, 3))
        self.assertEqual(model.get_layer(index=0).activation.__name__, 'relu')
        self.assertEqual(model.get_layer(index=0).input_shape, (None, 28, 28, 10, 1))
        self.assertEqual(model.get_layer(index=0).output_shape, (None, 26, 26, 8, 32))

    def test_Conv2DTranspose_layer(self):
        x_input = Input(shape=(28, 28, 1))
        x = Conv2D(1, kernel_size=3, activation='relu')(x_input)
        x = Conv2DTranspose(1, kernel_size=3, activation='relu')(x)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=0).input_shape, (None, 28, 28, 1))
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 28, 28, 1))
        self.assertEqual(model.get_layer(index=2).input_shape, (None, 26, 26, 1))
        self.assertEqual(model.get_layer(index=2).output_shape, (None, 28, 28, 1))

    def test_Conv3DTranspose_layer(self):
        x_input = Input(shape=(28, 28, 10, 1))
        x = Conv3D(1, kernel_size=3, activation='relu')(x_input)
        x = Conv3DTranspose(1, kernel_size=3, activation='relu')(x)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=0).input_shape, (None, 28, 28, 10, 1))
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 28, 28, 10, 1))
        self.assertEqual(model.get_layer(index=2).input_shape, (None, 26, 26, 8, 1))
        self.assertEqual(model.get_layer(index=2).output_shape, (None, 28, 28, 10, 1))

    # Recurrent Layers
    def test_LSTM_layer(self):
        model = Sequential()
        model.add(Embedding(1000, 64, input_length=10))
        model.add(LSTM(128))
        self.assertEqual(model.get_layer(index=1).units, 128)
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 10, 64))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 128))
        model.pop()
        model.add(LSTM(128, return_sequences=True))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 10, 128))

    def test_GRU_layer(self):
        model = Sequential()
        model.add(Embedding(1000, 64, input_length=10))
        model.add(GRU(128))
        self.assertEqual(model.get_layer(index=1).units, 128)
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 10, 64))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 128))
        model.pop()
        model.add(GRU(128, return_sequences=True))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 10, 128))

    def test_SimpleRNN_layer(self):
        x_input = Input(shape=(10,))
        x = Embedding(1000, 64)(x_input)
        x = GRU(128, return_sequences=True)(x)
        x = SimpleRNN(64)(x)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=3).units, 64)
        self.assertEqual(model.get_layer(index=3).input_shape, (None, 10, 128))
        self.assertEqual(model.get_layer(index=3).output_shape, (None, 64))

    def test_Bidirectional_wrapper(self):
        model = Sequential()
        model.add(Embedding(1000, 64, input_length=10))
        model.add(Bidirectional(LSTM(128)))
        self.assertEqual(model.get_layer(index=1).forward_layer.units, 128)
        self.assertEqual(model.get_layer(index=1).backward_layer.units, 128)
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 10, 64))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 256))
        model.pop()
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 10, 256))

    # Reshaping layers
    def test_Reshape_layer(self):
        x_input = Input(shape=(784,))
        x = Reshape((28, 28, 1))(x_input)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 784))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 28, 28, 1))

    def test_Flatten_layer(self):
        x_input = Input(shape=(28, 28, 1))
        x = Flatten()(x_input)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 28, 28, 1))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 784))

    def test_UpSampling1D_layer(self):
        x_input = Input(shape=(28,))
        x = UpSampling1D(size=2)(x_input)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=1).size, (2,))
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 28))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 56))

    def test_UpSampling2D_layer(self):
        x_input = Input(shape=(28, 28, 1))
        x = UpSampling2D(size=(2, 2))(x_input)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=1).size, (2, 2))
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 28, 28, 1))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 56, 56, 1))

    def test_UpSampling3D_laye(self):
        x_input = Input(shape=(28, 28, 10, 1))
        x = UpSampling3D(size=(2, 2, 2))(x_input)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.get_layer(index=1).size, (2, 2, 2))
        self.assertEqual(model.get_layer(index=1).input_shape, (None, 28, 28, 10, 1))
        self.assertEqual(model.get_layer(index=1).output_shape, (None, 56, 56, 20, 1))


if __name__ == '__main__':
    main()
