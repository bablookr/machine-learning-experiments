from unittest import TestCase
from unittest import main

from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import sgd

class TestModelsAPI(TestCase):
    def test_Model_class(self):
        x_input = Input(shape=(4,))
        x = Dense(2, activation='relu')(x_input)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=x_input, outputs=x)
        self.assertEqual(model.input_shape, (None, 4))
        self.assertEqual(model.output_shape, (None, 1))

    def test_Sequential_class(self):
        model = Sequential()
        model.add(Dense(2, activation='relu', input_shape=(4,)))
        model.add(Dense(1, activation='sigmoid'))
        self.assertEqual(model.input_shape, (None, 4))
        self.assertEqual(model.output_shape, (None, 1))

    def test_compile_method(self):
        model = Sequential()
        model.add(Dense(2, activation='relu', input_shape=(4,)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=binary_crossentropy,
                      optimizer=sgd(),
                      metrics=['accuracy'])
        self.assertEqual(model.loss_functions.__getitem__(0).name, 'binary_crossentropy')
        self.assertEqual(model.optimizer.__class__.__name__, 'SGD')
        self.assertEqual(model.metrics.__getitem__(0).name, 'accuracy')
        self.assertEqual(model.metrics_names, ['loss', 'accuracy'])

if __name__ == '__main__':
    main()
