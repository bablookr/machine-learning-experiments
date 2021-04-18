import tensorflow as tf
import unittest

batch_size = 1
channels = 1

kernel_size = 3
pool_size = 2

initializer = tf.initializers.glorot_normal()


class Testnn(unittest.TestCase):
    def test_conv2d(self):
        input_shape = [batch_size, 28, 28, channels]
        kernel_shape = [kernel_size, kernel_size, channels, 32]

        x_input = tf.Variable(initializer(input_shape))
        kernel = tf.Variable(initializer(kernel_shape))

        x = tf.nn.conv2d(x_input, kernel, strides=[1, 1], padding='VALID')
        self.assertEqual(x.shape, (batch_size, 26, 26, 32))

    def test_max_pool2d(self):
        input_shape = [batch_size, 28, 28, channels]

        x_input = tf.Variable(initializer(input_shape))
        x = tf.nn.max_pool2d(x_input, ksize=[pool_size, pool_size], strides=[pool_size, pool_size], padding='VALID')
        self.assertEqual(x.shape, (batch_size, 14, 14, channels))


if __name__ == '__main__':
    unittest.main()
