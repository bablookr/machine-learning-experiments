import tensorflow as tf
import numpy as np
import unittest

from tensorflow import debugging


class TestCore(unittest.TestCase):
    def test_constants(self):
        int_constant_1 = tf.constant(1)
        self.assertEqual(int_constant_1.dtype, 'int32')
        self.assertEqual(int_constant_1.shape, ())

        int_constant_2 = tf.constant(0, dtype=tf.int64, shape=(2, 3))
        self.assertEqual(int_constant_2.dtype, 'int64')
        self.assertEqual(int_constant_2.shape, (2, 3))

        float_constant_1 = tf.constant(1.)
        self.assertEqual(float_constant_1.dtype, 'float32')
        self.assertEqual(float_constant_1.shape, ())

        float_constant_2 = tf.constant(1.0, dtype=tf.float64, shape=(6,))
        self.assertEqual(float_constant_2.dtype, 'float64')
        self.assertEqual(float_constant_2.shape, (6,))

        list_constant_1 = tf.constant([1, 2, 3, 4, 5, 6])
        self.assertEqual(list_constant_1.dtype, 'int32')
        self.assertEqual(list_constant_1.shape, (6,))

        list_constant_2 = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32, shape=(2, 3))
        self.assertEqual(list_constant_2.dtype, 'float32')
        self.assertEqual(list_constant_2.shape, (2, 3))

        array_constant_1 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(array_constant_1.dtype, 'int32')
        self.assertEqual(array_constant_1.shape, (2, 3))

        array_constant_2 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.float64, shape=(6,))
        self.assertEqual(array_constant_2.dtype, 'float64')
        self.assertEqual(array_constant_2.shape, (6,))

    def test_zeros(self):
        pass

    def test_ones(self):
        pass

    def test_one_hot(self):
        pass

    def test_convert_to_tensor(self):
        pass

    def test_gradient(self):
        pass

if __name__ == '__main__':
    unittest.main()
