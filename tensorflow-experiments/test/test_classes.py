import tensorflow as tf
from tensorflow import AggregationMethod, Assert, CriticalSection, DeviceSpec, DType
from tensorflow import GradientTape, Graph, IndexedSlices, IndexedSlicesSpec
from tensorflow import Module, Operation, OptionalSpec
from tensorflow import RegisterGradient, RaggedTensor, RaggedTensorSpec, SparseTensor, SparseTensorSpec, TensorShape, Tensor, TensorArray, TensorArraySpec, TensorSpec, TypeSpec
from tensorflow import UnconnectedGradients, Variable, VariableAggregation, VariableSynchronization
import unittest


class TestClasses(unittest.TestCase):
    def test_Tensor(self):
        pass

    def test_Variable(self):
        int_variable = tf.Variable(0)
        int_variable.assign(1)
        self.assertEqual(int_variable.value(), 1)

        array_variable = tf.Variable(np.array([1, 2, 3, 4, 5, 6]), dtype=tf.float32)
        array_variable.assign(np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]))
        # self.assertEqual(array_variable.value(), np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]))

        list_variable = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.float32, shape=tf.TensorShape(None))
        list_variable.assign([1.0])
        self.assertEqual(list_variable.value(), [1.0])

    def test_GradientTape(self):
        w = tf.Variable(1.)
        x = tf.constant(3.)

        with tf.GradientTape() as tape:
            y = tf.math.square(w*x + 1)

        dw = tape.gradient(y, w)
        self.assertEqual(dw, 24.0)

    def test_Module(self):
        class MyModule(tf.Module):
            def __init__(self, in_features, out_features, name=None):
                super().__init__(name=name)
                self.w = tf.Variable(tf.random.normal([in_features, out_features]))
                self.b = tf.Variable(tf.zeros([out_features]))

            def __call__(self, x):
                y = tf.matmul(x, self.w) + self.b
                return tf.nn.relu(y)

        module = MyModule(in_features=3, out_features=2)
        input = tf.ones(shape=[1, 3])
        output = module(input)
        self.assertEqual(output.shape, (1, 2))


if __name__ == '__main__':
    unittest.main()
