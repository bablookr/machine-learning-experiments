from sklearn.datasets import load_iris
import numpy as np
import unittest


class TestSklearn(unittest.TestCase):
    def test_load(self):
        dataset = load_iris()
        self.assertEqual(dataset.data.shape, (150, 4))
        self.assertEqual(dataset.target.shape, (150,))
        self.assertListEqual(dataset.feature_names, ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    def test_load_as_frame(self):
        dataset = load_iris(as_frame=True)
        df = dataset.frame
        self.assertEqual(df.shape, (150, 5))
        np.testing.assert_array_equal(df.columns.values, np.array(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']))

    def test_load_X_y(self):
        x_train, y_train = load_iris(return_X_y=True)
        self.assertEqual(x_train.shape, (150, 4))
        self.assertEqual(y_train.shape, (150,))

if __name__ == '__main__':
    unittest.main()



