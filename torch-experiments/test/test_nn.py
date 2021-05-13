import torch
from torch.nn import Conv1d, Conv2d, Conv3d
from unittest import TestCase
from unittest import main

m = 10000


class TestNN(TestCase):
    def test_Conv1d(self):
        model = Conv1d(64, 32, kernel_size=3)
        x_input = torch.randn(m, 64, 10)
        x = model(x_input)
        self.assertEqual(x.shape, (m, 32, 8))

    def test_Conv2d(self):
        model = Conv2d(1, 32, kernel_size=3)
        x_input = torch.randn(m, 1, 28, 28)
        x = model(x_input)
        self.assertEqual(x.shape, (m, 32, 26, 26))

    def test_Conv3d(self):
        model = Conv3d(1, 32, kernel_size=3)
        x_input = torch.randn(m, 1, 28, 28, 10)
        x = model(x_input)
        self.assertEqual(x.shape, (m, 32, 26, 26, 8))


if __name__ == '__main__':
    main()