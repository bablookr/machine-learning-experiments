import torch
import numpy as np
from unittest import TestCase
from unittest import main


class TestTorch(TestCase):
    def test_randn(self):
        torch.manual_seed(0)
        x_1 = torch.randn(3)
        np.testing.assert_allclose(x_1.numpy(), np.array([ 1.5409961, -0.2934289, -2.1787894]))
        x_2 = torch.randn(2,2)
        np.testing.assert_allclose(x_2.numpy(), np.array([[0.56843126, -1.0845224], [-1.3985955, 0.40334684]]))

    def test_from_numpy(self):
        x_1 = np.array([ 1.5409961, -0.2934289, -2.1787894])
        tensor_1 = torch.from_numpy(x_1)
        np.testing.assert_allclose(tensor_1.numpy(), x_1)
        x_2 = np.array([[0.56843126, -1.0845224], [-1.3985955, 0.40334684]])
        tensor_2 = torch.from_numpy(x_2)
        np.testing.assert_allclose(tensor_2.numpy(), x_2)

    def test_tensor(self):
        x_1 = [1.5409961, -0.2934289, -2.1787894]
        tensor_1 = torch.tensor(x_1)
        np.testing.assert_allclose(tensor_1.numpy(), np.array(x_1))
        x_2 = [[0.56843126, -1.0845224], [-1.3985955, 0.40334684]]
        tensor_2 = torch.tensor(x_2)
        np.testing.assert_allclose(tensor_2.numpy(), np.array(x_2))

    def test_grad(self):
        w = torch.tensor(1., requires_grad=True)
        x = torch.tensor(3., requires_grad=True)
        y = torch.square(w*x+1)
        self.assertEqual(y, torch.tensor(16.))
        y.backward()
        self.assertEqual(w.grad, torch.tensor(24.))
        self.assertEqual(x.grad, torch.tensor(8.))

    def test_grad_multi(self):
        w = torch.tensor([2., 1.], requires_grad=True)
        x = torch.tensor([[3.], [4.]], requires_grad=True)

        y = torch.square(w*x+1)
        print(y)
        y.backward()
        print(w.grad)
        print(x.grad)

    def test(self):
        a = torch.tensor([2., 3.], requires_grad=True)
        b = torch.tensor([6., 4.], requires_grad=True)

        Q = 3 * a ** 3 - b ** 2
        Q.backward(gradient=torch.tensor([1., 1.]))

        print(9 * a ** 2 == a.grad)
        print(-2 * b == b.grad)

if __name__ == '__main__':
    main()