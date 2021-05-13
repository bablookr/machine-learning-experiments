import torch
from torch.nn import Module, Flatten, Sequential, Linear, ReLU
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

batch_size = 64
epochs = 5

input_shape = 784
num_categories = 10

learning_rate = 0.001
log_interval = 100

training_data = FashionMNIST(root='data',
                             train=True,
                             download=True,
                             transform=ToTensor())

test_data = FashionMNIST(root='data',
                         train=False,
                         download=True,
                         transform=ToTensor())

train_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

train_size = len(train_loader.dataset)
test_size = len(test_loader.dataset)


class ANN(Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = Flatten()
        self.sequential = Sequential(
            Linear(in_features=input_shape, out_features=512), ReLU(),
            Linear(in_features=512, out_features=512), ReLU(),
            Linear(in_features=512, out_features=10), ReLU()
        )

    def forward(self, x_input):
        x = self.flatten(x_input)
        x = self.sequential(x)
        return x


model = ANN().to(device)
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for i, (X, Y) in enumerate(train_loader):
        X = X.reshape(-1, input_shape).to(device)
        Y = Y.to(device)

        pred = model(X)
        loss = loss_fn(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        if (i + 1) % log_interval == 0:
            print('Epoch {}/{}, Loss: {:.4f} [{}/{}]'
                  .format(epoch + 1, epochs, train_loss, (i + 1) * batch_size, train_size))

model.eval()
with torch.no_grad():
    test_loss = 0
    correct = 0
    for X, Y in test_loader:
        X = X.reshape(-1, input_shape).to(device)
        Y = Y.to(device)

        pred = model(X)
        test_loss += loss_fn(pred, Y).item()
        correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

    print('Accuracy: {:.4f}, Avg Loss:{:.4f}'
          .format(correct / test_size, test_loss / test_size))

# torch.save(model.state_dict(), "mnist_ann.pt")
