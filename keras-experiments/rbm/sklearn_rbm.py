import numpy as np
from sklearn.neural_network import BernoulliRBM

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = BernoulliRBM(n_components=2, verbose=True, random_state=0)
model.fit(X)

print(model.transform(X))