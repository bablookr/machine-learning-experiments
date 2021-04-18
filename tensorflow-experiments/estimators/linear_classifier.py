from sklearn.datasets import load_diabetes

epochs = 1000
num_features = 10


def load_and_preprocess_data():
    x, y = load_diabetes(return_X_y=True)
    x_train, y_train = x[:-20], y[:-20]
    x_test, y_test = x[-20:], y[-20:]
    return (x_train, y_train), (x_test, y_test)


def create_model():
    pass


(X_train, Y_train), (X_test, Y_test) = load_and_preprocess_data()