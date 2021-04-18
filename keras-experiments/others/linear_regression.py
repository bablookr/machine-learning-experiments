from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mse
from keras.optimizers import sgd
from keras.callbacks import EarlyStopping

epochs = 1000
num_features = 10


def load_and_preprocess_data():
    x, y = load_diabetes(return_X_y=True)
    x_train, y_train = x[:-20], y[:-20]
    x_test, y_test = x[-20:], y[-20:]
    return (x_train, y_train), (x_test, y_test)


def run_keras_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(1, input_shape=(num_features,)))
    model.compile(sgd(),
                  loss=mse)

    es = EarlyStopping(mode='min', patience=50, verbose=1)
    model.fit(x_train, y_train,
              batch_size= 422,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[es])

    return  model.evaluate(x_test, y_test)


def run_sklearn_model(x_train, y_train, x_test, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return mean_squared_error(y_test, model.predict(x_test))


(X_train, Y_train), (X_test, Y_test) = load_and_preprocess_data()
sklearn_mse = run_sklearn_model(X_train, Y_train, X_test, Y_test)
keras_mse = run_keras_model(X_train, Y_train, X_test, Y_test)
print('sklearn_mse=', sklearn_mse)
print('keras_mse=', keras_mse)
