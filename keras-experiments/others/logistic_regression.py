from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.optimizers import sgd
from keras.callbacks import EarlyStopping

epochs = 1000
num_features = 4

def load_and_preprocess_data():
    x, y = load_iris(return_X_y=True)
    x_train, y_train = x[:-20], y[:-20]
    x_test, y_test = x[-20:], y[-20:]
    return (x_train, y_train), (x_test, y_test)


def run_keras_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_shape=(num_features,)))
    model.compile(sgd(),
                  loss=binary_crossentropy,
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=50)
    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[es])

    return model.evaluate(x_test, y_test)[1]

def run_sklearn_model(x_train, y_train, x_test, y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return log_loss(y_test, model.predict(x_test))


(X_train, Y_train), (X_test, Y_test) = load_and_preprocess_data()
sklearn_accuracy = run_sklearn_model(X_train, Y_train, X_test, Y_test)
keras_accuracy = run_keras_model(X_train, Y_train, X_test, Y_test)
print('sklearn_accuracy=', sklearn_accuracy)
print('keras_accuracy=', keras_accuracy)
