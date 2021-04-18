import os
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import adam
from keras.preprocessing.sequence import pad_sequences

batch_size = 32
epochs = 2

vocab_size = 20000
embedding_size = 128
input_length = 100


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=input_length)
    x_test = pad_sequences(x_test, maxlen=input_length)
    return (x_train, y_train), (x_test, y_test)


def create_model():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=input_length))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


(X_train, Y_train), (X_test, Y_test) = load_and_preprocess_data()
lstm_model = create_model()

lstm_model.compile(loss=binary_crossentropy(),
                   optimizer=adam,
                   metrics=['accuracy'])

lstm_model.fit(X_train, Y_train,
               batch_size=batch_size,
               epochs=epochs,
               validation_data=(X_test, Y_test))

scores = lstm_model.evaluate(X_test, Y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model_path = os.path.join(os.getcwd(), 'saved_models\\bidirectional_lstm_imdb')
lstm_model.save(model_path)
