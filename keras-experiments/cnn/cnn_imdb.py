import os
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import adam
from keras.preprocessing.sequence import pad_sequences

batch_size = 32
epochs = 2

vocab_size = 5000
embedding_size = 50
input_length = 400
default_kernel_size = 3


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=input_length)
    x_test = pad_sequences(x_test, maxlen=input_length)
    return (x_train, y_train), (x_test, y_test)


def create_model():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=input_length))  # Embedding
    model.add(Dropout(0.2))
    model.add(Conv1D(250, kernel_size=default_kernel_size, padding='valid', activation='relu'))  # Convolutional
    model.add(GlobalMaxPooling1D())
    model.add(Dense(250, activation='relu'))  # Dense
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


(X_train, Y_train), (X_test, Y_test) = load_and_preprocess_data()

cnn_model = create_model()
cnn_model.summary()

cnn_model.compile(adam,
                  loss=binary_crossentropy,
                  metrics=['accuracy'])

cnn_model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, Y_test))

scores = cnn_model.evaluate(X_test, Y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model_path = os.path.join(os.getcwd(), 'saved_models\cnn_imdb')
cnn_model.save(model_path)
