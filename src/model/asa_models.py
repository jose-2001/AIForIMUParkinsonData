from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Conv1D, Dropout

DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_LOSS_FUNCTION = 'binary_crossentropy'
DEFAULT_ACCURACY_FUNCTION = 'sigmoid'

INPUT_SHAPE = (16723, 24)


def build_basic_lstm() -> Sequential:
    model = Sequential([
        LSTM(64, input_shape=INPUT_SHAPE),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=DEFAULT_OPTIMIZER,
                  metrics=['accuracy'])

    return model


def build_2layer_lstm() -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=DEFAULT_OPTIMIZER,
                  metrics=['accuracy'])

    return model


def build_1d_conv_2layer_lstm() -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add()
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=DEFAULT_OPTIMIZER,
                  metrics=['accuracy'])

    return model


def build_1d_conv_2layer_lstm_do() -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add()
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=DEFAULT_OPTIMIZER,
                  metrics=['accuracy'])

    return model
