from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Conv1D
from tensorflow.keras.optimizers import Adam

DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LOSS_FUNCTION = 'binary_crossentropy'
DEFAULT_ACTIVATION_FUNCTION = 'sigmoid'

LEARNING_RATE = 1e-2
LSTM_UNITS = 128
DROPOUT_RATE = 0.6
KERNEL = 8
FILTERS = 9

INPUT_SHAPE = (16723, 24)


def build_basic_lstm() -> Sequential:
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=INPUT_SHAPE),
        Dense(1, activation='sigmoid')
    ])

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def build_2layer_lstm() -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(LSTM(LSTM_UNITS, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dense(units=1, activation=DEFAULT_ACTIVATION_FUNCTION))

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def build_1d_conv_1lstm() -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL, strides=1, padding='causal', activation='relu'))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(units=1, activation='sigmoid'))

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def build_1d_conv_1layer_lstm_do() -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(Conv1D(filters=9, kernel_size=8, strides=1, padding='causal', activation='relu'))
    model.add(LSTM(128, return_sequences=True, dropout=DROPOUT_RATE))
    model.add(Dense(units=1, activation='sigmoid'))

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
