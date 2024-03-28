from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Conv1D, Dropout
from tensorflow.keras.optimizers import Adam

DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_LOSS_FUNCTION = 'binary_crossentropy'
DEFAULT_ACCURACY_FUNCTION = 'sigmoid'

INPUT_SHAPE = (16723, 24)


def build_basic_lstm(hp) -> Sequential:
    hp_units = hp.Int('units_hp', min_value=32, max_value=128, step=32)
    model = Sequential([
        LSTM(hp_units, input_shape=INPUT_SHAPE),
        Dense(1, activation='sigmoid')
    ])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3,1e-4])
    opt = Adam(learning_rate=hp_learning_rate)
    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
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
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='causal', activation='relu'))
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
