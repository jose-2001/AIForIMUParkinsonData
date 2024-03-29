from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Conv1D, Dropout
from tensorflow.keras.optimizers import Adam

DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_LOSS_FUNCTION = 'binary_crossentropy'
DEFAULT_ACTIVATION_FUNCTION = 'sigmoid'

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


def build_2layer_lstm(hp) -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(LSTM(128, return_sequences=True))
    hp_units = hp.Int('units_hp', min_value=32, max_value=256, step=32)
    model.add(LSTM(hp_units, return_sequences=True))
    model.add(Dense(units=1, activation=DEFAULT_ACTIVATION_FUNCTION))

    opt = Adam(learning_rate=DEFAULT_LEARNING_RATE)
    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def build_1d_conv_2layer_lstm(hp) -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))

    hp_filters = hp.Int('filters_hp', min_value=5, max_value=32)
    hp_kernel = hp.Int('kernel_hp', min_value=8, max_value=64, step=8)
    model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel, strides=1, padding='causal', activation='relu'))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(units=1, activation='sigmoid'))

    opt = Adam(learning_rate=DEFAULT_LEARNING_RATE)
    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def build_1d_conv_2layer_lstm_do(hp) -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    # TODO: Add neurons according to results of previous models
    model.add(model.add(Conv1D(filters=13, kernel_size=5, strides=1, padding='causal', activation='relu')))
    model.add(LSTM(128, return_sequences=True))

    hp_do1 = hp.Float('hp_do1', min_value=0.2, max_value=0.8, step=0.05)
    model.add(Dropout(rate=hp_do1))
    model.add(LSTM(128, return_sequences=True))

    hp_do2 = hp.Float('hp_do2', min_value=0.2, max_value=0.8, step=0.05)
    model.add(Dropout(rate=hp_do2))
    model.add(Dense(units=1, activation='sigmoid'))

    opt = Adam(learning_rate=DEFAULT_LEARNING_RATE)
    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
