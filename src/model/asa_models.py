import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer

DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 10
DEFAULT_LOSS_FUNCTION = 'mse'
DEFAULT_ACCURACY_FUNCTION = 'sigmoid'


def build_basic_lstm(neurons: int) -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=(None, 24)))
    model.add(LSTM(neurons, input_shape=(24,), return_sequences=True))
    model.add(LSTM(neurons, return_sequences=True))
    model.add(Dense(units=1))

    model.compile(loss=DEFAULT_LOSS_FUNCTION,
                  optimizer=DEFAULT_OPTIMIZER,
                  metrics=['accuracy'])

    return model
