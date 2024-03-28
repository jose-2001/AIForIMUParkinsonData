import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from src.settings import ROOT_DIR


def get_callbacks(module: str, model: str):
    checkpoint = build_model_checkpoint(module, model)
    early_stopping = build_early_stopping()
    return checkpoint, early_stopping


def build_model_checkpoint(module: str, model: str):
    checkpoint_filepath = str(ROOT_DIR) + f'/results/{module}/{model}/tmp/ckpt/checkpoint.model.keras'
    return ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )


def build_early_stopping():
    return EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights='True',
        min_delta=0.1
    )


def build_tensorboard(module: str):
    log_dir = str(ROOT_DIR) + f'/results/{module}/lightning_logs' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir , histogram_freq=1)
    return tensorboard_callback

