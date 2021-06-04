import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from datetime import datetime


def get_call_backs(model_name='mobile_v2.h5', reduce_lr_p=10, early_stopping_p=25, log_dir='logs/scalars',
                   csv_dir='logs/csvs'):

    file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = log_dir + file_name
    tb_callback = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True,
                                   save_weights_only=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_p, verbose=1)

    reduce_lr = ReduceLROnPlateau(patience=reduce_lr_p, factor=0.2, cooldown=0, verbose=1)

    os.makedirs(csv_dir, exist_ok=True)
    csv_logger = CSVLogger(
        csv_dir + file_name, separator=',', append=False
    )

    return reduce_lr, tb_callback, checkpointer, early_stopping, csv_logger
